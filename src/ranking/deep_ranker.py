"""
Deep Learning Ranking Model for Route Recommendation.

Implements a multi-stage ranking system:
1. LightGBM baseline (fast, interpretable)
2. Deep Interest Network (DIN) - sequence-based personalization
3. Multi-task learning (CTR + CVR + dwell_time)

Compatible with RecBole user events and GoAfar POI data.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Feature Engineering
# ============================================================================

@dataclass
class FeatureConfig:
    """Feature configuration for ranking model."""

    # User features
    user_history_length: int = 10  # Number of recent POIs to consider
    user_preference_dim: int = 64  # Embedding dimension for user preference

    # POI features
    poi_category_embed_dim: int = 16
    poi_geo_bins: int = 20  # Geographic discretization bins

    # Context features
    time_of_day_buckets: int = 24
    day_of_week_buckets: int = 7

    # Sequence features
    max_seq_length: int = 20
    seq_embedding_dim: int = 64


class FeatureStore:
    """
    Feature store for ranking models.

    Computes and caches features for:
    - User: historical behavior, preferences, demographics
    - POI: category, geography, popularity, time windows
    - Context: time, location constraints
    - Interaction: user-POI cross features
    """

    def __init__(
        self,
        poi_df: pd.DataFrame,
        events_df: pd.DataFrame,
        cache_dir: str = "outputs/features",
    ):
        self.poi_df = poi_df
        self.events_df = events_df
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Build indices
        self._build_indices()

        # Pre-compute static features
        self._precompute_features()

    def _build_indices(self):
        """Build lookup indices for fast feature access."""
        self.poi_index = self.poi_df["poi_id"].astype(str).tolist()
        self.poi_to_idx = {pid: i for i, pid in enumerate(self.poi_index)}

        self.user_index = self.events_df["user_id"].unique().tolist()
        self.user_to_idx = {uid: i for i, uid in enumerate(self.user_index)}

    def _precompute_features(self):
        """Pre-compute and cache static POI features."""
        cache_path = self.cache_dir / "poi_features.pkl"

        if cache_path.exists():
            with open(cache_path, "rb") as f:
                self.poi_features = pickle.load(f)
            logger.info(f"Loaded cached POI features from {cache_path}")
            return

        # Compute POI features
        self.poi_features = {
            "category": pd.Categorical(self.poi_df["category"]).codes,
            "province": pd.Categorical(self.poi_df["province"]).codes,
            "popularity": self.poi_df.get("visit_count", 0).values,
            "avg_stay": self.poi_df.get("stay_min", 60).values,
            "open_hour": self.poi_df["open_min"].values // 60,
            "close_hour": self.poi_df["close_min"].values // 60,
        }

        with open(cache_path, "wb") as f:
            pickle.dump(self.poi_features, f)
        logger.info(f"Cached POI features to {cache_path}")

    def get_user_features(
        self, user_id: str, timestamp: Optional[int] = None
    ) -> Dict[str, Any]:
        """Extract user features."""
        user_events = self.events_df[
            self.events_df["user_id"] == user_id
        ].sort_values("timestamp", ascending=False)

        features = {
            "user_id": user_id,
            "history_length": len(user_events),
            "last_action": user_events.iloc[0]["event_type"] if len(user_events) > 0 else "none",
            "action_counts": user_events["event_type"].value_counts().to_dict(),
        }

        # Recent POI sequence
        recent_pois = user_events.head(10)["poi_id"].tolist()
        features["recent_pois"] = recent_pois

        # Category preferences
        recent_categories = self.poi_df.set_index("poi_id").loc[
            recent_pois, "category"
        ].values if recent_pois else []
        features["category_pref"] = pd.Series(recent_categories).value_counts().to_dict()

        return features

    def get_poi_features(self, poi_id: str) -> Dict[str, Any]:
        """Extract POI features."""
        if str(poi_id) not in self.poi_to_idx:
            return {}

        idx = self.poi_to_idx[str(poi_id)]
        return {
            "poi_id": str(poi_id),
            "category": self.poi_features["category"][idx],
            "province": self.poi_features["province"][idx],
            "popularity": self.poi_features["popularity"][idx],
            "avg_stay": self.poi_features["avg_stay"][idx],
            "open_hour": self.poi_features["open_hour"][idx],
            "close_hour": self.poi_features["close_hour"][idx],
        }

    def get_interaction_features(
        self, user_id: str, poi_id: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract user-POI interaction features."""
        user_features = self.get_user_features(user_id)
        poi_features = self.get_poi_features(poi_id)

        # Cross features
        features = {
            # Recency: last interaction with this POI
            "is_repeat": str(poi_id) in user_features.get("recent_pois", []),
            "repeat_position": (
                user_features["recent_pois"].index(str(poi_id))
                if str(poi_id) in user_features.get("recent_pois", [])
                else -1
            ),
            # Category match
            "category_preference_score": user_features.get("category_pref", {}).get(
                poi_features.get("category"), 0
            ),
        }

        # Time window feasibility
        if "hour" in context:
            hour = context["hour"]
            open_h = poi_features.get("open_hour", 0)
            close_h = poi_features.get("close_hour", 24)
            features["time_feasible"] = open_h <= hour <= close_h

        return features

    def build_ranking_features(
        self,
        user_id: str,
        candidate_pois: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Build feature matrix for ranking.

        Args:
            user_id: User identifier
            candidate_pois: List of candidate POI IDs
            context: Additional context (time, location, etc.)

        Returns:
            DataFrame with features for each candidate POI
        """
        context = context or {}
        features_list = []

        for poi_id in candidate_pois:
            poi_feat = self.get_poi_features(poi_id)
            inter_feat = self.get_interaction_features(user_id, poi_id, context)

            row = {
                "user_id": user_id,
                "poi_id": poi_id,
                **poi_feat,
                **inter_feat,
                **context,
            }
            features_list.append(row)

        return pd.DataFrame(features_list)


# ============================================================================
# LightGBM Ranker (Baseline)
# ============================================================================

class LightGBMRanker:
    """
    LightGBM-based ranking model.

    Fast, interpretable baseline for POI ranking.
    Supports learning-to-rank with LambdaRank.
    """

    def __init__(
        self,
        objective: str = "lambdarank",
        num_leaves: int = 64,
        learning_rate: float = 0.05,
        feature_fraction: float = 0.8,
        verbose: int = -1,
    ):
        self.params = {
            "objective": objective,
            "metric": "ndcg",
            "num_leaves": num_leaves,
            "learning_rate": learning_rate,
            "feature_fraction": feature_fraction,
            "verbose": verbose,
        }
        self.model = None
        self.feature_names = None

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        group_sizes: np.ndarray,
        query_ids: Optional[List[str]] = None,
        valid_data: Optional[Tuple[pd.DataFrame, np.ndarray, np.ndarray]] = None,
    ) -> None:
        """
        Train the ranking model.

        Args:
            X: Feature matrix
            y: Relevance labels (higher = more relevant)
            group_sizes: Number of candidates per query
            query_ids: Optional query identifiers
            valid_data: Optional validation set
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm is required. Install with: pip install lightgbm")

        # Create dataset
        train_data = lgb.Dataset(X, label=y, group=group_sizes)

        # Validation data
        valid_sets = [train_data]
        valid_names = ["train"]
        if valid_data is not None:
            valid_X, valid_y, valid_groups = valid_data
            valid_dataset = lgb.Dataset(valid_X, label=valid_y, group=valid_groups)
            valid_sets.append(valid_dataset)
            valid_names.append("valid")

        # Train
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=1000,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(100),
            ],
        )
        self.feature_names = self.model.feature_name()

        logger.info(f"Training completed. Best iteration: {self.model.best_iteration}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict relevance scores for candidates."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        return self.model.predict(X)

    def save_model(self, path: str) -> None:
        """Save model to file."""
        if self.model is None:
            raise RuntimeError("No model to save")

        self.model.booster_.save_model(path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """Load model from file."""
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm is required")

        self.model = lgb.Booster(model_file=path)
        self.feature_names = self.model.feature_name()
        logger.info(f"Model loaded from {path}")

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importance scores."""
        if self.model is None:
            raise RuntimeError("Model not trained")

        importance = self.model.feature_importance()
        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False)


# ============================================================================
# Deep Interest Network (Advanced)
# ============================================================================

class DeepInterestNetwork:
    """
    Deep Interest Network for POI recommendation.

    References:
        - Zhou et al. "Deep Interest Network for Click-Through Rate Prediction" (KDD 2018)
        - Uses attention mechanism over user history
        - Captures temporal dynamics and user interest diversity

    Note: PyTorch required. Falls back to feature extraction if unavailable.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        hidden_dims: List[int] = (256, 128, 64),
        attention_heads: int = 4,
        dropout: float = 0.1,
        num_pois: int = 10000,
        num_categories: int = 50,
        num_provinces: int = 50,
    ):
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.attention_heads = attention_heads
        self.dropout = dropout
        self.num_pois = num_pois
        self.num_categories = num_categories
        self.num_provinces = num_provinces
        self.seq_length = 20  # Default sequence length
        self.model = None

        # Check PyTorch availability
        try:
            import torch
            import torch.nn as nn
            self.torch_available = True
            self.nn = nn
            self.torch = torch
        except ImportError:
            self.torch_available = False
            logger.warning("PyTorch not available. DIN model will use feature extraction only.")

    def _build_model(
        self,
        num_pois: int,
        num_categories: int,
        num_provinces: int,
        seq_length: int = 20,
    ) -> None:
        """Build the DIN model architecture."""
        if not self.torch_available:
            return

        class DINModel(self.nn.Module):
            def __init__(
                self,
                num_pois: int,
                num_categories: int,
                num_provinces: int,
                embedding_dim: int,
                hidden_dims: List[int],
                seq_length: int,
                attention_heads: int,
                dropout: float,
            ):
                super().__init__()
                self.seq_length = seq_length
                self.embedding_dim = embedding_dim

                # Embeddings
                self.poi_embedding = self.nn.Embedding(num_pois, embedding_dim, padding_idx=0)
                self.category_embedding = self.nn.Embedding(num_categories, embedding_dim // 2)
                self.province_embedding = self.nn.Embedding(num_provinces, embedding_dim // 4)

                # Attention over history
                self.attention = self.nn.MultiheadAttention(
                    embedding_dim, attention_heads, dropout=dropout, batch_first=True
                )

                # Deep feedforward layers
                input_dim = embedding_dim * 3  # POI + category + province
                layers = []
                prev_dim = input_dim
                for dim in hidden_dims:
                    layers.extend([
                        self.nn.Linear(prev_dim, dim),
                        self.nn.ReLU(),
                        self.nn.Dropout(dropout),
                    ])
                    prev_dim = dim
                layers.append(self.nn.Linear(prev_dim, 1))
                self.mlp = self.nn.Sequential(*layers)

            def forward(
                self,
                candidate_poi_ids: self.torch.Tensor,
                history_poi_ids: self.torch.Tensor,
                history_categories: self.torch.Tensor,
                history_provinces: self.torch.Tensor,
            ) -> self.torch.Tensor:
                batch_size = candidate_poi_ids.shape[0]

                # Candidate encoding
                cand_embed = self.poi_embedding(candidate_poi_ids)

                # History encoding
                hist_poi_embed = self.poi_embedding(history_poi_ids)
                hist_cat_embed = self.category_embedding(history_categories)
                hist_prov_embed = self.province_embedding(history_provinces)

                # Concatenate history embeddings
                hist_embed = self.torch.cat([
                    hist_poi_embed,
                    hist_cat_embed,
                    hist_prov_embed,
                ], dim=-1)  # (batch, seq_len, embed_dim*3)

                # Attention over history (query = candidate)
                attended, _ = self.attention(
                    query=cand_embed.unsqueeze(1),
                    key=hist_embed,
                    value=hist_embed,
                )
                attended = attended.squeeze(1)

                # Combine with candidate
                combined = self.torch.cat([cand_embed, attended], dim=-1)

                # MLP scoring
                score = self.mlp(combined).squeeze(-1)
                return score

        self.model = DINModel(
            num_pois=num_pois,
            num_categories=num_categories,
            num_provinces=num_provinces,
            embedding_dim=self.embedding_dim,
            hidden_dims=self.hidden_dims,
            seq_length=self.seq_length,
            attention_heads=self.attention_heads,
            dropout=self.dropout,
        )

        logger.info("DIN model architecture built")

    def build_from_data(self, poi_df: pd.DataFrame, events_df: pd.DataFrame) -> None:
        """
        Build DIN model from data by inferring vocab sizes.

        Args:
            poi_df: POI DataFrame with poi_id column
            events_df: Events DataFrame with user_id and poi_id columns
        """
        # Infer vocab sizes
        num_pois = poi_df["poi_id"].nunique() + 1  # +1 for padding
        num_categories = poi_df.get("category", pd.Series()).nunique() + 1
        num_provinces = poi_df.get("province", pd.Series()).nunique() + 1

        # Update config
        self.num_pois = num_pois
        self.num_categories = num_categories
        self.num_provinces = num_provinces

        # Build model
        self._build_model(num_pois, num_categories, num_provinces, seq_length=20)

        logger.info(f"DIN model built: {num_pois} POIs, {num_categories} categories, {num_provinces} provinces")

    def fit(
        self,
        train_data: pd.DataFrame,
        poi_vocab_size: int,
        category_vocab_size: int,
        province_vocab_size: int,
        epochs: int = 10,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Train the DIN model.

        Args:
            train_data: Training DataFrame with columns:
                - user_id, poi_id, category, province, label
                - history_poi_ids (list), history_categories (list), history_provinces (list)
            poi_vocab_size: Size of POI vocabulary
            category_vocab_size: Size of category vocabulary
            province_vocab_size: Size of province vocabulary
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            device: Device to train on ('cpu' or 'cuda')

        Returns:
            Training metrics dictionary
        """
        if not self.torch_available:
            logger.warning("Cannot train DIN: PyTorch not available")
            return {}

        # Build model
        self._build_model(
            poi_vocab_size,
            category_vocab_size,
            province_vocab_size,
            seq_length=20,
        )

        # Move to device
        device_obj = self.torch.device(device)
        self.model = self.model.to(device_obj)

        # Optimizer
        optimizer = self.torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = self.nn.BCEWithLogitsLoss()

        # Prepare training data
        # Group by user to create sequences
        user_groups = train_data.groupby("user_id")

        # Training loop
        self.model.train()
        total_loss = 0
        num_batches = 0

        for epoch in range(epochs):
            epoch_loss = 0

            for user_id, user_data in user_groups:
                # Get history for this user
                history_pois = user_data["history_poi_ids"].iloc[0] if "history_poi_ids" in user_data.columns else []
                history_categories = user_data["history_categories"].iloc[0] if "history_categories" in user_data.columns else []
                history_provinces = user_data["history_provinces"].iloc[0] if "history_provinces" in user_data.columns else []

                # Pad/truncate sequences
                max_len = 20
                if len(history_pois) > max_len:
                    history_pois = history_pois[-max_len:]
                    history_categories = history_categories[-max_len:]
                    history_provinces = history_provinces[-max_len:]

                # Convert to tensors
                for _, row in user_data.iterrows():
                    candidate_poi_id = self.torch.tensor([row["poi_id"]], dtype=self.torch.long).to(device_obj)

                    hist_poi_tensor = self.torch.tensor(history_pois + [0] * (max_len - len(history_pois)), dtype=self.torch.long).unsqueeze(0).to(device_obj)
                    hist_cat_tensor = self.torch.tensor(history_categories + [0] * (max_len - len(history_categories)), dtype=self.torch.long).unsqueeze(0).to(device_obj)
                    hist_prov_tensor = self.torch.tensor(history_provinces + [0] * (max_len - len(history_provinces)), dtype=self.torch.long).unsqueeze(0).to(device_obj)

                    # Forward pass
                    optimizer.zero_grad()
                    score = self.model(candidate_poi_id, hist_poi_tensor, hist_cat_tensor, hist_prov_tensor)

                    # Compute loss
                    label = self.torch.tensor([row.get("label", 1)], dtype=self.torch.float).to(device_obj)
                    loss = criterion(score, label)

                    # Backward pass
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    num_batches += 1

            avg_loss = epoch_loss / num_batches
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        logger.info(f"DIN training completed. Total batches: {num_batches}")
        return {"total_batches": num_batches, "final_loss": avg_loss}

    def predict(
        self,
        candidates: pd.DataFrame,
        user_history: Optional[Dict[str, List]] = None,
        device: str = "cpu",
    ) -> np.ndarray:
        """
        Predict relevance scores for candidates.

        Args:
            candidates: DataFrame with candidate POIs (poi_id, category, province columns)
            user_history: User history dict with keys:
                - history_poi_ids: List[int]
                - history_categories: List[int]
                - history_provinces: List[int]
            device: Device for inference

        Returns:
            Array of relevance scores
        """
        if not self.torch_available or self.model is None:
            # Fallback to simple scoring
            return candidates.get("popularity", 0).values

        device_obj = self.torch.device(device)
        self.model.eval()

        scores = []
        max_len = 20

        # Get user history
        if user_history is None:
            history_pois = []
            history_categories = []
            history_provinces = []
        else:
            history_pois = user_history.get("history_poi_ids", [])
            history_categories = user_history.get("history_categories", [])
            history_provinces = user_history.get("history_provinces", [])

        # Pad sequences
        if len(history_pois) > max_len:
            history_pois = history_pois[-max_len:]
            history_categories = history_categories[-max_len:]
            history_provinces = history_provinces[-max_len:]

        hist_poi_tensor = self.torch.tensor(
            history_pois + [0] * (max_len - len(history_pois)),
            dtype=self.torch.long
        ).unsqueeze(0).to(device_obj)
        hist_cat_tensor = self.torch.tensor(
            history_categories + [0] * (max_len - len(history_categories)),
            dtype=self.torch.long
        ).unsqueeze(0).to(device_obj)
        hist_prov_tensor = self.torch.tensor(
            history_provinces + [0] * (max_len - len(history_provinces)),
            dtype=self.torch.long
        ).unsqueeze(0).to(device_obj)

        with self.torch.no_grad():
            for _, row in candidates.iterrows():
                candidate_poi_id = self.torch.tensor([row["poi_id"]], dtype=self.torch.long).to(device_obj)

                score = self.model(candidate_poi_id, hist_poi_tensor, hist_cat_tensor, hist_prov_tensor)
                scores.append(score.sigmoid().cpu().item())

        return np.array(scores)

    def save_model(self, path: str) -> None:
        """Save model to file."""
        if not self.torch_available or self.model is None:
            raise RuntimeError("No model to save")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "config": {
                "embedding_dim": self.embedding_dim,
                "hidden_dims": self.hidden_dims,
                "attention_heads": self.attention_heads,
                "dropout": self.dropout,
            },
        }

        self.torch.save(checkpoint, path)
        logger.info(f"DIN model saved to {path}")

    def load_model(self, path: str, device: str = "cpu") -> None:
        """Load model from file."""
        if not self.torch_available:
            raise ImportError("PyTorch is required")

        checkpoint = self.torch.load(path, map_location=device)

        # Restore config
        self.embedding_dim = checkpoint["config"]["embedding_dim"]
        self.hidden_dims = checkpoint["config"]["hidden_dims"]
        self.attention_heads = checkpoint["config"]["attention_heads"]
        self.dropout = checkpoint["config"]["dropout"]

        # Rebuild model and load weights
        # Note: Need to call _build_model first with correct vocab sizes
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"DIN model loaded from {path}")


# ============================================================================
# Multi-Task Ranker (CTR + CVR + Dwell)
# ============================================================================

@dataclass
class MultiTaskConfig:
    """Configuration for multi-task learning."""

    # Tasks
    predict_ctr: bool = True  # Click-through rate
    predict_cvr: bool = True  # Conversion/visit rate
    predict_dwell: bool = True  # Dwell time

    # Shared tower
    shared_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])

    # Task towers
    ctr_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    cvr_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    dwell_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])

    # Loss weights
    ctr_weight: float = 1.0
    cvr_weight: float = 1.0
    dwell_weight: float = 0.5


class MultiTaskRanker:
    """
    Multi-task learning model for POI ranking.

    Simultaneously predicts:
    - CTR: Likelihood of user clicking/viewing
    - CVR: Likelihood of user visiting/booking
    - Dwell time: Expected stay duration

    Uses MMoE (Multi-gate Mixture-of-Experts) architecture.
    """

    def __init__(self, config: Optional[MultiTaskConfig] = None):
        self.config = config or MultiTaskConfig()
        self.model = None

    def fit(self, train_data: pd.DataFrame, epochs: int = 10) -> None:
        """Train the multi-task model."""
        # TODO: Implement MMoE training
        logger.info("Multi-task ranker not yet fully implemented")

    def predict(
        self, candidates: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """Predict all task scores."""
        if self.model is None:
            # Fallback to simple heuristics
            return {
                "ctr": candidates.get("popularity", 0).values / 100,
                "cvr": candidates.get("popularity", 0).values / 200,
                "dwell": candidates.get("avg_stay", 60).values,
            }
        return {"ctr": None, "cvr": None, "dwell": None}


# ============================================================================
# Unified Ranking Interface
# ============================================================================

class DeepRanker:
    """
    Unified ranking interface supporting multiple models.

    Usage:
        ranker = DeepRanker(model_type="lightgbm")
        ranker.fit(train_data, ...)
        scores = ranker.rank(user_id, candidate_pois, context)
    """

    MODEL_TYPES = ["lightgbm", "din", "multitask"]

    def __init__(
        self,
        model_type: str = "lightgbm",
        model_config: Optional[Dict] = None,
        feature_store: Optional[FeatureStore] = None,
    ):
        if model_type not in self.MODEL_TYPES:
            raise ValueError(f"model_type must be one of {self.MODEL_TYPES}")

        self.model_type = model_type
        self.model_config = model_config or {}
        self.feature_store = feature_store

        # Initialize model
        if model_type == "lightgbm":
            self.model = LightGBMRanker(**self.model_config)
        elif model_type == "din":
            self.model = DeepInterestNetwork(**self.model_config)
        elif model_type == "multitask":
            self.model = MultiTaskRanker(**self.model_config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def fit(
        self,
        train_data: pd.DataFrame,
        labels: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        """
        Train the ranking model.

        Args:
            train_data: Training DataFrame with features
            labels: Relevance labels (if not in train_data)
            **kwargs: Model-specific arguments
        """
        if self.model_type == "lightgbm":
            y = labels or train_data.get("label", 0).values
            groups = kwargs.get("groups", np.ones(len(train_data)))
            self.model.fit(train_data, y, groups, **kwargs)
        else:
            self.model.fit(train_data, **kwargs)

    def rank(
        self,
        user_id: str,
        candidate_pois: List[str],
        context: Optional[Dict[str, Any]] = None,
        top_k: int = 50,
    ) -> List[Tuple[str, float]]:
        """
        Rank candidate POIs for a user.

        Args:
            user_id: User identifier
            candidate_pois: List of candidate POI IDs
            context: Additional context
            top_k: Return top-K results

        Returns:
            List of (poi_id, score) tuples, sorted by score descending
        """
        if self.feature_store:
            features = self.feature_store.build_ranking_features(
                user_id, candidate_pois, context
            )
        else:
            # Minimal features
            features = pd.DataFrame({"poi_id": candidate_pois})

        # Get scores
        if self.model_type == "multitask":
            # Combine multi-task scores
            scores_dict = self.model.predict(features)
            scores = (
                scores_dict.get("ctr", 0) * 0.4 +
                scores_dict.get("cvr", 0) * 0.5 +
                scores_dict.get("dwell", 0) / 300 * 0.1
            )
        else:
            scores = self.model.predict(features)

        # Sort and return top-k
        sorted_idx = np.argsort(-scores)[:top_k]
        results = [
            (candidate_pois[i], float(scores[i]))
            for i in sorted_idx
        ]

        return results

    def save(self, path: str) -> None:
        """Save model to file."""
        if hasattr(self.model, "save_model"):
            self.model.save_model(path)
        else:
            import pickle
            with open(path, "wb") as f:
                pickle.dump(self.model, f)
        logger.info(f"Ranker saved to {path}")

    def load(self, path: str) -> None:
        """Load model from file."""
        if hasattr(self.model, "load_model"):
            self.model.load_model(path)
        else:
            import pickle
            with open(path, "rb") as f:
                self.model = pickle.load(f)
        logger.info(f"Ranker loaded from {path}")


def create_ranker(
    model_type: str = "lightgbm",
    poi_df: Optional[pd.DataFrame] = None,
    events_df: Optional[pd.DataFrame] = None,
    **config,
) -> DeepRanker:
    """
    Factory function to create a ranker with feature store.

    Args:
        model_type: Type of ranking model
        poi_df: POI data for feature store
        events_df: User events for feature store
        **config: Additional model configuration

    Returns:
        Configured DeepRanker instance
    """
    feature_store = None
    if poi_df is not None and events_df is not None:
        feature_store = FeatureStore(poi_df, events_df)

    return DeepRanker(
        model_type=model_type,
        model_config=config,
        feature_store=feature_store,
    )


# CLI entry point
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train deep ranking model")
    parser.add_argument("--model", choices=DeepRanker.MODEL_TYPES, default="lightgbm")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--poi-csv", default="data/all/poi_expanded.csv")
    parser.add_argument("--events-csv", default="data/all/user_events.csv")
    args = parser.parse_args()

    # Load data
    poi_df = pd.read_csv(args.poi_csv)
    events_df = pd.read_csv(args.events_csv)
    train_df = pd.read_csv(args.train_data)

    # Create and train
    ranker = create_ranker(args.model, poi_df, events_df)
    ranker.fit(train_df)
    ranker.save(args.output)

    logger.info(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
