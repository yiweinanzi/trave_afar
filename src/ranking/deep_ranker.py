"""
Multi-Task Deep Learning Ranking Model for POI Recommendation.

Implements MMoE (Multi-gate Mixture-of-Experts) architecture for:
- CTR Prediction: Click-through rate estimation
- Visit Prediction: Visit/conversion probability
- Duration Prediction: Expected stay time

Architecture:
    UserTower -> User Embedding
    ItemTower -> Item Embedding
    MMoE -> Expert Networks + Gating
    Task Heads -> CTR, Visit, Duration outputs

Compatible with GoAfar POI data and user events.
"""
from __future__ import annotations

import logging
import os
import pickle
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class MMoEConfig:
    """Configuration for MMoE multi-task model."""

    # Embedding dimensions
    user_embed_dim: int = 64
    item_embed_dim: int = 64
    category_embed_dim: int = 16
    province_embed_dim: int = 16

    # Sequence encoding
    max_seq_length: int = 20
    seq_embed_dim: int = 64
    use_attention: bool = True
    attention_heads: int = 4

    # MMoE experts
    num_experts: int = 4
    expert_hidden_dims: List[int] = field(default_factory=lambda: [128, 64])

    # Tower dimensions
    user_tower_dims: List[int] = field(default_factory=lambda: [256, 128])
    item_tower_dims: List[int] = field(default_factory=lambda: [256, 128])

    # Task heads
    ctr_head_dims: List[int] = field(default_factory=lambda: [64, 32])
    visit_head_dims: List[int] = field(default_factory=lambda: [64, 32])
    duration_head_dims: List[int] = field(default_factory=lambda: [64, 32])

    # Training
    dropout: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 512
    epochs: int = 20
    early_stop_patience: int = 5

    # Loss weights
    ctr_weight: float = 1.0
    visit_weight: float = 1.0
    duration_weight: float = 0.1

    # Regularization
    l2_reg: float = 1e-5
    label_smoothing: float = 0.0

    # Device
    device: str = "cuda"  # cuda or cpu


@dataclass
class TrainingMetrics:
    """Metrics tracked during training."""
    epoch: int
    train_loss: float
    train_ctr_loss: float
    train_visit_loss: float
    train_duration_loss: float
    val_loss: Optional[float] = None
    val_ctr_auc: Optional[float] = None
    val_visit_auc: Optional[float] = None
    val_duration_mae: Optional[float] = None


# ============================================================================
# PyTorch Model Implementation
# ============================================================================

class MMoEDeepRanker:
    """
    Multi-Task Deep Ranking Model with MMoE architecture.

    Uses PyTorch for neural network implementation with fallback
    to scikit-learn when PyTorch is unavailable.

    Features:
    - UserTower: Encodes user history and preferences
    - ItemTower: Encodes POI features and embeddings
    - MMoE: Shared expert networks with task-specific gating
    - Multi-task heads: CTR, Visit, Duration prediction

    Usage:
        config = MMoEConfig(num_experts=4)
        model = MMoEDeepRanker(config)

        # Training
        model.fit(train_data, valid_data)

        # Inference
        predictions = model.predict(user_id, candidate_pois, context)
    """

    def __init__(self, config: Optional[MMoEConfig] = None):
        self.config = config or MMoEConfig()
        self.model = None
        self.optimizers = None
        self.vocab_sizes = {}
        self.feature_maps = {}
        self.user_id_map: Dict[str, int] = {}
        self.poi_id_map: Dict[str, int] = {}
        self.category_map: Dict[str, int] = {}
        self.province_map: Dict[str, int] = {}
        self.city_map: Dict[str, int] = {}
        self.training_history: List[TrainingMetrics] = []

        # Check PyTorch availability
        self._check_torch()

    def _check_torch(self) -> None:
        """Check if PyTorch is available."""
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F

            self.torch = torch
            self.nn = nn
            self.F = F
            self.torch_available = True
            logger.info("PyTorch is available. Using GPU acceleration.")
        except ImportError:
            self.torch = None
            self.nn = None
            self.F = None
            self.torch_available = False
            logger.warning(
                "PyTorch not available. Install with: pip install torch"
            )
            logger.warning("Falling back to sklearn-based ranking.")

    def _build_vocabularies(
        self,
        poi_df: pd.DataFrame,
        events_df: pd.DataFrame,
    ) -> Dict[str, int]:
        """
        Build vocabularies from data.

        Args:
            poi_df: POI DataFrame
            events_df: Events DataFrame

        Returns:
            Dictionary mapping feature names to vocab sizes
        """
        vocab_sizes = {}

        # POI vocabulary
        vocab_sizes["num_pois"] = poi_df["poi_id"].nunique() + 1

        # Category vocabulary
        if "category" in poi_df.columns:
            vocab_sizes["num_categories"] = poi_df["category"].nunique() + 1
        else:
            vocab_sizes["num_categories"] = 10  # Default

        # Province vocabulary
        if "province" in poi_df.columns:
            vocab_sizes["num_provinces"] = poi_df["province"].nunique() + 1
        else:
            vocab_sizes["num_provinces"] = 35  # Default

        # City vocabulary
        if "city" in poi_df.columns:
            vocab_sizes["num_cities"] = poi_df["city"].nunique() + 1
        else:
            vocab_sizes["num_cities"] = 100

        # User vocabulary
        vocab_sizes["num_users"] = events_df["user_id"].nunique() + 1

        self.vocab_sizes = vocab_sizes
        logger.info(f"Built vocabularies: {vocab_sizes}")
        return vocab_sizes

    def _create_model(self) -> None:
        """Create the MMoE PyTorch model."""
        if not self.torch_available:
            return

        nn = self.nn  # Capture nn for use in inner class
        torch = self.torch  # Capture torch for use in inner class

        class _MMoENet(nn.Module):
            """MMoE Network for POI ranking."""

            def __init__(
                self,
                vocab_sizes: Dict[str, int],
                config: MMoEConfig,
            ):
                super().__init__()
                self.config = config
                self.vocab_sizes = vocab_sizes

                # --- Embeddings ---
                self.user_embedding = nn.Embedding(
                    vocab_sizes["num_users"],
                    config.user_embed_dim,
                    padding_idx=0,
                )

                self.poi_embedding = nn.Embedding(
                    vocab_sizes["num_pois"],
                    config.item_embed_dim,
                    padding_idx=0,
                )

                self.category_embedding = nn.Embedding(
                    vocab_sizes["num_categories"],
                    config.category_embed_dim,
                )

                self.province_embedding = nn.Embedding(
                    vocab_sizes["num_provinces"],
                    config.province_embed_dim,
                )

                self.city_embedding = nn.Embedding(
                    vocab_sizes["num_cities"],
                    config.category_embed_dim,
                )

                # --- Sequence Encoder ---
                seq_input_dim = config.item_embed_dim
                if config.use_attention:
                    self.sequence_encoder = nn.MultiheadAttention(
                        embed_dim=seq_input_dim,
                        num_heads=config.attention_heads,
                        dropout=config.dropout,
                        batch_first=True,
                    )
                    if config.item_embed_dim != config.seq_embed_dim:
                        self.seq_projection = nn.Linear(config.item_embed_dim, config.seq_embed_dim)
                    else:
                        self.seq_projection = nn.Identity()
                else:
                    self.sequence_encoder = nn.LSTM(
                        seq_input_dim,
                        config.seq_embed_dim,
                        num_layers=1,
                        batch_first=True,
                    )
                    self.seq_projection = nn.Identity()

                # --- User Tower ---
                user_input_dim = (
                    config.user_embed_dim +
                    config.seq_embed_dim +
                    config.category_embed_dim * 2  # preference for categories
                )
                logger.info(f"User tower: input_dim={user_input_dim}, tower_dims={config.user_tower_dims}")
                user_layers = []
                prev_dim = user_input_dim
                for dim in config.user_tower_dims:
                    user_layers.extend([
                        nn.Linear(prev_dim, dim),
                        nn.ReLU(),
                        nn.Dropout(config.dropout),
                    ])
                    prev_dim = dim
                self.user_tower = nn.Sequential(*user_layers)

                # --- Item Tower ---
                item_input_dim = (
                    config.item_embed_dim +
                    config.category_embed_dim +
                    config.province_embed_dim +
                    config.category_embed_dim +  # city
                    7  # numeric features: popularity, stay, open, close, lat, lon, hour
                )
                logger.info(f"Item tower: input_dim={item_input_dim}, tower_dims={config.item_tower_dims}")
                item_layers = []
                prev_dim = item_input_dim
                for dim in config.item_tower_dims:
                    item_layers.extend([
                        nn.Linear(prev_dim, dim),
                        nn.ReLU(),
                        nn.Dropout(config.dropout),
                    ])
                    prev_dim = dim
                self.item_tower = nn.Sequential(*item_layers)

                # --- MMoE Experts ---
                tower_output_dim = config.user_tower_dims[-1] + config.item_tower_dims[-1]
                expert_input_dim = tower_output_dim

                self.experts = nn.ModuleList([
                    self._create_expert(expert_input_dim, config.expert_hidden_dims)
                    for _ in range(config.num_experts)
                ])

                # --- Gating Networks ---
                self.gates = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(expert_input_dim, config.num_experts),
                        nn.Softmax(dim=-1),
                    )
                    for _ in range(3)  # CTR, Visit, Duration
                ])

                # --- Task Heads ---
                expert_output_dim = config.expert_hidden_dims[-1]

                # CTR head
                ctr_layers = []
                prev_dim = expert_output_dim
                for dim in config.ctr_head_dims:
                    ctr_layers.extend([
                        nn.Linear(prev_dim, dim),
                        nn.ReLU(),
                        nn.Dropout(config.dropout),
                    ])
                    prev_dim = dim
                ctr_layers.append(nn.Linear(prev_dim, 1))
                self.ctr_head = nn.Sequential(*ctr_layers)

                # Visit head
                visit_layers = []
                prev_dim = expert_output_dim
                for dim in config.visit_head_dims:
                    visit_layers.extend([
                        nn.Linear(prev_dim, dim),
                        nn.ReLU(),
                        nn.Dropout(config.dropout),
                    ])
                    prev_dim = dim
                visit_layers.append(nn.Linear(prev_dim, 1))
                self.visit_head = nn.Sequential(*visit_layers)

                # Duration head
                duration_layers = []
                prev_dim = expert_output_dim
                for dim in config.duration_head_dims:
                    duration_layers.extend([
                        nn.Linear(prev_dim, dim),
                        nn.ReLU(),
                        nn.Dropout(config.dropout),
                    ])
                    prev_dim = dim
                duration_layers.append(nn.Linear(prev_dim, 1))
                # Use ReLU for duration (non-negative)
                duration_layers.append(nn.ReLU())
                self.duration_head = nn.Sequential(*duration_layers)

            def _create_expert(self, input_dim: int, hidden_dims: List[int]) -> nn.Module:
                """Create a single expert network."""
                layers = []
                prev_dim = input_dim
                for dim in hidden_dims:
                    layers.extend([
                        nn.Linear(prev_dim, dim),
                        nn.ReLU(),
                        nn.Dropout(self.config.dropout),
                    ])
                    prev_dim = dim
                return nn.Sequential(*layers)

            def forward(
                self,
                user_ids,
                poi_ids,
                category_ids,
                province_ids,
                city_ids,
                history_poi_ids,
                history_category_ids,
                numeric_features,
            ):
                """
                Forward pass.

                Args:
                    user_ids: User ID indices [batch]
                    poi_ids: POI ID indices [batch]
                    category_ids: Category indices [batch]
                    province_ids: Province indices [batch]
                    city_ids: City indices [batch]
                    history_poi_ids: Historical POI IDs [batch, seq_len]
                    history_category_ids: Historical category IDs [batch, seq_len]
                    numeric_features: Numeric features [batch, 7]

                Returns:
                    (ctr_logits, visit_logits, duration_pred)
                """
                batch_size = user_ids.shape[0]

                # --- User Tower ---
                user_embed = self.user_embedding(user_ids)  # [batch, user_dim]

                # Encode sequence
                history_embed = self.poi_embedding(history_poi_ids)  # [batch, seq, item_dim]

                if self.config.use_attention:
                    # Self-attention over sequence
                    seq_output, _ = self.sequence_encoder(
                        history_embed, history_embed, history_embed
                    )
                    # Aggregate with mean pooling and project to seq_embed_dim.
                    seq_repr = self.seq_projection(seq_output.mean(dim=1))
                else:
                    # LSTM aggregation
                    _, (h_n, _) = self.sequence_encoder(history_embed)
                    seq_repr = h_n.squeeze(0)  # [batch, seq_dim]

                # Category preferences (aggregated from history)
                history_cat_embed = self.category_embedding(history_category_ids)
                cat_preference = history_cat_embed.mean(dim=1)  # [batch, cat_dim]

                # Combine user tower inputs
                user_input = torch.cat([
                    user_embed,
                    seq_repr,
                    cat_preference,
                    torch.zeros(batch_size, self.config.category_embed_dim).to(user_embed.device),
                ], dim=-1)

                user_tower_output = self.user_tower(user_input)  # [batch, user_tower_dim]

                # --- Item Tower ---
                poi_embed = self.poi_embedding(poi_ids)
                cat_embed = self.category_embedding(category_ids)
                prov_embed = self.province_embedding(province_ids)
                city_embed = self.city_embedding(city_ids)

                item_input = torch.cat([
                    poi_embed,
                    cat_embed,
                    prov_embed,
                    city_embed,
                    numeric_features,
                ], dim=-1)

                item_tower_output = self.item_tower(item_input)  # [batch, item_tower_dim]

                # --- MMoE Layer ---
                combined = torch.cat([user_tower_output, item_tower_output], dim=-1)

                # Get expert outputs
                expert_outputs = [expert(combined) for expert in self.experts]
                expert_outputs = torch.stack(expert_outputs, dim=-1)  # [batch, expert_dim, num_experts]

                # Apply gating for each task
                gated_outputs = []
                for i, gate in enumerate(self.gates):
                    gate_weights = gate(combined)  # [batch, num_experts]
                    gated_output = (
                        expert_outputs * gate_weights.unsqueeze(1)
                    ).sum(dim=-1)  # [batch, expert_dim]
                    gated_outputs.append(gated_output)

                # --- Task Heads ---
                ctr_logits = self.ctr_head(gated_outputs[0]).squeeze(-1)
                visit_logits = self.visit_head(gated_outputs[1]).squeeze(-1)
                duration_pred = self.duration_head(gated_outputs[2]).squeeze(-1)

                return ctr_logits, visit_logits, duration_pred

        # Create model instance
        self.model = _MMoENet(self.vocab_sizes, self.config)
        logger.info("MMoE model created")

    def _prepare_training_data(
        self,
        poi_df: pd.DataFrame,
        events_df: pd.DataFrame,
        negative_sampling_ratio: int = 4,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Prepare training data with features and labels.

        Args:
            poi_df: POI DataFrame
            events_df: Events DataFrame
            negative_sampling_ratio: Negative samples per positive

        Returns:
            (training_df, metadata)
        """
        # Encode categorical variables
        poi_df = poi_df.copy()
        poi_df["poi_id"] = poi_df["poi_id"].astype(str)
        poi_df["poi_id_encoded"] = poi_df["poi_id"].astype("category").cat.codes.astype(int)
        # Use get with fillna for missing columns
        if "category" in poi_df.columns:
            poi_df["category"] = poi_df["category"].fillna("unknown").astype(str)
            poi_df["category_encoded"] = poi_df["category"].astype("category").cat.codes.astype(int)
        else:
            poi_df["category_encoded"] = 0  # unknown category

        if "province" in poi_df.columns:
            poi_df["province"] = poi_df["province"].fillna("unknown").astype(str)
            poi_df["province_encoded"] = poi_df["province"].astype("category").cat.codes.astype(int)
        else:
            poi_df["province_encoded"] = 0  # unknown province

        if "city" in poi_df.columns:
            poi_df["city"] = poi_df["city"].fillna("unknown").astype(str)
            poi_df["city_encoded"] = poi_df["city"].astype("category").cat.codes.astype(int)
        else:
            poi_df["city_encoded"] = 0  # unknown city

        events_df = events_df.copy()
        events_df["user_id"] = events_df["user_id"].astype(str)
        events_df["user_id_encoded"] = events_df["user_id"].astype("category").cat.codes.astype(int)

        # Build id maps for online inference.
        self.poi_id_map = {
            str(row["poi_id"]): int(row["poi_id_encoded"])
            for _, row in poi_df[["poi_id", "poi_id_encoded"]].drop_duplicates().iterrows()
        }
        self.category_map = {
            str(row["category"]): int(row["category_encoded"])
            for _, row in poi_df[["category", "category_encoded"]].drop_duplicates().iterrows()
        } if "category" in poi_df.columns else {"unknown": 0}
        self.province_map = {
            str(row["province"]): int(row["province_encoded"])
            for _, row in poi_df[["province", "province_encoded"]].drop_duplicates().iterrows()
        } if "province" in poi_df.columns else {"unknown": 0}
        self.city_map = {
            str(row["city"]): int(row["city_encoded"])
            for _, row in poi_df[["city", "city_encoded"]].drop_duplicates().iterrows()
        } if "city" in poi_df.columns else {"unknown": 0}
        self.user_id_map = {
            str(row["user_id"]): int(row["user_id_encoded"])
            for _, row in events_df[["user_id", "user_id_encoded"]].drop_duplicates().iterrows()
        }

        # Create labels
        action_to_label = {
            "visit": (1, 1),  # (ctr, visit)
            "fav": (1, 0.7),
            "click": (1, 0.3),
            "view": (0.5, 0),
        }

        # Merge with POI data
        merged = events_df.merge(
            poi_df[[
                "poi_id", "poi_id_encoded", "category_encoded",
                "province_encoded", "city_encoded", "stay_min"
            ]],
            on="poi_id",
            how="left",
        )

        # Create labels
        merged["ctr_label"] = merged["action"].map(lambda x: action_to_label.get(x, (0, 0))[0])
        merged["visit_label"] = merged["action"].map(lambda x: action_to_label.get(x, (0, 0))[1])
        merged["duration_label"] = merged.get("stay_min", 60).astype(float)

        # Build history sequences
        rows = []
        for user_id, user_data in merged.groupby("user_id_encoded"):
            user_data = user_data.sort_values("timestamp")

            for idx, row in user_data.iterrows():
                # Get history before this event
                history = user_data[user_data["timestamp"] < row["timestamp"]]

                # Take last N interactions
                recent = history.tail(self.config.max_seq_length)

                row_data = {
                    "user_id_encoded": int(row["user_id_encoded"]),
                    "poi_id_encoded": int(row["poi_id_encoded"]),
                    "category_encoded": int(row["category_encoded"]),
                    "province_encoded": int(row["province_encoded"]),
                    "city_encoded": int(row["city_encoded"]),
                    "history_poi_ids": recent["poi_id_encoded"].tolist(),
                    "history_category_ids": recent["category_encoded"].tolist(),
                    "ctr_label": float(row["ctr_label"]),
                    "visit_label": float(row["visit_label"]),
                    "duration_label": float(row["duration_label"]),
                    # Numeric features (normalized later)
                    "popularity": float(poi_df[poi_df["poi_id"] == row["poi_id"]].get("visit_count", pd.Series([0])).iloc[0] if len(poi_df[poi_df["poi_id"] == row["poi_id"]]) > 0 else 0),
                }
                rows.append(row_data)

        train_df = pd.DataFrame(rows)

        # Negative sampling
        positive_rows = rows.copy()
        negative_rows = []

        all_poi_ids = poi_df["poi_id_encoded"].tolist()
        all_category_ids = poi_df["category_encoded"].tolist()
        all_province_ids = poi_df["province_encoded"].tolist()
        all_city_ids = poi_df["city_encoded"].tolist()

        for pos_row in positive_rows:
            for _ in range(negative_sampling_ratio):
                neg_poi_id = random.choice(all_poi_ids)
                idx = all_poi_ids.index(neg_poi_id)

                neg_row = pos_row.copy()
                neg_row["poi_id_encoded"] = neg_poi_id
                neg_row["category_encoded"] = all_category_ids[idx]
                neg_row["province_encoded"] = all_province_ids[idx]
                neg_row["city_encoded"] = all_city_ids[idx]
                neg_row["ctr_label"] = 0.0
                neg_row["visit_label"] = 0.0
                neg_row["duration_label"] = 60.0  # Default duration
                neg_row["is_negative"] = True

                negative_rows.append(neg_row)

        # Combine
        train_df = pd.DataFrame(positive_rows + negative_rows)

        # Normalize numeric features
        for col in ["popularity"]:
            if col in train_df.columns:
                mean = train_df[col].mean()
                std = train_df[col].std()
                train_df[col] = (train_df[col] - mean) / (std + 1e-8)
                self.feature_maps[col] = {"mean": mean, "std": std}

        metadata = {
            "num_samples": len(train_df),
            "num_positives": len(positive_rows),
            "num_negatives": len(negative_rows),
        }

        return train_df, metadata

    def fit(
        self,
        poi_df: pd.DataFrame,
        events_df: pd.DataFrame,
        valid_poi_df: Optional[pd.DataFrame] = None,
        valid_events_df: Optional[pd.DataFrame] = None,
        negative_sampling_ratio: int = 4,
    ) -> Dict[str, Any]:
        """
        Train the MMoE model.

        Args:
            poi_df: POI DataFrame
            events_df: User events DataFrame
            valid_poi_df: Optional validation POI data
            valid_events_df: Optional validation events data
            negative_sampling_ratio: Negative samples per positive

        Returns:
            Training metrics dictionary
        """
        if not self.torch_available:
            logger.error("PyTorch required for training")
            return {}

        # Build vocabularies
        self._build_vocabularies(poi_df, events_df)

        # Prepare data
        logger.info("Preparing training data...")
        train_df, train_meta = self._prepare_training_data(
            poi_df, events_df, negative_sampling_ratio
        )
        logger.info(f"Training data: {train_meta}")

        if valid_poi_df is not None and valid_events_df is not None:
            valid_df, valid_meta = self._prepare_training_data(
                valid_poi_df, valid_events_df, negative_sampling_ratio
            )
        else:
            # Split train/valid
            user_ids = train_df["user_id_encoded"].unique()
            random.shuffle(user_ids)
            split = int(len(user_ids) * 0.8)

            train_users = user_ids[:split]
            valid_users = user_ids[split:]

            train_df = train_df[train_df["user_id_encoded"].isin(train_users)]
            valid_df = train_df[train_df["user_id_encoded"].isin(valid_users)]

            logger.info(f"Train users: {len(train_users)}, Valid users: {len(valid_users)}")
            logger.info(f"Train samples: {len(train_df)}, Valid samples: {len(valid_df)}")

            if len(valid_df) == 0:
                logger.warning("No validation samples! Using train set for validation.")
                valid_df = train_df.sample(min(1000, len(train_df)))

        # Create model
        self._create_model()

        # Setup device
        device = self.torch.device(self.config.device)
        self.model = self.model.to(device)

        # Optimizers
        optimizer = self.torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.l2_reg,
        )

        # Loss functions
        ctr_criterion = self.nn.BCEWithLogitsLoss()
        visit_criterion = self.nn.BCEWithLogitsLoss()
        duration_criterion = self.nn.MSELoss()

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.epochs):
            # Shuffle training data
            train_df_shuffled = train_df.sample(frac=1).reset_index(drop=True)

            # Mini-batch training
            epoch_losses = {"total": [], "ctr": [], "visit": [], "duration": []}
            num_batches = 0

            for start_idx in range(0, len(train_df_shuffled), self.config.batch_size):
                batch = train_df_shuffled.iloc[start_idx:start_idx + self.config.batch_size]

                # Prepare batch tensors
                batch_tensors = self._prepare_batch(batch, device)

                # Forward pass
                self.model.train()
                optimizer.zero_grad()

                ctr_logits, visit_logits, duration_pred = self.model(
                    batch_tensors["user_ids"],
                    batch_tensors["poi_ids"],
                    batch_tensors["category_ids"],
                    batch_tensors["province_ids"],
                    batch_tensors["city_ids"],
                    batch_tensors["history_poi_ids"],
                    batch_tensors["history_category_ids"],
                    batch_tensors["numeric_features"],
                )

                # Compute losses
                ctr_loss = ctr_criterion(ctr_logits, batch_tensors["ctr_labels"])
                visit_loss = visit_criterion(visit_logits, batch_tensors["visit_labels"])
                duration_loss = duration_criterion(duration_pred, batch_tensors["duration_labels"])

                # Combined loss
                total_loss = (
                    self.config.ctr_weight * ctr_loss +
                    self.config.visit_weight * visit_loss +
                    self.config.duration_weight * duration_loss
                )

                # Backward pass
                total_loss.backward()
                optimizer.step()

                # Track losses
                epoch_losses["total"].append(total_loss.item())
                epoch_losses["ctr"].append(ctr_loss.item())
                epoch_losses["visit"].append(visit_loss.item())
                epoch_losses["duration"].append(duration_loss.item())
                num_batches += 1

            # Compute epoch metrics
            train_loss = np.mean(epoch_losses["total"])
            train_ctr_loss = np.mean(epoch_losses["ctr"])
            train_visit_loss = np.mean(epoch_losses["visit"])
            train_duration_loss = np.mean(epoch_losses["duration"])

            # Validation
            val_metrics = self._evaluate(
                valid_df, device, ctr_criterion, visit_criterion, duration_criterion
            )

            # Log metrics
            logger.info(
                f"Epoch {epoch + 1}/{self.config.epochs} - "
                f"Loss: {train_loss:.4f} | "
                f"CTR: {train_ctr_loss:.4f} | "
                f"Visit: {train_visit_loss:.4f} | "
                f"Duration: {train_duration_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val AUC: {val_metrics['ctr_auc']:.4f}"
            )

            # Save metrics
            metric = TrainingMetrics(
                epoch=epoch,
                train_loss=train_loss,
                train_ctr_loss=train_ctr_loss,
                train_visit_loss=train_visit_loss,
                train_duration_loss=train_duration_loss,
                val_loss=val_metrics["loss"],
                val_ctr_auc=val_metrics["ctr_auc"],
                val_visit_auc=val_metrics["visit_auc"],
                val_duration_mae=val_metrics["duration_mae"],
            )
            self.training_history.append(metric)

            # Early stopping
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                # Save best model
                self._save_checkpoint("best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stop_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        # Load best model
        self._load_checkpoint("best_model.pt")

        return {
            "best_val_loss": best_val_loss,
            "final_metrics": self.training_history[-1].__dict__,
        }

    def _prepare_batch(self, batch: pd.DataFrame, device) -> Dict[str, Any]:
        """Prepare batch tensors for training."""
        # Pad sequences
        max_seq_len = self.config.max_seq_length

        def pad_sequence(seq, max_len, value=0):
            seq = list(seq)
            return seq + [value] * (max_len - len(seq))

        history_poi_ids = [
            pad_sequence(seq, max_seq_len) for seq in batch["history_poi_ids"]
        ]
        history_category_ids = [
            pad_sequence(seq, max_seq_len) for seq in batch["history_category_ids"]
        ]

        # Numeric features (use popularity, default others)
        numeric_features = self.torch.zeros(len(batch), 7, device=device)
        if "popularity" in batch.columns:
            numeric_features[:, 0] = self.torch.tensor(batch["popularity"].values)

        return {
            "user_ids": self.torch.tensor(batch["user_id_encoded"].values, dtype=self.torch.long, device=device),
            "poi_ids": self.torch.tensor(batch["poi_id_encoded"].values, dtype=self.torch.long, device=device),
            "category_ids": self.torch.tensor(batch["category_encoded"].values, dtype=self.torch.long, device=device),
            "province_ids": self.torch.tensor(batch["province_encoded"].values, dtype=self.torch.long, device=device),
            "city_ids": self.torch.tensor(batch["city_encoded"].values, dtype=self.torch.long, device=device),
            "history_poi_ids": self.torch.tensor(history_poi_ids, dtype=self.torch.long, device=device),
            "history_category_ids": self.torch.tensor(history_category_ids, dtype=self.torch.long, device=device),
            "numeric_features": numeric_features,
            "ctr_labels": self.torch.tensor(batch["ctr_label"].values, dtype=self.torch.float, device=device),
            "visit_labels": self.torch.tensor(batch["visit_label"].values, dtype=self.torch.float, device=device),
            "duration_labels": self.torch.tensor(batch["duration_label"].values, dtype=self.torch.float, device=device),
        }

    def _evaluate(
        self,
        valid_df: pd.DataFrame,
        device,
        ctr_criterion,
        visit_criterion,
        duration_criterion,
    ) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()

        all_ctr_preds = []
        all_visit_preds = []
        all_duration_preds = []
        all_ctr_labels = []
        all_visit_labels = []
        all_duration_labels = []

        total_loss = 0
        num_batches = 0

        with self.torch.no_grad():
            for start_idx in range(0, len(valid_df), self.config.batch_size * 2):
                batch = valid_df.iloc[start_idx:start_idx + self.config.batch_size * 2]
                batch_tensors = self._prepare_batch(batch, device)

                ctr_logits, visit_logits, duration_pred = self.model(
                    batch_tensors["user_ids"],
                    batch_tensors["poi_ids"],
                    batch_tensors["category_ids"],
                    batch_tensors["province_ids"],
                    batch_tensors["city_ids"],
                    batch_tensors["history_poi_ids"],
                    batch_tensors["history_category_ids"],
                    batch_tensors["numeric_features"],
                )

                ctr_loss = ctr_criterion(ctr_logits, batch_tensors["ctr_labels"])
                visit_loss = visit_criterion(visit_logits, batch_tensors["visit_labels"])
                duration_loss = duration_criterion(duration_pred, batch_tensors["duration_labels"])

                loss = (
                    self.config.ctr_weight * ctr_loss +
                    self.config.visit_weight * visit_loss +
                    self.config.duration_weight * duration_loss
                )
                total_loss += loss.item()
                num_batches += 1

                # Collect predictions
                all_ctr_preds.extend(self.torch.sigmoid(ctr_logits).cpu().numpy())
                all_visit_preds.extend(self.torch.sigmoid(visit_logits).cpu().numpy())
                all_duration_preds.extend(duration_pred.cpu().numpy())
                all_ctr_labels.extend(batch_tensors["ctr_labels"].cpu().numpy())
                all_visit_labels.extend(batch_tensors["visit_labels"].cpu().numpy())
                all_duration_labels.extend(batch_tensors["duration_labels"].cpu().numpy())

        # Compute metrics
        from sklearn.metrics import roc_auc_score, mean_absolute_error

        def _safe_auc(labels, preds, threshold: float = 0.5) -> float:
            # 训练标签支持软标签（例如 0.3/0.7），AUC 需要二值标签
            y_true = (np.array(labels, dtype=float) >= threshold).astype(int)
            if len(set(y_true.tolist())) <= 1:
                return 0.5
            return float(roc_auc_score(y_true, preds))

        ctr_auc = _safe_auc(all_ctr_labels, all_ctr_preds, threshold=0.5)
        visit_auc = _safe_auc(all_visit_labels, all_visit_preds, threshold=0.5)
        duration_mae = mean_absolute_error(all_duration_labels, all_duration_preds)

        return {
            "loss": total_loss / num_batches if num_batches > 0 else 0,
            "ctr_auc": ctr_auc,
            "visit_auc": visit_auc,
            "duration_mae": duration_mae,
        }

    def predict(
        self,
        user_id: Union[str, int],
        candidate_pois: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, Dict[str, float]]]:
        """
        Predict scores for candidate POIs.

        Args:
            user_id: User identifier
            candidate_pois: List of candidate POI dicts with features
            context: Optional context information

        Returns:
            List of (poi_id, scores) tuples, sorted by combined score
        """
        if not self.torch_available or self.model is None:
            # Fallback to popularity scoring
            results = []
            for poi in candidate_pois:
                score = poi.get("popularity", 0)
                results.append((poi.get("poi_id", ""), {
                    "ctr": min(score / 100, 1.0),
                    "visit": min(score / 200, 1.0),
                    "duration": poi.get("stay_min", 60),
                }))
            return sorted(results, key=lambda x: x[1]["ctr"], reverse=True)

        context = context or {}
        device = self.torch.device(self.config.device)
        self.model.eval()

        # Get user encoding
        user_str_id = str(user_id)
        if user_str_id not in getattr(self, "user_id_map", {}):
            # Unknown user - use average
            user_encoded = 0
        else:
            user_encoded = self.user_id_map[user_str_id]

        results = []

        with self.torch.no_grad():
            for poi in candidate_pois:
                # Prepare input
                poi_id = poi.get("poi_id", "")
                poi_encoded = self.poi_id_map.get(str(poi_id), 0)
                category_encoded = self.category_map.get(poi.get("category", "unknown"), 0)
                province_encoded = self.province_map.get(poi.get("province", "unknown"), 0)
                city_encoded = self.city_map.get(poi.get("city", "unknown"), 0)

                # Numeric features
                numeric = self.torch.zeros(7, device=device)
                numeric[0] = poi.get("popularity", 0) / 100  # Normalized

                # Create batch
                user_ids = self.torch.tensor([user_encoded], dtype=self.torch.long, device=device)
                poi_ids = self.torch.tensor([poi_encoded], dtype=self.torch.long, device=device)
                category_ids = self.torch.tensor([category_encoded], dtype=self.torch.long, device=device)
                province_ids = self.torch.tensor([province_encoded], dtype=self.torch.long, device=device)
                city_ids = self.torch.tensor([city_encoded], dtype=self.torch.long, device=device)

                # Empty history for new prediction
                history_poi_ids = self.torch.zeros(
                    1, self.config.max_seq_length,
                    dtype=self.torch.long, device=device
                )
                history_category_ids = self.torch.zeros(
                    1, self.config.max_seq_length,
                    dtype=self.torch.long, device=device
                )
                numeric_features = numeric.unsqueeze(0)

                # Predict
                ctr_logit, visit_logit, duration_pred = self.model(
                    user_ids, poi_ids, category_ids, province_ids, city_ids,
                    history_poi_ids, history_category_ids, numeric_features
                )

                scores = {
                    "ctr": float(self.torch.sigmoid(ctr_logit).item()),
                    "visit": float(self.torch.sigmoid(visit_logit).item()),
                    "duration": float(duration_pred.item()),
                }

                results.append((poi_id, scores))

        # Sort by combined score
        results.sort(
            key=lambda x: 0.4 * x[1]["ctr"] + 0.5 * x[1]["visit"] + 0.1 * (x[1]["duration"] / 300),
            reverse=True
        )

        return results

    def single_predict(
        self,
        user: Dict[str, Any],
        item: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Single prediction for a user-item pair.

        Args:
            user: User dict with keys: user_id, history_pois (optional)
            item: POI dict with keys: poi_id, category, province, etc.

        Returns:
            Dictionary with ctr, visit, duration scores
        """
        results = self.predict(
            user.get("user_id", ""),
            [item],
            user,
        )
        return results[0][1] if results else {"ctr": 0, "visit": 0, "duration": 60}

    def batch_predict(
        self,
        users: List[Dict[str, Any]],
        items: List[Dict[str, Any]],
    ) -> List[Dict[str, float]]:
        """
        Batch prediction for multiple user-item pairs.

        Args:
            users: List of user dicts
            items: List of item dicts (same length as users)

        Returns:
            List of score dicts
        """
        results = []
        for user, item in zip(users, items):
            results.append(self.single_predict(user, item))
        return results

    def export_model(self, path: str) -> None:
        """
        Export model to file.

        Args:
            path: Path to save the model
        """
        export_path = Path(path)
        export_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.torch_available or self.model is None:
            # Save config only
            with open(export_path, "wb") as f:
                pickle.dump({
                    "config": self.config,
                    "vocab_sizes": self.vocab_sizes,
                }, f)
            logger.info(f"Model config saved to {export_path}")
            return

        # Save full model
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "vocab_sizes": self.vocab_sizes,
            "feature_maps": self.feature_maps,
            "id_maps": {
                "user_id_map": self.user_id_map,
                "poi_id_map": self.poi_id_map,
                "category_map": self.category_map,
                "province_map": self.province_map,
                "city_map": self.city_map,
            },
            "training_history": [
                m.__dict__ for m in self.training_history
            ],
        }

        self.torch.save(checkpoint, export_path)
        logger.info(f"Model exported to {export_path}")

    def load_model(self, path: str) -> None:
        """
        Load model from file.

        Args:
            path: Path to the saved model
        """
        if not self.torch_available:
            raise ImportError("PyTorch required to load model")

        checkpoint = self._torch_load(path)

        # Restore config
        self.config = checkpoint["config"]
        self.vocab_sizes = checkpoint["vocab_sizes"]
        self.feature_maps = checkpoint.get("feature_maps", {})
        id_maps = checkpoint.get("id_maps", {})
        self.user_id_map = id_maps.get("user_id_map", {})
        self.poi_id_map = id_maps.get("poi_id_map", {})
        self.category_map = id_maps.get("category_map", {})
        self.province_map = id_maps.get("province_map", {})
        self.city_map = id_maps.get("city_map", {})

        # Recreate model
        self._create_model()
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Restore training history
        if "training_history" in checkpoint:
            self.training_history = [
                TrainingMetrics(**m) for m in checkpoint["training_history"]
            ]

        logger.info(f"Model loaded from {path}")

    def _save_checkpoint(self, filename: str) -> None:
        """Save temporary checkpoint."""
        checkpoint_dir = Path("outputs/ranking/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        path = checkpoint_dir / filename
        self.torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "vocab_sizes": self.vocab_sizes,
        }, path)

    def _load_checkpoint(self, filename: str) -> None:
        """Load temporary checkpoint."""
        path = Path("outputs/ranking/checkpoints") / filename
        if not path.exists():
            return

        checkpoint = self._torch_load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])

    def _torch_load(self, path: str | Path):
        """兼容 PyTorch 2.6+ 的 weights_only 默认行为。"""
        try:
            return self.torch.load(path, map_location=self.config.device, weights_only=False)
        except TypeError:
            return self.torch.load(path, map_location=self.config.device)

    def get_training_curves(self) -> Dict[str, List[float]]:
        """
        Get training curves for visualization.

        Returns:
            Dictionary with metric names to value lists
        """
        if not self.training_history:
            return {}

        curves = {
            "train_loss": [m.train_loss for m in self.training_history],
            "train_ctr_loss": [m.train_ctr_loss for m in self.training_history],
            "train_visit_loss": [m.train_visit_loss for m in self.training_history],
            "train_duration_loss": [m.train_duration_loss for m in self.training_history],
        }

        if self.training_history[0].val_loss is not None:
            curves.update({
                "val_loss": [m.val_loss for m in self.training_history],
                "val_ctr_auc": [m.val_ctr_auc for m in self.training_history],
                "val_visit_auc": [m.val_visit_auc for m in self.training_history],
                "val_duration_mae": [m.val_duration_mae for m in self.training_history],
            })

        return curves

    def save_training_curves(self, path: str) -> None:
        """Save training curves as image."""
        curves = self.get_training_curves()
        if not curves:
            logger.warning("No training curves to save")
            return

        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Loss curves
            axes[0, 0].plot(curves["train_loss"], label="Train")
            if "val_loss" in curves:
                axes[0, 0].plot(curves["val_loss"], label="Valid")
            axes[0, 0].set_title("Total Loss")
            axes[0, 0].legend()
            axes[0, 0].grid(True)

            # Task losses
            axes[0, 1].plot(curves["train_ctr_loss"], label="CTR")
            axes[0, 1].plot(curves["train_visit_loss"], label="Visit")
            axes[0, 1].plot(curves["train_duration_loss"], label="Duration")
            axes[0, 1].set_title("Task Losses")
            axes[0, 1].legend()
            axes[0, 1].grid(True)

            # AUC curves
            if "val_ctr_auc" in curves:
                axes[1, 0].plot(curves["val_ctr_auc"], label="CTR AUC")
                axes[1, 0].plot(curves["val_visit_auc"], label="Visit AUC")
                axes[1, 0].set_title("Validation AUC")
                axes[1, 0].legend()
                axes[1, 0].grid(True)

            # MAE curve
            if "val_duration_mae" in curves:
                axes[1, 1].plot(curves["val_duration_mae"])
                axes[1, 1].set_title("Duration MAE")
                axes[1, 1].grid(True)

            plt.tight_layout()
            plt.savefig(path, dpi=150)
            plt.close()

            logger.info(f"Training curves saved to {path}")
        except ImportError:
            logger.warning("matplotlib not available, skipping curve visualization")


# ============================================================================
# Factory Functions
# ============================================================================

def create_mmoe_ranker(
    num_experts: int = 4,
    user_embed_dim: int = 64,
    item_embed_dim: int = 64,
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
    device: str = "cuda",
) -> MMoEDeepRanker:
    """
    Create an MMoE deep ranking model with default configuration.

    Args:
        num_experts: Number of expert networks
        user_embed_dim: User embedding dimension
        item_embed_dim: Item embedding dimension
        dropout: Dropout rate
        learning_rate: Learning rate
        device: Device for training

    Returns:
        Configured MMoEDeepRanker instance
    """
    config = MMoEConfig(
        num_experts=num_experts,
        user_embed_dim=user_embed_dim,
        item_embed_dim=item_embed_dim,
        dropout=dropout,
        learning_rate=learning_rate,
        device=device,
    )
    return MMoEDeepRanker(config)


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train MMoE deep ranking model")
    parser.add_argument(
        "--poi-csv",
        default="data/all/poi_expanded.csv",
        help="Path to POI data CSV"
    )
    parser.add_argument(
        "--events-csv",
        default="data/all/user_events.csv",
        help="Path to user events CSV"
    )
    parser.add_argument(
        "--output",
        default="outputs/ranking/mmoe_model.pt",
        help="Path to save trained model"
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=4,
        help="Number of expert networks"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for training"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--neg-ratio",
        type=int,
        default=4,
        help="Negative sampling ratio"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for training (cuda or cpu)"
    )
    parser.add_argument(
        "--plot-curves",
        action="store_true",
        help="Save training curves visualization"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Load data
    logger.info(f"Loading POI data from {args.poi_csv}")
    poi_df = pd.read_csv(args.poi_csv)

    logger.info(f"Loading events from {args.events_csv}")
    events_df = pd.read_csv(args.events_csv)

    # Create model
    config = MMoEConfig(
        num_experts=args.num_experts,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
    )
    model = MMoEDeepRanker(config)

    # Train
    logger.info("Starting training...")
    metrics = model.fit(
        poi_df=poi_df,
        events_df=events_df,
        negative_sampling_ratio=args.neg_ratio,
    )

    logger.info(f"Training completed. Best validation loss: {metrics.get('best_val_loss', 'N/A')}")

    # Export model
    model.export_model(args.output)

    # Save training curves
    if args.plot_curves:
        curves_path = Path(args.output).parent / "training_curves.png"
        model.save_training_curves(str(curves_path))

    logger.info("Done!")


if __name__ == "__main__":
    main()
