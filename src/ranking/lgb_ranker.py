"""
LightGBM Ranking Model for POI Recommendation.

Implements a fast, interpretable baseline ranking model using LightGBM
with LambdaRank objective for learning-to-rank tasks.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LightGBMRanker:
    """
    LightGBM-based ranking model for POI recommendation.

    Features:
    - User-side: historical behavior statistics, preference vectors
    - POI-side: category, popularity, time windows, vector representations
    - Context: travel time, group size
    - Interaction: user-POI cross features

    Uses LambdaRank objective for optimizing ranking metrics (NDCG).
    """

    def __init__(
        self,
        objective: str = "lambdarank",
        metric: str = "ndcg",
        num_leaves: int = 64,
        learning_rate: float = 0.05,
        feature_fraction: float = 0.8,
        bagging_fraction: float = 0.8,
        bagging_freq: int = 5,
        min_data_in_leaf: int = 20,
        max_depth: int = -1,
        verbose: int = -1,
        random_state: int = 42,
    ):
        """
        Initialize LightGBM ranker.

        Args:
            objective: Learning objective (lambdarank, rank_xendcg, etc.)
            metric: Evaluation metric (ndcg, map, auc)
            num_leaves: Maximum number of leaves in one tree
            learning_rate: Learning rate for gradient boosting
            feature_fraction: Fraction of features used for each iteration
            bagging_fraction: Fraction of data used for bagging
            bagging_freq: Frequency for bagging
            min_data_in_leaf: Minimum data per leaf
            max_depth: Maximum tree depth (-1 for no limit)
            verbose: Verbosity level
            random_state: Random seed
        """
        self.params = {
            "objective": objective,
            "metric": metric,
            "num_leaves": num_leaves,
            "learning_rate": learning_rate,
            "feature_fraction": feature_fraction,
            "bagging_fraction": bagging_fraction,
            "bagging_freq": bagging_freq,
            "min_data_in_leaf": min_data_in_leaf,
            "max_depth": max_depth,
            "verbose": verbose,
            "random_state": random_state,
        }
        self.model = None
        self.feature_names: Optional[List[str]] = None
        self.feature_importance_: Optional[pd.DataFrame] = None

    def extract_features(
        self,
        user: Dict[str, Any],
        pois: List[Dict[str, Any]],
        context: Dict[str, Any],
        user_history: Optional[pd.DataFrame] = None,
        poi_embeddings: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Extract features for ranking.

        Args:
            user: User information dict with keys:
                - user_id: str
                - history_pois: List[str] (optional)
                - preference_vector: np.ndarray (optional)
            pois: List of POI dicts with keys:
                - poi_id: str
                - category: str
                - popularity: float (optional)
                - stay_min: int
                - open_min: int
                - close_min: int
                - lat: float
                - lon: float
                - embedding: np.ndarray (optional)
            context: Context information with keys:
                - hour: int (0-23)
                - day_of_week: int (0-6)
                - group_size: int (optional)
                - max_hours: float
            user_history: User's historical interactions DataFrame (optional)
            poi_embeddings: Pre-computed POI embeddings array (optional)

        Returns:
            DataFrame with extracted features for each POI
        """
        features_list = []

        # Extract user features once
        user_id = user.get("user_id", "unknown")
        history_pois = user.get("history_pois", [])
        preference_vec = user.get("preference_vector")

        # User statistics
        history_length = len(history_pois)
        history_categories = {}
        if user_history is not None and len(user_history) > 0 and "category" in user_history.columns:
            history_categories = user_history["category"].value_counts().to_dict()

        for idx, poi in enumerate(pois):
            row = {
                # IDs
                "user_id": user_id,
                "poi_id": poi.get("poi_id", ""),
                "poi_index": idx,

                # === User Features ===
                "user_history_length": history_length,
                "user_history_poi_count": history_length,

                # === POI Features ===
                "poi_stay_min": poi.get("stay_min", 60),
                "poi_open_hour": poi.get("open_min", 0) // 60,
                "poi_close_hour": poi.get("close_min", 1440) // 60,
                "poi_popularity": poi.get("popularity", 0),
                "poi_lat": poi.get("lat", 0),
                "poi_lon": poi.get("lon", 0),

                # === Context Features ===
                "ctx_hour": context.get("hour", 12),
                "ctx_day_of_week": context.get("day_of_week", 0),
                "ctx_group_size": context.get("group_size", 1),
                "ctx_max_hours": context.get("max_hours", 10),
            }

            # Category features
            category = poi.get("category", "unknown")
            row["poi_category"] = category
            row["user_category_preference"] = history_categories.get(category, 0)

            # Time feasibility
            hour = context.get("hour", 12)
            open_h = poi.get("open_min", 0) // 60
            close_h = poi.get("close_min", 1440) // 60
            row["time_feasible"] = 1 if open_h <= hour <= close_h else 0

            # Distance to closing time
            if hour < open_h:
                row["hours_until_open"] = open_h - hour
                row["hours_until_close"] = 24
            elif hour > close_h:
                row["hours_until_open"] = 24
                row["hours_until_close"] = 0
            else:
                row["hours_until_open"] = 0
                row["hours_until_close"] = close_h - hour

            # Stay time feasibility
            max_hours = context.get("max_hours", 10)
            stay_min = poi.get("stay_min", 60)
            row["stay_feasible"] = 1 if stay_min <= max_hours * 60 else 0
            row["stay_ratio"] = stay_min / (max_hours * 60 + 1)

            # History interaction features
            poi_id = poi.get("poi_id", "")
            if poi_id in history_pois:
                row["is_repeat"] = 1
                row["repeat_position"] = history_pois.index(poi_id)
            else:
                row["is_repeat"] = 0
                row["repeat_position"] = -1

            # Semantic similarity (if embeddings available)
            if preference_vec is not None and "embedding" in poi:
                poi_emb = poi["embedding"]
                if preference_vec.shape == poi_emb.shape:
                    similarity = np.dot(preference_vec, poi_emb) / (
                        np.linalg.norm(preference_vec) * np.linalg.norm(poi_emb) + 1e-8
                    )
                    row["semantic_similarity"] = float(similarity)
                else:
                    row["semantic_similarity"] = 0.0
            else:
                row["semantic_similarity"] = 0.0

            # Cross features: popularity * category preference
            row["popularity_category_score"] = (
                row["poi_popularity"] * (row["user_category_preference"] + 1)
            )

            # Time-of-day interaction
            row["hour_stay_interaction"] = row["ctx_hour"] * row["poi_stay_min"]

            features_list.append(row)

        df = pd.DataFrame(features_list)

        # Encode categorical features
        if "poi_category" in df.columns:
            df["poi_category_encoded"] = pd.Categorical(df["poi_category"]).codes

        return df

    def train(
        self,
        train_data: pd.DataFrame,
        label_column: str = "label",
        query_column: str = "user_id",
        valid_data: Optional[pd.DataFrame] = None,
        num_boost_round: int = 1000,
        early_stopping_rounds: int = 50,
    ) -> Dict[str, Any]:
        """
        Train the LightGBM ranking model.

        Args:
            train_data: Training DataFrame with features and labels
            label_column: Name of the relevance label column
            query_column: Name of the query/group column
            valid_data: Optional validation DataFrame
            num_boost_round: Maximum number of boosting rounds
            early_stopping_rounds: Early stopping patience

        Returns:
            Training metrics dictionary
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError(
                "lightgbm is required. Install with: pip install lightgbm"
            )

        # Prepare features and labels
        feature_cols = [col for col in train_data.columns if col not in
                       [label_column, query_column, "user_id", "poi_id"]]

        X = train_data[feature_cols]
        y = train_data[label_column].values

        # Compute group sizes (number of candidates per query)
        if query_column in train_data.columns:
            group_sizes = train_data.groupby(query_column, sort=False).size().values
        else:
            group_sizes = np.array([len(train_data)])

        # Create dataset
        train_dataset = lgb.Dataset(X, label=y, group=group_sizes)

        # Validation data
        valid_sets = [train_dataset]
        valid_names = ["train"]
        if valid_data is not None and len(valid_data) > 0:
            valid_X = valid_data[feature_cols]
            valid_y = valid_data[label_column].values
            if query_column in valid_data.columns:
                valid_groups = valid_data.groupby(query_column, sort=False).size().values
            else:
                valid_groups = np.array([len(valid_data)])

            valid_dataset = lgb.Dataset(valid_X, label=valid_y, group=valid_groups)
            valid_sets.append(valid_dataset)
            valid_names.append("valid")

        # Train
        self.model = lgb.train(
            self.params,
            train_dataset,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=100),
            ],
        )

        self.feature_names = self.model.feature_name()

        # Compute feature importance
        importance = self.model.feature_importance(importance_type="gain")
        self.feature_importance_ = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False)

        logger.info(f"Training completed. Best iteration: {self.model.best_iteration}")
        logger.info(f"Number of features: {len(self.feature_names)}")

        return {
            "best_iteration": self.model.best_iteration,
            "num_features": len(self.feature_names),
            "feature_importance": self.feature_importance_.head(10).to_dict(),
        }

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """
        Predict relevance scores for candidates.

        Args:
            X: Feature matrix (DataFrame or array)

        Returns:
            Array of relevance scores
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        # Convert array to DataFrame if needed
        if isinstance(X, np.ndarray):
            if self.feature_names is None:
                raise RuntimeError("Feature names not available. Cannot predict on array.")
            X = pd.DataFrame(X, columns=self.feature_names)

        return self.model.predict(X)

    def predict_rank(
        self,
        user: Dict[str, Any],
        pois: List[Dict[str, Any]],
        context: Dict[str, Any],
        user_history: Optional[pd.DataFrame] = None,
        top_k: int = 50,
    ) -> List[Tuple[str, float]]:
        """
        Rank POIs for a user and return top-K.

        Args:
            user: User information
            pois: List of candidate POIs
            context: Context information
            user_history: User's historical data
            top_k: Number of top results to return

        Returns:
            List of (poi_id, score) tuples sorted by score descending
        """
        # Extract features
        features = self.extract_features(user, pois, context, user_history)

        # Predict scores
        scores = self.predict(features)

        # Sort and return top-k
        sorted_idx = np.argsort(-scores)[:top_k]
        results = [
            (pois[i]["poi_id"], float(scores[i]))
            for i in sorted_idx
        ]

        return results

    def save_model(self, path: str) -> None:
        """
        Save model to file.

        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise RuntimeError("No model to save")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.model.booster_.save_model(str(path))
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """
        Load model from file.

        Args:
            path: Path to the saved model
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm is required")

        self.model = lgb.Booster(model_file=path)
        self.feature_names = self.model.feature_name()
        logger.info(f"Model loaded from {path}")

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance scores.

        Args:
            top_n: Return top N features

        Returns:
            DataFrame with feature importance
        """
        if self.feature_importance_ is None:
            if self.model is None:
                raise RuntimeError("Model not trained")
            importance = self.model.feature_importance(importance_type="gain")
            self.feature_importance_ = pd.DataFrame({
                "feature": self.feature_names,
                "importance": importance,
            }).sort_values("importance", ascending=False)

        return self.feature_importance_.head(top_n)

    def evaluate(
        self,
        test_data: pd.DataFrame,
        label_column: str = "label",
        query_column: str = "user_id",
        k_list: List[int] = [5, 10, 20],
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.

        Args:
            test_data: Test DataFrame
            label_column: Label column name
            query_column: Query column name
            k_list: List of K values for NDCG calculation

        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import ndcg_score

        # Prepare features
        feature_cols = [col for col in test_data.columns if col not in
                       [label_column, query_column, "user_id", "poi_id"]]
        X = test_data[feature_cols]
        y_true = test_data[label_column].values

        # Predict
        y_pred = self.predict(X)

        # Compute NDCG@K for each query
        ndcg_scores = {f"ndcg@{k}": [] for k in k_list}

        for query_id in test_data[query_column].unique():
            mask = test_data[query_column] == query_id
            true_relevance = y_true[mask].reshape(1, -1)
            pred_scores = y_pred[mask].reshape(1, -1)

            for k in k_list:
                # Handle case where k > num_items
                actual_k = min(k, mask.sum())
                if actual_k > 1:
                    score = ndcg_score(true_relevance, pred_scores, k=actual_k)
                    ndcg_scores[f"ndcg@{k}"].append(score)

        # Average scores
        results = {
            metric: np.mean(scores) if scores else 0.0
            for metric, scores in ndcg_scores.items()
        }

        # AUC
        from sklearn.metrics import roc_auc_score
        try:
            results["auc"] = roc_auc_score(y_true, y_pred)
        except ValueError:
            results["auc"] = 0.0

        return results


def create_training_data(
    events_df: pd.DataFrame,
    poi_df: pd.DataFrame,
    negative_sampling_ratio: int = 4,
    time_decay: bool = True,
) -> pd.DataFrame:
    """
    Create training data for ranking from user events.

    Args:
        events_df: User events DataFrame with columns:
            - user_id, poi_id, timestamp, action
        poi_df: POI data DataFrame
        negative_sampling_ratio: Number of negatives per positive
        time_decay: Apply time decay to relevance scores

    Returns:
        Training DataFrame with features and labels
    """
    # Merge events with POI data
    merged = events_df.merge(
        poi_df[["poi_id", "category", "popularity", "stay_min", "open_min", "close_min"]],
        on="poi_id",
        how="left",
    )

    # Assign relevance labels
    merged["label"] = merged["action"].map({
        "visit": 5,
        "click": 3,
        "view": 1,
    }).fillna(1)

    # Apply time decay (recent interactions more important)
    if time_decay and "timestamp" in merged.columns:
        max_time = merged["timestamp"].max()
        decay_days = (max_time - merged["timestamp"]) / 86400  # seconds to days
        merged["label"] = merged["label"] * np.exp(-decay_days / 30)  # 30-day half-life

    # Negative sampling
    positive_samples = merged[merged["label"] > 0]
    negative_samples = []

    for _, pos in positive_samples.iterrows():
        user_id = pos["user_id"]
        # Sample random POIs not in user's history
        user_pois = set(merged[merged["user_id"] == user_id]["poi_id"])
        candidate_pois = list(set(poi_df["poi_id"]) - user_pois)

        num_negatives = min(negative_sampling_ratio, len(candidate_pois))
        neg_pois = np.random.choice(candidate_pois, num_negatives, replace=False)

        for neg_poi_id in neg_pois:
            neg_row = pos.copy()
            neg_row["poi_id"] = neg_poi_id
            neg_row["label"] = 0

            # Update POI features
            neg_poi_data = poi_df[poi_df["poi_id"] == neg_poi_id].iloc[0]
            for col in ["category", "popularity", "stay_min", "open_min", "close_min"]:
                if col in neg_poi_data:
                    neg_row[col] = neg_poi_data[col]

            negative_samples.append(neg_row)

    # Combine
    train_df = pd.concat([positive_samples, pd.DataFrame(negative_samples)], ignore_index=True)

    # Add context features
    if "timestamp" in train_df.columns:
        train_df["hour"] = pd.to_datetime(train_df["timestamp"], unit="s").dt.hour
        train_df["day_of_week"] = pd.to_datetime(train_df["timestamp"], unit="s").dt.dayofweek
    else:
        train_df["hour"] = 12
        train_df["day_of_week"] = 0

    return train_df


# CLI entry point
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train LightGBM ranking model")
    parser.add_argument("--events-csv", required=True, help="Path to user events CSV")
    parser.add_argument("--poi-csv", required=True, help="Path to POI data CSV")
    parser.add_argument("--output", required=True, help="Path to save model")
    parser.add_argument("--neg-ratio", type=int, default=4, help="Negative sampling ratio")
    parser.add_argument("--num-leaves", type=int, default=64, help="LightGBM num_leaves")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger.info("Loading data...")

    # Load data
    events_df = pd.read_csv(args.events_csv)
    poi_df = pd.read_csv(args.poi_csv)

    logger.info(f"Loaded {len(events_df)} events, {len(poi_df)} POIs")

    # Create training data
    train_df = create_training_data(
        events_df,
        poi_df,
        negative_sampling_ratio=args.neg_ratio,
    )

    logger.info(f"Created {len(train_df)} training samples")

    # Split train/validation
    user_ids = train_df["user_id"].unique()
    np.random.shuffle(user_ids)
    split = int(len(user_ids) * 0.8)

    train_users = user_ids[:split]
    valid_users = user_ids[split:]

    train_data = train_df[train_df["user_id"].isin(train_users)]
    valid_data = train_df[train_df["user_id"].isin(valid_users)]

    logger.info(f"Train: {len(train_data)}, Valid: {len(valid_data)}")

    # Train model
    ranker = LightGBMRanker(num_leaves=args.num_leaves, learning_rate=args.lr)
    metrics = ranker.train(train_data, valid_data=valid_data)

    logger.info(f"Training metrics: {metrics}")

    # Evaluate
    eval_metrics = ranker.evaluate(valid_data)
    logger.info(f"Evaluation metrics: {eval_metrics}")

    # Save model
    ranker.save_model(args.output)

    # Save feature importance
    importance_path = Path(args.output).parent / "feature_importance.csv"
    ranker.get_feature_importance().to_csv(importance_path, index=False)
    logger.info(f"Feature importance saved to {importance_path}")


if __name__ == "__main__":
    main()
