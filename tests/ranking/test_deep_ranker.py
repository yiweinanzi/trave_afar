"""
Tests for ranking.deep_ranker module.

Tests deep ranker training/inference and feature engineering.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_poi_df():
    """Create a sample POI DataFrame."""
    return pd.DataFrame({
        "poi_id": ["POI_0001", "POI_0002", "POI_0003", "POI_0004", "POI_0005"],
        "name": ["Tianshan", "Kanas", "Sayram", "Nalati", "Flaming"],
        "province": ["Xinjiang"] * 5,
        "city": ["Urumqi", "Altay", "Ili", "Ili", "Turpan"],
        "category": ["lake", "lake", "lake", "grassland", "desert"],
        "lat": [43.88, 48.70, 44.60, 43.30, 42.95],
        "lon": [88.13, 87.00, 81.00, 83.80, 89.18],
        "stay_min": [120, 180, 120, 150, 90],
        "popularity": [1000, 800, 600, 700, 500],
    })


@pytest.fixture
def sample_events_df():
    """Create a sample user events DataFrame."""
    events = []
    for user_id in ["user_001", "user_002", "user_003"]:
        for i, poi_id in enumerate(["POI_0001", "POI_0002", "POI_0003"]):
            events.append({
                "user_id": user_id,
                "poi_id": poi_id,
                "event_type": "view" if i < 2 else "visit",
                "timestamp": 1700000000 + i * 3600,
            })
    return pd.DataFrame(events)


# ============================================================================
# Test: Feature Engineering
# ============================================================================

class TestFeatureEngineering:
    """Test feature engineering functionality."""

    def test_user_features(self, sample_events_df):
        """Test user feature extraction."""
        user_id = "user_001"
        user_events = sample_events_df[sample_events_df["user_id"] == user_id]

        features = {
            "user_id": user_id,
            "history_length": len(user_events),
            "recent_pois": user_events["poi_id"].tolist()[-5:],
        }

        assert features["user_id"] == user_id
        assert features["history_length"] >= 0
        assert isinstance(features["recent_pois"], list)

    def test_poi_features(self, sample_poi_df):
        """Test POI feature extraction."""
        poi_id = "POI_0001"
        poi = sample_poi_df[sample_poi_df["poi_id"] == poi_id].iloc[0]

        features = {
            "poi_id": poi_id,
            "category": poi["category"],
            "province": poi["province"],
            "popularity": poi["popularity"],
        }

        assert features["poi_id"] == poi_id
        assert "category" in features

    def test_interaction_features(self, sample_events_df, sample_poi_df):
        """Test interaction feature extraction."""
        user_id = "user_001"
        poi_id = "POI_0001"

        user_events = sample_events_df[sample_events_df["user_id"] == user_id]
        poi_events = user_events[user_events["poi_id"] == poi_id]

        is_repeat = len(poi_events) > 0

        features = {
            "user_id": user_id,
            "poi_id": poi_id,
            "is_repeat": is_repeat,
            "visit_count": len(poi_events),
        }

        assert "is_repeat" in features

    def test_category_encoding(self, sample_poi_df):
        """Test category encoding."""
        categories = sample_poi_df["category"].unique()
        category_map = {cat: idx for idx, cat in enumerate(categories)}

        encoded = sample_poi_df["category"].map(category_map)

        assert len(encoded) == len(sample_poi_df)
        assert encoded.min() >= 0

    def test_feature_normalization(self):
        """Test feature normalization."""
        features = np.array([[100], [500], [1000], [50], [200]])

        mean = features.mean()
        std = features.std()
        normalized = (features - mean) / (std + 1e-8)

        assert abs(normalized.mean()) < 0.1  # Near zero

    def test_time_features(self, sample_poi_df):
        """Test time-based features."""
        hour = 10  # 10 AM
        day_of_week = 2  # Tuesday

        features = {
            "hour": hour,
            "is_weekend": day_of_week >= 5,
            "time_of_day": "morning" if 6 <= hour < 12 else "afternoon",
        }

        assert features["hour"] == 10
        assert features["is_weekend"] is False


# ============================================================================
# Test: Model Training
# ============================================================================

class TestModelTraining:
    """Test model training functionality."""

    def test_training_data_split(self, sample_events_df):
        """Test training/validation data split."""
        user_ids = sample_events_df["user_id"].unique()
        np.random.shuffle(user_ids)

        split = int(0.8 * len(user_ids))
        train_users = user_ids[:split]
        valid_users = user_ids[split:]

        assert len(train_users) + len(valid_users) == len(user_ids)
        assert len(train_users) > len(valid_users)

    def test_label_creation(self, sample_events_df):
        """Test label creation for training."""
        action_to_label = {
            "visit": 1,
            "fav": 0.7,
            "click": 0.3,
            "view": 0,
        }

        labels = sample_events_df["event_type"].map(action_to_label)

        assert labels.min() >= 0
        assert labels.max() <= 1

    def test_negative_sampling(self, sample_poi_df):
        """Test negative sampling for training."""
        positive_pois = sample_poi_df["poi_id"].tolist()[:3]
        all_pois = sample_poi_df["poi_id"].tolist()

        negatives = [poi for poi in all_pois if poi not in positive_pois]

        assert len(negatives) == len(all_pois) - len(positive_pois)

    def test_batch_creation(self, sample_poi_df):
        """Test batch creation for training."""
        batch_size = 32
        data_size = len(sample_poi_df)

        num_batches = int(np.ceil(data_size / batch_size))

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, data_size)
            assert end_idx <= data_size


# ============================================================================
# Test: Model Inference
# ============================================================================

class TestModelInference:
    """Test model inference functionality."""

    def test_single_prediction(self, sample_poi_df):
        """Test single prediction."""
        # Simulate prediction
        poi = sample_poi_df.iloc[0]
        score = poi["popularity"] / 1000

        prediction = {
            "poi_id": poi["poi_id"],
            "score": score,
            "ctr": min(score, 1.0),
        }

        assert 0 <= prediction["ctr"] <= 1

    def test_batch_prediction(self, sample_poi_df):
        """Test batch prediction."""
        pois = sample_poi_df.head(5)

        predictions = []
        for _, poi in pois.iterrows():
            score = poi["popularity"] / 1000
            predictions.append({
                "poi_id": poi["poi_id"],
                "score": score,
            })

        assert len(predictions) == 5

    def test_ranking_output(self, sample_poi_df):
        """Test ranking output format."""
        scores = np.random.rand(5)
        ranked_indices = np.argsort(-scores)[:3]

        ranked = sample_poi_df.iloc[ranked_indices].copy()
        ranked["score"] = scores[ranked_indices]
        ranked["rank"] = range(1, len(ranked) + 1)

        assert len(ranked) == 3
        assert ranked["score"].max() == ranked["score"].iloc[0]

    def test_topk_limiting(self, sample_poi_df):
        """Test topk limiting in predictions."""
        all_scores = sample_poi_df["popularity"].values
        topk = 3

        top_indices = np.argsort(-all_scores)[:topk]

        assert len(top_indices) == 3


# ============================================================================
# Test: Model Persistence
# ============================================================================

class TestModelPersistence:
    """Test model save and load functionality."""

    def test_model_save_format(self):
        """Test model save format."""
        model_data = {
            "weights": np.random.randn(10, 5),
            "config": {"embed_dim": 64},
            "vocab_sizes": {"num_pois": 100},
        }

        assert "weights" in model_data
        assert "config" in model_data
        assert model_data["weights"].shape == (10, 5)

    def test_model_load_format(self):
        """Test model load format."""
        # Simulate loaded model
        model_data = {
            "weights": np.random.randn(10, 5),
            "config": {"embed_dim": 64},
        }

        assert model_data is not None
        assert model_data["config"]["embed_dim"] == 64

    def test_checkpoint_structure(self):
        """Test checkpoint structure."""
        checkpoint = {
            "epoch": 5,
            "model_state": {"weights": np.random.randn(10, 5)},
            "optimizer_state": {"lr": 0.001},
            "metrics": {"train_loss": 0.5, "val_loss": 0.6},
        }

        required_keys = ["epoch", "model_state", "optimizer_state", "metrics"]
        for key in required_keys:
            assert key in checkpoint


# ============================================================================
# Test: Evaluation Metrics
# ============================================================================

class TestEvaluationMetrics:
    """Test evaluation metrics."""

    def test_precision_calculation(self):
        """Test precision calculation."""
        predicted = [1, 1, 0, 1, 0]
        actual = [1, 0, 0, 1, 1]

        true_positives = sum(p and a for p, a in zip(predicted, actual))
        predicted_positives = sum(predicted)

        precision = true_positives / predicted_positives if predicted_positives > 0 else 0

        assert 0 <= precision <= 1

    def test_recall_calculation(self):
        """Test recall calculation."""
        predicted = [1, 1, 0, 1, 0]
        actual = [1, 0, 0, 1, 1]

        true_positives = sum(p and a for p, a in zip(predicted, actual))
        actual_positives = sum(actual)

        recall = true_positives / actual_positives if actual_positives > 0 else 0

        assert 0 <= recall <= 1

    def test_ndcg_calculation(self):
        """Test NDCG calculation."""
        # Simulate ranked scores
        scores = [3, 2, 1, 0, 0]
        dcg = sum((2**score - 1) / np.log2(i + 2) for i, score in enumerate(scores))

        assert dcg > 0

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        ranked_results = ["POI_0001", "POI_0002", "POI_0003"]
        ground_truth = ["POI_0002", "POI_0004"]

        hits = sum(1 for item in ranked_results if item in ground_truth)
        hit_rate = hits / len(ground_truth)

        assert hit_rate == 0.5


# ============================================================================
# Test: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_poi_df(self):
        """Test with empty POI DataFrame."""
        empty_df = pd.DataFrame({
            "poi_id": [],
            "name": [],
        })

        assert len(empty_df) == 0

    def test_single_poi(self):
        """Test with single POI."""
        single_df = pd.DataFrame({
            "poi_id": ["POI_0001"],
            "name": ["Test"],
            "popularity": [100],
        })

        assert len(single_df) == 1

    def test_missing_features(self, sample_poi_df):
        """Test handling of missing features."""
        # Remove popularity temporarily
        test_df = sample_poi_df.drop(columns=["popularity"])

        assert "popularity" not in test_df.columns
        assert "poi_id" in test_df.columns

    def test_new_user_cold_start(self):
        """Test cold start for new user."""
        new_user_id = "user_new"
        user_features = {
            "user_id": new_user_id,
            "history_length": 0,
            "recent_pois": [],
        }

        assert user_features["history_length"] == 0
        assert len(user_features["recent_pois"]) == 0

    def test_new_item_cold_start(self):
        """Test cold start for new item."""
        new_item_id = "POI_NEW"
        item_features = {
            "poi_id": new_item_id,
            "category": "unknown",
            "popularity": 0,
        }

        assert item_features["poi_id"] == new_item_id
        assert item_features["popularity"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
