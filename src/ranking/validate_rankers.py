#!/usr/bin/env python
"""
Validation script for ranking models.

Tests basic functionality without requiring training data.
"""
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_lightgbm_import():
    """Test LightGBM import and initialization."""
    logger.info("Testing LightGBM import...")
    try:
        from src.ranking import LightGBMRanker
        ranker = LightGBMRanker()
        logger.info("✓ LightGBMRanker imported and initialized")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to import LightGBMRanker: {e}")
        return False


def test_din_import():
    """Test DIN import and initialization."""
    logger.info("Testing DIN import...")
    try:
        from src.ranking import DeepInterestNetwork
        din = DeepInterestNetwork()
        logger.info("✓ DeepInterestNetwork imported and initialized")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to import DeepInterestNetwork: {e}")
        return False


def test_feature_extraction():
    """Test feature extraction."""
    logger.info("Testing feature extraction...")
    try:
        from src.ranking import LightGBMRanker

        ranker = LightGBMRanker()

        user = {
            "user_id": "U12345",
            "history_pois": ["1001", "1002"],
            "preference_vector": np.random.randn(64),
        }

        pois = [
            {
                "poi_id": "2001",
                "category": "文化景点",
                "popularity": 0.9,
                "stay_min": 120,
                "open_min": 570,
                "close_min": 1020,
                "lat": 39.9163,
                "lon": 116.3972,
            }
        ]

        context = {
            "hour": 10,
            "day_of_week": 2,
            "group_size": 2,
            "max_hours": 8,
        }

        features = ranker.extract_features(user, pois, context)

        assert features.shape[0] == len(pois), "Wrong number of rows"
        assert features.shape[1] > 10, "Too few features"

        logger.info(f"✓ Feature extraction successful: {features.shape}")
        return True
    except Exception as e:
        logger.error(f"✗ Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_store():
    """Test FeatureStore."""
    logger.info("Testing FeatureStore...")
    try:
        from src.ranking.deep_ranker import FeatureStore

        poi_df = pd.DataFrame({
            "poi_id": ["1001", "1002"],
            "category": ["文化景点", "自然风光"],
            "province": ["北京", "北京"],
            "visit_count": [1000, 800],
            "stay_min": [120, 180],
            "open_min": [570, 420],
            "close_min": [1020, 1140],
        })

        events_df = pd.DataFrame({
            "user_id": ["U001", "U001", "U002"],
            "poi_id": ["1001", "1002", "1001"],
            "timestamp": [1000000, 1000100, 1000200],
            "event_type": ["visit", "visit", "click"],
        })

        feature_store = FeatureStore(poi_df, events_df, cache_dir="/tmp/test_features")

        user_features = feature_store.get_user_features("U001")
        assert "user_id" in user_features, "Missing user_id"

        poi_features = feature_store.get_poi_features("1001")
        assert poi_features["poi_id"] == "1001", "Wrong POI ID"

        logger.info("✓ FeatureStore successful")
        return True
    except Exception as e:
        logger.error(f"✗ FeatureStore failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_unified_ranker():
    """Test unified DeepRanker interface."""
    logger.info("Testing unified DeepRanker...")
    try:
        from src.ranking.deep_ranker import create_ranker

        poi_df = pd.DataFrame({
            "poi_id": ["1001", "1002"],
            "category": ["文化景点", "自然风光"],
            "province": ["北京", "北京"],
            "visit_count": [1000, 800],
            "stay_min": [120, 180],
            "open_min": [570, 420],
            "close_min": [1020, 1140],
        })

        events_df = pd.DataFrame({
            "user_id": ["U001", "U001", "U002"],
            "poi_id": ["1001", "1002", "1001"],
            "timestamp": [1000000, 1000100, 1000200],
            "event_type": ["visit", "visit", "click"],
        })

        ranker_lgb = create_ranker(
            model_type="lightgbm",
            poi_df=poi_df,
            events_df=events_df,
        )
        assert ranker_lgb.model_type == "lightgbm", "Wrong model type"

        ranker_din = create_ranker(
            model_type="din",
            poi_df=poi_df,
            events_df=events_df,
        )
        assert ranker_din.model_type == "din", "Wrong model type"

        logger.info("✓ Unified DeepRanker successful")
        return True
    except Exception as e:
        logger.error(f"✗ Unified DeepRanker failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_data_creation():
    """Test training data creation utility."""
    logger.info("Testing training data creation...")
    try:
        from src.ranking.lgb_ranker import create_training_data

        events_df = pd.DataFrame({
            "user_id": ["U001", "U001", "U002"],
            "poi_id": ["1001", "1002", "1001"],
            "timestamp": [1000000, 1000100, 1000200],
            "action": ["visit", "click", "visit"],
        })

        poi_df = pd.DataFrame({
            "poi_id": ["1001", "1002"],
            "category": ["文化景点", "自然风光"],
            "province": ["北京", "北京"],
            "popularity": [0.9, 0.8],
            "stay_min": [120, 180],
            "open_min": [570, 420],
            "close_min": [1020, 1140],
        })

        train_df = create_training_data(
            events_df,
            poi_df,
            negative_sampling_ratio=2,
        )

        assert "label" in train_df.columns, "Missing label column"
        assert "hour" in train_df.columns, "Missing hour column"
        assert len(train_df) > len(events_df), "No negative samples added"

        logger.info(f"✓ Training data creation successful: {len(train_df)} samples")
        return True
    except Exception as e:
        logger.error(f"✗ Training data creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    logger.info("=" * 60)
    logger.info("Ranking Models Validation")
    logger.info("=" * 60)

    tests = [
        ("LightGBM Import", test_lightgbm_import),
        ("DIN Import", test_din_import),
        ("Feature Extraction", test_feature_extraction),
        ("FeatureStore", test_feature_store),
        ("Unified Ranker", test_unified_ranker),
        ("Training Data Creation", test_training_data_creation),
    ]

    results = []
    for name, test_func in tests:
        logger.info(f"\n{'─' * 40}")
        result = test_func()
        results.append((name, result))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Validation Summary")
    logger.info("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {name}")

    logger.info(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        logger.info("\n✓ All validation tests passed!")
        return 0
    else:
        logger.error(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
