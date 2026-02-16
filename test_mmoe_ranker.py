#!/usr/bin/env python3
"""
Quick test script for MMoE Deep Ranking Model.

This script verifies the MMoE model implementation by:
1. Testing model initialization
2. Running a small training loop
3. Testing inference with sample data

Usage:
    python test_mmoe_ranker.py
    python test_mmoe_ranker.py --device cpu
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.ranking.deep_ranker import MMoEDeepRanker, MMoEConfig, create_mmoe_ranker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_sample_data(num_users: int = 50, num_pois: int = 200, num_events: int = 1000):
    """Create synthetic data for testing."""
    logger.info(f"Creating sample data: {num_users} users, {num_pois} POIs, {num_events} events")

    # Create POI data
    categories = ["文化景点", "自然风光", "历史遗迹", "主题乐园", "美食街"]
    provinces = ["北京", "上���", "广东", "四川", "浙江"]

    poi_data = []
    for i in range(num_pois):
        poi_data.append({
            "poi_id": f"P{i:04d}",
            "name": f"景点{i}",
            "category": np.random.choice(categories),
            "province": np.random.choice(provinces),
            "city": np.random.choice(provinces),
            "lat": 30.0 + np.random.randn() * 5,
            "lon": 110.0 + np.random.randn() * 10,
            "stay_min": np.random.randint(30, 180),
            "open_min": np.random.randint(300, 600),
            "close_min": np.random.randint(1080, 1320),
            "visit_count": np.random.randint(10, 1000),
        })
    poi_df = pd.DataFrame(poi_data)

    # Create events data
    actions = ["click", "fav", "visit"]
    events_data = []
    for i in range(num_events):
        events_data.append({
            "user_id": f"U{np.random.randint(0, num_users):04d}",
            "poi_id": f"P{np.random.randint(0, num_pois):04d}",
            "timestamp": 1700000000 + i * 3600,
            "action": np.random.choice(actions, p=[0.5, 0.3, 0.2]),
        })
    events_df = pd.DataFrame(events_data)

    logger.info(f"Created {len(poi_df)} POIs, {len(events_df)} events")
    return poi_df, events_df


def test_model_creation(device: str = "cuda"):
    """Test model initialization."""
    logger.info("Testing model creation...")

    config = MMoEConfig(
        num_experts=2,
        user_embed_dim=32,
        item_embed_dim=32,
        epochs=2,
        batch_size=32,
        device=device,
    )

    model = MMoEDeepRanker(config)
    logger.info("Model created successfully")
    logger.info(f"Config: num_experts={config.num_experts}, device={config.device}")

    return model


def test_training(model: MMoEDeepRanker, poi_df: pd.DataFrame, events_df: pd.DataFrame):
    """Test model training."""
    logger.info("Testing training...")

    metrics = model.fit(
        poi_df=poi_df,
        events_df=events_df,
        negative_sampling_ratio=2,
    )

    logger.info(f"Training completed: {metrics}")
    return metrics


def test_inference(model: MMoEDeepRanker, poi_df: pd.DataFrame):
    """Test model inference."""
    logger.info("Testing inference...")

    # Create sample candidates
    candidates = poi_df.head(10).to_dict("records")
    candidates = [
        {
            "poi_id": str(c["poi_id"]),
            "name": c["name"],
            "category": c["category"],
            "province": c["province"],
            "city": c.get("city", ""),
            "stay_min": c["stay_min"],
            "popularity": c["visit_count"],
        }
        for c in candidates
    ]

    # Test single predict
    user = {"user_id": "U0001"}
    results = model.predict(user_id="U0001", candidate_pois=candidates)

    logger.info(f"Got {len(results)} predictions")
    if results:
        poi_id, scores = results[0]
        logger.info(f"Top prediction: {poi_id} with scores {scores}")

    return results


def test_export_import(model: MMoEDeepRanker, tmp_path: str = "/tmp/test_mmoe.pt"):
    """Test model export and import."""
    logger.info("Testing export/import...")

    # Export
    model.export_model(tmp_path)
    logger.info(f"Model exported to {tmp_path}")

    # Create new model and load
    config = MMoEConfig(device="cpu")
    new_model = MMoEDeepRanker(config)
    new_model.load_model(tmp_path)
    logger.info("Model loaded successfully")

    return new_model


def main():
    parser = argparse.ArgumentParser(description="Test MMoE Deep Ranking Model")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--small-data", action="store_true")
    args = parser.parse_args()

    # Determine data size
    if args.small_data:
        num_users, num_pois, num_events = 20, 50, 200
    else:
        num_users, num_pois, num_events = 50, 200, 1000

    logger.info("=" * 50)
    logger.info("MMoE Deep Ranking Model Test")
    logger.info("=" * 50)

    try:
        # 1. Create sample data
        poi_df, events_df = create_sample_data(num_users, num_pois, num_events)

        # 2. Test model creation
        model = test_model_creation(args.device)

        # 3. Test training
        if not args.skip_training:
            metrics = test_training(model, poi_df, events_df)

            # 4. Test inference with trained model
            results = test_inference(model, poi_df)

            # 5. Test export/import
            test_export_import(model)

            logger.info("=" * 50)
            logger.info("All tests passed!")
            logger.info("=" * 50)
        else:
            logger.info("Training skipped (--skip-training)")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
