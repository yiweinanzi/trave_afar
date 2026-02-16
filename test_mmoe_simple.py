#!/usr/bin/env python3
"""
Quick test script for MMoE Deep Ranking Model.

Usage:
    python test_mmoe_simple.py
    python test_mmoe_simple.py --device cpu
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


def create_sample_data(num_users=20, num_pois=50, num_events=200):
    """Create synthetic data for testing."""
    logger.info(f"Creating sample data: {num_users} users, {num_pois} POIs, {num_events} events")

    # Create POI data
    categories = ["文化景点", "自然风光", "历史遗迹", "主题乐园", "美食街"]
    provinces = ["北京", "上海", "广东", "四川", "浙江"]

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


def main():
    parser = argparse.ArgumentParser(description="Test MMoE Deep Ranking Model")
    parser.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    parser.add_argument("--quick", action="store_true", help="Skip training")
    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("MMoE Deep Ranking Model Test")
    logger.info("=" * 50)

    # Create sample data
    poi_df, events_df = create_sample_data()

    # Create model
    logger.info("Creating model...")
    config = MMoEConfig(
        num_experts=2,
        user_embed_dim=32,
        item_embed_dim=32,
        epochs=2,
        batch_size=32,
        device=args.device,
    )

    model = MMoEDeepRanker(config)
    logger.info("Model created successfully")

    # Test training (if not quick mode)
    if not args.quick:
        logger.info("Testing training...")
        try:
            metrics = model.fit(
                poi_df=poi_df,
                events_df=events_df,
                negative_sampling_ratio=2,
            )
            logger.info(f"Training completed: {metrics}")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return 1

    # Test inference
    logger.info("Testing inference...")
    candidates = poi_df.head(10).to_dict("records")
    candidates = [
        {
            "poi_id": str(c["poi_id"]),
            "name": c["name"],
            "category": c["category"],
            "province": c["province"],
            "stay_min": c["stay_min"],
            "popularity": c["visit_count"],
        }
        for c in candidates
    ]

    results = model.predict(user_id="U0001", candidate_pois=candidates)
    logger.info(f"Got {len(results)} predictions")
    if results:
        poi_id, scores = results[0]
        logger.info(f"Top prediction: {poi_id} with scores {scores}")

    # Test export
    logger.info("Testing export...")
    model.export_model("/tmp/test_mmoe.pt")
    logger.info("Model exported successfully")

    logger.info("=" * 50)
    logger.info("All tests passed!")
    logger.info("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())
