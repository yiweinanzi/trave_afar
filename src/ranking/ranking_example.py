"""
Example usage of ranking models in GoAfar recommendation pipeline.

Demonstrates:
1. Using LightGBMRanker for candidate ranking
2. Using DeepInterestNetwork (DIN) for personalized ranking
3. Integration with the recommendation pipeline
"""
import logging
import random
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from src.ranking.lgb_ranker import LightGBMRanker
from src.ranking.deep_ranker import DeepInterestNetwork, FeatureStore, create_ranker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_lightgbm_ranking():
    """Example: Using LightGBM ranker for POI ranking."""
    logger.info("=" * 60)
    logger.info("LightGBM Ranking Example")
    logger.info("=" * 60)

    # Initialize ranker
    ranker = LightGBMRanker(
        num_leaves=64,
        learning_rate=0.05,
    )

    # Sample user data
    user = {
        "user_id": "U12345",
        "history_pois": ["1001", "1002", "1003"],
        "preference_vector": np.random.randn(128),  # Assuming 128-dim embeddings
    }

    # Sample candidate POIs
    pois = [
        {
            "poi_id": "2001",
            "name": "故宫博物院",
            "category": "文化景点",
            "popularity": 0.9,
            "stay_min": 120,
            "open_min": 570,  # 9:30 AM
            "close_min": 1020,  # 5:00 PM
            "lat": 39.9163,
            "lon": 116.3972,
            "embedding": np.random.randn(128),
        },
        {
            "poi_id": "2002",
            "name": "颐和园",
            "category": "自然风光",
            "popularity": 0.85,
            "stay_min": 180,
            "open_min": 420,  # 7:00 AM
            "close_min": 1140,  # 7:00 PM
            "lat": 40.0005,
            "lon": 116.2756,
            "embedding": np.random.randn(128),
        },
        {
            "poi_id": "2003",
            "name": "天坛公园",
            "category": "文化景点",
            "popularity": 0.88,
            "stay_min": 90,
            "open_min": 480,  # 8:00 AM
            "close_min": 1080,  # 6:00 PM
            "lat": 39.8822,
            "lon": 116.4097,
            "embedding": np.random.randn(128),
        },
    ]

    # Context
    context = {
        "hour": 10,  # 10:00 AM
        "day_of_week": 2,  # Tuesday
        "group_size": 2,
        "max_hours": 8,
    }

    # Extract features
    features = ranker.extract_features(user, pois, context)
    logger.info(f"\nExtracted {len(features)} features for {len(pois)} POIs")
    logger.info(f"Feature columns: {features.columns.tolist()[:10]}...")

    # Simulate trained model (in practice, call ranker.train() first)
    logger.info("\nNote: For actual predictions, train the model first using:")
    logger.info("  ranker.train(train_data, valid_data=valid_data)")

    # Example ranking workflow
    logger.info("\nRanking workflow:")
    logger.info("  1. Extract features for user-POI pairs")
    logger.info("  2. Train model with: ranker.train(train_data)")
    logger.info("  3. Predict with: ranker.predict(features)")
    logger.info("  4. Rank with: ranker.predict_rank(user, pois, context)")

    return ranker


def example_din_ranking():
    """Example: Using DIN for personalized ranking."""
    logger.info("\n" + "=" * 60)
    logger.info("Deep Interest Network (DIN) Example")
    logger.info("=" * 60)

    # Initialize DIN
    din = DeepInterestNetwork(
        embedding_dim=64,
        hidden_dims=[256, 128, 64],
        attention_heads=4,
        dropout=0.1,
    )

    # Sample training data
    train_data = pd.DataFrame([
        {
            "user_id": "U001",
            "poi_id": 101,
            "category": 1,
            "province": 2,
            "label": 1,
            "history_poi_ids": [100, 99, 98, 97],
            "history_categories": [1, 2, 1, 3],
            "history_provinces": [2, 2, 2, 3],
        },
        {
            "user_id": "U001",
            "poi_id": 102,
            "category": 2,
            "province": 2,
            "label": 0,
            "history_poi_ids": [100, 99, 98, 97],
            "history_categories": [1, 2, 1, 3],
            "history_provinces": [2, 2, 2, 3],
        },
    ])

    # Build model from data
    logger.info("\nBuilding DIN model architecture...")
    logger.info("Note: Call din.fit() with training data to train the model")

    # Show prediction workflow
    logger.info("\nPrediction workflow:")
    logger.info("  1. Build model: din.build_from_data(poi_df, events_df)")
    logger.info("  2. Train: din.fit(train_data, poi_vocab_size, ...)")
    logger.info("  3. Predict: din.predict(candidates, user_history)")

    # Example candidates
    candidates = pd.DataFrame([
        {"poi_id": 201, "category": 1, "province": 2, "popularity": 0.9},
        {"poi_id": 202, "category": 2, "province": 3, "popularity": 0.85},
    ])

    user_history = {
        "history_poi_ids": [100, 99, 98],
        "history_categories": [1, 2, 1],
        "history_provinces": [2, 2, 3],
    }

    logger.info(f"\nSample candidates: {len(candidates)} POIs")
    logger.info(f"User history length: {len(user_history['history_poi_ids'])}")

    return din


def example_feature_store():
    """Example: Using FeatureStore for feature engineering."""
    logger.info("\n" + "=" * 60)
    logger.info("FeatureStore Example")
    logger.info("=" * 60)

    # Sample POI data
    poi_df = pd.DataFrame({
        "poi_id": ["1001", "1002", "1003"],
        "category": ["文化景点", "自然风光", "文化景点"],
        "province": ["北京", "北京", "上海"],
        "visit_count": [1000, 800, 600],
        "stay_min": [120, 180, 90],
        "open_min": [570, 420, 480],
        "close_min": [1020, 1140, 1080],
    })

    # Sample events data
    events_df = pd.DataFrame({
        "user_id": ["U001", "U001", "U002", "U002"],
        "poi_id": ["1001", "1002", "1001", "1003"],
        "timestamp": [1000000, 1000100, 1000200, 1000300],
        "event_type": ["visit", "visit", "click", "visit"],
    })

    # Create feature store
    feature_store = FeatureStore(poi_df, events_df, cache_dir="outputs/features")

    # Get user features
    user_features = feature_store.get_user_features("U001")
    logger.info(f"\nUser features: {user_features}")

    # Get POI features
    poi_features = feature_store.get_poi_features("1001")
    logger.info(f"POI features: {poi_features}")

    # Get interaction features
    interaction_features = feature_store.get_interaction_features("U001", "1002", {"hour": 10})
    logger.info(f"Interaction features: {interaction_features}")

    # Build ranking features
    candidate_pois = ["1001", "1002", "1003"]
    ranking_features = feature_store.build_ranking_features("U001", candidate_pois, {"hour": 10})
    logger.info(f"\nRanking features shape: {ranking_features.shape}")
    logger.info(f"Ranking features columns: {ranking_features.columns.tolist()}")

    return feature_store


def example_unified_ranker():
    """Example: Using unified DeepRanker interface."""
    logger.info("\n" + "=" * 60)
    logger.info("Unified DeepRanker Example")
    logger.info("=" * 60)

    # Sample data
    poi_df = pd.DataFrame({
        "poi_id": ["1001", "1002", "1003"],
        "category": ["文化景点", "自然风光", "文化景点"],
        "province": ["北京", "北京", "上海"],
        "visit_count": [1000, 800, 600],
        "stay_min": [120, 180, 90],
        "open_min": [570, 420, 480],
        "close_min": [1020, 1140, 1080],
    })

    events_df = pd.DataFrame({
        "user_id": ["U001", "U001", "U002", "U002"],
        "poi_id": ["1001", "1002", "1001", "1003"],
        "timestamp": [1000000, 1000100, 1000200, 1000300],
        "event_type": ["visit", "visit", "click", "visit"],
    })

    # Create ranker with LightGBM
    ranker_lgb = create_ranker(
        model_type="lightgbm",
        poi_df=poi_df,
        events_df=events_df,
        num_leaves=64,
        learning_rate=0.05,
    )
    logger.info("\nCreated LightGBM ranker")

    # Create ranker with DIN
    ranker_din = create_ranker(
        model_type="din",
        poi_df=poi_df,
        events_df=events_df,
        embedding_dim=64,
    )
    logger.info("Created DIN ranker")

    # Example ranking
    candidate_pois = ["1001", "1002", "1003"]
    context = {"hour": 10, "day_of_week": 2}

    logger.info(f"\nRanking {len(candidate_pois)} candidates for user U001...")
    logger.info("Note: Train the model first using ranker.fit()")
    logger.info("Then rank with: ranker.rank(user_id, candidate_pois, context)")

    return ranker_lgb, ranker_din


def example_pipeline_integration():
    """Example: Integration with recommendation pipeline."""
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline Integration Example")
    logger.info("=" * 60)

    logger.info("""
Integration steps:

1. After candidate recall (e.g., from RecBole, vector search):

   candidates = recall_stage.get_candidates(query, top_k=100)

2. Extract features and rank:

   from src.ranking import LightGBMRanker, create_ranker

   # Load trained model
   ranker = create_ranker(model_type="lightgbm", poi_df=poi_df, events_df=events_df)
   ranker.load("outputs/ranking/lightgbm_model.txt")

   # Rank candidates
   user = {
       "user_id": request.user_id,
       "history_pois": user_history,
       "preference_vector": user_embedding,
   }

   context = {
       "hour": current_hour,
       "day_of_week": current_day,
       "group_size": request.group_size,
       "max_hours": request.max_hours,
   }

   ranked_pois = ranker.rank(
       user_id=request.user_id,
       candidate_pois=[c["poi_id"] for c in candidates],
       context=context,
       top_k=50,
   )

3. Use ranked candidates for routing/optimization:

   selected_pois = routing_stage.optimize(ranked_pois, constraints)

4. Return to user in response:

   response.route = [RouteStop(poi_id=pid, ...) for pid, score in ranked_pois]
    """)


def main():
    """Run all examples."""
    logger.info("GoAfar Ranking Models - Usage Examples\n")

    # Run examples
    example_lightgbm_ranking()
    example_din_ranking()
    example_feature_store()
    example_unified_ranker()
    example_pipeline_integration()

    logger.info("\n" + "=" * 60)
    logger.info("Examples completed!")
    logger.info("=" * 60)
    logger.info("\nFor training, use:")
    logger.info("  python -m src.ranking.train_rankers --model lightgbm")
    logger.info("  python -m src.ranking.train_rankers --model din")
    logger.info("\nFor more information, see the module documentation.")


if __name__ == "__main__":
    main()
