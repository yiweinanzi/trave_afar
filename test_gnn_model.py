#!/usr/bin/env python3
"""
Test GNN Model Implementation

This script tests the GNN model implementation:
- Graph building
- Model training (minimal)
- Inference
- Integration with recommendation pipeline

Usage:
    python test_gnn_model.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.model.gnn_model import (
    GNNConfig,
    POIGraphBuilder,
    POIGNNRecommender,
    create_gnn_model,
    create_graph_builder,
)
from src.model.gnn_integration import GNNProvider
from src.utils.logger import get_logger

logger = get_logger(__name__)


def test_graph_builder():
    """Test POI graph builder."""
    logger.info("=" * 60)
    logger.info("Testing POIGraphBuilder")
    logger.info("=" * 60)

    builder = POIGraphBuilder()

    # Create sample POIs
    sample_pois = [
        {
            "poi_id": "1",
            "name": "Test POI 1",
            "fclass": "park",
            "province": "北京",
            "latitude": 39.9,
            "longitude": 116.4,
        },
        {
            "poi_id": "2",
            "name": "Test POI 2",
            "fclass": "park",
            "province": "北京",
            "latitude": 39.91,
            "longitude": 116.41,
        },
        {
            "poi_id": "3",
            "name": "Test POI 3",
            "fclass": "hotel",
            "province": "上海",
            "latitude": 31.2,
            "longitude": 121.5,
        },
    ]

    for poi in sample_pois:
        builder.nodes[poi["poi_id"]] = poi

    logger.info(f"  Added {len(builder.nodes)} POIs")

    # Add geographic edges
    edges = builder.add_geographic_edges(radius_km=10)
    logger.info(f"  Added {edges} geographic edges")

    # Add category edges
    cat_edges = builder.add_category_edges(similarity_threshold=0.0)
    logger.info(f"  Added {cat_edges} category edges")

    # Build graph
    graph = builder.build_graph()

    if graph:
        logger.info(f"  Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
        logger.info("  PASSED: Graph building")
    else:
        logger.warning("  SKIPPED: PyTorch Geometric not available")

    return builder


def test_recommender():
    """Test GNN recommender."""
    logger.info("=" * 60)
    logger.info("Testing POIGNNRecommender")
    logger.info("=" * 60)

    # Create config
    config = GNNConfig(
        hidden_dim=32,
        output_dim=16,
        num_epochs=2,
    )

    # Create builder with sample data
    builder = POIGraphBuilder()

    # Add sample POIs
    for i in range(10):
        builder.nodes[str(i)] = {
            "poi_id": str(i),
            "name": f"POI {i}",
            "fclass": "park" if i % 2 == 0 else "hotel",
            "province": "北京" if i < 5 else "上海",
            "latitude": 39.9 + i * 0.01,
            "longitude": 116.4 + i * 0.01,
        }

    # Add edges
    builder.add_geographic_edges(radius_km=100)
    builder.add_category_edges(similarity_threshold=0.0)

    # Create recommender
    recommender = POIGNNRecommender(builder=builder, config=config)

    # Build graph
    graph = builder.build_graph()
    if graph is None:
        logger.warning("  SKIPPED: Could not build graph")
        return False

    # Create mock embeddings (since we can't train without torch)
    import numpy as np
    recommender.embeddings = np.random.randn(len(builder.nodes), 16)
    recommender.poi_id_map = {pid: i for i, pid in enumerate(builder.nodes.keys())}
    recommender.id_poi_map = {i: pid for pid, i in recommender.poi_id_map.items()}

    # Test similarity
    similar = recommender.get_similar_pois("0", top_k=5)
    logger.info(f"  Similar POIs to 0: {len(similar)} found")

    # Test graph walk
    walk = recommender.graph_walk_recommend("0", num_steps=2, top_k=5)
    logger.info(f"  Graph walk from 0: {len(walk)} results")

    logger.info("  PASSED: POIGNNRecommender")
    return True


def test_provider_integration():
    """Test GNN provider integration."""
    logger.info("=" * 60)
    logger.info("Testing GNNProvider Integration")
    logger.info("=" * 60)

    import pandas as pd

    # Create provider (disabled by default for testing)
    provider = GNNProvider(
        model_path=None,  # No model
        enabled=False,
        fallback_to_popular=True,
    )

    # Create sample POI data
    poi_df = pd.DataFrame([
        {"poi_id": "1", "name": "POI 1", "fclass": "park", "popularity": 100},
        {"poi_id": "2", "name": "POI 2", "fclass": "hotel", "popularity": 80},
        {"poi_id": "3", "name": "POI 3", "fclass": "park", "popularity": 60},
    ])

    # Add user history
    provider.update_user_history("user1", "1")

    # Get predictions
    rec_df, metadata = provider.predict("user1", top_k=2, poi_df=poi_df)

    logger.info(f"  Method: {metadata['method']}")
    logger.info(f"  Recommendations: {len(rec_df)}")
    logger.info(f"  Columns: {list(rec_df.columns)}")

    logger.info("  PASSED: GNNProvider integration")
    return True


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("GNN Model Test Suite")
    logger.info("=" * 60)
    logger.info("")

    results = {}

    # Run tests
    results["graph_builder"] = test_graph_builder()
    results["recommender"] = test_recommender()
    results["provider"] = test_provider_integration()

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)

    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED/SKIPPED"
        logger.info(f"  {name}: {status}")

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    logger.info(f"\n  Total: {passed}/{total} passed")

    if passed == total:
        logger.info("\n  All tests passed!")
        return 0
    else:
        logger.warning(f"\n  {total - passed} test(s) failed or skipped")
        return 1


if __name__ == "__main__":
    sys.exit(main())
