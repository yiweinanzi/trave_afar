#!/usr/bin/env python3
"""
GNN Model Inference Script

This script demonstrates inference with a trained GNN model:
- Get POI embeddings
- Find similar POIs
- Generate recommendations via graph walk
- Integrate with GoAfar recommendation pipeline

Usage:
    python scripts/infer_gnn_model.py --model outputs/gnn/model.pkl --poi-id 87494665
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.gnn_model import POIGNNRecommender, create_gnn_model
from src.utils.logger import get_logger

logger = get_logger(__name__)


def inference_demo(
    model_path: Path,
    poi_id: str = None,
    top_k: int = 10,
) -> None:
    """
    Run inference demo with trained GNN model.

    Args:
        model_path: Path to trained model
        poi_id: Query POI ID (None = random)
        top_k: Number of results to return
    """
    logger.info("=" * 60)
    logger.info("GNN Model Inference")
    logger.info("=" * 60)

    # Load model
    logger.info(f"Loading model from {model_path}")
    recommender = create_gnn_model()
    recommender.load_model(model_path)

    if recommender.embeddings is None:
        logger.error("No embeddings available")
        return

    logger.info(f"  Embeddings: {recommender.embeddings.shape}")
    logger.info(f"  POIs: {len(recommender.poi_id_map)}")

    # Pick a POI
    if poi_id is None:
        import random
        poi_id = random.choice(list(recommender.poi_id_map.keys()))
        logger.info(f"Randomly selected POI: {poi_id}")
    else:
        if str(poi_id) not in recommender.poi_id_map:
            logger.error(f"POI {poi_id} not found in model")
            return

    # Get POI info
    if poi_id in recommender.builder.nodes:
        node = recommender.builder.nodes[poi_id]
        logger.info(f"\nQuery POI:")
        logger.info(f"  ID: {poi_id}")
        logger.info(f"  Name: {node.get('name', 'N/A')}")
        logger.info(f"  Category: {node.get('fclass', 'N/A')}")
        logger.info(f"  Province: {node.get('province', 'N/A')}")

    # Get embedding
    emb = recommender.get_poi_embedding(poi_id)
    if emb is not None:
        logger.info(f"\nEmbedding (first 10 dims): {emb[:10]}")

    # Find similar POIs
    logger.info(f"\nTop {top_k} Similar POIs:")
    logger.info("-" * 60)

    similar = recommender.get_similar_pois(poi_id, top_k=top_k)

    for i, (sim_poi_id, score) in enumerate(similar, 1):
        if sim_poi_id in recommender.builder.nodes:
            node = recommender.builder.nodes[sim_poi_id]
            name = node.get("name", "N/A")
            fclass = node.get("fclass", "N/A")
            logger.info(f"{i}. {sim_poi_id} | {score:.4f} | {name} ({fclass})")

    # Graph walk recommendations
    logger.info(f"\nGraph Walk Recommendations (starting from {poi_id}):")
    logger.info("-" * 60)

    walk_results = recommender.graph_walk_recommend(poi_id, num_steps=3, top_k=top_k)

    for i, (rec_poi_id, score) in enumerate(walk_results, 1):
        if rec_poi_id in recommender.builder.nodes:
            node = recommender.builder.nodes[rec_poi_id]
            name = node.get("name", "N/A")
            fclass = node.get("fclass", "N/A")
            logger.info(f"{i}. {rec_poi_id} | {score:.4f} | {name} ({fclass})")

    # Example: Integration with recommendation pipeline
    logger.info(f"\nExample: Integration with Recommendation Pipeline")
    logger.info("-" * 60)

    # Simulate a user with some visited POIs
    visited = [poi_id]
    if similar:
        visited.append(similar[0][0])

    # Get candidate POIs (similar + walk results)
    candidate_ids = list(set([p for p, _ in similar] + [p for p, _ in walk_results]))
    candidates = []
    for cid in candidate_ids[:50]:  # Limit for demo
        if cid in recommender.builder.nodes:
            node = recommender.builder.nodes[cid]
            candidates.append({
                "poi_id": cid,
                "name": node.get("name", ""),
                "fclass": node.get("fclass", ""),
                "province": node.get("province", ""),
                "popularity": 0.5,  # Placeholder
            })

    # Predict
    predictions = recommender.predict(
        user_id="demo_user",
        candidate_pois=candidates,
        context={"visited_pois": visited},
    )

    logger.info(f"Top {top_k} Recommendations:")
    for i, (rec_id, scores) in enumerate(predictions[:top_k], 1):
        if rec_id in recommender.builder.nodes:
            node = recommender.builder.nodes[rec_id]
            name = node.get("name", "N/A")
            gnn_score = scores.get("gnn_score", 0)
            logger.info(f"{i}. {rec_id} | {gnn_score:.4f} | {name}")


def batch_embedding_export(
    model_path: Path,
    output_path: Path,
    format: str = "npy",
) -> None:
    """
    Export POI embeddings for use in other systems.

    Args:
        model_path: Path to trained model
        output_path: Output path
        format: Output format (npy, csv, json)
    """
    logger.info(f"Exporting embeddings from {model_path}")

    recommender = create_gnn_model()
    recommender.load_model(model_path)

    if recommender.embeddings is None:
        logger.error("No embeddings available")
        return

    if format == "npy":
        import numpy as np
        np.save(output_path, recommender.embeddings)
        logger.info(f"Saved embeddings to {output_path}")

    elif format == "csv":
        import pandas as pd
        df_data = []
        for poi_id, idx in recommender.poi_id_map.items():
            emb = recommender.embeddings[idx]
            row = {"poi_id": poi_id}
            row.update({f"dim_{i}": v for i, v in enumerate(emb)})
            df_data.append(row)

        df = pd.DataFrame(df_data)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved embeddings to {output_path}")

    elif format == "json":
        data = {}
        for poi_id, idx in recommender.poi_id_map.items():
            emb = recommender.embeddings[idx]
            data[poi_id] = emb.tolist()

        with open(output_path, "w") as f:
            json.dump(data, f)
        logger.info(f"Saved embeddings to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="GNN model inference"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="outputs/gnn/model.pkl",
        help="Path to trained model",
    )
    parser.add_argument(
        "--poi-id",
        type=str,
        default=None,
        help="Query POI ID (default: random)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results",
    )
    parser.add_argument(
        "--export-embeddings",
        type=str,
        default=None,
        help="Export embeddings to file (path)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="npy",
        choices=["npy", "csv", "json"],
        help="Export format",
    )

    args = parser.parse_args()

    # Convert paths
    model_path = Path(args.model)

    # Validate
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        sys.exit(1)

    # Export mode
    if args.export_embeddings:
        batch_embedding_export(
            model_path=model_path,
            output_path=Path(args.export_embeddings),
            format=args.format,
        )
    else:
        # Inference demo
        inference_demo(
            model_path=model_path,
            poi_id=args.poi_id,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()
