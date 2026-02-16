#!/usr/bin/env python3
"""
Build POI Graph for GoAfar GNN Model

This script constructs a POI graph from various data sources:
- POI attributes (category, location)
- Co-occurrence in routes
- Geographic proximity
- Category similarity

The graph is saved for use in GNN training.

Usage:
    python scripts/build_poi_graph.py --poi-dir data/shengfen_pois --output outputs/gnn/graph.pkl
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.gnn_model import POIGraphBuilder
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_graph_from_poi_data(
    poi_dir: Path,
    output_path: Path,
    geo_radius_km: float = 1.0,
    cooccurrence_window: int = 5,
    category_threshold: float = 0.8,
) -> None:
    """
    Build POI graph from directory of POI CSV files.

    Args:
        poi_dir: Directory containing province POI files
        output_path: Path to save graph
        geo_radius_km: Radius for geographic edges
        cooccurrence_window: Window for co-occurrence edges
        category_threshold: Threshold for category similarity edges
    """
    logger.info("=" * 60)
    logger.info("Building POI Graph")
    logger.info("=" * 60)

    # Initialize builder
    builder = POIGraphBuilder(cache_dir=str(output_path.parent))

    # Load POIs
    logger.info(f"Loading POIs from {poi_dir}")
    num_pois = builder.add_pois_from_csv(poi_dir)
    logger.info(f"  Total POIs: {num_pois}")

    if num_pois == 0:
        logger.error("No POIs loaded, exiting")
        return

    # Add geographic edges
    logger.info("Adding geographic proximity edges...")
    geo_edges = builder.add_geographic_edges(
        radius_km=geo_radius_km,
        max_neighbors=10,
    )
    logger.info(f"  Geographic edges: {geo_edges}")

    # Add category similarity edges
    logger.info("Adding category similarity edges...")
    cat_edges = builder.add_category_edges(
        similarity_threshold=category_threshold,
    )
    logger.info(f"  Category edges: {cat_edges}")

    # Build and save graph
    logger.info("Building PyG graph structure...")
    graph = builder.build_graph()

    if graph is not None:
        logger.info(f"  Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")

    # Save
    logger.info(f"Saving graph to {output_path}")
    saved_path = builder.save_graph(output_path)
    logger.info(f"  Saved: {saved_path}")

    # Summary
    logger.info("=" * 60)
    logger.info("Graph Construction Summary")
    logger.info("=" * 60)
    logger.info(f"  Nodes (POIs): {len(builder.nodes)}")
    logger.info(f"  Co-occurrence edges: {len(builder.edges['cooccurrence'])}")
    logger.info(f"  Geographic edges: {len(builder.edges['geographic'])}")
    logger.info(f"  Category edges: {len(builder.edges['category'])}")
    logger.info(f"  Categories: {len(builder.category_map)}")
    logger.info(f"  Provinces: {len(builder.province_map)}")

    logger.info("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Build POI graph for GNN training"
    )
    parser.add_argument(
        "--poi-dir",
        type=str,
        default="data/shengfen_pois",
        help="Directory containing POI CSV files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/gnn/graph.pkl",
        help="Output path for graph",
    )
    parser.add_argument(
        "--geo-radius",
        type=float,
        default=1.0,
        help="Geographic edge radius (km)",
    )
    parser.add_argument(
        "--category-threshold",
        type=float,
        default=0.8,
        help="Category similarity threshold (0-1)",
    )
    parser.add_argument(
        "--cooc-window",
        type=int,
        default=5,
        help="Co-occurrence window size",
    )
    parser.add_argument(
        "--provinces",
        type=str,
        nargs="*",
        default=None,
        help="Specific provinces to include (default: all)",
    )

    args = parser.parse_args()

    # Convert paths
    poi_dir = Path(args.poi_dir)
    output_path = Path(args.output)

    # Validate
    if not poi_dir.exists():
        logger.error(f"POI directory not found: {poi_dir}")
        sys.exit(1)

    # Build
    build_graph_from_poi_data(
        poi_dir=poi_dir,
        output_path=output_path,
        geo_radius_km=args.geo_radius,
        cooccurrence_window=args.cooc_window,
        category_threshold=args.category_threshold,
    )


if __name__ == "__main__":
    main()
