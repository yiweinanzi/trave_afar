#!/usr/bin/env python3
"""
Inference script for MMoE Deep Ranking Model.

Demonstrates how to use the trained MMoE model for:
1. Single user-item prediction
2. Batch ranking of candidate POIs
3. Integration with the recommendation pipeline

Usage:
    python scripts/infer_mmoe_ranker.py --help
    python scripts/infer_mmoe_ranker.py --user U0001 --top-k 10
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ranking.deep_ranker import MMoEDeepRanker, MMoEConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference with MMoE Deep Ranking Model"
    )

    parser.add_argument(
        "--model-path",
        default="outputs/ranking/mmoe_model.pt",
        help="Path to trained model checkpoint"
    )
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
        "--user-id",
        help="Specific user ID for prediction"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top POIs to return"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference"
    )
    parser.add_argument(
        "--output",
        help="Path to save predictions JSON"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )

    return parser.parse_args()


def load_model(model_path: str, device: str) -> MMoEDeepRanker:
    """Load trained model from checkpoint."""
    logger.info(f"Loading model from {model_path}")

    config = MMoEConfig(device=device)
    model = MMoEDeepRanker(config)
    model.load_model(model_path)

    logger.info("Model loaded successfully")
    return model


def load_poi_data(poi_csv: str) -> pd.DataFrame:
    """Load POI data for inference."""
    logger.info(f"Loading POI data from {poi_csv}")
    poi_df = pd.read_csv(poi_csv)
    logger.info(f"Loaded {len(poi_df)} POIs")
    return poi_df


def get_user_history(user_id: str, events_df: pd.DataFrame, poi_df: pd.DataFrame) -> Dict[str, Any]:
    """Get user history for context."""
    user_events = events_df[events_df["user_id"] == user_id].sort_values("timestamp", ascending=False)

    history = user_events.head(10).merge(
        poi_df[["poi_id", "name", "category"]],
        on="poi_id",
        how="left",
    )

    return {
        "user_id": user_id,
        "history_pois": history["poi_id"].tolist(),
        "history_count": len(history),
    }


def create_candidate_pois(poi_df: pd.DataFrame, limit: int = 100) -> List[Dict[str, Any]]:
    """Create candidate POIs for ranking."""
    candidates = []

    # Sample or top candidates
    if len(poi_df) > limit:
        # Use top by some criteria (e.g., popularity)
        sampled = poi_df.head(limit)
    else:
        sampled = poi_df

    for _, row in sampled.iterrows():
        candidates.append({
            "poi_id": str(row["poi_id"]),
            "name": row.get("name", ""),
            "category": row.get("category", "unknown"),
            "province": row.get("province", "unknown"),
            "city": row.get("city", "unknown"),
            "lat": row.get("lat", 0),
            "lon": row.get("lon", 0),
            "stay_min": row.get("stay_min", 60),
            "popularity": row.get("visit_count", 0),
        })

    return candidates


def format_prediction_results(
    results: List[tuple],
    poi_df: pd.DataFrame,
    user_context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Format prediction results with POI details."""
    formatted = []

    for poi_id, scores in results:
        poi_info = poi_df[poi_df["poi_id"].astype(str) == poi_id]

        if len(poi_info) > 0:
            poi_row = poi_info.iloc[0]
            formatted.append({
                "poi_id": poi_id,
                "name": poi_row.get("name", ""),
                "category": poi_row.get("category", ""),
                "province": poi_row.get("province", ""),
                "city": poi_row.get("city", ""),
                "scores": scores,
                "combined_score": (
                    0.4 * scores["ctr"] +
                    0.5 * scores["visit"] +
                    0.1 * (scores["duration"] / 300)
                ),
            })

    return formatted


def interactive_inference(model: MMoEDeepRanker, poi_df: pd.DataFrame, events_df: pd.DataFrame):
    """Run interactive inference loop."""
    print("\n" + "=" * 50)
    print("MMoE Ranking - Interactive Mode")
    print("=" * 50)
    print("\nCommands:")
    print("  <user_id> - Rank for user")
    print("  top <k> <user_id> - Get top K for user")
    print("  compare <user1> <user2> - Compare two users")
    print("  quit - Exit")
    print()

    while True:
        try:
            cmd = input("\n> ").strip()

            if not cmd or cmd == "quit":
                break

            parts = cmd.split()
            cmd_type = parts[0].lower()

            if cmd_type == "top":
                if len(parts) >= 3:
                    k = int(parts[1])
                    user_id = parts[2]
                    rank_for_user(model, poi_df, events_df, user_id, k)
                else:
                    print("Usage: top <k> <user_id>")

            elif cmd_type == "compare":
                if len(parts) >= 3:
                    user1 = parts[1]
                    user2 = parts[2]
                    compare_users(model, poi_df, events_df, user1, user2)
                else:
                    print("Usage: compare <user1> <user2>")

            else:
                # Treat as user_id
                rank_for_user(model, poi_df, events_df, cmd, 10)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def rank_for_user(
    model: MMoEDeepRanker,
    poi_df: pd.DataFrame,
    events_df: pd.DataFrame,
    user_id: str,
    top_k: int,
):
    """Rank POIs for a specific user."""
    print(f"\nRanking for user: {user_id}")

    # Get user context
    user_context = get_user_history(user_id, events_df, poi_df)
    print(f"User history: {user_context['history_count']} interactions")

    # Create candidates
    candidates = create_candidate_pois(poi_df, limit=200)

    # Predict
    results = model.predict(
        user_id=user_id,
        candidate_pois=candidates,
        context=user_context,
    )[:top_k]

    # Format results
    formatted = format_prediction_results(results, poi_df, user_context)

    # Display
    print(f"\nTop {top_k} Recommendations:")
    print("-" * 80)
    print(f"{'Rank':<5} {'POI':<30} {'Category':<15} {'CTR':<6} {'Visit':<6} {'Dur':<6}")
    print("-" * 80)

    for i, item in enumerate(formatted):
        print(
            f"{i+1:<5} "
            f"{item['name'][:30]:<30} "
            f"{item['category'][:15]:<15} "
            f"{item['scores']['ctr']:.3f} "
            f"{item['scores']['visit']:.3f} "
            f"{item['scores']['duration']:.0f}m"
        )


def compare_users(
    model: MMoEDeepRanker,
    poi_df: pd.DataFrame,
    events_df: pd.DataFrame,
    user1_id: str,
    user2_id: str,
):
    """Compare recommendations for two users."""
    print(f"\nComparing recommendations for {user1_id} vs {user2_id}")

    candidates = create_candidate_pois(poi_df, limit=50)

    # Get recommendations for both users
    results1 = model.predict(user_id=user1_id, candidate_pois=candidates)[:10]
    results2 = model.predict(user_id=user2_id, candidate_pois=candidates)[:10]

    # Extract POI IDs
    top1 = {poi_id for poi_id, _ in results1}
    top2 = {poi_id for poi_id, _ in results2}

    overlap = top1 & top2

    print(f"\nUser 1 top 10: {len(top1)} unique POIs")
    print(f"User 2 top 10: {len(top2)} unique POIs")
    print(f"Overlap: {len(overlap)} POIs ({len(overlap)/10*100:.0f}%)")

    if overlap:
        print("\nCommon recommendations:")
        for poi_id in overlap:
            poi_info = poi_df[poi_df["poi_id"].astype(str) == poi_id]
            if len(poi_info) > 0:
                print(f"  - {poi_info.iloc[0]['name']}")


def main():
    args = parse_args()

    # Load model
    model = load_model(args.model_path, args.device)

    # Load data
    poi_df = load_poi_data(args.poi_csv)
    events_path = Path(args.events_csv)
    events_df = pd.read_csv(events_path) if events_path.exists() else None

    # Interactive mode
    if args.interactive:
        interactive_inference(model, poi_df, events_df)
        return

    # Single user prediction
    if args.user_id:
        user_context = get_user_history(args.user_id, events_df, poi_df) if events_df is not None else {"user_id": args.user_id}
        candidates = create_candidate_pois(poi_df)

        results = model.predict(
            user_id=args.user_id,
            candidate_pois=candidates,
            context=user_context,
        )[:args.top_k]

        formatted = format_prediction_results(results, poi_df, user_context)

        print(f"\nTop {args.top_k} Recommendations for {args.user_id}:")
        for i, item in enumerate(formatted):
            print(f"{i+1}. {item['name']} - {item['category']}")
            print(f"   CTR: {item['scores']['ctr']:.3f}, Visit: {item['scores']['visit']:.3f}, Duration: {item['scores']['duration']:.0f}m")

        # Save if requested
        if args.output:
            output_data = {
                "user_id": args.user_id,
                "recommendations": formatted,
            }
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to {args.output}")

    else:
        print("Please specify --user-id or use --interactive mode")


if __name__ == "__main__":
    main()
