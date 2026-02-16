#!/usr/bin/env python3
"""
Training script for MMoE Deep Ranking Model.

This script provides a convenient interface for training the MMoE multi-task
ranking model with various configurations.

Usage:
    python scripts/train_mmoe_ranker.py --help
    python scripts/train_mmoe_ranker.py --quick
    python scripts/train_mmoe_ranker.py --full --num-experts 8
"""
import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ranking.deep_ranker import MMoEDeepRanker, MMoEConfig, create_mmoe_ranker
from src.ranking.lgb_ranker import LightGBMRanker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MMoE Deep Ranking Model for POI Recommendation"
    )

    # Data paths
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
        "--output-dir",
        default="outputs/ranking",
        help="Output directory for models"
    )

    # Model architecture
    parser.add_argument(
        "--num-experts",
        type=int,
        default=4,
        help="Number of expert networks in MMoE"
    )
    parser.add_argument(
        "--user-embed-dim",
        type=int,
        default=64,
        help="User embedding dimension"
    )
    parser.add_argument(
        "--item-embed-dim",
        type=int,
        default=64,
        help="Item embedding dimension"
    )
    parser.add_argument(
        "--use-attention",
        action="store_true",
        default=True,
        help="Use attention for sequence encoding"
    )
    parser.add_argument(
        "--category-embed-dim",
        type=int,
        default=16,
        help="Category embedding dimension"
    )
    parser.add_argument(
        "--seq-embed-dim",
        type=int,
        default=64,
        help="Sequence embedding dimension"
    )
    parser.add_argument(
        "--province-embed-dim",
        type=int,
        default=16,
        help="Province embedding dimension"
    )
    parser.add_argument(
        "--user-tower-dims",
        type=int,
        nargs="+",
        default=[256, 128],
        help="User tower hidden dimensions"
    )
    parser.add_argument(
        "--item-tower-dims",
        type=int,
        nargs="+",
        default=[256, 128],
        help="Item tower hidden dimensions"
    )
    parser.add_argument(
        "--expert-hidden-dims",
        type=int,
        nargs="+",
        default=[128, 64],
        help="Expert network hidden dimensions"
    )

    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for training"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate"
    )
    parser.add_argument(
        "--neg-ratio",
        type=int,
        default=4,
        help="Negative sampling ratio"
    )

    # Task weights
    parser.add_argument(
        "--ctr-weight",
        type=float,
        default=1.0,
        help="Weight for CTR task loss"
    )
    parser.add_argument(
        "--visit-weight",
        type=float,
        default=1.0,
        help="Weight for Visit task loss"
    )
    parser.add_argument(
        "--duration-weight",
        type=float,
        default=0.1,
        help="Weight for Duration task loss"
    )

    # Device
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for training"
    )

    # Presets
    parser.add_argument(
        "--preset",
        choices=["quick", "base", "large"],
        help="Use preset configuration"
    )

    # Compare with baseline
    parser.add_argument(
        "--compare-with-lightgbm",
        action="store_true",
        help="Also train LightGBM baseline for comparison"
    )

    # Visualization
    parser.add_argument(
        "--plot-curves",
        action="store_true",
        help="Save training curves visualization"
    )

    return parser.parse_args()


def apply_preset(args):
    """Apply preset configuration."""
    if args.preset == "quick":
        args.epochs = 5
        args.num_experts = 2
        args.batch_size = 256
        args.user_embed_dim = 32
        args.item_embed_dim = 32
        args.category_embed_dim = 16
        args.seq_embed_dim = 32  # Match item_embed_dim to avoid projection
        args.province_embed_dim = 16
        # Adjust tower dims for smaller embeddings
        # user_input_dim = 32 + 32 + 32 = 96
        # item_input_dim = 32 + 16 + 16 + 16 + 7 = 87
        args.user_tower_dims = [96, 64]
        args.item_tower_dims = [96, 64]  # >= 87
        args.expert_hidden_dims = [64, 32]
    elif args.preset == "base":
        args.epochs = 20
        args.num_experts = 4
        args.batch_size = 512
    elif args.preset == "large":
        args.epochs = 50
        args.num_experts = 8
        args.batch_size = 1024
        args.user_embed_dim = 128
        args.item_embed_dim = 128
        args.dropout = 0.2


def train_lightgbm_baseline(poi_df, events_df, output_dir):
    """Train LightGBM baseline for comparison."""
    logger.info("=" * 50)
    logger.info("Training LightGBM baseline...")
    logger.info("=" * 50)

    from src.ranking.lgb_ranker import create_training_data

    # Prepare training data
    train_df = create_training_data(
        events_df,
        poi_df,
        negative_sampling_ratio=4,
    )

    # Split train/valid
    user_ids = train_df["user_id"].unique()
    import numpy as np
    np.random.seed(42)
    np.random.shuffle(user_ids)
    split = int(len(user_ids) * 0.8)

    train_data = train_df[train_df["user_id"].isin(user_ids[:split])]
    valid_data = train_df[train_df["user_id"].isin(user_ids[split:])]

    # Train model
    lgb_ranker = LightGBMRanker(
        num_leaves=64,
        learning_rate=0.05,
    )

    lgb_ranker.train(
        train_data,
        valid_data=valid_data,
        num_boost_round=500,
        early_stopping_rounds=30,
    )

    # Evaluate
    eval_metrics = lgb_ranker.evaluate(valid_data, k_list=[5, 10, 20])

    # Save model
    lgb_path = Path(output_dir) / "lightgbm_baseline.txt"
    lgb_ranker.save_model(str(lgb_path))

    logger.info(f"LightGBM metrics: {eval_metrics}")
    logger.info(f"LightGBM model saved to {lgb_path}")

    return eval_metrics


def main():
    args = parse_args()

    # Apply preset if specified
    if args.preset:
        apply_preset(args)
        logger.info(f"Using preset: {args.preset}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Log configuration
    logger.info("=" * 50)
    logger.info("MMoE Deep Ranking Model Training")
    logger.info("=" * 50)
    logger.info(f"POI CSV: {args.poi_csv}")
    logger.info(f"Events CSV: {args.events_csv}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Experts: {args.num_experts}")
    logger.info(f"Embeddings: user={args.user_embed_dim}, item={args.item_embed_dim}")
    logger.info(f"Training: epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 50)

    # Load data
    logger.info("Loading data...")
    poi_df = pd.read_csv(args.poi_csv)
    events_df = pd.read_csv(args.events_csv)

    logger.info(f"Loaded {len(poi_df)} POIs, {len(events_df)} events")

    # Train LightGBM baseline if requested
    baseline_metrics = None
    if args.compare_with_lightgbm:
        baseline_metrics = train_lightgbm_baseline(poi_df, events_df, output_dir)

    # Create MMoE configuration
    config = MMoEConfig(
        num_experts=args.num_experts,
        user_embed_dim=args.user_embed_dim,
        item_embed_dim=args.item_embed_dim,
        category_embed_dim=args.category_embed_dim,
        seq_embed_dim=args.seq_embed_dim,
        province_embed_dim=args.province_embed_dim,
        user_tower_dims=args.user_tower_dims,
        item_tower_dims=args.item_tower_dims,
        expert_hidden_dims=args.expert_hidden_dims,
        dropout=args.dropout,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        ctr_weight=args.ctr_weight,
        visit_weight=args.visit_weight,
        duration_weight=args.duration_weight,
        device=args.device,
        use_attention=args.use_attention,
    )

    # Create and train model
    model = MMoEDeepRanker(config)

    logger.info("Starting training...")
    start_time = datetime.now()

    metrics = model.fit(
        poi_df=poi_df,
        events_df=events_df,
        negative_sampling_ratio=args.neg_ratio,
    )

    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()

    logger.info(f"Training completed in {training_time:.1f} seconds")
    logger.info(f"Best validation loss: {metrics.get('best_val_loss', 'N/A')}")

    # Export model
    model_path = output_dir / "mmoe_model.pt"
    model.export_model(str(model_path))
    logger.info(f"Model saved to {model_path}")

    # Save training curves
    if args.plot_curves:
        curves_path = output_dir / "training_curves.png"
        model.save_training_curves(str(curves_path))
        logger.info(f"Training curves saved to {curves_path}")

    # Generate comparison report
    if baseline_metrics:
        report_path = output_dir / "model_comparison.md"
        with open(report_path, "w") as f:
            f.write("# Model Comparison Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## LightGBM Baseline\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            for k, v in baseline_metrics.items():
                f.write(f"| {k} | {v:.4f} |\n")

            f.write("\n## MMoE Deep Model\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Training Time | {training_time:.1f}s |\n")
            if metrics.get('final_metrics'):
                final = metrics['final_metrics']
                f.write(f"| Final Val Loss | {final.get('val_loss', 'N/A')} |\n")
                f.write(f"| CTR AUC | {final.get('val_ctr_auc', 'N/A')} |\n")
                f.write(f"| Visit AUC | {final.get('val_visit_auc', 'N/A')} |\n")
                f.write(f"| Duration MAE | {final.get('val_duration_mae', 'N/A')} |\n")

        logger.info(f"Comparison report saved to {report_path}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
