"""
Training script for ranking models (LightGBM and DIN).

Example usage:
    python -m src.ranking.train_rankers --model lightgbm --train-data data/all/user_events.csv
    python -m src.ranking.train_rankers --model din --train-data data/all/user_events.csv
"""
import argparse
import logging
from pathlib import Path

import pandas as pd

from .lgb_ranker import LightGBMRanker, create_training_data
from .deep_ranker import DeepInterestNetwork, DeepRanker, FeatureStore, create_ranker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def prepare_din_training_data(
    events_df: pd.DataFrame,
    poi_df: pd.DataFrame,
    max_history_length: int = 20,
) -> pd.DataFrame:
    """
    Prepare training data for DIN model.

    Creates sequences of user history for each interaction.
    """
    # Encode categorical variables
    poi_df["poi_id_encoded"] = poi_df["poi_id"].astype("category").cat.codes
    poi_df["category_encoded"] = poi_df.get("category", "unknown").astype("category").cat.codes
    poi_df["province_encoded"] = poi_df.get("province", "unknown").astype("category").cat.codes

    # Merge events with POI data
    merged = events_df.merge(
        poi_df[["poi_id", "poi_id_encoded", "category_encoded", "province_encoded"]],
        on="poi_id",
        how="left",
    )

    # Create labels
    merged["label"] = merged["action"].map({
        "visit": 1,
        "click": 1,
        "view": 0,
    }).fillna(0)

    # Group by user and create history sequences
    result_rows = []

    for user_id, user_data in merged.groupby("user_id"):
        user_data = user_data.sort_values("timestamp")

        # Build history sequences
        for idx, row in user_data.iterrows():
            # Get history up to this point
            history_data = user_data[user_data["timestamp"] < row["timestamp"]]

            # Take last N interactions
            recent_history = history_data.tail(max_history_length)

            result_row = {
                "user_id": user_id,
                "poi_id": row["poi_id_encoded"],
                "category": row["category_encoded"],
                "province": row["province_encoded"],
                "label": row["label"],
                "history_poi_ids": recent_history["poi_id_encoded"].tolist(),
                "history_categories": recent_history["category_encoded"].tolist(),
                "history_provinces": recent_history["province_encoded"].tolist(),
            }
            result_rows.append(result_row)

    return pd.DataFrame(result_rows)


def train_lightgbm(args):
    """Train LightGBM ranking model."""
    logger.info("Training LightGBM ranking model...")

    # Load data
    logger.info(f"Loading events from {args.events_csv}")
    events_df = pd.read_csv(args.events_csv)
    logger.info(f"Loading POIs from {args.poi_csv}")
    poi_df = pd.read_csv(args.poi_csv)

    # Create training data
    logger.info("Creating training data...")
    train_df = create_training_data(
        events_df,
        poi_df,
        negative_sampling_ratio=args.neg_ratio,
    )
    logger.info(f"Created {len(train_df)} training samples")

    # Split train/validation
    user_ids = train_df["user_id"].unique()
    import numpy as np
    np.random.seed(42)
    np.random.shuffle(user_ids)
    split = int(len(user_ids) * 0.8)

    train_users = user_ids[:split]
    valid_users = user_ids[split:]

    train_data = train_df[train_df["user_id"].isin(train_users)]
    valid_data = train_df[train_df["user_id"].isin(valid_users)]

    logger.info(f"Train: {len(train_data)}, Valid: {len(valid_data)}")

    # Initialize and train model
    ranker = LightGBMRanker(
        num_leaves=args.num_leaves,
        learning_rate=args.lr,
    )

    metrics = ranker.train(
        train_data,
        valid_data=valid_data,
        num_boost_round=args.max_iters,
        early_stopping_rounds=args.early_stop,
    )

    logger.info(f"Training metrics: {metrics}")

    # Evaluate
    eval_metrics = ranker.evaluate(valid_data, k_list=[5, 10, 20])
    logger.info(f"Evaluation metrics: {eval_metrics}")

    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ranker.save_model(str(output_path))
    logger.info(f"Model saved to {output_path}")

    # Save feature importance
    importance_path = output_path.parent / "feature_importance.csv"
    ranker.get_feature_importance().to_csv(importance_path, index=False)
    logger.info(f"Feature importance saved to {importance_path}")

    return ranker, eval_metrics


def train_din(args):
    """Train DIN model."""
    logger.info("Training Deep Interest Network...")

    # Load data
    logger.info(f"Loading events from {args.events_csv}")
    events_df = pd.read_csv(args.events_csv)
    logger.info(f"Loading POIs from {args.poi_csv}")
    poi_df = pd.read_csv(args.poi_csv)

    # Prepare training data
    logger.info("Preparing DIN training data...")
    train_df = prepare_din_training_data(events_df, poi_df)

    # Get vocab sizes
    poi_df["poi_id_encoded"] = poi_df["poi_id"].astype("category").cat.codes
    num_pois = poi_df["poi_id_encoded"].nunique() + 1
    num_categories = poi_df.get("category", pd.Series()).nunique() + 1
    num_provinces = poi_df.get("province", pd.Series()).nunique() + 1

    logger.info(f"Vocab sizes: POIs={num_pois}, Categories={num_categories}, Provinces={num_provinces}")

    # Split train/validation
    user_ids = train_df["user_id"].unique()
    import numpy as np
    np.random.seed(42)
    np.random.shuffle(user_ids)
    split = int(len(user_ids) * 0.8)

    train_users = user_ids[:split]
    valid_users = user_ids[split:]

    train_data = train_df[train_df["user_id"].isin(train_users)]
    valid_data = train_df[train_df["user_id"].isin(valid_users)]

    logger.info(f"Train: {len(train_data)}, Valid: {len(valid_data)}")

    # Initialize and train model
    din = DeepInterestNetwork(
        embedding_dim=args.embedding_dim,
        hidden_dims=[int(x) for x in args.hidden_dims.split(",")],
        attention_heads=args.attention_heads,
        dropout=args.dropout,
    )

    metrics = din.fit(
        train_data,
        poi_vocab_size=num_pois,
        category_vocab_size=num_categories,
        province_vocab_size=num_provinces,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
    )

    logger.info(f"Training metrics: {metrics}")

    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    din.save_model(str(output_path))
    logger.info(f"Model saved to {output_path}")

    return din, metrics


def main():
    parser = argparse.ArgumentParser(description="Train ranking models")
    parser.add_argument(
        "--model",
        choices=["lightgbm", "din"],
        default="lightgbm",
        help="Model type to train"
    )
    parser.add_argument(
        "--events-csv",
        default="data/all/user_events.csv",
        help="Path to user events CSV"
    )
    parser.add_argument(
        "--poi-csv",
        default="data/all/poi_expanded.csv",
        help="Path to POI data CSV"
    )
    parser.add_argument(
        "--output",
        default="outputs/ranking/model.txt",
        help="Path to save model"
    )
    parser.add_argument(
        "--neg-ratio",
        type=int,
        default=4,
        help="Negative sampling ratio (for LightGBM)"
    )
    parser.add_argument(
        "--num-leaves",
        type=int,
        default=64,
        help="LightGBM num_leaves"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.05,
        help="Learning rate"
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=1000,
        help="Max boosting iterations (for LightGBM)"
    )
    parser.add_argument(
        "--early-stop",
        type=int,
        default=50,
        help="Early stopping rounds (for LightGBM)"
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=64,
        help="Embedding dimension (for DIN)"
    )
    parser.add_argument(
        "--hidden-dims",
        default="256,128,64",
        help="Hidden layer dimensions (for DIN, comma-separated)"
    )
    parser.add_argument(
        "--attention-heads",
        type=int,
        default=4,
        help="Number of attention heads (for DIN)"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate (for DIN)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs (for DIN)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size (for DIN)"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for training (for DIN, 'cpu' or 'cuda')"
    )

    args = parser.parse_args()

    if args.model == "lightgbm":
        model, metrics = train_lightgbm(args)
    elif args.model == "din":
        model, metrics = train_din(args)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
