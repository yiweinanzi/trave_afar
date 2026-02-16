#!/usr/bin/env python3
"""
Train GNN Model for POI Recommendation

This script trains a Graph Neural Network model on POI graph data.
Supports GraphSAGE, GAT, and GCN architectures.

Usage:
    python scripts/train_gnn_model.py --graph outputs/gnn/graph.pkl --output outputs/gnn/model.pkl
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.gnn_model import (
    POIGraphBuilder,
    POIGNNRecommender,
    GNNConfig,
    create_gnn_model,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_gnn(
    graph_path: Path,
    output_path: Path,
    model_type: str = "graphsage",
    hidden_dim: int = 128,
    output_dim: int = 64,
    num_layers: int = 2,
    num_epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    device: str = "cuda",
) -> None:
    """
    Train GNN model on POI graph.

    Args:
        graph_path: Path to pre-built graph
        output_path: Path to save trained model
        model_type: GNN architecture (graphsage, gat, gcn)
        hidden_dim: Hidden layer dimension
        output_dim: Output embedding dimension
        num_layers: Number of GNN layers
        num_epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Training device
    """
    logger.info("=" * 60)
    logger.info("GNN Model Training")
    logger.info("=" * 60)

    # Load graph
    logger.info(f"Loading graph from {graph_path}")
    builder = POIGraphBuilder()
    builder.load_graph(graph_path)

    if not builder.nodes:
        logger.error("Empty graph, exiting")
        return

    # Create config
    config = GNNConfig(
        model_type=model_type,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
    )

    # Create recommender
    recommender = POIGNNRecommender(
        builder=builder,
        config=config,
    )

    # Build PyG graph
    graph = builder.build_graph()
    if graph is None:
        logger.error("Failed to build graph")
        return

    logger.info(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    logger.info(f"Feature dim: {graph.x.shape[1]}")

    # Train
    logger.info("Starting training...")
    logger.info(f"  Model: {model_type}")
    logger.info(f"  Hidden dim: {hidden_dim}")
    logger.info(f"  Output dim: {output_dim}")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Device: {device}")

    # Create trainer
    from src.model.gnn_model import GNNTrainer

    # Create model
    input_dim = graph.x.shape[1]
    if model_type == "graphsage":
        from src.model.gnn_model import POIGraphSAGE
        model = POIGraphSAGE(input_dim, hidden_dim, output_dim, num_layers)
    elif model_type == "gat":
        from src.model.gnn_model import POIGAT
        model = POIGAT(input_dim, hidden_dim, output_dim, num_layers)
    elif model_type == "gcn":
        from src.model.gnn_model import POIGCN
        model = POIGCN(input_dim, hidden_dim, output_dim, num_layers)
    else:
        logger.error(f"Unknown model type: {model_type}")
        return

    trainer = GNNTrainer(model, config)
    metrics = trainer.train_link_prediction(graph)

    logger.info(f"Training complete!")
    logger.info(f"  Best val loss: {metrics['best_val_loss']:.4f}")

    # Generate embeddings
    logger.info("Generating POI embeddings...")
    recommender.model = model
    recommender._generate_embeddings(graph)

    # Save
    logger.info(f"Saving model to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    recommender.save_model(output_path)

    logger.info("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Train GNN model for POI recommendation"
    )
    parser.add_argument(
        "--graph",
        type=str,
        default="outputs/gnn/graph.pkl",
        help="Path to pre-built graph",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/gnn/model.pkl",
        help="Output path for trained model",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="graphsage",
        choices=["graphsage", "gat", "gcn"],
        help="GNN architecture",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension",
    )
    parser.add_argument(
        "--output-dim",
        type=int,
        default=64,
        help="Output embedding dimension",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of GNN layers",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Training device (cuda/cpu)",
    )

    args = parser.parse_args()

    # Convert paths
    graph_path = Path(args.graph)
    output_path = Path(args.output)

    # Validate
    if not graph_path.exists():
        logger.error(f"Graph file not found: {graph_path}")
        sys.exit(1)

    # Train
    train_gnn(
        graph_path=graph_path,
        output_path=output_path,
        model_type=args.model_type,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_layers=args.num_layers,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
    )


if __name__ == "__main__":
    main()
