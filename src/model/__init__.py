"""
GoAfar Model Package

This package contains various models for POI recommendation and ranking:
- GNN models for POI representation learning
- Integration with existing ranking models
"""

from .gnn_model import (
    GNNConfig,
    GraphMetrics,
    POIGraphBuilder,
    POIGNNRecommender,
    create_gnn_model,
    create_graph_builder,
)

# Only import torch-dependent classes if available
try:
    from .gnn_model import (
        POIGraphSAGE,
        POIGAT,
        POIGCN,
        GNNTrainer,
    )
    _TORCH_MODELS = True
except ImportError:
    _TORCH_MODELS = False

from .gnn_integration import (
    GNNProvider,
    GNNScorer,
    create_gnn_provider,
)

__all__ = [
    "GNNConfig",
    "GraphMetrics",
    "POIGraphBuilder",
    "POIGNNRecommender",
    "create_gnn_model",
    "create_graph_builder",
    "GNNProvider",
    "GNNScorer",
    "create_gnn_provider",
]

# Conditionally add torch models
if _TORCH_MODELS:
    __all__.extend([
        "POIGraphSAGE",
        "POIGAT",
        "POIGCN",
        "GNNTrainer",
    ])
