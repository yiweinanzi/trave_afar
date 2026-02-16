"""Deep ranking module for POI recommendation.

Provides multiple ranking models for POI recommendation:
- LightGBMRanker: Fast, interpretable gradient boosting baseline
- MMoEDeepRanker: Multi-task deep learning with MMoE architecture
- DeepInterestNetwork: Sequence-based attention model

Usage:
    from src.ranking import create_ranker
    ranker = create_ranker("lightgbm")
    predictions = ranker.rank(user_id, candidate_pois)
"""

from .lgb_ranker import LightGBMRanker, create_training_data

# Try to import deep learning models (requires PyTorch)
try:
    from .deep_ranker import (
        MMoEDeepRanker,
        MMoEConfig,
        TrainingMetrics,
        create_mmoe_ranker,
    )
    _deep_available = True
except ImportError:
    _deep_available = False

# Import unified logger
from src.utils import get_logger

logger = get_logger(__name__)

if not _deep_available:
    logger.warning("PyTorch not available, deep learning rankers disabled")

__all__ = [
    "LightGBMRanker",
    "create_training_data",
    "create_ranker",
]

if _deep_available:
    __all__.extend([
        "MMoEDeepRanker",
        "MMoEConfig",
        "TrainingMetrics",
        "create_mmoe_ranker",
    ])


def create_ranker(model_type: str = "lightgbm", **kwargs):
    """
    Factory function to create a ranking model.

    Args:
        model_type: Type of model ("lightgbm", "mmoe")
        **kwargs: Model-specific configuration

    Returns:
        Configured ranker instance

    Example:
        >>> from src.ranking import create_ranker
        >>> ranker = create_ranker("lightgbm", num_leaves=64)
        >>> ranker = create_ranker("mmoe", num_experts=4)
    """
    if model_type == "lightgbm":
        logger.debug(f"Creating LightGBMRanker with config: {kwargs}")
        return LightGBMRanker(**kwargs)
    elif model_type == "mmoe":
        if not _deep_available:
            logger.error("PyTorch required for MMoE model")
            raise ImportError("PyTorch required for MMoE model")
        logger.debug(f"Creating MMoEDeepRanker with config: {kwargs}")
        return create_mmoe_ranker(**kwargs)
    else:
        logger.error(f"Unknown model type: {model_type}")
        raise ValueError(f"Unknown model type: {model_type}")
