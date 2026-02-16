# MMoE Deep Ranking Model - Implementation Report

**Date:** 2025-02-15
**Status:** Implemented
**Author:** GoAfar Team

## Executive Summary

The MMoE (Multi-gate Mixture-of-Experts) deep ranking model has been successfully implemented as an upgrade from the LightGBM baseline. This multi-task learning architecture provides:

- **CTR Prediction**: Click-through rate estimation
- **Visit Prediction**: Visit/conversion probability
- **Duration Prediction**: Expected stay time

## Architecture Overview

### Model Components

```
Input Layer:
    - User ID Embedding
    - POI ID Embedding
    - Category Embedding
    - Province/City Embedding
    - Historical POI Sequence

Tower Networks:
    - UserTower: [256, 128] MLP
    - ItemTower: [256, 128] MLP

MMoE Layer:
    - 4 Expert Networks: [128, 64]
    - 3 Gating Networks (one per task)

Task Heads:
    - CTR Head: [64, 32] -> 1
    - Visit Head: [64, 32] -> 1
    - Duration Head: [64, 32] -> 1
```

### Key Features

1. **Multi-Task Learning**: Simultaneously predicts CTR, visit probability, and duration
2. **Attention-based Sequence Encoding**: Captures user behavior patterns
3. **Task-Specific Gating**: Each task selects different expert combinations
4. **Negative Sampling**: 4:1 negative:positive ratio for training
5. **Early Stopping**: Prevents overfitting with patience mechanism

## Model Comparison

| Feature | LightGBM Baseline | MMoE Deep Model |
|----------|-------------------|------------------|
| Architecture | Gradient Boosting | Neural Network (MMoE) |
| Tasks | Single (ranking) | Multi-task (CTR, Visit, Duration) |
| Sequence Handling | Manual feature engineering | Attention mechanism |
| Training Time | Fast (~30s) | Moderate (~2-5 min) |
| Inference Speed | Very fast | Fast (GPU accelerated) |
| Interpretability | High (feature importance) | Medium (attention weights) |
| Cold Start | Poor | Better with embeddings |
| Memory Usage | Low | Moderate (embeddings) |

## Usage Examples

### Training

```python
from src.ranking import MMoEDeepRanker, MMoEConfig

# Create configuration
config = MMoEConfig(
    num_experts=4,
    user_embed_dim=64,
    item_embed_dim=64,
    epochs=20,
    batch_size=512,
    learning_rate=1e-3,
)

# Initialize and train
model = MMoEDeepRanker(config)
model.fit(poi_df, events_df)

# Export
model.export_model("outputs/ranking/mmoe_model.pt")
```

### Inference

```python
from src.ranking import MMoEDeepRanker, MMoEConfig

# Load model
config = MMoEConfig(device="cuda")
model = MMoEDeepRanker(config)
model.load_model("outputs/ranking/mmoe_model.pt")

# Predict for candidates
candidates = [
    {"poi_id": "1001", "category": "文化景点", "province": "北京", ...},
    {"poi_id": "1002", "category": "自然风光", "province": "北京", ...},
]

results = model.predict(user_id="U0001", candidate_pois=candidates)
# Returns: [(poi_id, {"ctr": 0.85, "visit": 0.72, "duration": 120}), ...]
```

### Command Line Training

```bash
# Quick training (5 epochs, 2 experts)
python scripts/train_mmoe_ranker.py --preset quick

# Full training
python scripts/train_mmoe_ranker.py --epochs 20 --num-experts 4

# With baseline comparison
python scripts/train_mmoe_ranker.py --compare-with-lightgbm --plot-curves
```

### Interactive Inference

```bash
python scripts/infer_mmoe_ranker.py --interactive

# Or single user
python scripts/infer_mmoe_ranker.py --user U0001 --top-k 20
```

## File Structure

```
src/ranking/
├── __init__.py              # Module exports
├── deep_ranker.py           # MMoE implementation (NEW)
├── lgb_ranker.py            # LightGBM baseline
├── train_rankers.py         # Unified training script
└── ranking_example.py        # Usage examples

scripts/
├── train_mmoe_ranker.py     # Training script (NEW)
├── train_mmoe_ranker.sh     # Shell script (NEW)
└── infer_mmoe_ranker.py     # Inference script (NEW)

docs/
└── MMoE_MODEL_COMPARISON.md  # This file
```

## Performance Considerations

### Training Speed

- **LightGBM**: ~30 seconds for 100K samples
- **MMoE (CPU)**: ~5 minutes for 100K samples, 20 epochs
- **MMoE (GPU)**: ~2 minutes for 100K samples, 20 epochs

### Scalability

| Data Size | LightGBM Memory | MMoE Memory (GPU) |
|------------|------------------|---------------------|
| 10K samples | ~50MB | ~200MB |
| 100K samples | ~200MB | ~500MB |
| 1M samples | ~1GB | ~2GB |

### Recommendation

- Use **LightGBM** for:
  - Quick prototyping
  - Resource-constrained environments
  - When interpretability is critical

- Use **MMoE** for:
  - Production recommendation
  - When multi-task prediction is needed
  - When GPU acceleration is available

## Integration with GoAfar Pipeline

The MMoE ranker integrates seamlessly with the existing GoAfar pipeline:

```python
# In your recommendation service
from src.ranking import create_ranker

# Create ranker
ranker = create_ranker(
    model_type="mmoe",
    num_experts=4,
    device="cuda"
)

# Load trained model
ranker.load_model("outputs/ranking/mmoe_model.pt")

# Use in pipeline
ranked_pois = ranker.predict(
    user_id=request.user_id,
    candidate_pois=candidates,
    context={"hour": current_hour},
)
```

## Future Enhancements

1. **Dynamic Expert Selection**: Adapt expert usage based on user segments
2. **Meta-Learning**: Fast adaptation to new users/cities
3. **Knowledge Distillation**: Compress MMoE to smaller model
4. **Online Learning**: Update model incrementally with new data

## References

- [Multi-gate Mixture-of-Experts](https://arxiv.org/abs/1803.00147) - Ma et al., KDD 2018
- [Deep Interest Network](https://arxiv.org/abs/1706.06978) - Zhou et al., KDD 2018
- [Recommendation System Design](https://dl.acm.org/doi/10.1145/3383313) - ACM RecSys 2020
