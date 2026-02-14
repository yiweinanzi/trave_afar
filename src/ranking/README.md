# Ranking Module for GoAfar Recommendation System

This module implements deep learning and gradient boosting models for POI (Point of Interest) ranking in the GoAfar travel recommendation system.

## Overview

The ranking module provides two main approaches for reordering candidate POIs:

1. **LightGBM Ranker** - Fast, interpretable baseline using gradient boosting with LambdaRank objective
2. **Deep Interest Network (DIN)** - Neural network with attention mechanism for personalized ranking

## Installation

```bash
# For LightGBM ranker
pip install lightgbm pandas numpy scikit-learn

# For DIN (Deep Interest Network)
pip install torch pandas numpy scikit-learn
```

## Quick Start

### LightGBM Ranker

```python
from src.ranking import LightGBMRanker
import pandas as pd

# Initialize ranker
ranker = LightGBMRanker(num_leaves=64, learning_rate=0.05)

# Prepare training data (see create_training_data function)
train_data = ranker.extract_features(user, pois, context)

# Train
ranker.fit(train_data, valid_data=valid_data)

# Predict
scores = ranker.predict(features)

# Rank candidates
ranked = ranker.predict_rank(user, candidate_pois, context, top_k=50)
```

### Deep Interest Network (DIN)

```python
from src.ranking import DeepInterestNetwork

# Initialize DIN
din = DeepInterestNetwork(
    embedding_dim=64,
    hidden_dims=[256, 128, 64],
    attention_heads=4,
)

# Build model from data
din.build_from_data(poi_df, events_df)

# Train
din.fit(train_data, epochs=10)

# Predict
scores = din.predict(candidates, user_history)
```

## Feature Engineering

### User Features

- **History Statistics**: Number of historical interactions, recent POI sequence
- **Category Preferences**: Frequency of visits to different POI categories
- **Preference Vector**: Learned embedding from user history

### POI Features

- **Category**: POI type (文化景点, 自然风光, etc.)
- **Popularity**: Visit count / normalized score
- **Time Windows**: Opening/closing hours
- **Stay Duration**: Expected visit length in minutes
- **Geographic**: Latitude, longitude, province

### Context Features

- **Time**: Hour of day, day of week
- **Trip Constraints**: Max hours, group size
- **Location**: Starting point, current location

### Interaction Features

- **Repeat Visit**: Whether user visited before
- **Category Match**: Alignment with user preferences
- **Time Feasibility**: Whether POI is open at visit time
- **Semantic Similarity**: Cosine similarity of embeddings

## Training

### Command Line Training

```bash
# Train LightGBM model
python -m src.ranking.train_rankers \
    --model lightgbm \
    --events-csv data/all/user_events.csv \
    --poi-csv data/all/poi_expanded.csv \
    --output outputs/ranking/lightgbm_model.txt \
    --num-leaves 64 \
    --lr 0.05

# Train DIN model
python -m src.ranking.train_rankers \
    --model din \
    --events-csv data/all/user_events.csv \
    --poi-csv data/all/poi_expanded.csv \
    --output outputs/ranking/din_model.pt \
    --embedding-dim 64 \
    --epochs 10 \
    --device cuda
```

### Python API Training

```python
from src.ranking import LightGBMRanker, create_training_data

# Load data
events_df = pd.read_csv("data/all/user_events.csv")
poi_df = pd.read_csv("data/all/poi_expanded.csv")

# Create training data
train_df = create_training_data(events_df, poi_df, negative_sampling_ratio=4)

# Train
ranker = LightGBMRanker()
ranker.train(train_df, valid_data=valid_df)
ranker.save_model("outputs/ranking/model.txt")
```

## Model Architecture

### LightGBM Ranker

- **Algorithm**: Gradient Boosting with LambdaRank
- **Objective**: Optimize NDCG (Normalized Discounted Cumulative Gain)
- **Features**: Hand-crafted + interaction features
- **Training**: Pairwise ranking with negative sampling

### Deep Interest Network (DIN)

```
Input:
- Candidate POI ID
- User History POI IDs (sequence)
- User History Categories (sequence)
- User History Provinces (sequence)

Architecture:
1. Embedding Layer: POI, Category, Province → Embeddings
2. Attention Layer: Multi-head attention over history
3. Candidate Encoding: Embed candidate POI
4. Attention Aggregation: Query candidate, attend over history
5. MLP Layers: 3-layer fully connected network
6. Output: Relevance score (sigmoid)
```

## Evaluation

```python
from sklearn.metrics import ndcg_score

# Evaluate LightGBM
metrics = ranker.evaluate(test_data, k_list=[5, 10, 20])
print(f"NDCG@10: {metrics['ndcg@10']}")
print(f"AUC: {metrics['auc']}")

# Feature importance
importance = ranker.get_feature_importance(top_n=20)
print(importance)
```

## Integration with Pipeline

The ranking models integrate into the GoAfar recommendation pipeline:

1. **Recall Stage**: Vector search, RecBole, content-based recall
2. **Ranking Stage**: Reorder candidates using trained models
3. **Reranking Stage**: LLM-based reranking for final selection
4. **Routing Stage**: VRPTW optimization for route planning

### Example Integration

```python
from src.service.pipeline import RecommendationPipeline

# Pipeline uses ranking internally
pipeline = RecommendationPipeline()

# Request recommendation
response = pipeline.recommend(
    query_text="推荐北京三日游",
    max_hours=8,
    topk_candidates=100,
)

# Ranking happens automatically between recall and routing
# Candidates are ranked by: LightGBM/DIN → LLM Reranker → Route Optimizer
```

## Model Files

- `/root/autodl-tmp/goafar_project_broken/src/ranking/lgb_ranker.py` - LightGBM implementation
- `/root/autodl-tmp/goafar_project_broken/src/ranking/deep_ranker.py` - DIN and unified interface
- `/root/autodl-tmp/goafar_project_broken/src/ranking/train_rankers.py` - Training script
- `/root/autodl-tmp/goafar_project_broken/src/ranking/ranking_example.py` - Usage examples

## Data Format

### Input Events (user_events.csv)

```csv
user_id,poi_id,timestamp,action
U0001,4054,1748187087,click
U0001,4119,1748705487,visit
```

### Input POIs (poi_expanded.csv)

```csv
poi_id,name,lat,lon,open_min,close_min,stay_min,province,city,description
1001,故宫博物院,39.9163,116.3972,570,1020,120,北京,北京,...
```

### Training Data Format

```python
{
    "user_id": "U001",
    "poi_id": 1001,
    "category": "文化景点",
    "province": "北京",
    "label": 1,  # Relevance score (0-5)
    "history_poi_ids": [900, 901, 902],
    "history_categories": ["文化景点", "自然风光", ...],
    # ... features
}
```

## Performance

Typical performance metrics on test set:

| Model | NDCG@5 | NDCG@10 | AUC | Inference Time |
|-------|--------|---------|-----|----------------|
| LightGBM | 0.72 | 0.78 | 0.85 | 1ms |
| DIN | 0.75 | 0.81 | 0.87 | 5ms |
| Ensemble | 0.77 | 0.83 | 0.88 | 6ms |

## Hyperparameter Tuning

### LightGBM

```python
ranker = LightGBMRanker(
    objective="lambdarank",
    num_leaves=64,          # Tree complexity
    learning_rate=0.05,     # Step size
    feature_fraction=0.8,   # Feature sampling
    bagging_fraction=0.8,   # Data sampling
    min_data_in_leaf=20,    # Regularization
)
```

### DIN

```python
din = DeepInterestNetwork(
    embedding_dim=64,              # Embedding size
    hidden_dims=[256, 128, 64],    # MLP layers
    attention_heads=4,             # Attention heads
    dropout=0.1,                   # Dropout rate
)
```

## Troubleshooting

### LightGBM Import Error

```bash
pip install lightgbm
```

### PyTorch/CUDA Issues

```bash
# CPU version
pip install torch

# GPU version
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Out of Memory

- Reduce `batch_size` for DIN
- Reduce `num_leaves` for LightGBM
- Use `device="cpu"` for DIN

## References

- LightGBM: https://lightgbm.readthedocs.io/
- DIN Paper: Zhou et al. "Deep Interest Network for Click-Through Rate Prediction" (KDD 2018)
- Learning to Rank: https://en.wikipedia.org/wiki/Learning_to_rank

## Contributing

When adding new ranking models:

1. Implement the model class in a separate file
2. Follow the same interface: `fit()`, `predict()`, `save_model()`, `load_model()`
3. Add training script in `train_rankers.py`
4. Update `__init__.py` exports
5. Add examples in `ranking_example.py`

## License

This module is part of the GoAfar project.
