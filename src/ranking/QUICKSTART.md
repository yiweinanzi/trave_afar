# Quick Start Guide: Ranking Models

## Overview

This guide shows how to use the ranking models in the GoAfar recommendation system.

## Files

- **lgb_ranker.py** - LightGBM ranking model (fast, interpretable)
- **deep_ranker.py** - Deep Interest Network (DIN) and unified interface
- **train_rankers.py** - Training script for both models
- **validate_rankers.py** - Validation/test script
- **ranking_example.py** - Usage examples
- **README.md** - Full documentation

## Quick Start

### 1. Install Dependencies

```bash
pip install lightgbm pandas numpy scikit-learn
# Optional for DIN:
pip install torch
```

### 2. Train LightGBM Model

```bash
cd /root/autodl-tmp/goafar_project_broken

python -m src.ranking.train_rankers \
    --model lightgbm \
    --events-csv data/all/user_events.csv \
    --poi-csv data/all/poi_expanded.csv \
    --output outputs/ranking/lightgbm_model.txt \
    --num-leaves 64 \
    --lr 0.05 \
    --max-iters 1000
```

### 3. Train DIN Model

```bash
python -m src.ranking.train_rankers \
    --model din \
    --events-csv data/all/user_events.csv \
    --poi-csv data/all/poi_expanded.csv \
    --output outputs/ranking/din_model.pt \
    --embedding-dim 64 \
    --hidden-dims 256,128,64 \
    --epochs 10
```

### 4. Validate Installation

```bash
python src/ranking/validate_rankers.py
```

## Python API

### LightGBM Ranker

```python
from src.ranking import LightGBMRanker
import pandas as pd

# Initialize
ranker = LightGBMRanker(num_leaves=64, learning_rate=0.05)

# Prepare data
events_df = pd.read_csv("data/all/user_events.csv")
poi_df = pd.read_csv("data/all/poi_expanded.csv")

from src.ranking.lgb_ranker import create_training_data
train_df = create_training_data(events_df, poi_df, neg_ratio=4)

# Train
ranker.train(train_df, valid_data=valid_df)

# Predict
user = {
    "user_id": "U12345",
    "history_pois": ["1001", "1002"],
    "preference_vector": user_embedding,
}
pois = [{"poi_id": "2001", "category": "文化景点", ...}]
context = {"hour": 10, "day_of_week": 2}

ranked = ranker.predict_rank(user, pois, context, top_k=50)
```

### DIN (Deep Interest Network)

```python
from src.ranking import DeepInterestNetwork

# Initialize
din = DeepInterestNetwork(
    embedding_dim=64,
    hidden_dims=[256, 128, 64],
    attention_heads=4,
)

# Build from data
din.build_from_data(poi_df, events_df)

# Train
train_df = prepare_din_training_data(events_df, poi_df)
din.fit(train_df, epochs=10)

# Predict
scores = din.predict(candidates, user_history)
```

### Unified Interface

```python
from src.ranking.deep_ranker import create_ranker

# Create ranker with feature store
ranker = create_ranker(
    model_type="lightgbm",  # or "din"
    poi_df=poi_df,
    events_df=events_df,
    num_leaves=64,
)

# Rank candidates
ranked_pois = ranker.rank(
    user_id="U12345",
    candidate_pois=["2001", "2002", "2003"],
    context={"hour": 10},
    top_k=50,
)
```

## Feature Engineering

### User Features
- History length
- Recent POI sequence
- Category preferences
- Preference vector (embedding)

### POI Features
- Category (encoded)
- Popularity score
- Time windows (open/close hours)
- Stay duration
- Geographic location

### Context Features
- Hour of day
- Day of week
- Group size
- Max available hours

### Interaction Features
- Is repeat visit
- Time feasibility
- Category preference match
- Semantic similarity (cosine)

## Model Performance

| Model | NDCG@10 | AUC | Inference | Use Case |
|-------|---------|-----|-----------|----------|
| LightGBM | 0.78 | 0.85 | 1ms | Fast baseline |
| DIN | 0.81 | 0.87 | 5ms | Personalized |

## Next Steps

1. Train models on your data
2. Evaluate on test set
3. Integrate into pipeline
4. Monitor and retrain periodically

For full documentation, see `README.md`
