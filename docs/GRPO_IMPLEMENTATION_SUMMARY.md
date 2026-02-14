# GRPO Training Implementation Summary

## Overview

This document summarizes the GRPO (Group Relative Policy Optimization) training implementation for the GoAfar route planning system.

## Implementation Status

✅ **Completed**: Full GRPO training pipeline with two backends

## Files Created/Modified

### Core Implementation

1. **`src/rl/grpo_trainer.py`** (Enhanced)
   - Native PyTorch implementation of GRPO
   - No external dependencies beyond PyTorch and transformers
   - Supports LoRA for efficient training
   - Includes:
     - `GRPOConfig`: Training configuration
     - `RoutePlanningDataset`: Dataset class for training
     - `GRPOTrainer`: Main training class with group sampling
     - Enhanced reward computation with POI data support
     - Command-line interface for easy training

2. **`src/rl/grpo_trainer_trl.py`** (New)
   - Alternative implementation using TRL library
   - Leverages TRL's optimized `GRPOTrainer`
   - Better performance and more features
   - Includes:
     - `TourismGRPOConfig`: Configuration for TRL training
     - `TourismRewardFunction`: Custom reward function
     - `TourismGRPODataset`: Dataset for TRL trainer
     - Integration with TRL's training infrastructure

### Training Scripts

3. **`scripts/train_grpo.sh`** (New)
   - Full-featured training script
   - Supports all hyperparameters
   - Includes error handling and validation
   - Supports optional POI data for advanced rewards

4. **`scripts/train_grpo_quick.sh`** (New)
   - Quick start script with sensible defaults
   - Automatic dependency checking
   - Simple one-command training

### Testing

5. **`scripts/test_grpo_trainer.py`** (New)
   - Comprehensive test suite
   - Tests data loading, configuration, reward computation
   - Validates advantage calculation
   - Checks TRL availability

### Documentation

6. **`docs/GRPO_TRAINING_GUIDE.md`** (New)
   - Complete training guide
   - Algorithm explanation
   - Usage examples
   - Troubleshooting guide

7. **`docs/GRPO_IMPLEMENTATION_SUMMARY.md`** (This file)
   - Implementation overview
   - Architecture details
   - Quick reference

## Key Features

### 1. Dual Backend Support

**Native Implementation** (`grpo_trainer.py`):
- Works with just PyTorch and transformers
- Self-contained GRPO algorithm
- Full control over training loop
- Suitable for research and customization

**TRL Implementation** (`grpo_trainer_trl.py`):
- Uses HuggingFace TRL library
- Better optimized and tested
- Supports distributed training
- Recommended for production use

### 2. Group-Relative Advantage

```python
# For each prompt, sample N responses
responses = [model.generate(prompt) for _ in range(N)]

# Compute rewards
rewards = [compute_reward(r) for r in responses]

# Group-relative advantage (no critic needed!)
group_mean = mean(rewards)
advantages = [r - group_mean for r in rewards]
```

### 3. Hybrid Reward System

**Basic Rewards** (always available):
- Format validity: Is output a valid POI ID? (+0.5)
- Target matching: Does it match ground truth? (+2.0)
- Database validity: Does POI exist? (+0.5/-0.5)

**Advanced Rewards** (with POI data):
- Route feasibility (time windows)
- Travel time penalties
- Category diversity
- User preference matching

### 4. Efficient Training

- **LoRA Support**: Train only 0.1-1% of parameters
- **Gradient Accumulation**: Simulate larger batch sizes
- **Mixed Precision**: FP16 training for speed and memory
- **Group Sampling**: No critic network needed

## Usage Examples

### Basic Training

```bash
# Using native implementation
bash scripts/train_grpo.sh

# Quick start
bash scripts/train_grpo_quick.sh
```

### Custom Configuration

```bash
python -m src.rl.grpo_trainer \
  --model models/Qwen3-8B \
  --data outputs/datasets/grpo_planner_prompts.jsonl \
  --output outputs/grpo/my-model \
  --epochs 5 \
  --batch-size 8 \
  --group-size 8 \
  --lr 5e-6
```

### With Advanced Rewards

```bash
python -m src.rl.grpo_trainer \
  --model models/Qwen3-8B \
  --data outputs/datasets/grpo_planner_prompts.jsonl \
  --output outputs/grpo/my-model \
  --poi-data data/processed/pois_with_embeddings.parquet \
  --time-matrix data/processed/time_matrix.npy
```

### Using TRL (Recommended)

```bash
# Install TRL first
pip install trl>=0.12.0

# Train with TRL backend
python -m src.rl.grpo_trainer_trl \
  --model models/Qwen3-8B \
  --data outputs/datasets/grpo_planner_prompts.jsonl \
  --output outputs/grpo/qwen3-grpo-planner
```

## Architecture

### Training Pipeline

```
┌─────────────────┐
│  Raw Trajectories│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Dataset Builder │ → grpo_planner_prompts.jsonl
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  GRPO Trainer   │
│  - Load model   │
│  - Load data    │
│  - Group sample │
│  - Compute adv  │
│  - Update policy│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Trained Model  │ → LoRA adapters
└─────────────────┘
```

### GRPO Algorithm

```
For each training step:
  1. Sample batch of prompts P
  2. For each prompt p in P:
     - Generate N responses: [r_1, ..., r_N]
     - Compute rewards: [R_1, ..., R_N]
     - Compute advantages: A_i = R_i - mean(R)
  3. Compute policy loss: L = -E[log π(a|s) * A(s,a)]
  4. Add KL penalty: L_total = L + β * KL(π || π_ref)
  5. Update policy with gradient descent
```

## Configuration

### Recommended Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `model` | models/Qwen3-8B | Base model |
| `group_size` | 4-8 | Samples per prompt |
| `batch_size` | 4-8 | Depends on GPU memory |
| `learning_rate` | 1e-5 | Conservative for stability |
| `kl_coef` | 0.1 | KL penalty weight |
| `epochs` | 3-5 | Usually converges quickly |
| `use_lora` | True | Strongly recommended |

### Reward Weights

Adjust in `src/rl/reward_manager.py`:

```python
RewardWeights(
    pref_weight=1.0,        # User preferences
    feasible_penalty=1.5,   # Time window violations
    overtime_penalty=1.2,   # Exceeding time limits
    travel_penalty=0.003,   # Travel time
    diversity_weight=0.3,   # Category diversity
)
```

## Output

### Trained Model Structure

```
outputs/grpo/qwen3-grpo-planner/
├── adapter_config.json       # LoRA configuration
├── adapter_model.safetensors # Trainable weights (~1-2% of base)
├── tokenizer_config.json
├── tokenizer.json
├── special_tokens_map.json
└── grpo_config.json          # Training metadata
```

### Model Size

- Base model: ~15GB (Qwen3-8B)
- LoRA adapters: ~15-150MB
- Total training overhead: Minimal with LoRA

## Monitoring

### Key Metrics

Track these during training:

1. **loss**: Total training loss (should decrease)
2. **policy_loss**: Policy gradient loss
3. **kl_penalty**: KL divergence (keep < 0.1)
4. **mean_reward**: Average reward per batch

### TensorBoard

```bash
tensorboard --logdir outputs/grpo/qwen3-grpo-planner/runs
```

## Integration

### Using Trained Model

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
base = AutoModelForCausalLM.from_pretrained("models/Qwen3-8B")

# Load GRPO-trained adapters
model = PeftModel.from_pretrained(base, "outputs/grpo/qwen3-grpo-planner")

# Generate next POI
prompt = "已选景点：S094645, S151634\n\n请推荐下一个景点。"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
next_poi = tokenizer.decode(outputs[0])
```

## Future Enhancements

Potential improvements:

1. **Curriculum Learning**: Start with easy routes, increase difficulty
2. **Reward Shaping**: Add intermediate rewards for partial routes
3. **Multi-Objective**: Balance multiple objectives (time, cost, preferences)
4. **Ensemble**: Combine multiple GRPO-trained models
5. **Online Learning**: Update model with user feedback

## References

- [DeepSeekMath: GRPO Paper](https://arxiv.org/abs/2406.01806)
- [TRL GRPO Documentation](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [veRL Framework](https://github.com/volcengine/verl)

## Troubleshooting

### Common Issues

1. **Out of Memory**:
   - Reduce `batch_size`
   - Reduce `group_size`
   - Enable gradient checkpointing

2. **Low Rewards**:
   - Check data quality
   - Adjust reward weights
   - Verify POI database

3. **High KL Divergence**:
   - Lower learning rate
   - Increase `kl_coef`
   - Reduce training steps

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Conclusion

The GRPO training implementation provides a complete pipeline for training route planning policies. It supports both native PyTorch and TRL backends, includes hybrid reward computation, and is optimized for efficient training with LoRA.

For detailed usage instructions, see `docs/GRPO_TRAINING_GUIDE.md`.
