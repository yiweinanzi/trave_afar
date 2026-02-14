# GRPO Training for Route Planning

This document describes the GRPO (Group Relative Policy Optimization) training implementation for the GoAfar route planning system.

## Overview

GRPO is a reinforcement learning algorithm that trains a policy to generate POI sequences for tourism routes. Unlike traditional PPO, GRPO uses group-relative advantages, eliminating the need for a value network (critic).

### Key Features

- **Group Sampling**: Generate multiple (N=4 by default) responses per prompt
- **Relative Advantages**: Compute advantages within each group instead of using a critic
- **Hybrid Rewards**:
  - Format validity (is the output a valid POI ID?)
  - Target matching (does it match the ground truth?)
  - Route feasibility (time windows, travel time)
  - User preferences (category diversity)
- **Efficient Training**: Supports LoRA for memory-efficient fine-tuning

## Architecture

### Files

- `src/rl/grpo_trainer.py` - Native PyTorch implementation (works without TRL)
- `src/rl/grpo_trainer_trl.py` - TRL-based implementation (recommended if TRL available)
- `src/rl/reward_manager.py` - Reward computation for route quality
- `src/rl/dataset_builder.py` - Build training datasets from trajectories
- `scripts/train_grpo.sh` - Training script

### Data Flow

```
Raw Trajectories -> Dataset Builder -> GRPO Prompts -> GRPO Trainer -> Trained Policy
                    (jsonl)           (jsonl)          (group sampling)   (Qwen3-8B+LoRA)
```

## Usage

### 1. Prepare Training Data

The training data should be in JSONL format with the following structure:

```json
{
  "prompt": "{\"task\": \"plan_next_poi\", \"user_id\": \"SYNTH_0000\", \"day\": \"2025-08-17\", \"state_prefix\": [\"S094645\", \"S151634\"], \"instruction\": \"请基于当前已选景点，生成下一步最合理的 poi_id，并保证时间窗可行。\"}",
  "target_next_poi": "S292793",
  "full_target_route": ["S094645", "S151634", "S292793", "S153992"]
}
```

Generate data from trajectories:

```bash
python -m src.rl.dataset_builder \
  --rl-prompts outputs/datasets/planner_rl_prompts.jsonl \
  --out-rl outputs/datasets/grpo_planner_prompts.jsonl
```

### 2. Train with Native Implementation

Basic training (no external dependencies beyond PyTorch):

```bash
bash scripts/train_grpo.sh \
  --model models/Qwen3-8B \
  --data outputs/datasets/grpo_planner_prompts.jsonl \
  --output outputs/grpo/qwen3-grpo-planner \
  --epochs 3 \
  --batch-size 4 \
  --group-size 4
```

With advanced rewards (requires POI data and time matrix):

```bash
bash scripts/train_grpo.sh \
  --model models/Qwen3-8B \
  --data outputs/datasets/grpo_planner_prompts.jsonl \
  --output outputs/grpo/qwen3-grpo-planner \
  --poi-data data/processed/pois_with_embeddings.parquet \
  --time-matrix data/processed/time_matrix.npy
```

### 3. Train with TRL (Recommended)

First install TRL:

```bash
pip install trl>=0.12.0
```

Then run:

```bash
python -m src.rl.grpo_trainer_trl \
  --model models/Qwen3-8B \
  --data outputs/datasets/grpo_planner_prompts.jsonl \
  --output outputs/grpo/qwen3-grpo-planner \
  --epochs 3 \
  --batch-size 4 \
  --group-size 4
```

## Configuration

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `group_size` | 4 | Number of samples per prompt for advantage estimation |
| `kl_coef` | 0.1 | KL divergence penalty coefficient |
| `learning_rate` | 1e-5 | Learning rate for policy updates |
| `batch_size` | 4 | Batch size per device |
| `num_epochs` | 3 | Number of training epochs |
| `use_lora` | True | Use LoRA adapters for efficient training |

### Reward Weights

Configure in `RewardWeights`:

```python
@dataclass
class RewardWeights:
    pref_weight: float = 1.0          # Preference matching
    feasible_penalty: float = 1.5     # Time window violation
    overtime_penalty: float = 1.2     # Exceeding day time limit
    travel_penalty: float = 0.003     # Excessive travel time
    diversity_weight: float = 0.3     # Category diversity bonus
```

## Algorithm Details

### GRPO Overview

GRPO differs from PPO in how advantages are computed:

**PPO**:
```
Advantage(s) = R(s) - V_φ(s)  # Uses learned value function
```

**GRPO**:
```
Advantage(s_i) = R(s_i) - mean({R(s_1), ..., R(s_N)})  # Group-relative
```

For each prompt, GRPO samples N responses and computes advantages relative to the group mean. This eliminates the need for a critic network.

### Training Loop

1. **Sample Prompts**: Get a batch of prompts from dataset
2. **Group Generation**: For each prompt, generate N responses
3. **Compute Rewards**: Score each generated POI sequence
4. **Compute Advantages**: Calculate group-relative advantages
5. **Policy Update**: Optimize policy with PPO-style objective + KL penalty
6. **Repeat**: Until convergence

### Loss Function

```
L = L_policy + β * L_kl

where:
L_policy = -E[log π_θ(a|s) * A(s,a)]
L_kl = KL(π_θ || π_ref)
```

## Output

The trained model is saved to the specified output directory:

```
outputs/grpo/qwen3-grpo-planner/
├── adapter_config.json       # LoRA configuration
├── adapter_model.safetensors # LoRA weights
├── tokenizer_config.json     # Tokenizer config
├── tokenizer.json            # Tokenizer vocab
└── grpo_config.json          # Training config
```

## Inference

Load the trained model for inference:

```python
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("models/Qwen3-8B")
tokenizer = AutoTokenizer.from_pretrained("models/Qwen3-8B")

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "outputs/grpo/qwen3-grpo-planner")

# Generate
messages = [
    {"role": "system", "content": "你是一位专业的旅游规划助手。"},
    {"role": "user", "content": "已选景点：S094645, S151634\n\n请推荐下一个景点。"}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=100)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Monitoring Training

View training logs with TensorBoard:

```bash
tensorboard --logdir outputs/grpo/qwen3-grpo-planner/runs
```

Key metrics to monitor:
- `loss`: Total training loss
- `policy_loss`: Policy gradient loss
- `kl_penalty`: KL divergence (should stay small, < 0.1)
- `mean_reward`: Average reward per batch

## Troubleshooting

### Low Rewards

If rewards are consistently low:
- Check if POI IDs in training data match POI database
- Verify time matrix is correctly loaded
- Adjust reward weights in `RewardWeights`
- Increase model capacity or training time

### High KL Divergence

If KL penalty is too high:
- Decrease learning rate
- Increase `kl_coef` to penalize large policy shifts
- Reduce number of training steps

### Memory Issues

If running out of GPU memory:
- Reduce `batch_size`
- Reduce `group_size`
- Enable gradient checkpointing
- Use smaller base model or quantization

## References

- [DeepSeekMath: GRPO Paper](https://arxiv.org/abs/2406.01806)
- [TRL GRPO Documentation](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [veRL Framework](https://github.com/volcengine/verl) (optional backend)
