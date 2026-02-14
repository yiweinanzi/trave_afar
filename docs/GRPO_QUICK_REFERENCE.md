# GRPO Training Quick Reference

## Quick Start Commands

### 1. Prepare Data (if needed)
```bash
python -m src.rl.dataset_builder \
  --rl-prompts outputs/datasets/planner_rl_prompts.jsonl \
  --out-rl outputs/datasets/grpo_planner_prompts.jsonl
```

### 2. Train Model (Basic)
```bash
bash scripts/train_grpo.sh
```

### 3. Train Model (Advanced)
```bash
python -m src.rl.grpo_trainer \
  --model models/Qwen3-8B \
  --data outputs/datasets/grpo_planner_prompts.jsonl \
  --output outputs/grpo/qwen3-grpo-planner \
  --epochs 3 \
  --batch-size 4 \
  --group-size 4 \
  --lr 1e-5 \
  --use-lora
```

### 4. Train with TRL (Recommended)
```bash
pip install trl>=0.12.0

python -m src.rl.grpo_trainer_trl \
  --model models/Qwen3-8B \
  --data outputs/datasets/grpo_planner_prompts.jsonl \
  --output outputs/grpo/qwen3-grpo-planner
```

### 5. Test Implementation
```bash
python scripts/test_grpo_trainer.py
```

## Environment Variables

```bash
export MODEL_PATH="models/Qwen3-8B"
export DATA_PATH="outputs/datasets/grpo_planner_prompts.jsonl"
export OUTPUT_DIR="outputs/grpo/qwen3-grpo-planner"
export EPOCHS=3
export BATCH_SIZE=4
export GROUP_SIZE=4
export LR=1e-5
export USE_LORA=true
export POI_DATA="data/processed/pois_with_embeddings.parquet"
export TIME_MATRIX="data/processed/time_matrix.npy"
```

## Common Training Scenarios

### Quick Test (1 epoch, small batch)
```bash
bash scripts/train_grpo_quick.sh
```

### Production Training (with all data)
```bash
python -m src.rl.grpo_trainer \
  --model models/Qwen3-8B \
  --data outputs/datasets/grpo_planner_prompts.jsonl \
  --output outputs/grpo/qwen3-grpo-planner \
  --epochs 5 \
  --batch-size 8 \
  --group-size 8 \
  --poi-data data/processed/pois_with_embeddings.parquet \
  --time-matrix data/processed/time_matrix.npy
```

### Fine-tune Existing Model
```bash
python -m src.rl.grpo_trainer \
  --model outputs/grpo/qwen3-grpo-planner \
  --data outputs/datasets/grpo_planner_prompts.jsonl \
  --output outputs/grpo/qwen3-grpo-planner-v2 \
  --epochs 2 \
  --lr 5e-6
```

## Monitoring

### View Training Progress
```bash
tensorboard --logdir outputs/grpo/qwen3-grpo-planner/runs
```

### Check Model Outputs
```bash
ls -lh outputs/grpo/qwen3-grpo-planner/
```

## File Locations

| Item | Path |
|------|------|
| Training script | `scripts/train_grpo.sh` |
| Quick start | `scripts/train_grpo_quick.sh` |
| Native trainer | `src/rl/grpo_trainer.py` |
| TRL trainer | `src/rl/grpo_trainer_trl.py` |
| Test script | `scripts/test_grpo_trainer.py` |
| Training data | `outputs/datasets/grpo_planner_prompts.jsonl` |
| Model output | `outputs/grpo/qwen3-grpo-planner/` |
| Documentation | `docs/GRPO_TRAINING_GUIDE.md` |

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | models/Qwen3-8B | Base model path |
| `--data` | outputs/datasets/grpo_planner_prompts.jsonl | Training data |
| `--output` | outputs/grpo/qwen3-grpo-planner | Output directory |
| `--epochs` | 3 | Number of training epochs |
| `--batch-size` | 4 | Batch size per device |
| `--group-size` | 4 | Samples per prompt |
| `--lr` | 1e-5 | Learning rate |
| `--use-lora` | true | Use LoRA adapters |
| `--poi-data` | data/processed/pois_with_embeddings.parquet | POI data |
| `--time-matrix` | data/processed/time_matrix.npy | Time matrix |

## Troubleshooting Commands

### Check GPU Memory
```bash
nvidia-smi
```

### Verify Data Format
```bash
head -1 outputs/datasets/grpo_planner_prompts.jsonl | python -m json.tool
```

### Check Model Files
```bash
ls -lh models/Qwen3-8B/
```

### Test Import
```bash
python -c "from src.rl.grpo_trainer import GRPOTrainer; print('OK')"
```

## Performance Tips

1. **Increase Speed**: Use larger `batch-size` if GPU memory allows
2. **Better Quality**: Increase `group-size` for more accurate advantages
3. **Faster Convergence**: Use `--use-lora` for faster training
4. **Stability**: Reduce `--lr` if training is unstable
5. **Memory**: Reduce `batch-size` or `group-size` if OOM

## Next Steps

After training:

1. Evaluate model performance
2. Test on validation set
3. Deploy to production
4. Monitor and iterate

See `docs/GRPO_TRAINING_GUIDE.md` for detailed instructions.
