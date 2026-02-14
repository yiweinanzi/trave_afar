# DPO Training Quick Reference

## Files
- **Training**: `src/content_generation/train_dpo.py`
- **Scripts**: `scripts/train_dpo.sh`, `scripts/train_dpo_quick.sh`
- **Data**: `outputs/datasets/dpo_prefs.csv`
- **Docs**: `outputs/datasets/README_DPO_TRAINING.md`

## Quick Start

```bash
# 1. Quick test (1 epoch)
bash scripts/train_dpo_quick.sh

# 2. Full training (3 epochs)
bash scripts/train_dpo.sh

# 3. Custom training
EPOCHS=5 BATCH_SIZE=4 LR=5e-6 bash scripts/train_dpo.sh
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 3 | Training epochs |
| `--batch-size` | 2 | Per-device batch size |
| `--grad-accum` | 4 | Gradient accumulation |
| `--lr` | 1e-5 | Learning rate |
| `--beta` | 0.1 | DPO temperature (lower = stronger) |
| `--use-qlora` | true | Use 4-bit quantization |

## Memory

- QLoRA: ~8 GB
- LoRA: ~16 GB
- Full: ~40 GB

## Using Trained Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
model = PeftModel.from_pretrained(base, "outputs/dpo/qwen3-8b-dpo")
tokenizer = AutoTokenizer.from_pretrained("outputs/dpo/qwen3-8b-dpo")
```
