#!/bin/bash
# Quick start GRPO training script
#
# For production training, use scripts/train_grpo.sh

set -e

echo "🚀 Quick Start: GRPO Training for Route Planning"
echo ""

# Set paths
export MODEL_PATH="models/Qwen3-8B"
export DATA_PATH="outputs/datasets/grpo_planner_prompts.jsonl"
export OUTPUT_DIR="outputs/grpo/qwen3-grpo-planner"
export EPOCHS=3
export BATCH_SIZE=4
export GROUP_SIZE=4

# Check dependencies
echo "�� Checking dependencies..."
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')" 2>/dev/null || echo "✗ PyTorch not found"
python -c "import transformers; print('✓ transformers installed')" 2>/dev/null || echo "✗ transformers not found"
python -c "import peft; print('✓ peft installed')" 2>/dev/null || echo "✗ peft not found"
echo ""

# Check data
if [ ! -f "$DATA_PATH" ]; then
    echo "⚠ Training data not found at $DATA_PATH"
    echo "Building dataset first..."
    python -m src.rl.dataset_builder \
        --rl-prompts outputs/datasets/planner_rl_prompts.jsonl \
        --out-rl "$DATA_PATH"
fi

# Run training
echo "🎯 Starting GRPO training..."
echo "   Model: $MODEL_PATH"
echo "   Data: $DATA_PATH"
echo "   Output: $OUTPUT_DIR"
echo ""

python -m src.rl.grpo_trainer \
    --model "$MODEL_PATH" \
    --data "$DATA_PATH" \
    --output "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --group-size "$GROUP_SIZE" \
    --use-lora

echo ""
echo "✅ Training completed!"
echo "📁 Model saved to: $OUTPUT_DIR"
