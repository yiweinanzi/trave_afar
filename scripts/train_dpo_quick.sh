#!/bin/bash
# Quick DPO Training Script (for testing)
# Uses minimal epochs and smaller dataset for quick validation

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-models/Qwen3-8B}"
DATA_PATH="${DATA_PATH:-outputs/datasets/dpo_prefs.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/dpo/qwen3-8b-dpo-quick}"

# Quick test settings
EPOCHS=1
BATCH_SIZE=2
GRAD_ACCUM=2
LR=1e-5
BETA=0.1
MAX_LENGTH=512

echo "=========================================="
echo "Quick DPO Training Test"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Data: ${DATA_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Quick Test Settings:"
echo "  Epochs: ${EPOCHS}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Gradient Accumulation: ${GRAD_ACCUM}"
echo "  Learning Rate: ${LR}"
echo "=========================================="

# Check data
if [ ! -f "${DATA_PATH}" ]; then
    echo "Error: Data file not found: ${DATA_PATH}"
    exit 1
fi

NUM_SAMPLES=$(wc -l < "${DATA_PATH}")
echo "Dataset: ${NUM_SAMPLES} preference pairs"
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run quick training
echo "Starting quick DPO training..."

python -m src.content_generation.train_dpo \
    --model "${MODEL_PATH}" \
    --prefs "${DATA_PATH}" \
    --output "${OUTPUT_DIR}" \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --grad-accum ${GRAD_ACCUM} \
    --lr ${LR} \
    --beta ${BETA} \
    --max-length ${MAX_LENGTH} \
    --use-lora \
    --use-qlora

echo ""
echo "=========================================="
echo "Quick test completed!"
echo "Model saved to: ${OUTPUT_DIR}"
echo "=========================================="
