#!/bin/bash
# DPO Training Script for Content Generation
#
# This script trains a content generation model using DPO (Direct Preference Optimization).
# It uses TRL's DPOTrainer with QLoRA support for efficient fine-tuning.
#
# Usage:
#   bash scripts/train_dpo.sh [--model MODEL_PATH] [--data DATA_PATH] [--output OUTPUT_DIR]
#
# Example:
#   bash scripts/train_dpo.sh \
#     --model models/Qwen3-8B \
#     --data outputs/datasets/dpo_prefs.csv \
#     --output outputs/dpo/qwen3-8b-dpo

set -e  # Exit on error

# Default paths
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-models/Qwen3-8B}"
DATA_PATH="${DATA_PATH:-outputs/datasets/dpo_prefs.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/dpo/qwen3-8b-dpo}"

# Training hyperparameters
EPOCHS=${EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-2}
GRAD_ACCUM=${GRAD_ACCUM:-4}
LR=${LR:-1e-5}
BETA=${BETA:-0.1}
MAX_LENGTH=${MAX_LENGTH:-512}
USE_LORA=${USE_LORA:-true}
USE_QLORA=${USE_QLORA:-true}

echo "=========================================="
echo "DPO Training for Content Generation"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Data: ${DATA_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Hyperparameters:"
echo "  Epochs: ${EPOCHS}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Gradient Accumulation: ${GRAD_ACCUM}"
echo "  Effective Batch Size: $((BATCH_SIZE * GRAD_ACCUM))"
echo "  Learning Rate: ${LR}"
echo "  Beta (DPO temp): ${BETA}"
echo "  Max Length: ${MAX_LENGTH}"
echo "  Use LoRA: ${USE_LORA}"
echo "  Use QLoRA: ${USE_QLORA}"
echo "=========================================="

# Check if data file exists
if [ ! -f "${DATA_PATH}" ]; then
    echo "Error: Training data not found at ${DATA_PATH}"
    echo "Please ensure the DPO preference dataset exists."
    exit 1
fi

# Show dataset info
NUM_SAMPLES=$(wc -l < "${DATA_PATH}")
echo ""
echo "Dataset: ${NUM_SAMPLES} preference pairs"
echo ""

# Check if model exists
if [ ! -d "${MODEL_PATH}" ]; then
    echo "Warning: Model directory not found at ${MODEL_PATH}"
    echo "Will attempt to download from HuggingFace..."
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Activate Python environment if needed
if [ -n "${VIRTUAL_ENV}" ]; then
    echo "Using virtual environment: ${VIRTUAL_ENV}"
fi

# Build command arguments
TRAIN_ARGS=(
    --model "${MODEL_PATH}"
    --prefs "${DATA_PATH}"
    --output "${OUTPUT_DIR}"
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --grad-accum "${GRAD_ACCUM}"
    --lr "${LR}"
    --beta "${BETA}"
    --max-length "${MAX_LENGTH}"
)

# Handle LoRA/QLoRA flags
if [ "${USE_LORA}" = "true" ]; then
    TRAIN_ARGS+=(--use-lora)
    if [ "${USE_QLORA}" = "true" ]; then
        TRAIN_ARGS+=(--use-qlora)
    else
        TRAIN_ARGS+=(--no-qlora)
    fi
else
    TRAIN_ARGS+=(--no-lora)
fi

# Run training
echo ""
echo "Starting DPO training..."
echo ""

python -m src.content_generation.train_dpo "${TRAIN_ARGS[@]}"

echo ""
echo "=========================================="
echo "Training completed!"
echo "Model saved to: ${OUTPUT_DIR}"
echo ""
echo "To use the trained model:"
echo "  python -c \"from transformers import AutoModelForCausalLM; model = AutoModelForCausalLM.from_pretrained('${OUTPUT_DIR}')\""
echo "=========================================="
