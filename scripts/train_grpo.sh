#!/bin/bash
# GRPO Training Script for Route Planning
#
# This script trains a route planning policy using GRPO (Group Relative Policy Optimization).
# It uses the existing GRPO trainer implementation which supports both TRL and veRL backends.
#
# Usage:
#   bash scripts/train_grpo.sh [--model MODEL_PATH] [--data DATA_PATH] [--output OUTPUT_DIR]
#
# Example:
#   bash scripts/train_grpo.sh \
#     --model models/Qwen3-8B \
#     --data outputs/datasets/grpo_planner_prompts.jsonl \
#     --output outputs/grpo/qwen3-grpo-planner

set -e  # Exit on error

# Default paths
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-models/Qwen3-8B}"
DATA_PATH="${DATA_PATH:-outputs/datasets/grpo_planner_prompts.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/grpo/qwen3-grpo-planner}"

# Training hyperparameters
EPOCHS=${EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-4}
GROUP_SIZE=${GROUP_SIZE:-4}
LR=${LR:-1e-5}
USE_LORA=${USE_LORA:-true}

# Optional data for advanced reward computation
POI_DATA="${POI_DATA:-data/processed/pois_with_embeddings.parquet}"
TIME_MATRIX="${TIME_MATRIX:-data/processed/time_matrix.npy}"

echo "=========================================="
echo "GRPO Training for Route Planning"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Data: ${DATA_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Epochs: ${EPOCHS}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Group Size: ${GROUP_SIZE}"
echo "Learning Rate: ${LR}"
echo "Use LoRA: ${USE_LORA}"
echo "=========================================="

# Check if data file exists
if [ ! -f "${DATA_PATH}" ]; then
    echo "Error: Training data not found at ${DATA_PATH}"
    echo "Please run the dataset builder first:"
    echo "  python -m src.rl.dataset_builder --rl-prompts outputs/datasets/planner_rl_prompts.jsonl --out-rl ${DATA_PATH}"
    exit 1
fi

# Check if model exists
if [ ! -d "${MODEL_PATH}" ]; then
    echo "Error: Model not found at ${MODEL_PATH}"
    echo "Please download the model first or specify a valid path"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Activate Python environment if needed
if [ -n "${VIRTUAL_ENV}" ]; then
    echo "Using virtual environment: ${VIRTUAL_ENV}"
fi

# Run training
echo ""
echo "Starting GRPO training..."
echo ""

python -m src.rl.grpo_trainer \
    --model "${MODEL_PATH}" \
    --data "${DATA_PATH}" \
    --output "${OUTPUT_DIR}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --group-size "${GROUP_SIZE}" \
    --lr "${LR}" \
    $( [ "${USE_LORA}" = "true" ] && echo "--use-lora" || echo "--no-lora" ) \
    --poi-data "${POI_DATA}" \
    --time-matrix "${TIME_MATRIX}"

echo ""
echo "=========================================="
echo "Training completed!"
echo "Model saved to: ${OUTPUT_DIR}"
echo "=========================================="
