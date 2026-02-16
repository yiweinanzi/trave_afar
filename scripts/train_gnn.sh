#!/bin/bash
# Train GNN Model for GoAfar
# Complete pipeline: graph building -> training -> inference

set -e

# Paths
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
POI_DIR="${PROJECT_ROOT}/data/shengfen_pois"
GRAPH_DIR="${PROJECT_ROOT}/outputs/gnn"
GRAPH_PATH="${GRAPH_DIR}/graph.pkl"
MODEL_PATH="${GRAPH_DIR}/model.pkl"

# Config
GEO_RADIUS_KM=1.0
MODEL_TYPE="graphsage"
HIDDEN_DIM=128
OUTPUT_DIM=64
NUM_LAYERS=2
NUM_EPOCHS=50
BATCH_SIZE=256
LR=1e-3
DEVICE="cuda"  # or "cpu"

echo "========================================"
echo "GNN Training Pipeline for GoAfar"
echo "========================================"

# Step 1: Build Graph
echo ""
echo "Step 1: Building POI graph..."
echo "  POI directory: ${POI_DIR}"
echo "  Graph output: ${GRAPH_PATH}"

python3 "${PROJECT_ROOT}/scripts/build_poi_graph.py" \
    --poi-dir "${POI_DIR}" \
    --output "${GRAPH_PATH}" \
    --geo-radius ${GEO_RADIUS_KM}

# Step 2: Train Model
echo ""
echo "Step 2: Training GNN model..."
echo "  Graph input: ${GRAPH_PATH}"
echo "  Model output: ${MODEL_PATH}"
echo "  Architecture: ${MODEL_TYPE}"
echo "  Hidden dim: ${HIDDEN_DIM}"
echo "  Output dim: ${OUTPUT_DIM}"
echo "  Epochs: ${NUM_EPOCHS}"

python3 "${PROJECT_ROOT}/scripts/train_gnn_model.py" \
    --graph "${GRAPH_PATH}" \
    --output "${MODEL_PATH}" \
    --model-type ${MODEL_TYPE} \
    --hidden-dim ${HIDDEN_DIM} \
    --output-dim ${OUTPUT_DIM} \
    --num-layers ${NUM_LAYERS} \
    --epochs ${NUM_EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --lr ${LR} \
    --device ${DEVICE}

# Step 3: Inference Demo
echo ""
echo "Step 3: Running inference demo..."

python3 "${PROJECT_ROOT}/scripts/infer_gnn_model.py" \
    --model "${MODEL_PATH}" \
    --top-k 10

echo ""
echo "========================================"
echo "Training complete!"
echo "========================================"
echo "Graph: ${GRAPH_PATH}"
echo "Model: ${MODEL_PATH}"
