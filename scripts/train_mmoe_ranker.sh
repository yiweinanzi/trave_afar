#!/bin/bash
# Training script for MMoE Deep Ranking Model
#
# Usage:
#   bash scripts/train_mmoe_ranker.sh
#   bash scripts/train_mmoe_ranker.sh --quick

set -e

# Default parameters
POI_CSV="data/all/poi_expanded.csv"
EVENTS_CSV="data/all/user_events.csv"
OUTPUT_DIR="outputs/ranking"
NUM_EXPERTS=4
EPOCHS=20
BATCH_SIZE=512
LR=0.001
DEVICE="cuda"
NEG_RATIO=4

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            EPOCHS=5
            BATCH_SIZE=256
            NUM_EXPERTS=2
            shift
            ;;
        --cpu)
            DEVICE="cpu"
            shift
            ;;
        --debug)
            EPOCHS=2
            BATCH_SIZE=32
            NEG_RATIO=2
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR/checkpoints"

echo "======================================"
echo "MMoE Deep Ranking Model Training"
echo "======================================"
echo "POI CSV: $POI_CSV"
echo "Events CSV: $EVENTS_CSV"
echo "Output: $OUTPUT_DIR"
echo "Experts: $NUM_EXPERTS"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LR"
echo "Device: $DEVICE"
echo "======================================"

# Run training
python -m src.ranking.deep_ranker \
    --poi-csv "$POI_CSV" \
    --events-csv "$EVENTS_CSV" \
    --output "$OUTPUT_DIR/mmoe_model.pt" \
    --num-experts $NUM_EXPERTS \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --neg-ratio $NEG_RATIO \
    --device "$DEVICE" \
    --plot-curves

echo ""
echo "Training completed!"
echo "Model saved to: $OUTPUT_DIR/mmoe_model.pt"
echo "Training curves: $OUTPUT_DIR/training_curves.png"
