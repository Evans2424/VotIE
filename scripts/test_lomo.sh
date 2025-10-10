#!/bin/bash
# Test script for LOMO experiment - runs only one municipality
# Usage: ./scripts/test_lomo.sh [municipality_name]

set -e

MUNI=${1:-"alandroal"}
CONFIG_DIR="configs/cross_municipality/deberta_crf"
OUTPUT_DIR="results/lomo_test"

mkdir -p $OUTPUT_DIR

echo "🧪 Testing LOMO experiment with DeBERTa-CRF"
echo "🏛️  Testing municipality: $MUNI"
echo ""

CONFIG_FILE="${CONFIG_DIR}/test_${MUNI}.yaml"
MODEL_NAME="lomo_deberta_crf_test_${MUNI}_test"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "📁 Config: $CONFIG_FILE"
echo "🎯 Model: $MODEL_NAME"
echo ""

# Run the pipeline (train + predict + evaluate)
python scripts/run_pipeline.py \
    --config "$CONFIG_FILE" \
    --name "$MODEL_NAME" \
    --force

# Move results to test output directory
if [ -d "experiments/pt_stratified_split/$MODEL_NAME" ]; then
    cp -r "experiments/pt_stratified_split/$MODEL_NAME" "$OUTPUT_DIR/$MODEL_NAME"
    echo "📊 Test results copied to: $OUTPUT_DIR/$MODEL_NAME"
fi

echo "✅ Test completed for $MUNI"

# Show results
RESULTS_FILE="$OUTPUT_DIR/$MODEL_NAME/evaluation_results.json"
if [ -f "$RESULTS_FILE" ]; then
    F1_SCORE=$(python -c "import json; data=json.load(open('$RESULTS_FILE')); print(f\"{data.get('entity_f1_strict', 0.0):.4f}\")" 2>/dev/null || echo "N/A")
    echo "🎯 Test F1 Score: $F1_SCORE"
else
    echo "❌ Results file not found"
fi