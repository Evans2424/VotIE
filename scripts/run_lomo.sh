#!/bin/bash
# scripts/run_lomo.sh
# Run LOMO (Leave-One-Municipality-Out) experiments using DeBERTa-CRF
# Sequential execution across all municipalities

set -e

# Municipality configurations
MUNICIPALITIES=("alandroal" "campomaior" "covilha" "fundao" "guimaraes" "porto")
CONFIG_DIR="configs/cross_municipality/deberta_crf"
OUTPUT_DIR="results/lomo"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "🚀 Starting LOMO experiments with DeBERTa-CRF"
echo "Running sequentially across all municipalities..."
echo ""

for MUNI in "${MUNICIPALITIES[@]}"
do
    echo "========================================="
    echo "🏛️  LOMO Experiment: Test on $MUNI"
    echo "📊 Training on 5 other municipalities"
    echo "========================================="
    
    CONFIG_FILE="${CONFIG_DIR}/test_${MUNI}.yaml"
    MODEL_NAME="lomo_deberta_crf_test_${MUNI}"
    
    # Check if config file exists
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "❌ Config file not found: $CONFIG_FILE"
        continue
    fi
    
    echo "📁 Config: $CONFIG_FILE"
    echo "🎯 Model: $MODEL_NAME"
    echo ""
    
    # Run the pipeline (train + predict + evaluate)
    python scripts/run_pipeline.py \
        --config "$CONFIG_FILE" \
        --name "$MODEL_NAME" \
        --force
    
    # Move results to LOMO output directory
    if [ -d "experiments/pt_stratified_split/$MODEL_NAME" ]; then
        cp -r "experiments/pt_stratified_split/$MODEL_NAME" "$OUTPUT_DIR/$MODEL_NAME"
        echo "📊 Results copied to: $OUTPUT_DIR/$MODEL_NAME"
    fi
    
    echo "✅ Completed LOMO experiment for $MUNI"
    echo ""
done

echo "========================================="
echo "🎉 All LOMO experiments completed!"
echo "📁 Results directory: $OUTPUT_DIR"
echo ""
echo "📊 Summary of experiments:"
for MUNI in "${MUNICIPALITIES[@]}"
do
    MODEL_NAME="lomo_deberta_crf_test_${MUNI}"
    RESULTS_FILE="$OUTPUT_DIR/$MODEL_NAME/evaluation_results.json"
    
    if [ -f "$RESULTS_FILE" ]; then
        # Extract F1 score from results
        F1_SCORE=$(python -c "import json; data=json.load(open('$RESULTS_FILE')); print(f\"{data.get('entity_f1_strict', 0.0):.4f}\")" 2>/dev/null || echo "N/A")
        echo "  🏛️  $MUNI: F1 = $F1_SCORE"
    else
        echo "  🏛️  $MUNI: Results not found"
    fi
done
echo "========================================="