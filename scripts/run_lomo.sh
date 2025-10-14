#!/bin/bash
# scripts/run_lomo.sh
# Run LOMO (Leave-One-Municipality-Out) experiments using DeBERTa-CRF
# Sequential execution across all municipalities
# Runs with nohup for background execution

set -e

# Municipality configurations
MUNICIPALITIES=("alandroal" "campomaior" "covilha" "fundao" "guimaraes" "porto")
CONFIG_DIR="configs/cross_municipality/bert_crf"
OUTPUT_DIR="experiments/lomo_bert_crf"
LOG_DIR="logs/lomo_bert_crf"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create output and log directories
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

echo "🚀 Starting LOMO experiments with DeBERTa-CRF"
echo "Running sequentially across all municipalities..."
echo "📁 Output directory: $OUTPUT_DIR"
echo "📝 Logs directory: $LOG_DIR"
echo "⏰ Session timestamp: $TIMESTAMP"
echo ""

# Main log file for the entire session
MAIN_LOG_FILE="$LOG_DIR/lomo_session_${TIMESTAMP}.log"
echo "Starting LOMO experiments at $(date)" > "$MAIN_LOG_FILE"

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
    
    # Check if data directory exists
    DATA_DIR=$(python -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['data']['data_dir'])" 2>/dev/null || echo "")
    if [ -n "$DATA_DIR" ] && [ ! -d "$DATA_DIR" ]; then
        echo "❌ Data directory not found: $DATA_DIR"
        echo "⚠️  Please ensure LOMO data is properly set up"
        continue
    fi
    
    # Individual log file for this municipality
    MUNI_LOG_FILE="$LOG_DIR/lomo_${MUNI}_${TIMESTAMP}.log"
    
    # Run the pipeline (train + predict + evaluate)
    echo "🚀 Starting pipeline for $MODEL_NAME..."
    echo "📝 Using DeBERTa-CRF model for $MUNI municipality..."
    echo "📋 Logs: $MUNI_LOG_FILE"
    
    echo "$(date): Starting experiment for $MUNI" >> "$MAIN_LOG_FILE"
    
    if nohup python scripts/run_pipeline.py \
        --config "$CONFIG_FILE" \
        --name "$MODEL_NAME" \
        --force > "$MUNI_LOG_FILE" 2>&1; then
        
        # Move results to LOMO output directory
        if [ -d "experiments/pt_stratified_split/$MODEL_NAME" ]; then
            cp -r "experiments/pt_stratified_split/$MODEL_NAME" "$OUTPUT_DIR/$MODEL_NAME"
            echo "📊 Results copied to: $OUTPUT_DIR/$MODEL_NAME"
            
            # Clean up the temporary results
            rm -rf "experiments/pt_stratified_split/$MODEL_NAME"
        else
            echo "⚠️  Results directory not found: experiments/pt_stratified_split/$MODEL_NAME"
        fi
        
        echo "✅ Completed LOMO experiment for $MUNI"
        echo "$(date): Completed experiment for $MUNI" >> "$MAIN_LOG_FILE"
    else
        echo "❌ Pipeline failed for $MUNI"
        echo "⚠️  Check logs: $MUNI_LOG_FILE"
        echo "$(date): Failed experiment for $MUNI" >> "$MAIN_LOG_FILE"
    fi
    echo ""
done

echo "========================================="
echo "🎉 All LOMO experiments completed!"
echo "📁 Results directory: $OUTPUT_DIR"
echo "📝 Session logs: $MAIN_LOG_FILE"
echo ""

# Log completion
echo "$(date): All LOMO experiments completed" >> "$MAIN_LOG_FILE"
echo "📊 Summary of experiments:"
for MUNI in "${MUNICIPALITIES[@]}"
do
    MODEL_NAME="lomo_deberta_crf_test_${MUNI}"
    RESULTS_FILE="$OUTPUT_DIR/$MODEL_NAME/evaluation_results.json"
    
    if [ -f "$RESULTS_FILE" ]; then
        # Extract F1 score from results
        F1_SCORE=$(python -c "import json; data=json.load(open('$RESULTS_FILE')); print(f\"{data['entity_level_metrics']['entity_f1']:.4f}\")" 2>/dev/null || echo "N/A")
        echo "  🏛️  $MUNI: F1 = $F1_SCORE"
    else
        echo "  🏛️  $MUNI: Results not found"
    fi
done
echo "========================================="