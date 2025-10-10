#!/bin/bash
# Test cross-lingual evaluation on one model
# Usage: ./scripts/test_cross_lingual.sh [model_name]

set -e

MODEL_NAME=${1:-"CRF"}
ENGLISH_DATA="data/votie_test_en/test.jsonl"
OUTPUT_DIR="results/cross_lingual_test"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p $OUTPUT_DIR

echo "🧪 Testing cross-lingual evaluation"
echo "Model: $MODEL_NAME"
echo "English data: $ENGLISH_DATA"
echo ""

# Define model path based on name
case $MODEL_NAME in
    "CRF")
        MODEL_PATH="models/crf/crf_20251010_064309"
        ;;
    "BERT-CRF")
        MODEL_PATH="models/bert_crf/bert_crf_20251010_064309"
        ;;
    "DeBERTa-CRF")
        MODEL_PATH="models/deberta_crf/deberta_crf_20251010_064309"
        ;;
    *)
        echo "❌ Unknown model: $MODEL_NAME"
        exit 1
        ;;
esac

echo "📁 Model path: $MODEL_PATH"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Model directory not found: $MODEL_PATH"
    exit 1
fi

# For transformer models, find the best checkpoint
ACTUAL_MODEL_PATH="$MODEL_PATH"
if [[ "$MODEL_NAME" == *"BERT"* ]] || [[ "$MODEL_NAME" == *"DeBERTa"* ]]; then
    if [ ! -f "$MODEL_PATH/pytorch_model.bin" ]; then
        LATEST_EPOCH=$(find "$MODEL_PATH" -name "epoch_*" -type d | sort -V | tail -1)
        if [ -n "$LATEST_EPOCH" ]; then
            ACTUAL_MODEL_PATH="$LATEST_EPOCH"
            echo "📂 Using checkpoint: $ACTUAL_MODEL_PATH"
        fi
    fi
fi

# Output files
MODEL_OUTPUT="$OUTPUT_DIR/${MODEL_NAME,,}_test_predictions_${TIMESTAMP}.jsonl"
EVAL_OUTPUT="$OUTPUT_DIR/${MODEL_NAME,,}_test_evaluation_${TIMESTAMP}.json"

echo "🔮 Generating predictions..."
if python scripts/predict.py "$ACTUAL_MODEL_PATH" "$ENGLISH_DATA" "$MODEL_OUTPUT" cpu; then
    echo "✅ Predictions saved to: $MODEL_OUTPUT"
    
    echo "📊 Evaluating predictions..."
    if python scripts/evaluate.py "$MODEL_OUTPUT" "$EVAL_OUTPUT"; then
        echo "✅ Evaluation completed: $EVAL_OUTPUT"
        
        # Show results
        if [ -f "$EVAL_OUTPUT" ]; then
            F1=$(python -c "import json; data=json.load(open('$EVAL_OUTPUT')); print(f\"{data.get('entity_f1_strict', 0.0):.4f}\")" 2>/dev/null || echo "N/A")
            echo "🎯 Entity F1 Score: $F1"
        fi
    else
        echo "❌ Evaluation failed"
        exit 1
    fi
else
    echo "❌ Prediction failed"
    exit 1
fi

echo "✅ Test completed successfully!"