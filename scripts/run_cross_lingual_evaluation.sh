#!/bin/bash
# scripts/run_cross_lingual_evaluation.sh
# Evaluate all baseline models on English test data for cross-lingual transfer study

set -e

# Configuration
ENGLISH_DATA="data/votie_test_en/test.jsonl"
OUTPUT_DIR="results/cross_lingual_evaluation"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create output directory
mkdir -p $OUTPUT_DIR

echo "🌍 Cross-Lingual Evaluation Study"
echo "Evaluating Portuguese-trained models on English test data"  
echo "English test data: $ENGLISH_DATA"
echo "Output directory: $OUTPUT_DIR"
echo "Timestamp: $TIMESTAMP"
echo ""

# Check if English test data exists
if [ ! -f "$ENGLISH_DATA" ]; then
    echo "❌ English test data not found: $ENGLISH_DATA"
    exit 1
fi

# Find all trained models from the main experiment
MODELS=(
    "models/bert_crf/bert_crf_20251010_064309"
    "models/bert_linear/bert_linear_20251010_064309"
    "models/deberta_crf/deberta_crf_20251010_064309" 
    "models/deberta_linear/deberta_linear_20251010_064309"
    "models/xlmr_crf/xlmr_crf_20251010_064309"
    "models/xlmr_linear/xlmr_linear_20251010_064309"
    "models/bilstm_fasttext/bilstm_fasttext_20251010_064309"
    "models/crf/crf_20251010_064309"
)

MODEL_NAMES=(
    "BERT-CRF"
    "BERT-Linear"
    "DeBERTa-CRF"
    "DeBERTa-Linear" 
    "XLM-R-CRF"
    "XLM-R-Linear"
    "BiLSTM-FastText"
    "CRF"
)

# Initialize results summary
SUMMARY_FILE="$OUTPUT_DIR/cross_lingual_summary_${TIMESTAMP}.csv"
echo "Model,F1,Precision,Recall,Event_Detection,Complete_Events,Model_Path" > $SUMMARY_FILE

echo "📊 Starting cross-lingual evaluation..."
echo ""

for i in "${!MODELS[@]}"; do
    MODEL_PATH="${MODELS[$i]}"
    MODEL_NAME="${MODEL_NAMES[$i]}"
    
    echo "========================================="
    echo "🔤 Evaluating: $MODEL_NAME"
    echo "📁 Model: $MODEL_PATH"
    echo "========================================="
    
    # Check if model exists
    if [ ! -d "$MODEL_PATH" ]; then
        echo "⚠️  Model directory not found: $MODEL_PATH"
        echo "❌ Skipping $MODEL_NAME"
        echo ""
        continue
    fi
    
    # For transformer models, find the best epoch/checkpoint
    ACTUAL_MODEL_PATH="$MODEL_PATH"
    if [[ "$MODEL_NAME" == *"BERT"* ]] || [[ "$MODEL_NAME" == *"DeBERTa"* ]] || [[ "$MODEL_NAME" == *"XLM-R"* ]]; then
        # Look for best model or latest epoch
        if [ -f "$MODEL_PATH/pytorch_model.bin" ]; then
            ACTUAL_MODEL_PATH="$MODEL_PATH"
        else
            # Find the latest epoch directory
            LATEST_EPOCH=$(find "$MODEL_PATH" -name "epoch_*" -type d | sort -V | tail -1)
            if [ -n "$LATEST_EPOCH" ]; then
                ACTUAL_MODEL_PATH="$LATEST_EPOCH"
                echo "📂 Using checkpoint: $ACTUAL_MODEL_PATH"
            else
                echo "❌ No valid checkpoint found in $MODEL_PATH"
                continue
            fi
        fi
    fi
    
    # Create output file for this model
    MODEL_OUTPUT="$OUTPUT_DIR/${MODEL_NAME,,}_english_results_${TIMESTAMP}.jsonl"
    EVAL_OUTPUT="$OUTPUT_DIR/${MODEL_NAME,,}_english_evaluation_${TIMESTAMP}.json"
    
    echo "🔮 Generating predictions..."
    
    # Generate predictions
    if python scripts/predict.py "$ACTUAL_MODEL_PATH" "$ENGLISH_DATA" "$MODEL_OUTPUT" cpu; then
        echo "✅ Predictions generated: $MODEL_OUTPUT"
        
        echo "📊 Evaluating predictions..."
        
        # Evaluate predictions
        if python scripts/evaluate.py "$MODEL_OUTPUT" "$EVAL_OUTPUT"; then
            echo "✅ Evaluation completed: $EVAL_OUTPUT"
            
            # Extract metrics from evaluation results
            if [ -f "$EVAL_OUTPUT" ]; then
                F1=$(python -c "import json; data=json.load(open('$EVAL_OUTPUT')); print(f\"{data.get('entity_f1_strict', 0.0):.4f}\")" 2>/dev/null || echo "0.0000")
                PRECISION=$(python -c "import json; data=json.load(open('$EVAL_OUTPUT')); print(f\"{data.get('entity_precision', 0.0):.4f}\")" 2>/dev/null || echo "0.0000")
                RECALL=$(python -c "import json; data=json.load(open('$EVAL_OUTPUT')); print(f\"{data.get('entity_recall', 0.0):.4f}\")" 2>/dev/null || echo "0.0000")
                EVENT_DET=$(python -c "import json; data=json.load(open('$EVAL_OUTPUT')); print(f\"{data.get('event_detection_accuracy', 0.0):.4f}\")" 2>/dev/null || echo "0.0000")
                COMPLETE_EVT=$(python -c "import json; data=json.load(open('$EVAL_OUTPUT')); print(f\"{data.get('complete_event_accuracy', 0.0):.4f}\")" 2>/dev/null || echo "0.0000")
                
                # Add to summary
                echo "$MODEL_NAME,$F1,$PRECISION,$RECALL,$EVENT_DET,$COMPLETE_EVT,$ACTUAL_MODEL_PATH" >> $SUMMARY_FILE
                
                echo "📈 Results: F1=$F1, P=$PRECISION, R=$RECALL"
                echo "🎯 Event Detection=$EVENT_DET, Complete Events=$COMPLETE_EVT"
            else
                echo "❌ Evaluation results file not found"
                echo "$MODEL_NAME,ERROR,ERROR,ERROR,ERROR,ERROR,$ACTUAL_MODEL_PATH" >> $SUMMARY_FILE
            fi
        else
            echo "❌ Evaluation failed for $MODEL_NAME"
            echo "$MODEL_NAME,EVAL_FAILED,EVAL_FAILED,EVAL_FAILED,EVAL_FAILED,EVAL_FAILED,$ACTUAL_MODEL_PATH" >> $SUMMARY_FILE
        fi
    else
        echo "❌ Prediction failed for $MODEL_NAME"
        echo "$MODEL_NAME,PRED_FAILED,PRED_FAILED,PRED_FAILED,PRED_FAILED,PRED_FAILED,$ACTUAL_MODEL_PATH" >> $SUMMARY_FILE
    fi
    
    echo "✅ Completed evaluation for $MODEL_NAME"
    echo ""
done

echo "========================================="
echo "🎉 Cross-lingual evaluation completed!"
echo "========================================="
echo ""
echo "📊 Summary of results:"
echo ""

# Display summary table
if [ -f "$SUMMARY_FILE" ]; then
    echo "Model                | F1     | Precision | Recall | Event Det. | Complete"
    echo "--------------------+--------+-----------+--------+------------+---------"
    
    tail -n +2 "$SUMMARY_FILE" | while IFS=',' read -r model f1 precision recall event_det complete_evt model_path; do
        printf "%-19s | %-6s | %-9s | %-6s | %-10s | %-8s\n" "$model" "$f1" "$precision" "$recall" "$event_det" "$complete_evt"
    done
    
    echo ""
    echo "📁 Detailed results saved to: $OUTPUT_DIR/"
    echo "📈 Summary table: $SUMMARY_FILE"
    echo ""
    echo "🔍 Key insights for cross-lingual transfer:"
    echo "  • Compare multilingual models (XLM-R) vs monolingual (BERT, DeBERTa)"
    echo "  • Analyze feature-based (CRF, BiLSTM) vs contextual approaches"
    echo "  • Examine if Portuguese training helps English performance"
else
    echo "❌ Summary file not created"
fi

echo "========================================="