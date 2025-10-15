#!/bin/bash
# scripts/re_evaluate_all_experiments.sh
# Re-evaluate all models in models directory with updated metrics
# This will generate predictions and evaluate them using strict subject evaluation

set -e

# Configuration
MODELS_DIR="models"
TEST_DATA_PATH="data/votie_bio/test.jsonl"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "🔄 Re-evaluating all models with updated metrics"
echo "📁 Models directory: $MODELS_DIR"
echo "📄 Test data: $TEST_DATA_PATH"
echo "⏰ Timestamp: $TIMESTAMP"
echo ""

# Check if models directory exists
if [ ! -d "$MODELS_DIR" ]; then
    echo "❌ Models directory not found: $MODELS_DIR"
    exit 1
fi

# Check if test data exists
if [ ! -f "$TEST_DATA_PATH" ]; then
    echo "❌ Test data not found: $TEST_DATA_PATH"
    exit 1
fi

# Find all model directories
MODEL_DIRS=()
for dir in "$MODELS_DIR"/*; do
    if [ -d "$dir" ]; then
        model_name=$(basename "$dir")
        # Skip README files
        if [ "$model_name" != "README.md" ]; then
            # Check if model has a config.json (indicates it's a trained model)
            if [ -f "$dir/config.json" ]; then
                MODEL_DIRS+=("$model_name")
            fi
        fi
    fi
done

if [ ${#MODEL_DIRS[@]} -eq 0 ]; then
    echo "❌ No trained models found in $MODELS_DIR"
    exit 1
fi

echo "📊 Found ${#MODEL_DIRS[@]} trained models:"
for model in "${MODEL_DIRS[@]}"; do
    echo "  🤖 $model"
done
echo ""

# Initialize summary
SUMMARY_FILE="$MODELS_DIR/re_evaluation_summary_${TIMESTAMP}.csv"
echo "Model,Entity_F1,Entity_Precision,Entity_Recall,Event_Detection,Subject_F1,Participant_Vote_F1,Complete_Event_Accuracy,Status" > "$SUMMARY_FILE"

SUCCESS_COUNT=0
FAILED_COUNT=0

for model_name in "${MODEL_DIRS[@]}"; do
    MODEL_DIR="$MODELS_DIR/$model_name"
    PREDICTIONS_FILE="$MODEL_DIR/predictions.jsonl"
    OLD_RESULTS_FILE="$MODEL_DIR/evaluation_results.json"
    NEW_RESULTS_FILE="$MODEL_DIR/evaluation_results_new.json"
    BACKUP_FILE="$MODEL_DIR/evaluation_results_backup_${TIMESTAMP}.json"
    
    echo "========================================="
    echo "🤖 Processing model: $model_name"
    echo "========================================="
    
    # Step 1: Generate predictions
    echo "� Generating predictions..."
    if python scripts/predict.py "$MODEL_DIR" "$TEST_DATA_PATH" "$PREDICTIONS_FILE"; then
        echo "✅ Predictions generated successfully"
    else
        echo "❌ Prediction failed for $model_name"
        echo "$model_name,PRED_FAILED,PRED_FAILED,PRED_FAILED,PRED_FAILED,PRED_FAILED,PRED_FAILED,PRED_FAILED,PRED_FAILED" >> "$SUMMARY_FILE"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        echo ""
        continue
    fi
    
    # Backup old results if they exist
    if [ -f "$OLD_RESULTS_FILE" ]; then
        cp "$OLD_RESULTS_FILE" "$BACKUP_FILE"
        echo "💾 Backed up old results to: evaluation_results_backup_${TIMESTAMP}.json"
    fi
    
    # Step 2: Run evaluation
    echo "🔍 Running evaluation..."
    if python scripts/evaluate.py "$PREDICTIONS_FILE" "$NEW_RESULTS_FILE"; then
        # Replace old results with new ones
        mv "$NEW_RESULTS_FILE" "$OLD_RESULTS_FILE"
        echo "✅ Evaluation completed successfully"
        
        # Extract metrics for summary
        if [ -f "$OLD_RESULTS_FILE" ]; then
            ENTITY_F1=$(python -c "import json; data=json.load(open('$OLD_RESULTS_FILE')); print(f\"{data['entity_level_metrics']['entity_f1']:.4f}\")" 2>/dev/null || echo "N/A")
            ENTITY_PRECISION=$(python -c "import json; data=json.load(open('$OLD_RESULTS_FILE')); print(f\"{data['entity_level_metrics']['entity_precision']:.4f}\")" 2>/dev/null || echo "N/A")
            ENTITY_RECALL=$(python -c "import json; data=json.load(open('$OLD_RESULTS_FILE')); print(f\"{data['entity_level_metrics']['entity_recall']:.4f}\")" 2>/dev/null || echo "N/A")
            EVENT_DETECTION=$(python -c "import json; data=json.load(open('$OLD_RESULTS_FILE')); print(f\"{data['event_level_metrics']['event_detection_accuracy']:.4f}\")" 2>/dev/null || echo "N/A")
            SUBJECT_F1=$(python -c "import json; data=json.load(open('$OLD_RESULTS_FILE')); print(f\"{data['event_level_metrics']['subject_f1']:.4f}\")" 2>/dev/null || echo "N/A")
            PARTICIPANT_F1=$(python -c "import json; data=json.load(open('$OLD_RESULTS_FILE')); print(f\"{data['event_level_metrics']['participant_vote_f1']:.4f}\")" 2>/dev/null || echo "N/A")
            COMPLETE_EVENT=$(python -c "import json; data=json.load(open('$OLD_RESULTS_FILE')); print(f\"{data['event_level_metrics']['complete_event_accuracy']:.4f}\")" 2>/dev/null || echo "N/A")
            
            echo "📈 Results:"
            echo "  Entity F1: $ENTITY_F1"
            echo "  Subject F1 (Event-level): $SUBJECT_F1"
            echo "  Event Detection: $EVENT_DETECTION"
            echo "  Complete Events: $COMPLETE_EVENT"
            
            # Add to summary
            echo "$model_name,$ENTITY_F1,$ENTITY_PRECISION,$ENTITY_RECALL,$EVENT_DETECTION,$SUBJECT_F1,$PARTICIPANT_F1,$COMPLETE_EVENT,SUCCESS" >> "$SUMMARY_FILE"
            
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            echo "⚠️ Results file not found after evaluation"
            echo "$model_name,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,NO_RESULTS" >> "$SUMMARY_FILE"
            FAILED_COUNT=$((FAILED_COUNT + 1))
        fi
    else
        echo "❌ Evaluation failed for $model_name"
        echo "$model_name,EVAL_FAILED,EVAL_FAILED,EVAL_FAILED,EVAL_FAILED,EVAL_FAILED,EVAL_FAILED,EVAL_FAILED,EVAL_FAILED" >> "$SUMMARY_FILE"
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
    
    echo ""
done

echo "========================================="
echo "🎉 Re-evaluation completed!"
echo "========================================="
echo ""
echo "📊 Summary:"
echo "  ✅ Successful: $SUCCESS_COUNT"
echo "  ❌ Failed: $FAILED_COUNT"
echo "  🤖 Total models: ${#MODEL_DIRS[@]}"
echo ""

if [ -f "$SUMMARY_FILE" ]; then
    echo "📈 Results summary:"
    echo ""
    echo "Model                           | Entity F1 | Subject F1 | Event Det. | Complete"
    echo "-------------------------------+-----------+------------+------------+---------"
    
    tail -n +2 "$SUMMARY_FILE" | while IFS=',' read -r model entity_f1 entity_p entity_r event_det subject_f1 participant_f1 complete_evt status; do
        if [ "$status" = "SUCCESS" ]; then
            printf "%-30s | %-9s | %-10s | %-10s | %-8s\n" "$model" "$entity_f1" "$subject_f1" "$event_det" "$complete_evt"
        else
            printf "%-30s | %-9s | %-10s | %-10s | %-8s\n" "$model" "$status" "$status" "$status" "$status"
        fi
    done
    
    echo ""
    echo "📄 Detailed summary saved to: $SUMMARY_FILE"
    echo ""
    echo "📝 Notes:"
    echo "  • Predictions generated from trained models using test data"
    echo "  • Old evaluation results backed up with timestamp suffix"
    echo "  • Subject F1 now uses strict set-based matching (may be lower than before)"
    echo "  • All metrics reflect updated evaluation methodology"
    echo ""
else
    echo "❌ Summary file not created"
fi

echo "========================================="