#!/bin/bash
# scripts/re_evaluate_all_experiments.sh
# Re-evaluate all models in models directory with updated metrics
# This will evaluate existing predictions.jsonl files without re-generating predictions

set -e

# Configuration
MODELS_DIR="models"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "🔄 Re-evaluating all models with updated metrics"
echo "📁 Models directory: $MODELS_DIR"
echo "⏰ Timestamp: $TIMESTAMP"
echo ""

# Check if models directory exists
if [ ! -d "$MODELS_DIR" ]; then
    echo "❌ Models directory not found: $MODELS_DIR"
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

# Initialize summary (simplified - only F1 scores)
SUMMARY_FILE="$MODELS_DIR/re_evaluation_summary_${TIMESTAMP}.csv"
echo "Model,Entity_F1,Event_Detection_F1,Subject_F1,Counting_F1,Participant_Vote_F1,Complete_Event_F1,Status" > "$SUMMARY_FILE"

SUCCESS_COUNT=0
FAILED_COUNT=0

for model_name in "${MODEL_DIRS[@]}"; do
    MODEL_DIR="$MODELS_DIR/$model_name"
    PREDICTIONS_FILE="$MODEL_DIR/predictions.jsonl"
    RESULTS_FILE="$MODEL_DIR/evaluation_results.json"
    
    echo "========================================="
    echo "🤖 Processing model: $model_name"
    echo "========================================="
    
    # Check if predictions file exists
    if [ ! -f "$PREDICTIONS_FILE" ]; then
        echo "❌ Predictions file not found: $PREDICTIONS_FILE"
        echo "   Skipping $model_name - please run predictions first"
        echo "$model_name,NO_PREDICTIONS,NO_PREDICTIONS,NO_PREDICTIONS,NO_PREDICTIONS,NO_PREDICTIONS,NO_PREDICTIONS,NO_PREDICTIONS,NO_PREDICTIONS" >> "$SUMMARY_FILE"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        echo ""
        continue
    fi
    
    echo "✓ Found predictions file: predictions.jsonl"
    
    # Run evaluation (overwrites existing results)
    echo "🔍 Running evaluation..."
    if python scripts/evaluate.py "$PREDICTIONS_FILE" "$RESULTS_FILE"; then
        echo "✅ Evaluation completed successfully"
        
        # Extract metrics for summary (5 key F1 scores only)
        if [ -f "$RESULTS_FILE" ]; then
            ENTITY_F1=$(python -c "import json; data=json.load(open('$RESULTS_FILE')); print(f\"{data['entity_level_metrics']['entity_f1']:.4f}\")" 2>/dev/null || echo "N/A")
            EVENT_DETECTION_F1=$(python -c "import json; data=json.load(open('$RESULTS_FILE')); print(f\"{data['event_level_metrics']['event_detection_f1']:.4f}\")" 2>/dev/null || echo "N/A")
            SUBJECT_F1=$(python -c "import json; data=json.load(open('$RESULTS_FILE')); print(f\"{data['event_level_metrics']['subject_f1']:.4f}\")" 2>/dev/null || echo "N/A")
            COUNTING_F1=$(python -c "import json; data=json.load(open('$RESULTS_FILE')); print(f\"{data['event_level_metrics']['counting_f1']:.4f}\")" 2>/dev/null || echo "N/A")
            PARTICIPANT_F1=$(python -c "import json; data=json.load(open('$RESULTS_FILE')); print(f\"{data['event_level_metrics']['participant_vote_f1']:.4f}\")" 2>/dev/null || echo "N/A")
            COMPLETE_EVENT_F1=$(python -c "import json; data=json.load(open('$RESULTS_FILE')); print(f\"{data['event_level_metrics']['complete_event_f1']:.4f}\")" 2>/dev/null || echo "N/A")
            
            echo "📈 Results (5 Key F1 Scores):"
            echo "  Entity F1:          $ENTITY_F1"
            echo "  Event Detection F1: $EVENT_DETECTION_F1"
            echo "  Subject F1:         $SUBJECT_F1"
            echo "  Counting F1:        $COUNTING_F1"
            echo "  Participant F1:     $PARTICIPANT_F1"
            echo "  Complete Event F1:  $COMPLETE_EVENT_F1"
            
            # Add to summary
            echo "$model_name,$ENTITY_F1,$EVENT_DETECTION_F1,$SUBJECT_F1,$COUNTING_F1,$PARTICIPANT_F1,$COMPLETE_EVENT_F1,SUCCESS" >> "$SUMMARY_FILE"
            
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            echo "⚠️ Results file not found after evaluation"
            echo "$model_name,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,NO_RESULTS" >> "$SUMMARY_FILE"
            FAILED_COUNT=$((FAILED_COUNT + 1))
        fi
    else
        echo "❌ Evaluation failed for $model_name"
        echo "$model_name,EVAL_FAILED,EVAL_FAILED,EVAL_FAILED,EVAL_FAILED,EVAL_FAILED,EVAL_FAILED,EVAL_FAILED" >> "$SUMMARY_FILE"
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
    echo "📈 Results Summary (6 F1 Scores):"
    echo ""
    echo "Model                           | Entity F1 | Event Det. | Subject F1 | Counting   | Part. Vote | Complete"
    echo "-------------------------------+-----------+------------+------------+------------+------------+----------"
    
    tail -n +2 "$SUMMARY_FILE" | while IFS=',' read -r model entity_f1 event_det_f1 subject_f1 counting_f1 participant_f1 complete_f1 status; do
        if [ "$status" = "SUCCESS" ]; then
            printf "%-30s | %-9s | %-10s | %-10s | %-10s | %-10s | %-8s\n" "$model" "$entity_f1" "$event_det_f1" "$subject_f1" "$counting_f1" "$participant_f1" "$complete_f1"
        else
            printf "%-30s | %-9s | %-10s | %-10s | %-10s | %-10s | %-8s\n" "$model" "$status" "$status" "$status" "$status" "$status" "$status"
        fi
    done
    
    echo ""
    echo "📄 Detailed summary saved to: $SUMMARY_FILE"
    echo ""
    echo "📝 Metrics Included:"
    echo "  1. Entity F1           - Span-level NER performance (seqeval)"
    echo "  2. Event Detection F1  - Binary classification (has voting event)"
    echo "  3. Subject F1          - Span-level subject extraction"
    echo "  4. Counting F1         - Span-level counting expression extraction"
    echo "  5. Participant Vote F1 - Set-based (participant, position) tuples"
    echo "  6. Complete Event F1   - Strict matching (all components must match)"
    echo ""
else
    echo "❌ Summary file not created"
fi

echo "========================================="