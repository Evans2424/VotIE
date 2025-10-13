#!/bin/bash
# scripts/re_evaluate_all_experiments.sh
# Re-evaluate all experiments in experiments/pt_stratified_split with updated metrics
# This will replace existing evaluation results with new ones using strict subject evaluation

set -e

# Configuration
EXPERIMENTS_DIR="experiments/pt_stratified_split"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "🔄 Re-evaluating all experiments with updated metrics"
echo "📁 Experiments directory: $EXPERIMENTS_DIR"
echo "⏰ Timestamp: $TIMESTAMP"
echo ""

# Check if experiments directory exists
if [ ! -d "$EXPERIMENTS_DIR" ]; then
    echo "❌ Experiments directory not found: $EXPERIMENTS_DIR"
    exit 1
fi

# Find all experiment directories with predictions
# Look for predictions.jsonl or any .jsonl file (e.g., bilstm_fastext.jsonl)
EXPERIMENT_DIRS=()
for dir in "$EXPERIMENTS_DIR"/*; do
    if [ -d "$dir" ]; then
        # Check for predictions.jsonl first
        if [ -f "$dir/predictions.jsonl" ]; then
            EXPERIMENT_DIRS+=("$(basename "$dir")")
        # Check for any other .jsonl file (e.g., bilstm_fastext.jsonl)
        elif ls "$dir"/*.jsonl 1> /dev/null 2>&1; then
            EXPERIMENT_DIRS+=("$(basename "$dir")")
        fi
    fi
done

if [ ${#EXPERIMENT_DIRS[@]} -eq 0 ]; then
    echo "❌ No experiments with predictions found in $EXPERIMENTS_DIR"
    exit 1
fi

echo "📊 Found ${#EXPERIMENT_DIRS[@]} experiments with predictions:"
for exp in "${EXPERIMENT_DIRS[@]}"; do
    echo "  📁 $exp"
done
echo ""

# Initialize summary
SUMMARY_FILE="$EXPERIMENTS_DIR/re_evaluation_summary_${TIMESTAMP}.csv"
echo "Experiment,Entity_F1,Entity_Precision,Entity_Recall,Event_Detection,Subject_F1,Participant_Vote_F1,Complete_Event_Accuracy,Status" > "$SUMMARY_FILE"

SUCCESS_COUNT=0
FAILED_COUNT=0

for exp_name in "${EXPERIMENT_DIRS[@]}"; do
    EXP_DIR="$EXPERIMENTS_DIR/$exp_name"
    
    # Find prediction file - check for predictions.jsonl first, then any .jsonl file
    if [ -f "$EXP_DIR/predictions.jsonl" ]; then
        PREDICTIONS_FILE="$EXP_DIR/predictions.jsonl"
    else
        # Find first .jsonl file in directory
        PREDICTIONS_FILE=$(find "$EXP_DIR" -maxdepth 1 -name "*.jsonl" -type f | head -n 1)
        if [ -z "$PREDICTIONS_FILE" ]; then
            echo "⚠️ No prediction file found in $EXP_DIR, skipping..."
            continue
        fi
    fi
    
    OLD_RESULTS_FILE="$EXP_DIR/evaluation_results.json"
    NEW_RESULTS_FILE="$EXP_DIR/evaluation_results_new.json"
    BACKUP_FILE="$EXP_DIR/evaluation_results_backup_${TIMESTAMP}.json"
    
    echo "========================================="
    echo "🔬 Re-evaluating: $exp_name"
    echo "📄 Predictions: $(basename "$PREDICTIONS_FILE")"
    echo "========================================="
    
    # Backup old results if they exist
    if [ -f "$OLD_RESULTS_FILE" ]; then
        cp "$OLD_RESULTS_FILE" "$BACKUP_FILE"
        echo "💾 Backed up old results to: evaluation_results_backup_${TIMESTAMP}.json"
    fi
    
    # Run evaluation
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
            echo "$exp_name,$ENTITY_F1,$ENTITY_PRECISION,$ENTITY_RECALL,$EVENT_DETECTION,$SUBJECT_F1,$PARTICIPANT_F1,$COMPLETE_EVENT,SUCCESS" >> "$SUMMARY_FILE"
            
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            echo "⚠️ Results file not found after evaluation"
            echo "$exp_name,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,NO_RESULTS" >> "$SUMMARY_FILE"
            FAILED_COUNT=$((FAILED_COUNT + 1))
        fi
    else
        echo "❌ Evaluation failed for $exp_name"
        echo "$exp_name,EVAL_FAILED,EVAL_FAILED,EVAL_FAILED,EVAL_FAILED,EVAL_FAILED,EVAL_FAILED,EVAL_FAILED,EVAL_FAILED" >> "$SUMMARY_FILE"
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
echo "  📁 Total: ${#EXPERIMENT_DIRS[@]}"
echo ""

if [ -f "$SUMMARY_FILE" ]; then
    echo "📈 Results summary:"
    echo ""
    echo "Experiment                      | Entity F1 | Subject F1 | Event Det. | Complete"
    echo "-------------------------------+-----------+------------+------------+---------"
    
    tail -n +2 "$SUMMARY_FILE" | while IFS=',' read -r exp entity_f1 entity_p entity_r event_det subject_f1 participant_f1 complete_evt status; do
        if [ "$status" = "SUCCESS" ]; then
            printf "%-30s | %-9s | %-10s | %-10s | %-8s\n" "$exp" "$entity_f1" "$subject_f1" "$event_det" "$complete_evt"
        else
            printf "%-30s | %-9s | %-10s | %-10s | %-8s\n" "$exp" "$status" "$status" "$status" "$status"
        fi
    done
    
    echo ""
    echo "📄 Detailed summary saved to: $SUMMARY_FILE"
    echo ""
    echo "📝 Notes:"
    echo "  • Old evaluation results backed up with timestamp suffix"
    echo "  • Subject F1 now uses strict set-based matching (may be lower than before)"
    echo "  • All metrics reflect updated evaluation methodology"
    echo ""
else
    echo "❌ Summary file not created"
fi

echo "========================================="