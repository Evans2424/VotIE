#!/bin/bash
# Cross-Lingual Zero-Shot Transfer Experiments
# Evaluate all Portuguese-trained models on English test set
#
# This is the main script for cross-lingual experiments.
# Simply run: ./experiments/cross_lingual/run_all.sh

set -e

cd "$(dirname "$0")/../.."  # Go to NER_BASELINES_PUBLICATION root

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Cross-Lingual Evaluation: Portuguese → English           ║"
echo "║  Zero-Shot Transfer Experiments                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Configuration
EN_TEST_FILE="data/publication_datasets/ner_datasets/en/splits/test.jsonl"
OUTPUT_BASE="results/cross_lingual"
DEVICE=0

if [ ! -f "$EN_TEST_FILE" ]; then
    echo "❌ Error: English test file not found: $EN_TEST_FILE"
    echo "Please ensure the data is in the correct location."
    exit 1
fi

echo "📊 Test Set: $EN_TEST_FILE"
echo "📁 Output: $OUTPUT_BASE/"
echo "🖥️  Device: GPU $DEVICE"
echo ""

# Count available models
model_count=0
for model_dir in results/*/*/best_model; do
    [ -d "$model_dir" ] && ((model_count++))
done

if [ $model_count -eq 0 ]; then
    echo "❌ No trained models found in results/"
    echo ""
    echo "Please train models first:"
    echo "  python train.py bert_crf --experiment-name pt_baseline"
    echo "  python train.py deberta_crf --experiment-name pt_baseline"
    echo ""
    exit 1
fi

echo "Found $model_count trained model(s)"
echo ""
echo "Starting evaluations..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Function to evaluate a model
evaluate_model() {
    local MODEL_PATH=$1
    local MODEL_TYPE=$(basename $(dirname $(dirname "$MODEL_PATH")))
    local RUN_NAME=$(basename $(dirname "$MODEL_PATH"))
    local OUTPUT_DIR="${OUTPUT_BASE}/${MODEL_TYPE}_${RUN_NAME}"

    echo "🔬 Evaluating: ${MODEL_TYPE}/${RUN_NAME}"

    python evaluate.py \
        --model-path "$MODEL_PATH" \
        --test-file "$EN_TEST_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --device $DEVICE \
        2>&1 | grep -E "(Test F1|Precision|Recall|Results saved)"

    # Extract F1 score
    if [ -f "${OUTPUT_DIR}/evaluation_results.json" ]; then
        F1=$(python -c "import json; print(f\"{json.load(open('${OUTPUT_DIR}/evaluation_results.json'))['metrics']['entity_f1_strict']:.4f}\")")
        echo "   ✅ F1 Score: $F1"
    fi

    echo ""
}

# Evaluate all trained models
for model_path in results/*/*/best_model; do
    if [ -d "$model_path" ]; then
        evaluate_model "$model_path"
    fi
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All evaluations complete!"
echo ""
echo "📊 Summary of Results:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for result_file in ${OUTPUT_BASE}/*/evaluation_results.json; do
    if [ -f "$result_file" ]; then
        MODEL_NAME=$(basename $(dirname "$result_file"))
        F1=$(python -c "import json; print(f\"{json.load(open('$result_file'))['metrics']['entity_f1_strict']:.4f}\")")
        PRECISION=$(python -c "import json; print(f\"{json.load(open('$result_file'))['metrics']['precision']:.4f}\")")
        RECALL=$(python -c "import json; print(f\"{json.load(open('$result_file'))['metrics']['recall']:.4f}\")")
        printf "%-40s F1: %s  P: %s  R: %s\n" "$MODEL_NAME" "$F1" "$PRECISION" "$RECALL"
    fi
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📁 Detailed results: $OUTPUT_BASE/"
echo ""
echo "View detailed results:"
echo "  cat ${OUTPUT_BASE}/<model_name>/evaluation_results.json | jq"
echo ""
