#!/bin/bash
# scripts/run_lomo.sh
# Run LOMO experiments for all municipalities

set -e

MUNICIPALITIES=("M01_alandroal" "M02_campomaior" "M03_covilha" "M04_fundao" "M05_guimaraes" "M06_porto")
CONFIG="configs/deberta_crf.yaml"
OUTPUT_DIR="results/lomo"

mkdir -p $OUTPUT_DIR

for MUNI in "${MUNICIPALITIES[@]}"
do
    echo "========================================="
    echo "Training on: $MUNI (held out for test)"
    echo "========================================="
    
    # Train
    python scripts/train.py \
        --config $CONFIG \
        --train data/votie_lomo/${MUNI}/train.jsonl \
        --dev data/votie_lomo/${MUNI}/dev.jsonl \
        --test data/votie_lomo/${MUNI}/test.jsonl \
        --output models/lomo_${MUNI} \
        --experiment_name "lomo_${MUNI}"
    
    # Evaluate on held-out municipality
    python scripts/evaluate.py \
        --model models/lomo_${MUNI} \
        --test data/votie_lomo/${MUNI}/test.jsonl \
        --output $OUTPUT_DIR/${MUNI}_results.json
    
    echo "✓ Completed $MUNI"
    echo ""
done

# Aggregate results
python scripts/aggregate_lomo_results.py \
    --input $OUTPUT_DIR \
    --output $OUTPUT_DIR/lomo_summary.csv

echo "========================================="
echo "✓ All LOMO experiments completed"
echo "Results saved to: $OUTPUT_DIR/lomo_summary.csv"
echo "========================================="