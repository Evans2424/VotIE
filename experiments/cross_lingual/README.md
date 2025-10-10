# Cross-Lingual Experiments

Zero-shot cross-lingual transfer evaluation for NER models.

## Overview

**Goal:** Test cross-lingual generalization by evaluating Portuguese-trained models on English test data.

**Experimental Setup:**
1. Train models on Portuguese data
2. Evaluate trained models on English test set (zero-shot)
3. Compare performance to in-language (Portuguese) results

This tests whether the models learn language-independent features or language-specific patterns.

## Quick Start

### 1. Train Models on Portuguese

```bash
# Train baseline models
python train.py bert_crf --experiment-name pt_baseline
python train.py deberta_crf --experiment-name pt_baseline
python train.py xlmr_crf --experiment-name pt_baseline
```

### 2. Evaluate on English (Single Model)

```bash
# Evaluate specific model on English
python evaluate.py \
  --model-path results/bert_crf/pt_baseline/best_model \
  --test-file data/publication_datasets/ner_datasets/en/splits/test.jsonl \
  --output-dir results/cross_lingual/bert_crf_pt_to_en
```

### 3. Batch Evaluate All Models

```bash
# Evaluate all trained PT models on EN test set
./scripts/evaluate_cross_lingual.sh
```

## Usage

### Evaluate Specific Model

```bash
python evaluate.py \
  --model-path <path_to_trained_model> \
  --test-file <path_to_test_file> \
  --output-dir <output_directory> \
  [--model-type <model_type>] \
  [--device 0]
```

**Arguments:**
- `--model-path`: Path to trained model checkpoint (e.g., `results/bert_crf/run_XXX/best_model`)
- `--test-file`: JSONL test file (e.g., `data/publication_datasets/ner_datasets/en/splits/test.jsonl`)
- `--output-dir`: Where to save results
- `--model-type`: Model architecture (auto-detected if omitted)
- `--device`: CUDA device ID (default: 0, use -1 for CPU)

**Example:**
```bash
python evaluate.py \
  --model-path results/deberta_crf/pt_baseline/best_model \
  --test-file data/publication_datasets/ner_datasets/en/splits/test.jsonl \
  --output-dir results/cross_lingual/deberta_crf_pt_to_en \
  --device 0
```

### Batch Evaluation

The helper script automatically finds all trained models and evaluates them:

```bash
./scripts/evaluate_cross_lingual.sh
```

This will:
1. Find all `best_model` checkpoints in `results/`
2. Evaluate each on English test set
3. Save results to `results/cross_lingual/<model_name>_pt_to_en/`

## Output Structure

Results are saved in JSON format:

```
results/cross_lingual/
├── bert_crf_pt_to_en/
│   └── evaluation_results.json
├── deberta_crf_pt_to_en/
│   └── evaluation_results.json
└── xlmr_crf_pt_to_en/
    └── evaluation_results.json
```

**evaluation_results.json** contains:
```json
{
  "model_path": "results/bert_crf/pt_baseline/best_model",
  "model_type": "bert_crf",
  "base_model": "neuralmind/bert-base-portuguese-cased",
  "test_file": "data/publication_datasets/ner_datasets/en/splits/test.jsonl",
  "test_examples": 433,
  "evaluation_date": "2025-10-05T...",
  "metrics": {
    "entity_f1_strict": 0.4521,
    "precision": 0.4892,
    "recall": 0.4201,
    "accuracy": 0.8234,
    "per_type_metrics": {...}
  },
  "config": {
    "max_length": 512,
    "overlap_tokens": 50,
    "batch_size": 32
  }
}
```

## Viewing Results

### Quick Summary

```bash
# View F1 scores for all evaluations
for file in results/cross_lingual/*/evaluation_results.json; do
  echo "$(dirname $file): $(jq -r '.metrics.entity_f1_strict' $file)"
done
```

### Detailed Results

```bash
# View full metrics for specific model
cat results/cross_lingual/bert_crf_pt_to_en/evaluation_results.json | jq '.metrics'
```

### Compare with Portuguese Results

```bash
# Portuguese results (training)
cat results/bert_crf/pt_baseline/training_results.json | jq '.test_results.entity_f1_strict'

# English results (cross-lingual)
cat results/cross_lingual/bert_crf_pt_to_en/evaluation_results.json | jq '.metrics.entity_f1_strict'
```

## Expected Results

**Hypothesis:**
- **Multilingual models** (XLM-R) should show better cross-lingual transfer
- **Monolingual models** (BERTimbau) will show larger performance drop
- **CRF models** may be more robust to language differences

**Typical Performance Drop:**
- BERTimbau (PT-only): ~30-50% F1 drop PT→EN
- XLM-RoBERTa (multilingual): ~10-20% F1 drop PT→EN
- DeBERTa v3 (multilingual): ~15-25% F1 drop PT→EN

## Publication Notes

This experiment demonstrates:
1. **Zero-shot cross-lingual transfer** capability
2. **Language-independent vs language-specific** feature learning
3. **Model architecture impact** on cross-lingual generalization

Compare:
- **In-language performance** (PT model on PT test)
- **Cross-lingual performance** (PT model on EN test)
- **Performance gap** = language-dependence measure

## Data

**Portuguese Test Set:**
- Location: `data/publication_datasets/ner_datasets/pt/splits/test.jsonl`
- Used during training for in-language evaluation

**English Test Set:**
- Location: `data/publication_datasets/ner_datasets/en/splits/test.jsonl`
- Same domain and entity types as Portuguese
- Never seen during training (zero-shot)

## Troubleshooting

**Model path not found:**
- Ensure you've trained the model first
- Check the exact path in `results/<model_name>/<experiment_name>/best_model`

**CUDA out of memory:**
- Reduce `--batch-size` (default: 32)
- Use `--device -1` for CPU evaluation

**Different label sets:**
- The script automatically handles label mapping
- Entity types must match between train and test data

## Advanced Usage

### Evaluate on Custom Test Set

```bash
python evaluate.py \
  --model-path results/bert_crf/pt_baseline/best_model \
  --test-file path/to/custom_test.jsonl \
  --output-dir results/custom_evaluation
```

### Evaluate Multiple Checkpoints

```bash
# Evaluate different training epochs
for epoch in 1 2 3 4 5; do
  python evaluate.py \
    --model-path results/bert_crf/pt_baseline/checkpoint-epoch-${epoch} \
    --test-file data/publication_datasets/ner_datasets/en/splits/test.jsonl \
    --output-dir results/cross_lingual/bert_crf_epoch${epoch}_pt_to_en
done
```

## Integration with Other Experiments

This evaluation framework can be used for:
- **Cross-municipality evaluation:** Evaluate on different municipality test sets
- **Cross-domain evaluation:** Evaluate on different document types
- **Ablation studies:** Test different model configurations
- **Ensemble evaluation:** Combine predictions from multiple models

Simply point `--test-file` to your target test set!
