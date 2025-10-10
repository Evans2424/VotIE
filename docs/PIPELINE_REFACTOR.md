# VotIE Training Pipeline Refactor - Implementation Plan

## Overview
Separate training, prediction, and evaluation into independent, reusable components following best practices for ML pipelines.

## Current Problems
1. `trainer.py` trains AND evaluates on test set
2. `train.py` also runs test evaluation
3. `evaluate.py` loads model and generates predictions (mixing concerns)
4. Can't reuse predictions without retraining
5. Hard to debug predictions independently

## New Architecture

```
Training Pipeline:
┌──────────┐      ┌────────────┐      ┌─────────────┐
│  train   │─────>│  predict   │─────>│  evaluate   │
│          │      │            │      │             │
│ Model    │      │ Predictions│      │  Metrics    │
│ Checkpoint      │ .jsonl     │      │  .json      │
└──────────┘      └────────────┘      └─────────────┘
```

### 1. Training (scripts/train.py + src/trainer.py)
**Responsibility**: Train model, validate for early stopping, save checkpoints
- Input: Config file
- Output: Model checkpoint + training_results.json (NO test metrics)
- Training results contain: best_dev_f1, epochs, training_time, hyperparameters

### 2. Prediction (scripts/predict.py)
**Responsibility**: Load model and generate predictions
- Input: Model checkpoint + test data
- Output: predictions.jsonl
- Format: `{id, tokens, pred_labels, gold_labels}`
- Handles windowing automatically

### 3. Evaluation (scripts/evaluate.py)
**Responsibility**: Load predictions and compute metrics
- Input: predictions.jsonl + gold data (optional, can be in predictions)
- Output: evaluation_results.json
- Computes: token-level + event-level metrics
- Pure evaluation logic, no model loading

### 4. Orchestration (scripts/run_pipeline.py)
**Responsibility**: Chain all steps together
- Run: train → predict → evaluate
- Support resuming from any step
- Clean CLI interface

## Implementation Steps

### Step 1: Clean predict.py ✓
- [x] Simple `predict(model_path, test_data_path, output_path)` function
- [x] Robust windowing support
- [x] Save predictions.jsonl

### Step 2: Refactor evaluate.py
- [ ] Remove model loading
- [ ] Load predictions from JSONL
- [ ] Keep all evaluation logic
- [ ] Save evaluation_results.json

### Step 3: Modify trainer.py
- [ ] Remove test evaluation from `train()` method
- [ ] Keep only training + validation
- [ ] Remove test_results from training_results.json

### Step 4: Update train.py
- [ ] Remove test evaluation call
- [ ] Only save training results

### Step 5: Create run_pipeline.py
- [ ] Orchestrate train → predict → evaluate
- [ ] Support --skip-train, --skip-predict flags
- [ ] Clean output and logging

### Step 6: Documentation
- [ ] Update README with new workflow
- [ ] Add usage examples
- [ ] Document file formats
- [ ] Add troubleshooting guide

## File Formats

### predictions.jsonl
```json
{
  "id": "example_001",
  "tokens": ["O", "prefeito", "votou", "..."],
  "pred_labels": ["O", "B-VOTE", "I-VOTE", "..."],
  "gold_labels": ["O", "B-VOTE", "I-VOTE", "..."]
}
```

### evaluation_results.json
```json
{
  "metadata": {
    "predictions_file": "predictions.jsonl",
    "timestamp": "2025-10-10T12:00:00"
  },
  "token_level": {
    "precision": 0.95,
    "recall": 0.93,
    "f1": 0.94
  },
  "event_level": {
    "event_detection_accuracy": 0.92,
    "complete_event_accuracy": 0.88
  }
}
```

## Benefits
1. **Reusability**: Evaluate same predictions multiple times
2. **Debugging**: Inspect predictions.jsonl for errors
3. **Efficiency**: Don't reload model for evaluation
4. **Flexibility**: Train once, evaluate on multiple test sets
5. **Clarity**: Each component has single responsibility
6. **Testing**: Can test evaluation with mock predictions

## Usage Examples

```bash
# Full pipeline
python scripts/run_pipeline.py --config configs/bert_crf.yaml --test-data data/votie_bio/test.jsonl

# Individual steps
python scripts/train.py --config configs/bert_crf.yaml
python scripts/predict.py results/bert_crf/run_XXX/best_model data/votie_bio/test.jsonl predictions.jsonl cuda
python scripts/evaluate.py --predictions predictions.jsonl --output eval_results.json

# Resume from prediction (training already done)
python scripts/run_pipeline.py --config configs/bert_crf.yaml --test-data data/votie_bio/test.jsonl --skip-train --model-path results/bert_crf/run_XXX/best_model
```
