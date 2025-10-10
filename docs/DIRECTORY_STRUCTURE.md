# VotIE Directory Structure

## Overview

VotIE now follows a clean separation between models and experiments:

```
VotIE/
├── models/                          # Trained models (Hugging Face format)
│   ├── bert_crf/
│   │   └── experiment_name/         # Model files directly here (no best_model/ subdir)
│   │       ├── config.json
│   │       ├── model.safetensors
│   │       ├── tokenizer.json
│   │       └── training_results.json
│   ├── deberta_crf/
│   └── xlmr_crf/
│
├── experiments/                     # Experiment results and predictions
│   └── pt_stratified_split/         # Portuguese stratified split experiments  
│       └── experiment_name/
│           ├── predictions.jsonl    # Model predictions
│           ├── evaluation_results.json  # Evaluation metrics
│           └── pipeline_metadata.json   # Pipeline run metadata
│
└── data/                           # Original datasets
    ├── votie_bio/
    ├── votie_events/
    └── votie_spans/
```

## Key Changes

### 1. Model Storage
- **Before**: `results/{model}/{experiment}/best_model/`
- **After**: `models/{model}/{experiment}/`
- Models are saved directly in standard Hugging Face format
- No nested `best_model/` subdirectory

### 2. Experiment Results
- **Before**: Mixed with model files
- **After**: Separate `experiments/pt_stratified_split/{experiment}/`
- Contains predictions and evaluation results
- Clean separation from model artifacts

### 3. Pipeline Components

**Training** (`scripts/train.py`)
- Saves models to: `models/{baseline}/{experiment}/`
- Only performs training and validation
- No test evaluation (separated for better pipeline design)

**Prediction** (`scripts/predict.py`)  
- Loads models from: `models/{baseline}/{experiment}/`
- Saves predictions to: `experiments/pt_stratified_split/{experiment}/predictions.jsonl`

**Evaluation** (`scripts/evaluate.py`)
- Loads predictions from: `experiments/pt_stratified_split/{experiment}/predictions.jsonl`
- Saves results to: `experiments/pt_stratified_split/{experiment}/evaluation_results.json`

**Pipeline** (`scripts/run_pipeline.py`)
- Orchestrates: train → predict → evaluate
- Handles proper directory management
- Supports partial runs (--predict-only, --evaluate-only)

## Usage Examples

### Complete Pipeline
```bash
python scripts/run_pipeline.py --config configs/main_experiment/bert_crf.yaml --name my_experiment
```

### Partial Pipeline (predict + evaluate only)
```bash  
python scripts/run_pipeline.py --config configs/main_experiment/bert_crf.yaml --name my_experiment --predict-only
```

### Individual Components
```bash
# Train only
python scripts/train.py --config configs/main_experiment/bert_crf.yaml --experiment_name my_experiment

# Predict only  
python scripts/predict.py --model-path models/bert_crf/my_experiment --test-data data/votie_bio/test.jsonl --output experiments/pt_stratified_split/my_experiment/predictions.jsonl

# Evaluate only
python scripts/evaluate.py --predictions experiments/pt_stratified_split/my_experiment/predictions.jsonl --output experiments/pt_stratified_split/my_experiment/results.json
```

## Benefits

1. **Clean Separation**: Models vs. experiment results are clearly separated
2. **Hugging Face Standard**: Models follow standard HF directory structure  
3. **Reusability**: Models can be used for multiple experiments
4. **Organization**: Easy to find and manage experiment results
5. **Pipeline Flexibility**: Can run partial pipelines as needed