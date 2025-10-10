# Cross-Municipality Experiments

This directory contains configurations for **leave-one-municipality-out** experiments to evaluate model generalization across different Portuguese municipalities.

## Experimental Design

**Goal:** Test model generalization to unseen municipalities

**Approach:** Leave-one-out cross-validation
- **Train:** 5 municipalities
- **Dev:** Same 5 municipalities
- **Test:** 1 held-out municipality (ALL its data: train+dev+test splits)

## Municipalities

| Code | Municipality | Train Examples | Dev Examples | Test Examples (All Data) |
|------|-------------|----------------|--------------|--------------------------|
| M01  | Campomaior  | 1738           | 360          | 396                      |
| M02  | Guimarães   | 1628           | 345          | 554                      |
| M03  | Porto       | 1682           | 371          | 472                      |
| M04  | Fundão      | 1852           | 394          | 235                      |
| M05  | Alandroal   | 1663           | 356          | 503                      |
| M06  | Covilhã     | 1512           | 329          | 719                      |

*Note: Train/Dev counts are from the other 5 municipalities. Test uses ALL data from the held-out municipality.*

## Available Models

- `bert_crf/` - BERT + CRF configurations
- `deberta_crf/` - DeBERTa + CRF configurations

Each model directory contains 6 configs (one per held-out municipality):
- `test_campomaior.yaml`
- `test_guimaraes.yaml`
- `test_porto.yaml`
- `test_fundao.yaml`
- `test_alandroal.yaml`
- `test_covilha.yaml`

## Running Experiments

### Single Experiment

```bash
# Test generalization to Campomaior
python train.py bert_crf \
  --config configs/cross_municipality/bert_crf/test_campomaior.yaml \
  --experiment-name cross_muni_campomaior

# Test generalization to Alandroal
python train.py deberta_crf \
  --config configs/cross_municipality/deberta_crf/test_alandroal.yaml \
  --experiment-name cross_muni_alandroal
```

### Run All Municipalities (BERT-CRF)

```bash
for muni in campomaior guimaraes porto fundao alandroal covilha; do
  python train.py bert_crf \
    --config configs/cross_municipality/bert_crf/test_${muni}.yaml \
    --experiment-name cross_muni_${muni}
done
```

### Run All Municipalities (DeBERTa-CRF)

```bash
for muni in campomaior guimaraes porto fundao alandroal covilha; do
  python train.py deberta_crf \
    --config configs/cross_municipality/deberta_crf/test_${muni}.yaml \
    --experiment-name cross_muni_${muni}
done
```

## Data Location

Pre-merged cross-municipality splits are in:
```
data/publication_datasets/ner_datasets/pt/cross_municipality/
├── test_campomaior/
│   ├── train.jsonl  (5 municipalities combined)
│   ├── dev.jsonl    (5 municipalities combined)
│   └── test.jsonl   (ALL Campomaior data)
├── test_guimaraes/
│   └── ...
└── (etc.)
```

## Results Interpretation

- **High test F1** → Model generalizes well to unseen municipality
- **Low test F1** → Municipality has different patterns/vocabulary
- **Compare across municipalities** → Identify which are easiest/hardest to generalize to

## Expected Outputs

Results saved in:
```
results/
├── bert_crf/
│   ├── cross_muni_campomaior/
│   │   ├── training_results.json
│   │   ├── best_model/
│   │   └── ...
│   └── cross_muni_guimaraes/
│       └── ...
└── deberta_crf/
    └── ...
```

## Regenerating Data/Configs

If you need to regenerate the splits or configs:

```bash
# Regenerate data splits
cd data/publication_datasets/ner_datasets/pt
python create_cross_municipality_splits.py

# Regenerate configs
cd NER_BASELINES_PUBLICATION/baselines/transformers/configs/cross_municipality
python generate_configs.py
```

## Publication Notes

These experiments test:
1. **Domain adaptation:** How well models transfer across municipalities
2. **Data diversity:** Which municipalities are more/less representative
3. **Model robustness:** Which architectures generalize best

Compare to standard in-municipality splits to quantify generalization gap.
