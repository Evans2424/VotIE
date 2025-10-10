# VotIE LOMO: Leave-One-Municipality-Out Splits

## Overview

This directory contains **Leave-One-Municipality-Out (LOMO)** splits for evaluating cross-domain generalization in the VotIE dataset. LOMO splits are designed to test how well models trained on some municipalities can generalize to unseen municipalities.

## What is LOMO?

Leave-One-Municipality-Out is a cross-validation strategy where:
- **Training**: Model is trained on 5 municipalities
- **Development**: Dev set comes from the same 5 municipalities (stratified 15% split)
- **Testing**: Model is tested on 1 held-out municipality

This setup evaluates:
- **Domain Shift**: Geographic and administrative variance between municipalities
- **Generalization**: Model robustness to new data sources
- **Transfer Learning**: How well models transfer across similar but distinct domains

## Municipalities

VotIE includes 6 Portuguese municipalities:

| ID   | Name        | Region      | Segments | Entities | Avg Tokens |
|------|-------------|-------------|----------|----------|------------|
| M01  | Alandroal   | Alentejo    | 396      | 1,421    | 398.3      |
| M02  | Campo Maior | Alentejo    | 554      | 1,702    | 372.1      |
| M03  | Covilhã     | Centro      | 472      | 1,720    | 358.5      |
| M04  | Fundão      | Centro      | 235      | 985      | 787.3      |
| M05  | Guimarães   | Norte       | 503      | 1,713    | 95.6       |
| M06  | Porto       | Norte       | 719      | 2,671    | 313.4      |

## Split Structure

Each municipality directory contains all three formats:

```
M01_alandroal/
├── events/
│   ├── train.jsonl    # M02-M06 combined (85%)
│   ├── dev.jsonl      # M02-M06 combined (15%, stratified)
│   └── test.jsonl     # M01 only (100%)
├── spans/
│   ├── train.jsonl
│   ├── dev.jsonl
│   └── test.jsonl
├── conll/
│   ├── train.conll
│   ├── dev.conll
│   └── test.conll
└── statistics.json
```

## Expected Results

### Performance Patterns

LOMO evaluation typically shows:

1. **Lower F1 than Main Splits**: 5-15% drop expected due to domain shift
   - Main splits: F1 ~0.85-0.90
   - LOMO: F1 ~0.70-0.85

2. **Variance Across Municipalities**: Some municipalities are harder to generalize to
   - Document structure variations
   - Linguistic patterns

3. **Entity Type Performance**:
   - Common entities (SUBJECT, VOTING) generalize better
   - Rare entities (VOTER-ABSENT, VOTER-AGAINST) more difficult


## Statistics

### Train/Dev/Test Sizes per LOMO Split

| Target Muni | Train Examples | Dev Examples | Test Examples |
|-------------|----------------|--------------|---------------|
| M01 (test)  | ~2,108         | ~375         | 396           |
| M02 (test)  | ~1,950         | ~375         | 554           |
| M03 (test)  | ~2,032         | ~375         | 472           |
| M04 (test)  | ~2,269         | ~375         | 235           |
| M05 (test)  | ~2,001         | ~375         | 503           |
| M06 (test)  | ~1,785         | ~375         | 719           |

*Note: Train/Dev sizes vary because test municipality is excluded*

### Entity Distribution Across Splits

Each LOMO split maintains similar entity distributions in train/dev sets, but test set distribution reflects the specific municipality's characteristics.
