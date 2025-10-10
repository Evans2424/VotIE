# LangExtract NER Evaluation Report

## Model Information
- **Model**: gemini-2.5-flash
- **Language**: Portuguese
- **Date**: 2025-10-06 05:03:28
- **Evaluation Method**: seqeval_iob2, strict, exact, partial, type

## Extraction Performance
- **Total Extraction Time**: 5113.27s
- **Avg Time per Document**: 11.81s
- **Documents Processed**: 433

## Overall Performance

| Metric | Value |
|--------|-------|
| **Precision** | 0.5187 |
| **Recall** | 0.6408 |
| **F1-Score** | 0.5733 |
| **Predicted Entities** | 1944 |
| **Ground Truth Entities** | 1534 |

## Per-Entity Type Performance

| Entity Type | Precision | Recall | F1-Score |
|-------------|-----------|--------|----------|
| VOTER-FAVOR | 0.5833 | 0.7808 | 0.6678 |
| VOTER-AGAINST | 0.2051 | 0.2222 | 0.2133 |
| VOTER-ABSTENTION | 0.6329 | 0.6250 | 0.6289 |
| VOTER-ABSENT | 0.0714 | 0.0833 | 0.0769 |
| SUBJECT | 0.1543 | 0.2285 | 0.1842 |
| VOTING | 0.6768 | 0.8396 | 0.7494 |
| COUNTING-MAJORITY | 0.7727 | 0.7234 | 0.7473 |
| COUNTING-UNANIMITY | 0.8821 | 0.8750 | 0.8785 |

## Enhanced Metrics (Nervaluate)

*Note: Strict/Exact/Partial metrics may be identical when entity boundaries match exactly.*

| Scheme | Precision | Recall | F1-Score | Correct | Partial | Missed | Spurious |
|--------|-----------|--------|----------|---------|---------|--------|----------|
| Strict | 0.4990 | 0.6323 | 0.5578 | 970 | 0 | 256 | 666 |
| Exact | 0.4990 | 0.6323 | 0.5578 | 970 | 0 | 256 | 666 |
| Partial | 0.4990 | 0.6323 | 0.5578 | 970 | 308 | 256 | 666 |
| Type | 0.6538 | 0.8286 | 0.7309 | 1271 | 0 | 256 | 666 |
