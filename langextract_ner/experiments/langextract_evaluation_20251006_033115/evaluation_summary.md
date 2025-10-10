# LangExtract NER Evaluation Report

## Model Information
- **Model**: gemini-2.5-flash
- **Language**: Portuguese
- **Date**: 2025-10-06 03:34:11
- **Evaluation Method**: seqeval_iob2, strict, exact, partial, type

## Extraction Performance
- **Total Extraction Time**: 174.97s
- **Avg Time per Document**: 17.50s
- **Documents Processed**: 10

## Overall Performance

| Metric | Value |
|--------|-------|
| **Precision** | 0.6400 |
| **Recall** | 0.6957 |
| **F1-Score** | 0.6667 |
| **Predicted Entities** | 55 |
| **Ground Truth Entities** | 46 |

## Per-Entity Type Performance

| Entity Type | Precision | Recall | F1-Score |
|-------------|-----------|--------|----------|
| VOTER-FAVOR | 0.9000 | 0.9000 | 0.9000 |
| VOTER-ABSTENTION | 0.0000 | 0.0000 | 0.0000 |
| SUBJECT | 0.0667 | 0.1000 | 0.0800 |
| VOTING | 0.9167 | 0.9167 | 0.9167 |
| COUNTING-MAJORITY | 1.0000 | 1.0000 | 1.0000 |
| COUNTING-UNANIMITY | 0.9091 | 0.9091 | 0.9091 |

## Enhanced Metrics (Nervaluate)

*Note: Strict/Exact/Partial metrics may be identical when entity boundaries match exactly.*

| Scheme | Precision | Recall | F1-Score | Correct | Partial | Missed | Spurious |
|--------|-----------|--------|----------|---------|---------|--------|----------|
| Strict | 0.4909 | 0.5870 | 0.5347 | 27 | 0 | 6 | 15 |
| Exact | 0.4909 | 0.5870 | 0.5347 | 27 | 0 | 6 | 15 |
| Partial | 0.4909 | 0.5870 | 0.5347 | 27 | 13 | 6 | 15 |
| Type | 0.7273 | 0.8696 | 0.7921 | 40 | 0 | 6 | 15 |
