# LangExtract Event Extraction Evaluation Report

## Model Information
- **Model**: gemini-2.5-flash
- **Language**: Portuguese
- **Date**: 2025-10-14 09:19:40
- **Evaluation Method**: Event-based extraction

## Event-Level Performance

| Metric | Value |
|--------|-------|
| **Precision** | 0.2000 |
| **Recall** | 0.2500 |
| **F1-Score** | 0.2222 |
| **Correct** | 1 |
| **Partial** | 3 |
| **Missed** | 0 |
| **Spurious** | 1 |

## Span-Level Performance (for comparison)

| Metric | Value |
|--------|-------|
| **Precision** | 0.2000 |
| **Recall** | 0.3571 |
| **F1-Score** | 0.2564 |
| **Correct Spans** | 5 |
| **Predicted Spans** | 25 |
| **Gold Spans** | 14 |

## Interpretation

- **Event-level metrics** measure how well the model extracts complete voting events with all components
- **Span-level metrics** measure how well individual entities (subject, voters, etc.) are extracted
- Higher event-level scores indicate better structured extraction of complete voting information
