# LangExtract Event Extraction Evaluation Report

## Model Information
- **Model**: gemini-2.5-flash
- **Language**: Portuguese
- **Date**: 2025-10-14 00:15:41
- **Evaluation Method**: Event-based extraction

## Event-Level Performance

| Metric | Value |
|--------|-------|
| **Precision** | 0.2857 |
| **Recall** | 0.2500 |
| **F1-Score** | 0.2667 |
| **Correct** | 2 |
| **Partial** | 4 |
| **Missed** | 1 |
| **Spurious** | 0 |

## Span-Level Performance (for comparison)

| Metric | Value |
|--------|-------|
| **Precision** | 0.3333 |
| **Recall** | 0.2667 |
| **F1-Score** | 0.2963 |
| **Correct Spans** | 8 |
| **Predicted Spans** | 24 |
| **Gold Spans** | 30 |

## Interpretation

- **Event-level metrics** measure how well the model extracts complete voting events with all components
- **Span-level metrics** measure how well individual entities (subject, voters, etc.) are extracted
- Higher event-level scores indicate better structured extraction of complete voting information
