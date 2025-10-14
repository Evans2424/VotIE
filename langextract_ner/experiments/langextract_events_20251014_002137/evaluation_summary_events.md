# LangExtract Event Extraction Evaluation Report

## Model Information
- **Model**: gemini-2.5-flash
- **Language**: Portuguese
- **Date**: 2025-10-14 00:27:56
- **Evaluation Method**: Event-based extraction

## Event-Level Performance

| Metric | Value |
|--------|-------|
| **Precision** | 0.2750 |
| **Recall** | 0.2683 |
| **F1-Score** | 0.2716 |
| **Correct** | 11 |
| **Partial** | 29 |
| **Missed** | 1 |
| **Spurious** | 0 |

## Span-Level Performance (for comparison)

| Metric | Value |
|--------|-------|
| **Precision** | 0.4000 |
| **Recall** | 0.3832 |
| **F1-Score** | 0.3914 |
| **Correct Spans** | 64 |
| **Predicted Spans** | 160 |
| **Gold Spans** | 167 |

## Interpretation

- **Event-level metrics** measure how well the model extracts complete voting events with all components
- **Span-level metrics** measure how well individual entities (subject, voters, etc.) are extracted
- Higher event-level scores indicate better structured extraction of complete voting information
