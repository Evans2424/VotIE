# LangExtract Event Extraction Evaluation Report

## Model Information
- **Model**: gemini-2.5-flash
- **Language**: Portuguese
- **Date**: 2025-10-15 03:51:01
- **Evaluation Method**: Event-based extraction

## Event-Level Performance

| Metric | Value |
|--------|-------|
| **Precision** | 0.5183 |
| **Recall** | 0.5308 |
| **F1-Score** | 0.5245 |
| **Correct** | 198 |
| **Partial** | 214 |
| **Missed** | 1 |
| **Spurious** | 10 |

## Span-Level Performance (for comparison)

| Metric | Value |
|--------|-------|
| **Precision** | 0.6452 |
| **Recall** | 0.7259 |
| **F1-Score** | 0.6832 |
| **Correct Spans** | 1091 |
| **Predicted Spans** | 1691 |
| **Gold Spans** | 1503 |

## Interpretation

- **Event-level metrics** measure how well the model extracts complete voting events with all components
- **Span-level metrics** measure how well individual entities (subject, voters, etc.) are extracted
- Higher event-level scores indicate better structured extraction of complete voting information
