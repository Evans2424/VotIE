# LangExtract Span-to-Event Extraction

This approach uses LangExtract to extract entity spans (VOTER-FAVOR, SUBJECT, etc.), then converts them to structured events for evaluation.

## Overview

**Pipeline:**
1. **Span Extraction** → Extract entity spans using LangExtract with Gemini
2. **Event Construction** → Convert spans to structured events using `src/utils/event_constructor.py`
3. **Event Evaluation** → Evaluate using event-level metrics from `src/evaluation/event_metrics.py`

## Setup

### 1. Install Dependencies

```bash
# Install requirements
pip install -r requirements.txt

# Install spacy Portuguese model
python -m spacy download pt_core_news_sm
```

### 2. Configure API Key

Set your Google API key:

```bash
export GOOGLE_API_KEY="your-api-key-here"
```

Or create a `.env` file:

```
GOOGLE_API_KEY=your-api-key-here
```

## Usage

### Run Span-to-Event Extraction

```bash
# Full evaluation on test set
python run_langextract_spans.py

# Test mode (10 documents only)
python run_langextract_spans.py --test

# Custom number of documents
python run_langextract_spans.py --max-docs 50
```

### Output

Results are saved to `experiments/langextract_spans_YYYYMMDD_HHMMSS/`:

- `predictions_spans.jsonl` - Entity spans extracted by LangExtract
- `predictions_events.jsonl` - Structured events constructed from spans
- `evaluation_results.json` - Complete evaluation metrics
- `evaluation_summary.md` - Human-readable summary

## Few-Shot Examples

The span extraction uses the same curated examples as the direct event extraction approach:

- **Source:** `langextract_ner/shared/selected_event_examples.json`
- **Conversion:** Event examples are automatically converted to span format
- **Entities:** VOTER-FAVOR, VOTER-AGAINST, VOTER-ABSTENTION, VOTER-ABSENT, SUBJECT, VOTING, COUNTING-MAJORITY, COUNTING-UNANIMITY

This ensures fair comparison between span-based and direct event extraction approaches.

## Comparison with Direct Event Extraction

| Approach | Script | Output Format | Evaluation |
|----------|--------|---------------|------------|
| **Direct Event** | `run_langextract_events.py` | JSON events | Event metrics |
| **Span-to-Event** | `run_langextract_spans.py` | Spans → Events | Event metrics |

Both approaches:
- Use the same few-shot examples (from `selected_event_examples.json`)
- Evaluate using event-level metrics (`src/evaluation/event_metrics.py`)
- Process the same test data (`data/votie_events/test.jsonl`)

## Event Metrics

The evaluation computes:

1. **Event Detection Accuracy** - Binary classification of voting event presence
2. **Subject Extraction Accuracy** - Exact string matching for subject
3. **Participant-Vote F1** - Set-based F1 for (participant, position) tuples
4. **Complete Event Accuracy** - Strict accuracy where all components must match

See `src/evaluation/event_metrics.py` for details.

## Troubleshooting

### ModuleNotFoundError: No module named 'spacy'

```bash
pip install spacy>=3.7.0
python -m spacy download pt_core_news_sm
```

### API Key Not Found

Make sure `GOOGLE_API_KEY` is set in your environment or `.env` file.

### Rate Limiting

The script includes a 1-second delay between requests to avoid rate limits. For large evaluations, consider running in batches.

## File Structure

```
langextract_ner/
├── run_langextract_spans.py          # Main script (span extraction)
├── run_langextract_events.py         # Direct event extraction (for comparison)
├── shared/
│   ├── prompts_spans.py              # Span extraction prompts
│   ├── prompts_events.py             # Event extraction prompts
│   └── selected_event_examples.json  # Curated examples (shared)
├── backup/
│   ├── models.py                     # GeminiModel for span extraction
│   ├── prompts.py                    # Old prompts (not used)
│   └── run_langextract.py           # Old script (not used)
└── experiments/                      # Output directory
```

## Notes

- The span-to-event approach allows comparison with traditional NER models that output BIO tags
- Event constructor handles conversion from BIO tags to structured events
- Both approaches can be run to compare direct vs. indirect event extraction
