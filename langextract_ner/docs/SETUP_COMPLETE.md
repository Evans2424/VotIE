# LangExtract Span Extraction Setup - Complete ✓

## What Has Been Set Up

The span labeling approach is now available and ready to run. It uses the **same few-shot examples** as the event extraction approach for fair comparison.

## New Files Created

1. **`run_langextract_spans.py`** - Main script for span-to-event extraction pipeline
   - Extracts entity spans using LangExtract
   - Converts spans to BIO tags
   - Uses `src/utils/event_constructor.py` to build structured events
   - Evaluates with `src/evaluation/event_metrics.py`

2. **`shared/prompts_spans.py`** - Span extraction prompts
   - Automatically loads examples from `shared/selected_event_examples.json`
   - Converts event format to span format (LangExtract Extraction objects)
   - Uses English prompts (matching the event extraction approach)
   - Includes fallback examples if file not found

3. **`README_SPAN_EXTRACTION.md`** - Complete usage documentation

## Pipeline Flow

```
Test Data (votie_events/test.jsonl)
    ↓
[1] LangExtract Span Extraction
    - Extract: VOTER-FAVOR, VOTER-AGAINST, SUBJECT, VOTING, etc.
    - Uses: backup/models.py (GeminiModel)
    - Prompts: shared/prompts_spans.py
    ↓
[2] Convert Spans to BIO Tags
    - Character offsets → Token-level tags
    - Uses spacy tokenization
    ↓
[3] Event Construction
    - BIO tags → Structured events
    - Uses: src/utils/event_constructor.py
    ↓
[4] Event Evaluation
    - Evaluate structured events
    - Uses: src/evaluation/event_metrics.py
    ↓
Results saved to experiments/langextract_spans_*/
```

## Key Features

✓ **Same Few-Shot Examples** - Uses `shared/selected_event_examples.json` (same as event extraction)
✓ **Same Evaluation** - Uses `src/evaluation/event_metrics.py` (event-level metrics)
✓ **Same Test Data** - Processes `data/votie_events/test.jsonl`
✓ **Fair Comparison** - Directly comparable with `run_langextract_events.py`

## How to Run

### Prerequisites

```bash
# Install dependencies
pip install -r langextract_ner/requirements.txt

# Install spacy Portuguese model
python -m spacy download pt_core_news_sm

# Set API key
export GOOGLE_API_KEY="your-key-here"
```

### Run Evaluation

```bash
# Navigate to project root
cd /Users/josepedroevans/Desktop/VotIE

# Test mode (10 documents)
python langextract_ner/run_langextract_spans.py --test

# Full evaluation
python langextract_ner/run_langextract_spans.py
```

## Output Files

Results saved to `langextract_ner/experiments/langextract_spans_YYYYMMDD_HHMMSS/`:

- **predictions_spans.jsonl** - Entity spans extracted by LangExtract
- **predictions_events.jsonl** - Structured events after construction
- **evaluation_results.json** - Complete metrics (JSON)
- **evaluation_summary.md** - Human-readable summary

## Comparison with Direct Event Extraction

| Aspect | Span Extraction | Event Extraction |
|--------|----------------|------------------|
| Script | `run_langextract_spans.py` | `run_langextract_events.py` |
| Output | Spans → Events | Events (direct) |
| Prompts | `shared/prompts_spans.py` | `shared/prompts_events.py` |
| Examples | From `selected_event_examples.json` | From `selected_event_examples.json` |
| Evaluation | Event metrics | Event metrics |
| Use Case | Compare with BIO-based models | Direct structured extraction |

## Files Used from Other Parts of the Project

- **`src/utils/event_constructor.py`** - Converts BIO tags to events
- **`src/evaluation/event_metrics.py`** - Event-level evaluation
- **`langextract_ner/backup/models.py`** - GeminiModel for LangExtract
- **`langextract_ner/shared/selected_event_examples.json`** - Few-shot examples

## What's in the Backup Directory

The `backup/` directory contains:
- `models.py` - ✓ **USED** by run_langextract_spans.py
- `prompts.py` - ✗ Not used (old prompts, replaced by shared/prompts_spans.py)
- `run_langextract.py` - ✗ Not used (old script, replaced by run_langextract_spans.py)
- `evaluation.py` - ✗ Not used (old evaluation, using src/evaluation/event_metrics.py instead)

## Next Steps

1. **Install dependencies** (if not already done)
2. **Set API key** in environment or `.env` file
3. **Run test mode** to verify setup: `python langextract_ner/run_langextract_spans.py --test`
4. **Compare results** with direct event extraction

## Troubleshooting

### ImportError: No module named 'spacy'
```bash
pip install spacy>=3.7.0
python -m spacy download pt_core_news_sm
```

### ImportError: No module named 'langextract'
```bash
pip install langextract>=0.2.0
```

### API Key Not Found
```bash
export GOOGLE_API_KEY="your-api-key-here"
```

Or create `.env` file:
```
GOOGLE_API_KEY=your-api-key-here
```

## Architecture Notes

The span extraction approach is designed to:
1. **Be compatible with traditional NER** - Outputs BIO tags like BERT-based models
2. **Enable fair comparison** - Same examples and evaluation as event extraction
3. **Use existing infrastructure** - Leverages src/utils/event_constructor.py
4. **Provide detailed outputs** - Saves both spans and constructed events

This allows you to compare:
- **LangExtract (spans)** vs **LangExtract (events)** vs **BERT/DeBERTa (spans)**

All using the same event-level evaluation metrics!
