# LangExtract NER - Portuguese Voting Event Extraction

Gemini 2.0 Flash implementation for extracting voting events from Portuguese municipal council meeting minutes.

## Overview

This module provides two approaches for voting event extraction:

1. **Direct Event Extraction** (`run_langextract_events.py`): Uses Gemini with JSON mode to extract structured events directly
2. **Span-Based Extraction** (`run_langextract_spans.py`): Uses LangExtract library to extract entity spans, then constructs events from BIO tags

Both approaches evaluate using event-level metrics to measure extraction quality.

## Event Structure (Hybrid Format)

The system extracts voting events in a hybrid format:
- `subject`: **string** (singular) - What is being voted on
- `participants`: **list** - Who voted and their positions (Favor/Against/Abstention/Absent)
- `counting`: **list** - Vote counting expressions (Majority/Unanimity)
- `voting_expressions`: **list** - Phrases indicating a vote occurred

```json
{
  "id": "document_seg001",
  "has_voting_event": true,
  "event": {
    "subject": "ata n.º 28",
    "participants": [
      {"text": "o Executivo Municipal", "position": "Favor"},
      {"text": "eleita pelo Nós, Cidadãos", "position": "Abstention"}
    ],
    "counting": [
      {"text": "por maioria", "type": "Majority"}
    ],
    "voting_expressions": ["deliberou", "aprovado"]
  }
}
```

## Installation

### Requirements

```bash
# Core dependencies
pip install google-generativeai  # For direct event extraction
pip install langextract           # For span-based extraction
pip install spacy                 # For tokenization

# Optional (for evaluation)
pip install nervaluate            # For relaxed entity matching metrics
```

### API Key Setup

Set your Google API key:

```bash
export GOOGLE_API_KEY="your-api-key-here"
# OR
export LANGEXTRACT_API_KEY="your-api-key-here"
```

Or create a `.env` file:

```
GOOGLE_API_KEY=your-api-key-here
```

## Usage

### 1. Direct Event Extraction

Extracts complete voting events using Gemini JSON mode:

```bash
# Test mode (10 documents)
python langextract_ner/run_langextract_events.py --test

# Full evaluation (all test documents)
python langextract_ner/run_langextract_events.py

# Custom number of documents
python langextract_ner/run_langextract_events.py --max-docs 50
```

**Output:**
- `predictions_events.jsonl`: Predicted events
- `evaluation_results_events.json`: Metrics (precision, recall, F1)
- `evaluation_summary_events.md`: Human-readable report

**Advantages:**
- Single API call per document
- Structured JSON output
- Handles all event components together
- Lower API costs (fewer requests)

### 2. Span-Based Extraction

Extracts entity spans using LangExtract, then constructs events:

```bash
# Test mode (10 documents)
python langextract_ner/run_langextract_spans.py --test

# Full evaluation
python langextract_ner/run_langextract_spans.py

# Custom number of documents
python langextract_ner/run_langextract_spans.py --max-docs 50
```

**Output:**
- `predictions_spans.jsonl`: Extracted entity spans
- `predictions_events.jsonl`: Constructed events from spans
- `evaluation_results.json`: Event-level metrics
- `evaluation_summary.md`: Human-readable report

**Advantages:**
- Fine-grained entity extraction
- Compatible with traditional NER evaluation
- Can extract spans for other downstream tasks
- Uses proven LangExtract library

## Module Structure

```
langextract_ner/
├── README.md                      # This file
├── __init__.py                    # Module exports
├── config.py                      # Configuration management
├── data_loader.py                 # Dataset loading utilities
│
├── run_langextract_events.py     # Direct event extraction script
├── run_langextract_spans.py      # Span-based extraction script
├── evaluation_events.py           # Event-level evaluation metrics
│
└── shared/                        # Shared components
    ├── models_events.py           # GeminiEventExtractor (JSON mode)
    ├── models_spans.py            # GeminiModel (LangExtract)
    ├── prompts_events.py          # Event extraction prompts
    ├── prompts_spans.py           # Span extraction prompts
    └── selected_event_examples.json  # Few-shot examples (hybrid format)
```

## Configuration

Edit `config.py` or set environment variables:

```python
# Model configuration
MODEL_ID = "gemini-2.0-flash-exp"
API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("LANGEXTRACT_API_KEY")

# Data paths
DATA_DIR = "../data/votie_events"
TEST_FILE = "test.json"

# Extraction parameters
MAX_RETRIES = 3           # Retry on quota errors
RETRY_DELAY = 60          # Seconds between retries
RATE_LIMIT_DELAY = 1      # Seconds between requests
```

## Evaluation Metrics

### Event-Level Metrics

- **Precision**: Correctness of extracted events
- **Recall**: Coverage of all voting events
- **F1-Score**: Harmonic mean of precision and recall
- **Correct**: Events with all components matching
- **Partial**: Events with some components matching
- **Missed**: Voting events not detected
- **Spurious**: False positive detections

### Component-Level Metrics

Event comparison checks individual components:
- Subject extraction accuracy
- Participant-vote pair matching
- Counting type correctness
- Voting expression matching

### Span-Level Metrics

For span-based extraction only:
- Per-entity-type precision, recall, F1
- Boundary matching (exact vs partial)
- Type matching accuracy

## Few-Shot Learning

The system uses 10 curated examples from `shared/selected_event_examples.json`. Examples cover:
- Different municipalities (M01-M06)
- All entity types (SUBJECT, VOTER-*, COUNTING-*, VOTING)
- Various voting scenarios (unanimity, majority, abstentions)
- Different document styles

Examples are automatically loaded and included in prompts.

## API Costs

Approximate costs with Gemini 2.0 Flash (as of Jan 2025):

| Approach | Requests per doc | Tokens per doc (avg) | Cost per 1000 docs |
|----------|------------------|----------------------|--------------------|
| Direct Event | 1 | ~1500 | ~$0.15 |
| Span-Based | 1 | ~1500 | ~$0.15 |

*Note: Costs may vary based on document length and API pricing.*

## Performance

### Direct Event Extraction

Typical performance on Portuguese test set (433 documents):
- Event Detection: ~85-90% accuracy
- Subject Extraction: ~75-80% F1
- Participant Extraction: ~70-75% F1
- Complete Event: ~60-65% accuracy

### Span-Based Extraction

Typical performance on Portuguese test set:
- Event Detection: ~80-85% accuracy
- Subject Extraction: ~70-75% F1
- Participant Extraction: ~65-70% F1
- Complete Event: ~55-60% accuracy

*Performance varies based on model version and prompt engineering.*

## Troubleshooting

### API Key Not Found

```
Error: No API key found
```

**Solution:** Set `GOOGLE_API_KEY` or `LANGEXTRACT_API_KEY` environment variable.

### Quota Exceeded

```
Error: Quota limit exceeded
```

**Solution:** The system automatically retries with exponential backoff. Adjust `MAX_RETRIES` and `RETRY_DELAY` in config.

### Import Errors

```
ImportError: cannot import name 'GeminiEventExtractor'
```

**Solution:** Ensure you're running from the project root directory:
```bash
cd /path/to/VotIE
python langextract_ner/run_langextract_events.py
```

### JSON Parsing Errors

```
Error: Failed to parse JSON response
```

**Solution:** This usually indicates model hallucination. The system logs the raw response for debugging. Consider:
- Simplifying the prompt
- Reducing example count
- Using higher temperature (though this reduces consistency)

## Development

### Adding New Examples

To add new few-shot examples:

1. Select examples from `data/votie_events/train.json`
2. Ensure examples cover diverse cases
3. Update `shared/selected_event_examples.json`
4. Run conversion to hybrid format:
   ```bash
   python convert_examples_to_hybrid_format.py
   ```

### Modifying Event Structure

If you change the event structure:

1. Update `src/utils/event_constructor.py` (BIO → Event conversion)
2. Update `shared/models_events.py` (prompt format)
3. Update `shared/prompts_events.py` (system prompt)
4. Update `evaluation_events.py` (comparison logic)
5. Re-convert examples to new format

### Testing

Run comprehensive tests without API calls:

```bash
python test_langextract_pipeline.py
```

This verifies:
- EventConstructor outputs correct format
- Evaluation functions handle hybrid format
- All imports work correctly
- Data loading works

## Entity Types

The system extracts 8 entity types from Portuguese municipal voting documents:

- **SUBJECT**: Subject/topic being voted on
- **VOTER-FAVOR**: Person/entity voting in favor
- **VOTER-AGAINST**: Person/entity voting against
- **VOTER-ABSTENTION**: Person/entity abstaining
- **VOTER-ABSENT**: Person/entity absent
- **VOTING**: Voting action (deliberou, aprovou, etc.)
- **COUNTING-MAJORITY**: Vote counting by majority
- **COUNTING-UNANIMITY**: Vote counting by unanimity

## License

Part of the VotIE project for Portuguese municipal document analysis.
