# LangExtract NER - Portuguese Voting Entity Extraction

Publication-ready implementation of voting entity extraction for Portuguese municipal documents using Google's Gemini 2.0 Flash model via LangExtract.

## Quick Start

### 1. Install Dependencies

```bash
# Activate virtual environment (if available)
source ../NER_BASELINES_PUBLICATION/venv/bin/activate

# Install required packages
pip install langextract seqeval nervaluate spacy python-dotenv

# Install Portuguese spaCy model
python -m spacy download pt_core_news_sm
```

### 2. Set API Key

```bash
export LANGEXTRACT_API_KEY="your-google-api-key"
# OR
export GOOGLE_API_KEY="your-google-api-key"
```

### 3. Run Evaluation

**Test Mode (10 documents):**
```bash
python run_langextract.py --test
```

**Custom number of documents:**
```bash
python run_langextract.py --max-docs 50
```

**Full evaluation (all 433 documents):**
```bash
python run_langextract.py
```

This will:
- Load the test dataset from `data/publication_datasets/ner_datasets/pt/splits/test.jsonl`
- Extract entities using Gemini 2.0 Flash
- Evaluate using seqeval (same as NER baselines)
- Save results in `experiments/langextract_evaluation_<timestamp>/`

## Entity Types

The system extracts 8 entity types from Portuguese municipal voting documents:

- **VOTER-FAVOR**: Person/entity voting in favor
- **VOTER-AGAINST**: Person/entity voting against
- **VOTER-ABSTENTION**: Person/entity abstaining
- **VOTER-ABSENT**: Person/entity absent
- **SUBJECT**: Subject/topic being voted on
- **VOTING**: Voting action (deliberou, aprovou, etc.)
- **COUNTING-MAJORITY**: Vote counting by majority
- **COUNTING-UNANIMITY**: Vote counting by unanimity

## Evaluation

The evaluation system uses:
- **spaCy tokenization** (matching the dataset converter)
- **seqeval** with IOB2 scheme for F1 calculation
- **nervaluate** for enhanced metrics (strict, exact, partial, type)

Results are compatible with NER baseline evaluations (NER_BASELINES_PUBLICATION/core/evaluation/metrics.py).

## Output Format

Results are saved in two formats:

1. **evaluation_report.json**: Complete metrics in JSON format
   - Overall metrics (precision, recall, F1)
   - Per-entity type metrics
   - Nervaluate enhanced metrics (if available)
   - Error analysis

2. **evaluation_summary.md**: Human-readable markdown summary

## Project Structure

```
langextract_ner/
├── run_langextract.py          # Main entry point
├── config.py                   # Configuration (Gemini 2.0, Portuguese)
├── data_loader.py              # Data loading utilities
├── evaluation.py               # Evaluation system (seqeval + nervaluate)
├── results_reporter.py         # Results reporting utilities
├── shared/
│   ├── models.py              # GeminiModel implementation
│   └── prompts.py             # Portuguese voting prompts
└── experiments/               # Evaluation results output
```

## Configuration

Default configuration (config.py):
- **Model**: gemini-2.0-flash-exp
- **Language**: Portuguese (pt)
- **Domain**: Municipal documents
- **Entity types**: 8 voting-related entities

## Notes

- Uses Portuguese-only extraction for publication
- Evaluation on test set: `data/publication_datasets/ner_datasets/pt/splits/test.jsonl`
- Comparable with NER baselines (BERT, XLM-R, DeBERTa models)
- Simple, standalone implementation - no unnecessary dependencies

## License

Part of the Citilink NLP project for Portuguese municipal document analysis.
