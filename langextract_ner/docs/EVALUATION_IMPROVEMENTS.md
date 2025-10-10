# Evaluation System Improvements

**Date:** October 3, 2025  
**Status:** Implemented

## Overview

This document describes the improvements made to the LangExtract evaluation system to ensure robust, accurate, and comparable evaluation of NER systems against span-based datasets.

## Problems Addressed

### 1. Tokenization Mismatch
**Problem:** The original evaluator used naive whitespace tokenization (`text.split()`), which differs from the spaCy tokenization used to create the datasets in `convert_publication_dataset_to_jsonl.py`.

**Impact:** Boundary misalignment between ground truth and predictions, leading to inflated or deflated scores when converting to BIO tags for seqeval evaluation.

**Solution:** Replaced whitespace tokenization with spaCy tokenization matching the converter:
- Portuguese: `pt_core_news_sm`
- English: `en_core_web_sm`
- Fallback to whitespace if spaCy unavailable

### 2. Non-Unique Entity Matching
**Problem:** The original `calculate_entity_matches` allowed multiple predicted entities to match the same ground truth entity, violating the one-to-one constraint expected in NER evaluation.

**Impact:** 
- Inflated match counts
- Spurious predictions not properly counted as false positives
- Incorrect precision/recall calculations

**Solution:** Implemented one-to-one matching with greedy assignment:
- Each ground truth entity can be matched at most once
- Predictions are greedily assigned to the best available GT entity
- Once a GT is matched, it's removed from the candidate pool

### 3. IOB2 Conversion Correctness
**Problem:** BIO conversion didn't explicitly follow IOB2 scheme requirements and lacked proper token-entity alignment verification.

**Impact:** Risk of invalid BIO sequences and incorrect seqeval metrics.

**Solution:**
- Explicit IOB2 scheme implementation (B- for first token, I- for continuation)
- Token-entity alignment using character spans from spaCy
- Language-aware tokenization parameter

## Implementation Details

### New Functions

#### `load_spacy_model(language: str)`
Loads the appropriate spaCy model for the given language, with caching to avoid repeated loads.

```python
model_name = "pt_core_news_sm" if language == "pt" else "en_core_web_sm"
```

#### `tokenize_text(text: str, language: str) -> Tuple[List[str], List[Tuple[int, int]]]`
Tokenizes text using spaCy and returns both tokens and character spans:
- Uses spaCy's linguistic tokenization (handles punctuation, contractions, etc.)
- Returns character positions for precise span alignment
- Fallback to whitespace tokenization if spaCy unavailable

### Updated Functions

#### `convert_to_bio_tags(text, entities, language)`
**Key changes:**
- Added `language` parameter
- Uses `tokenize_text()` for consistent tokenization
- Explicit IOB2 scheme: B-LABEL for first token, I-LABEL for continuation
- Character-span-based token-entity overlap detection

**Algorithm:**
1. Tokenize text using spaCy
2. Initialize all tags as "O"
3. For each entity:
   - Find all tokens that overlap the entity span
   - Assign B-LABEL to first overlapping token
   - Assign I-LABEL to remaining overlapping tokens

#### `calculate_entity_matches(predicted_entities, ground_truth_entities)`
**Key changes:**
- Added `matched_gt_indices: Set[int]` to track matched GT entities
- Skip already-matched GT entities when finding best match
- Mark GT entity as matched after assignment (one-to-one enforcement)

**Algorithm:**
1. For each predicted entity:
   - Find best unmatched GT entity by maximum overlap
   - Determine match type (exact/partial/label_only/no_match)
   - Record match and mark GT as used
2. Each GT can be matched at most once

#### `evaluate_extraction_results(predictions, language, municipalities)`
**Key changes:**
- Pass `language` parameter to `convert_to_bio_tags()`
- Ensures consistent tokenization across all documents

## Evaluation Schemes

The system now supports multiple evaluation schemes with consistent tokenization:

### 1. Seqeval (IOB2)
- Token-level sequence labeling metrics
- Uses spaCy tokenization for proper alignment
- Micro-averaged precision/recall/F1
- Per-entity-type metrics

### 2. Span-Based (Custom)
- Character-span matching (exact/partial/label-only)
- One-to-one assignment
- Direct comparison without tokenization artifacts

### 3. Nervaluate (Optional)
- Multi-scheme evaluation (strict/exact/partial/type)
- Span-based with different matching criteria
- Provides detailed error breakdowns

## Comparability Guarantees

### Within LangExtract
✅ **Fully comparable** - All evaluations use the same tokenization and matching rules.

### With Token-Classification Systems
✅ **Comparable if** you convert both gold and system outputs using the same tokenizer (spaCy recommended).
- Document the tokenizer used
- Use the same spaCy model version
- Report both span-based and seqeval metrics

### With Generative/Span-Extraction Systems
✅ **Directly comparable** - Span-based evaluation is the natural format.
- No BIO conversion needed
- Use nervaluate or custom span matching
- Prefer strict/exact/partial scheme metrics

## Testing

Run the test suite to validate improvements:

```bash
cd langextract_ner
python test_evaluation_improvements.py
```

Tests cover:
1. ✓ Tokenization consistency with converter
2. ✓ BIO tag conversion correctness
3. ✓ One-to-one matching enforcement
4. ✓ Real data evaluation with span datasets

## Usage Examples

### Basic Evaluation
```python
from evaluation import VotingEntityEvaluator
from data_loader import VotingDataLoader

# Initialize
data_loader = VotingDataLoader(data_root)
evaluator = VotingEntityEvaluator(data_loader)

# Evaluate
predictions = {...}  # document_id -> list of entities
report = evaluator.evaluate_extraction_results(
    predictions=predictions,
    language="en",  # or "pt"
    municipalities=["M01", "M02"]
)

# Access metrics
print(f"Overall F1 (seqeval): {report.overall_f1:.3f}")
print(f"Strict F1 (nervaluate): {report.nervaluate_metrics['strict']['overall']['f1']:.3f}")
```

### Convert to BIO Tags
```python
# Using spaCy tokenization (recommended)
bio_tags = evaluator.convert_to_bio_tags(
    text=document.text,
    entities=document.entities,
    language="en"
)

# Tokens and tags align with spaCy tokenization
tokens, char_spans = tokenize_text(document.text, "en")
for token, tag in zip(tokens, bio_tags):
    if tag != "O":
        print(f"{token} -> {tag}")
```

### One-to-One Matching
```python
# Matches are now unique - each GT matched at most once
matches = evaluator.calculate_entity_matches(
    predicted_entities=predictions,
    ground_truth_entities=ground_truth
)

# Verify uniqueness
gt_matched = set()
for match in matches:
    if match.match_type != "no_match":
        gt_key = (match.ground_truth_start, match.ground_truth_end)
        assert gt_key not in gt_matched  # Enforced by algorithm
        gt_matched.add(gt_key)
```

## Performance Considerations

### spaCy Model Loading
- Models are cached in `_SPACY_MODELS` dict
- First load takes ~1-2 seconds
- Subsequent uses are instant

### Tokenization Speed
- spaCy: ~10,000 tokens/second (typical)
- Whitespace fallback: ~100,000 tokens/second
- For large datasets, spaCy overhead is negligible (<1% of total time)

### Memory Usage
- spaCy models: ~50-100 MB per language
- Evaluation structures: minimal (proportional to entity count)

## Migration Guide

### From Old Evaluator
No changes needed to calling code - the API is backward compatible:

```python
# Old code - still works
report = evaluator.evaluate_extraction_results(predictions, "en", municipalities)

# New features automatically applied:
# - spaCy tokenization
# - One-to-one matching
# - Correct IOB2
```

### Dependencies
Add to `requirements.txt` if not present:
```
spacy>=3.0.0
pt-core-news-sm @ https://github.com/explosion/spacy-models/releases/download/pt_core_news_sm-3.7.0/pt_core_news_sm-3.7.0-py3-none-any.whl
en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.0/en_core_web_sm-3.7.0-py3-none-any.whl
```

Or install via:
```bash
pip install spacy
python -m spacy download pt_core_news_sm
python -m spacy download en_core_web_sm
```

## Future Improvements

### Potential Enhancements
1. **Hungarian matching** - Replace greedy with optimal bipartite matching
2. **IoU thresholds** - Configurable overlap threshold for partial matches
3. **Confusion matrix** - Full label confusion analysis
4. **Nested entities** - Support for overlapping entity spans
5. **Cross-lingual alignment** - Evaluate translation consistency

### Known Limitations
1. **Nested entities**: IOB2 cannot represent nesting; span evaluation can
2. **Tokenization variance**: Different spaCy versions may tokenize slightly differently
3. **Partial match definition**: Overlap-based, not IoU-based (future: add IoU option)

## References

- **seqeval**: https://github.com/chakki-works/seqeval
- **nervaluate**: https://github.com/MantisAI/nervaluate
- **spaCy**: https://spacy.io/
- **IOB2 scheme**: Sang & Veenstra (1999) - Representing Text Chunks

## Validation

All improvements validated against:
- ✅ Unit tests (token alignment, matching uniqueness, BIO correctness)
- ✅ Real span datasets (`span_datasets/en` and `span_datasets/pt`)
- ✅ Backward compatibility with existing evaluation reports
- ✅ Consistent results across languages

---

**Questions or Issues?**  
Contact: [Your contact info]  
Last updated: October 3, 2025
