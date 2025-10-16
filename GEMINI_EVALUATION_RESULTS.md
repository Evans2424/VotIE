# Gemini 2.0 Flash Event Extraction - Evaluation Results

## Overview

This document presents the evaluation results for the Gemini 2.0 Flash model on the VotIE event extraction task.

## Evaluation Script

**Location**: `evaluate_gemini_events.py`

**Usage**:
```bash
# Basic evaluation
python evaluate_gemini_events.py

# With verbose mode (shows error analysis)
python evaluate_gemini_events.py --verbose

# Custom paths
python evaluate_gemini_events.py \
    --predictions path/to/predictions.jsonl \
    --gold path/to/gold.json \
    --output path/to/output.json
```

## Results

### Event-Level Metrics (F1 Scores)

| Metric | F1 Score | Description |
|--------|----------|-------------|
| **Event Detection F1** | **0.9663** | Binary classification of segments with/without voting events |
| **Subject Extraction F1** | **0.5317** | Span-level exact matching of voting subjects |
| **Counting Extraction F1** | **0.9620** | Span-level exact matching of counting expressions with type |
| **Participant-Vote F1** | **0.6965** | Set-based F1 over (participant, position) tuples |
| **Complete Event F1** | **0.3239** | Strict accuracy - all components must match |
| **Macro-Average F1** | **0.6961** | Average across all 5 metrics |

### Analysis

#### Strengths ✓
1. **Excellent Event Detection** (96.63% F1)
   - Very accurate at identifying segments containing voting events
   - Only 25 errors out of 433 segments (~5.8% error rate)

2. **Strong Counting Extraction** (96.20% F1)
   - Nearly perfect at extracting voting counting expressions
   - Correctly identifies types (Unanimity, Majority, etc.)

3. **Good Participant-Vote Extraction** (69.65% F1)
   - Reasonably accurate at extracting participant-position pairs
   - Handles multiple participants per event

#### Weaknesses ✗
1. **Moderate Subject Extraction** (53.17% F1)
   - Most challenging component
   - Often fails to match exact subject text
   - May be due to:
     - Paraphrasing vs. exact text spans
     - Different levels of detail/abstraction
     - Boundary mismatch (e.g., including/excluding articles)

2. **Low Complete Event Accuracy** (32.39% F1)
   - Strict metric requiring ALL components to match
   - Driven by subject extraction weaknesses
   - Even small errors in any component cause complete failure

### Error Analysis

**Event Detection Errors**: 25 total
- Most errors are false positives (model predicts event when there isn't one)
- Example segments: M02_cm_011_2023-06-07_seg002, M02_cm_011_2023-06-07_seg003

**Common Error Patterns**:
1. Subject text doesn't match exactly (boundaries, paraphrasing)
2. Missing or extra participants
3. Position labels may differ (e.g., "Favor" vs. "A favor")

## Dataset Information

- **Test Set Size**: 433 segments
- **Predictions File**: `langextract_votie/experiments/langextract_events_20251015_024557/predictions_events.jsonl`
- **Gold File**: `data/votie_events/test.json`

## Evaluation Methodology

### Metrics Definitions

1. **Event Detection F1**: Binary F1 for has_voting_event classification
2. **Subject Extraction F1**: Span-level F1 with exact string matching (case-insensitive)
3. **Counting Extraction F1**: Span-level F1 matching both text and type
4. **Participant-Vote F1**: Set-based F1 over (participant_text, position) tuples
5. **Complete Event F1**: Strict F1 where all components must match exactly

### Important Notes

- All text comparisons are case-insensitive after normalization
- Counting and voting expressions: Only first item is compared (hybrid format support)
- Participant-vote pairs are compared as sets (order doesn't matter)
- Segment IDs are automatically normalized (municipality names → codes)

## ID Normalization

The script automatically handles ID format differences:
- **Predictions format**: `Covilha_cm_002_2023-02-03_seg045`
- **Gold format**: `M03_cm_002_2023-02-03_seg045`

Municipality code mapping:
```python
{
    'Alandroal': 'M01',
    'Campomaior': 'M02',
    'Covilha': 'M03',
    'Fundao': 'M04',
    'Guimaraes': 'M05',
    'Porto': 'M06'
}
```

## Comparison with Paper Metrics

The evaluation implements the exact metrics described in the VotIE paper:
- ✓ Event detection (binary classification)
- ✓ Subject extraction (span-level)
- ✓ Counting extraction (span-level with type)
- ✓ Participant-vote extraction (set-based)
- ✓ Complete event extraction (strict)

All metrics use **F1 score** (not accuracy) for proper evaluation of imbalanced classes.

## Files Generated

1. **Metrics JSON**: `langextract_votie/experiments/langextract_events_20251015_024557/evaluation_metrics.json`
   ```json
   {
     "event_detection_f1": 0.9663,
     "subject_f1": 0.5317,
     "counting_f1": 0.9620,
     "participant_vote_f1": 0.6965,
     "complete_event_f1": 0.3239
   }
   ```

2. **Evaluation Script**: `evaluate_gemini_events.py`
   - Standalone Python script
   - Full command-line interface
   - Comprehensive logging and error analysis

## Next Steps

To improve performance:
1. **Subject Extraction**: 
   - Fine-tune prompts for more accurate subject boundaries
   - Consider fuzzy matching or similarity metrics
   - Analyze specific failure cases

2. **Participant-Vote Extraction**:
   - Standardize position labels
   - Improve participant entity recognition
   - Handle implicit participants

3. **Complete Event**:
   - Will improve automatically as component metrics improve
   - Focus on subject extraction first (biggest weakness)
