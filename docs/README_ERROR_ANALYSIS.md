# Event-Level Error Analysis

This script analyzes event extraction errors and categorizes them into specific error types.

## Usage

```bash
# Basic usage
python scripts/analyze_event_errors.py \
    --predictions experiments/pt_stratified_split/deberta_crf_20251010_070842/predictions.jsonl \
    --gold data/votie_events/test.json

# Save detailed JSON output
python scripts/analyze_event_errors.py \
    --predictions experiments/pt_stratified_split/deberta_crf_20251010_070842/predictions.jsonl \
    --gold data/votie_events/test.json \
    --output error_analysis.json
```

## Input Formats

The script supports two prediction formats:
1. **BIO Format** (`.jsonl` with `tokens` and `pred_labels` fields) - automatically converted to events
2. **Event Format** (`.jsonl` with `event` and `has_voting_event` fields) - used directly

Gold data should be in event format (`.json` or `.jsonl`).

## Error Categories

The script categorizes errors into the following types:

### 1. Subject Errors
- **Missing**: Gold has subject but prediction doesn't
- **Boundary**: Partial text overlap (detected vs gold subject)
- **Incorrect**: Completely different subject text

### 2. Participant-Vote Errors
- **Missing**: Gold participants not found in predictions
- **Extra**: Predicted participants not in gold
- **Position Wrong**: Same participant text but wrong vote position

### 3. Counting Errors
- **Missing**: Gold has counting but prediction doesn't
- **Type Wrong**: Incorrect counting type (Unanimity vs Majority)
- **Incorrect**: Counting text doesn't match

### 4. Voting Expression Errors
- **Missing**: Gold has voting expressions but prediction doesn't
- **Incorrect**: Voting expression text doesn't overlap with gold

### 5. Event Detection Errors
- **False Positives**: Predicted event where none exists
- **False Negatives**: Missed event detection

### 6. Multiple Errors
Tracks events with more than one error type simultaneously.

## Output

The script provides:

1. **Console Output**: Summary statistics with percentages
2. **JSON Output** (if `--output` specified): Detailed breakdown including:
   - Overall statistics (accuracy, total events, errors)
   - Error distribution by category with percentages
   - Up to 5 examples for each error type
   - Detailed breakdowns for each error category

### Example Output

```
EVENT-LEVEL ERROR ANALYSIS
================================================================================

Total Events:    433
Correct Events:  54 (12.5%)
Failed Events:   379

================================================================================
ERROR DISTRIBUTION
================================================================================

Subject Errors: 301 (79.4%)
  - Missing:    15
  - Boundary:   278
  - Incorrect:  8

Participant-Vote Errors: 234 (61.7%)
  - Missing:        114
  - Extra:          120
  - Position Wrong: 0

Counting Errors: 9 (2.4%)
  - Missing:    6
  - Type Wrong: 2
  - Incorrect:  1

Voting Expression Errors: 357 (94.2%)
  - Missing:    357
  - Incorrect:  0

Event Detection Errors: 9 (2.4%)
  - False Positives: 6
  - False Negatives: 3

Multiple Simultaneous Errors: 313 (82.6%)

================================================================================
SUMMARY FOR PAPER
================================================================================

Error analysis of 379 failed events reveals:
79% from subject errors, 62% from participant-vote errors, 2% from counting
errors, 94% from voting expression errors, 2% from event detection errors,
and 83% from multiple simultaneous errors.
```

## Paper-Ready Summary

The script generates a publication-ready summary sentence that can be directly used in papers:

```
Error analysis of [N] failed events reveals:
[X]% from subject errors, [Y]% from participant-vote errors, [Z]% from counting
errors, [W]% from voting expression errors, [V]% from event detection errors,
and [U]% from multiple simultaneous errors.
```

## Comparison Methodology

### Text Matching
- **Exact Match**: Normalized text (lowercase, stripped) must be identical
- **Boundary Match**: At least 50% word overlap between texts
- **Partial Match**: At least 30% word overlap for expressions/counting

### Participant Matching
Participants are matched by both text AND position:
- `(text, position)` tuples must match exactly
- Position errors detected when same text has different position

### Counting Matching
- Type must match (Unanimity vs Majority)
- Text must have at least 30% word overlap

## Dependencies

- `src.utils.event_constructor` - for converting BIO to events
- Standard library: `json`, `argparse`, `pathlib`, `collections`, `dataclasses`

## Notes

- **ID Normalization**: Automatically handles ID format differences (with/without municipality prefix)
- **Format Flexibility**: Handles both list and non-list formats for counting/voting expressions
- **Multiple Errors**: Percentages may sum to >100% because events can have multiple error types
- **Boundary Errors**: The most common error type, indicating correct entity detection but imprecise boundaries
