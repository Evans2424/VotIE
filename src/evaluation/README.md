# Evaluation Metrics for Vote Identification

This directory contains comprehensive evaluation metrics for Named Entity Recognition (NER) and structured event extraction in the context of voting event identification from municipal meeting minutes.

## Module Structure

The evaluation code is organized into three specialized modules, each handling a different granularity level:

### 1. `entity_metrics.py` - Entity-Level Evaluation

**Purpose**: Evaluates NER performance at the token level using the standard seqeval library.

**Key Features**:
- Strict entity-level evaluation using IOB2 tagging scheme
- Overall metrics: precision, recall, F1, accuracy
- Per-entity-type metrics with support counts
- Standard in NER research and widely adopted

**Usage**:
```python
from src.evaluation.entity_metrics import EntityLevelEvaluator

evaluator = EntityLevelEvaluator()
metrics = evaluator.compute_metrics(pred_labels, true_labels)

print(f"Token-level F1: {metrics['token_f1']:.3f}")
print(f"Per-type metrics: {metrics['per_type_metrics']}")
```

**Metrics Returned**:
- `token_precision`: Precision at token level
- `token_recall`: Recall at token level
- `token_f1`: F1 score at token level
- `token_accuracy`: Token-level accuracy
- `per_type_metrics`: Dict mapping entity types to their metrics

---

### 2. `span_metrics.py` - Span-Level Evaluation

**Purpose**: Evaluates NER performance at the span level using exact matching paradigm.

**Key Features**:
- **Exact match paradigm**: A predicted span matches a gold span if and only if:
  1. The text is identical (character-level exact match)
  2. The entity type agrees
- **Macro F1 computation**: Averages F1 scores across entity types
- Micro metrics: Aggregated across all spans
- Per-entity-type metrics with support counts

**Usage**:
```python
from src.evaluation.span_metrics import SpanLevelEvaluator

evaluator = SpanLevelEvaluator()
metrics = evaluator.compute_metrics(pred_labels, true_labels, tokens)

print(f"Span-level F1 (micro): {metrics['span_f1']:.3f}")
print(f"Span-level F1 (macro): {metrics['span_macro_f1']:.3f}")
```

**Metrics Returned**:
- `span_precision`: Micro-averaged precision
- `span_recall`: Micro-averaged recall
- `span_f1`: Micro-averaged F1
- `span_macro_f1`: Macro-averaged F1 across entity types
- `per_type_metrics`: Dict mapping entity types to their span metrics
- `total_pred_spans`: Total number of predicted spans
- `total_gold_spans`: Total number of gold spans

---

### 3. `event_metrics.py` - Event-Level Evaluation

**Purpose**: Evaluates structured voting event extraction at the event level.

**Key Components**:

#### a) Event Detection
Binary classification accuracy for identifying segments containing voting events.

#### b) Participant-Vote Extraction
Set-based F1 over (participant, position) tuples:
- Given predicted set $\hat{P}$ and gold set $P$:
  - Precision = $|\hat{P} \cap P| / |\hat{P}|$
  - Recall = $|\hat{P} \cap P| / |P|$
  - F1 = $2 \cdot \text{precision} \cdot \text{recall} / (\text{precision} + \text{recall})$

#### c) Subject and Counting Extraction
Span-level exact matching (each event contains exactly one of each).

#### d) Complete Event Extraction
Strict accuracy where **all** components must match:
- Subject
- All participant-vote pairs
- Counting information

**Span Matching**: Predicted and gold spans match if their text is identical and types agree. For participant-vote pairs, both the participant mention and position value must match.

**Usage**:
```python
from src.evaluation.event_metrics import EventLevelEvaluator

evaluator = EventLevelEvaluator()
metrics = evaluator.compute_metrics(pred_events, gold_events)

print(f"Event detection accuracy: {metrics['event_detection_accuracy']:.3f}")
print(f"Participant-vote F1: {metrics['participant_vote_f1']:.3f}")
print(f"Complete event accuracy: {metrics['complete_event_accuracy']:.3f}")
```

**Metrics Returned**:
- `event_detection_accuracy`: Binary accuracy for event detection
- `true_positives`, `false_positives`, `true_negatives`, `false_negatives`: Confusion matrix
- `participant_vote_precision`: Precision for participant-vote extraction
- `participant_vote_recall`: Recall for participant-vote extraction
- `participant_vote_f1`: F1 for participant-vote extraction
- `subject_precision`: Precision for subject extraction
- `subject_recall`: Recall for subject extraction
- `subject_f1`: F1 for subject extraction
- `complete_event_accuracy`: Strict accuracy for complete events
- `complete_matches`: Number of perfectly matched events
- `total_events`: Total number of events in gold data

---

### 4. `metrics.py` - Unified Interface

**Purpose**: Provides a unified interface that maintains backward compatibility while delegating to specialized evaluators.

**Usage**:
```python
from src.evaluation.metrics import NEREvaluator

evaluator = NEREvaluator()
metrics = evaluator.compute_metrics(
    pred_labels,
    true_labels,
    tokens=tokens,
    include_span_metrics=True
)

# Access both token and span metrics
print(f"Token F1: {metrics['token_f1']:.3f}")
print(f"Span F1: {metrics['span_f1']:.3f}")
```

---

## Data Format Examples

### Token/Span Format
```python
# BIO labels
true_labels = [
    ['O', 'B-VOTER-FAVOR', 'I-VOTER-FAVOR', 'B-VOTING', 'O'],
    ['O', 'O', 'B-SUBJECT', 'I-SUBJECT', 'I-SUBJECT']
]

pred_labels = [
    ['O', 'B-VOTER-FAVOR', 'I-VOTER-FAVOR', 'B-VOTING', 'O'],
    ['O', 'O', 'B-SUBJECT', 'I-SUBJECT', 'O']
]

# Tokens (optional for span evaluation)
tokens = [
    ['O', 'Executivo', 'Municipal', 'deliberou', 'aprovar'],
    ['Aprovar', 'a', 'proposta', 'de', 'protocolo']
]
```

### Event Format
```python
gold_event = {
    "id": "M01_Alandroal_cm_003_2023-02-01_seg020",
    "text": "...",
    "has_voting_event": True,
    "event": {
        "subjects": ["atribuição de três apoios de vales"],
        "participants": [
            {"text": "o Executivo Municipal", "position": "Favor"}
        ],
        "counting": [
            {"text": "por unanimidade", "type": "Unanimity"}
        ]
    },
    "spans": [...]
}
```

---

## Migration Guide

### Old Code
```python
from src.evaluation.metrics import NEREvaluator

evaluator = NEREvaluator()
results = evaluator.compute_metrics(pred_labels, true_labels)
```

### New Code (Recommended)
```python
# For token-level only
from src.evaluation.token_metrics import TokenLevelEvaluator
evaluator = TokenLevelEvaluator()
token_results = evaluator.compute_metrics(pred_labels, true_labels)

# For span-level
from src.evaluation.span_metrics import SpanLevelEvaluator
evaluator = SpanLevelEvaluator()
span_results = evaluator.compute_metrics(pred_labels, true_labels, tokens)

# For event-level
from src.evaluation.event_metrics import EventLevelEvaluator
evaluator = EventLevelEvaluator()
event_results = evaluator.compute_metrics(pred_events, gold_events)

# Or use unified interface
from src.evaluation.metrics import NEREvaluator
evaluator = NEREvaluator()
all_results = evaluator.compute_metrics(
    pred_labels, 
    true_labels, 
    tokens=tokens,
    include_span_metrics=True
)
```

---

## Key Differences from Previous Version

1. **Removed nervaluate dependency**: The previous version used nervaluate for partial matching. This has been removed in favor of exact matching at the span level.

2. **Added macro F1**: Span-level evaluation now includes macro-averaged F1, which averages per-type F1 scores.

3. **Added event-level evaluation**: Comprehensive structured event extraction evaluation including:
   - Event detection
   - Participant-vote pair extraction
   - Subject and counting extraction
   - Complete event extraction

4. **Modular structure**: Evaluation logic is now separated into focused modules for easier maintenance and testing.

5. **Exact match paradigm**: All evaluations use strict exact matching (text and type must be identical).

---

## References

- **seqeval**: Standard evaluation framework for sequence labeling
  - Paper: "Introduction to the CoNLL-2003 Shared Task" (Tjong Kim Sang & De Meulder, 2003)
  - Library: https://github.com/chakki-works/seqeval

- **Voting Event Extraction**: Based on the methodology described in your paper's event-level metrics section.
