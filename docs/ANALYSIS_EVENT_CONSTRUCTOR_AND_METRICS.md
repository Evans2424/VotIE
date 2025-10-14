# Deep Analysis: Event Constructor and Evaluation Metrics

## Executive Summary

**CRITICAL ISSUES FOUND:**
1. ❌ `EventLevelEvaluator` docstring describes OLD format, not hybrid format
2. ❌ `compare_counting()` method is completely broken (copy-paste error)
3. ❌ `evaluate_complete_event()` expects singular `counting` dict, but receives list
4. ❌ `evaluate_complete_event()` expects singular `voting_expression`, but receives list
5. ⚠️  No handling of multiple counting/voting_expressions in evaluation

**VERDICT:** EventConstructor is correct ✓, but EventLevelEvaluator needs major fixes to be consistent.

---

## Part 1: EventConstructor Analysis

### 1.1 Purpose
Converts BIO-tagged predictions into structured voting events.

### 1.2 Event Structure (Hybrid Format)
```python
{
    'id': str,
    'has_voting_event': bool,
    'event': {
        'subject': str or None,                       # SINGULAR
        'participants': [{'text': str, 'position': str}],  # LIST
        'counting': [{'text': str, 'type': str}],     # LIST  ← HYBRID
        'voting_expressions': [str]                    # LIST  ← HYBRID
    }
}
```

**Design Rationale:**
- **subject**: Singular because conceptually there's ONE thing being voted on
- **participants**: List because multiple people vote
- **counting**: List because text may mention both "por maioria" and "unanimidade"
- **voting_expressions**: List because text may have multiple verbs ("deliberou", "aprovou")

### 1.3 Method Analysis

#### `_has_voting_event(labels)` ✓
```python
return any(label.startswith('B-VOTING') for label in labels)
```
**Status:** CORRECT
- Checks if any token has B-VOTING tag
- Simple boolean detection

#### `_extract_subject(tokens, labels)` ✓
```python
spans = self._extract_spans_by_type(tokens, labels, 'SUBJECT')
if not spans:
    return None
subject_text = ' '.join(spans[0])  # Take first
return subject_text
```
**Status:** CORRECT for hybrid format
- Extracts all SUBJECT spans
- Returns only the FIRST one as a string
- If multiple subjects exist, silently takes first (could log warning)

**Recommendation:** Add warning if `len(spans) > 1` to track multiple subjects for paper.

#### `_extract_participants(tokens, labels)` ✓
```python
# Parses B-VOTER-FAVOR, I-VOTER-FAVOR, etc.
participants.append({
    'text': participant_text,
    'position': position.capitalize()  # FAVOR, AGAINST, etc.
})
```
**Status:** CORRECT
- Extracts all VOTER-* spans
- Parses position from label (FAVOR, AGAINST, ABSTENTION, ABSENT)
- Returns list of dicts

#### `_extract_counting_list(tokens, labels)` ✓
```python
# Parses B-COUNTING-UNANIMITY, B-COUNTING-MAJORITY
counting_list.append({
    'text': counting_text,
    'type': counting_type.capitalize()
})
return counting_list  # Returns LIST
```
**Status:** CORRECT
- Extracts ALL counting spans
- Returns LIST (not single dict)
- Type extracted from BIO label

#### `_extract_voting_expressions_list(tokens, labels)` ✓
```python
spans = self._extract_spans_by_type(tokens, labels, 'VOTING')
return [' '.join(span) for span in spans]  # Returns LIST of strings
```
**Status:** CORRECT
- Extracts ALL voting expression spans
- Returns LIST of strings (not single string)

### 1.4 EventConstructor Verdict
✅ **FULLY CORRECT** - Implements hybrid format as specified.

---

## Part 2: EventLevelEvaluator Analysis

### 2.1 Docstring Issues ❌

**Current (lines 26-34):**
```python
Evaluates structured event extraction with the following event schema:
{
    'has_voting_event': bool,
    'event': {
        'subject': str,  # Single string, not list
        'participants': [{'text': str, 'position': str}, ...],
        'counting': {'text': str, 'type': str} or None,  # ❌ WRONG
        'voting_expression': str or None  # ❌ WRONG
    }
}
```

**Problems:**
1. `counting` shown as single dict, but EventConstructor outputs LIST
2. `voting_expression` (singular), but EventConstructor outputs `voting_expressions` (plural list)

**Should be:**
```python
Evaluates structured event extraction with the following event schema (HYBRID):
{
    'has_voting_event': bool,
    'event': {
        'subject': str or None,  # SINGULAR
        'participants': [{'text': str, 'position': str}],
        'counting': [{'text': str, 'type': str}],  # LIST
        'voting_expressions': [str]  # LIST
    }
}
```

### 2.2 Method-by-Method Analysis

#### `evaluate_event_detection()` ✓
```python
pred_has_event = pred_event.get('has_voting_event', False)
gold_has_event = gold_event.get('has_voting_event', False)
if pred_has_event == gold_has_event:
    correct += 1
```
**Status:** CORRECT
- Binary classification: does segment have voting event?
- Returns accuracy + confusion matrix (TP, FP, TN, FN)
- Independent of event structure

#### `evaluate_subject_extraction()` ✓
```python
pred_subject = pred_event.get('event', {}).get('subject', '')
gold_subject = gold_event.get('event', {}).get('subject', '')
pred_subject = pred_subject.strip().lower()
gold_subject = gold_subject.strip().lower()
if pred_subject == gold_subject:
    correct += 1
```
**Status:** CORRECT for hybrid format
- Treats subject as single string
- Exact match after normalization
- Returns accuracy

#### `evaluate_participant_votes()` ✓
```python
pred_pairs = self.extract_participant_vote_pairs(pred_ev)
gold_pairs = self.extract_participant_vote_pairs(gold_ev)
metrics = self.compute_set_based_f1(pred_pairs, gold_pairs)
```
**Status:** CORRECT
- Extracts (text, position) tuples
- Set-based F1 (handles multiple participants)
- Averages F1 across all events

#### `compare_counting()` ❌ BROKEN
```python
def compare_counting(self, pred_event: Dict, gold_event: Dict) -> bool:
    """Evaluate subject extraction (strict set-based matching)."""  # ← WRONG docstring
    ...
    for pred_event, gold_event in zip(pred_events, gold_events):  # ← pred_events not defined!
```

**Problems:**
1. Method signature takes 2 arguments: `(pred_event, gold_event)`
2. Body tries to iterate over `pred_events` and `gold_events` (undefined!)
3. Docstring says "subject extraction" but method is called `compare_counting`
4. This is clearly copy-pasted code that was never fixed
5. Method is NOT CALLED anywhere in the codebase

**Status:** BROKEN - Not used, should be removed or fixed

#### `evaluate_complete_event()` ❌ INCONSISTENT
```python
# Line 363-383: Comparing counting
pred_counting_item = pred_ev.get('counting', None)  # ❌ Expects single dict
gold_counting_item = gold_ev.get('counting', None)  # ❌ Expects single dict

if pred_counting_item is None and gold_counting_item is None:
    counting_match = True
elif pred_counting_item is None or gold_counting_item is None:
    counting_match = False
else:
    # Compares as single dict
    pred_text = pred_counting_item.get('text', '').strip().lower()
    pred_type = pred_counting_item.get('type', '').strip().upper()
    gold_text = gold_counting_item.get('text', '').strip().lower()
    gold_type = gold_counting_item.get('type', '').strip().upper()
    counting_match = (pred_text == gold_text and pred_type == gold_type)
```

**Problem:** EventConstructor outputs `counting` as a LIST, but this code treats it as a single dict!

**What happens:**
```python
# EventConstructor output:
counting: [{'text': 'por maioria', 'type': 'Majority'}]

# evaluate_complete_event tries:
pred_counting_item = counting  # This is a LIST, not dict!
pred_text = pred_counting_item.get('text', '')  # ❌ Lists don't have .get()!
# → AttributeError: 'list' object has no attribute 'get'
```

**Same issue with voting_expression (lines 386-391):**
```python
pred_expr = pred_ev.get('voting_expression', '').strip().lower()  # ❌ singular
gold_expr = gold_ev.get('voting_expression', '').strip().lower()  # ❌ singular
```

But EventConstructor outputs `voting_expressions` (plural, list)!

### 2.3 EventLevelEvaluator Verdict
❌ **INCONSISTENT** - Does not match EventConstructor's hybrid format.

---

## Part 3: Consistency Check

| Component | EventConstructor Output | EventMetrics Expects | Match? |
|-----------|------------------------|---------------------|--------|
| subject | `str or None` | `str` | ✓ |
| participants | `[{text, position}]` | `[{text, position}]` | ✓ |
| counting | `[{text, type}]` (LIST) | `{text, type} or None` (DICT) | ❌ |
| voting_expressions | `[str]` (LIST) | `str or None` (STRING) | ❌ |

**Conclusion:** Major inconsistency in counting and voting_expressions.

---

## Part 4: Realistic Metrics for Paper

### 4.1 Current Metrics

1. **Event Detection Accuracy**: Binary classification (has_voting_event)
   - ✓ Realistic
   - Standard metric for event detection

2. **Subject Extraction Accuracy**: Exact string match
   - ✓ Realistic for singular subject
   - Could consider partial match (token overlap) for robustness

3. **Participant-Vote F1**: Set-based F1 over (participant, position) pairs
   - ✓ Realistic
   - Standard for multi-label classification
   - Handles multiple participants correctly

4. **Complete Event Accuracy**: All components must match exactly
   - ⚠️ Problematic due to bugs
   - Too strict? (94% missing voting expressions in error analysis!)

### 4.2 Issues with Current Approach

**Problem 1: Multiple Counting Expressions**
- Gold might have: `["por maioria", "por unanimidade"]` (rare but possible)
- Pred might have: `["por maioria"]`
- Current code would crash trying to compare list to dict

**Question:** Should we require ALL counting expressions to match, or just first?

**Recommendation:**
- **For publication**: Compare as sets (all must match)
- **Rationale**: Counting info is critical for voting event semantics
- **Alternative**: Only compare first item (simpler but less accurate)

**Problem 2: Multiple Voting Expressions**
- Gold might have: `["deliberou", "aprovou", "votou"]`
- Pred might have: `["deliberou"]`
- Should this be considered correct?

**Recommendation:**
- **For publication**: Compare as sets OR only compare first item
- **Rationale**: Multiple expressions are stylistic, not semantic
- **Alternative**: Set-based comparison (stricter)

**Problem 3: Complete Event is Too Strict**
From error analysis: 94% of events had "missing" voting expressions!

This suggests:
1. Either the models truly fail at extracting voting expressions
2. OR the evaluation is broken/too strict

Given that voting expressions are common words like "deliberou", "aprovou", I suspect the evaluation is the issue.

### 4.3 Proposed Evaluation Strategy

**Option A: Strict (For perfectionist paper)**
- All counting expressions must match (set equality)
- All voting expressions must match (set equality)
- Subject must match exactly
- All participants must match exactly

**Option B: Relaxed (More realistic)**
- First counting expression must match
- First voting expression must match
- Subject must match exactly
- All participants must match exactly

**Option C: Hybrid (Recommended)**
- All counting expressions must match (critical info)
- First voting expression must match (less critical)
- Subject must match exactly
- All participants must match exactly

### 4.4 Missing Metrics

Consider adding:
1. **Partial event match**: Count how many components are correct (0-4 scale)
2. **Component-level F1**: Separate F1 for each component type
3. **Relaxed subject match**: Token overlap instead of exact match
4. **Boundary tolerance**: Allow ±1 token boundary errors

---

## Part 5: Recommendations

### 5.1 Immediate Fixes Required

1. **Fix EventLevelEvaluator docstring**
   - Update to show hybrid format
   - Correct counting and voting_expressions fields

2. **Remove or fix `compare_counting()` method**
   - Currently broken and unused
   - Either fix properly or delete

3. **Fix `evaluate_complete_event()` to handle lists**
   ```python
   # For counting (list)
   pred_counting = pred_ev.get('counting', [])
   gold_counting = gold_ev.get('counting', [])

   # Option A: Compare first item only
   pred_first = pred_counting[0] if pred_counting else None
   gold_first = gold_counting[0] if gold_counting else None
   # Compare pred_first and gold_first

   # Option B: Compare as sets (stricter)
   pred_set = {(c['text'].lower(), c['type'].upper()) for c in pred_counting}
   gold_set = {(c['text'].lower(), c['type'].upper()) for c in gold_counting}
   counting_match = (pred_set == gold_set)
   ```

4. **Add backward compatibility**
   - Some data might still have old format
   - Add helper to normalize both formats:
   ```python
   def normalize_counting(counting):
       if counting is None:
           return []
       if isinstance(counting, list):
           return counting
       if isinstance(counting, dict):
           return [counting]
       return []
   ```

### 5.2 Documentation Needed

1. **Event Format Specification Document**
   - Clearly define hybrid format
   - Show examples for all cases
   - Explain why hybrid (not all lists, not all singular)

2. **Evaluation Metrics Document**
   - Define each metric precisely
   - Show calculation examples
   - Justify design choices

3. **Method Docstrings**
   - Add detailed docstrings to all methods
   - Include examples of input/output
   - Document edge cases

### 5.3 Testing Needed

Create comprehensive unit tests:
```python
def test_event_constructor_output_format():
    """Verify EventConstructor outputs hybrid format"""
    constructor = EventConstructor()
    tokens = ["A", "proposta", "foi", "aprovada", "por", "unanimidade"]
    labels = ["O", "B-SUBJECT", "O", "B-VOTING", "B-COUNTING-UNANIMITY", "I-COUNTING-UNANIMITY"]

    event = constructor.construct_event(tokens, labels, "test_001")

    assert isinstance(event['event']['subject'], str)  # Singular
    assert isinstance(event['event']['counting'], list)  # List
    assert isinstance(event['event']['voting_expressions'], list)  # List
```

---

## Part 6: Summary

### Current State
- ✅ EventConstructor: Correct, implements hybrid format properly
- ❌ EventLevelEvaluator: Inconsistent, expects old format

### Critical Issues
1. Evaluator cannot handle list-based counting/voting_expressions
2. Broken `compare_counting()` method
3. Incorrect docstrings

### Impact on Paper
- **Data Quality**: EventConstructor outputs are correct ✓
- **Evaluation Validity**: Evaluation metrics are BROKEN ❌
- **Results Reliability**: Published results may be incorrect or inflated

### Priority Actions
1. **URGENT**: Fix evaluate_complete_event() to handle lists
2. **URGENT**: Update docstrings to show correct format
3. **HIGH**: Remove broken compare_counting() method
4. **HIGH**: Add unit tests to prevent regression
5. **MEDIUM**: Add comprehensive documentation

### Questions for Paper Authors
1. Should complete event require ALL counting expressions or just first?
2. Should complete event require ALL voting expressions or just first?
3. Are partial matches acceptable, or only exact matches?
4. Should we report per-component F1 in addition to complete event accuracy?
