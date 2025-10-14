#!/usr/bin/env python
"""
Event-Level Evaluation for Voting Event Extraction.

This module provides event-level evaluation metrics for structured voting event
extraction, including:
1. Event Detection: Binary accuracy for identifying segments with voting events
2. Subject Extraction: Exact string matching (single subject per event)
3. Participant-Vote Extraction: Set-based F1 over (participant, position) tuples
4. Complete Event Extraction: Strict accuracy where all components must match

Span matching uses exact text comparison after normalization (lowercase, strip).
"""

import logging
from typing import List, Dict, Any, Set, Tuple

# Note: Added Any to handle backward compatibility with different data formats

logger = logging.getLogger(__name__)


class EventLevelEvaluator:
    """
    Event-level evaluation for voting event extraction.

    Evaluates structured event extraction with the HYBRID event schema:
    {
        'has_voting_event': bool,
        'event': {
            'subject': str or None,  # SINGULAR - single string
            'participants': [{'text': str, 'position': str}],  # LIST
            'counting': [{'text': str, 'type': str}],  # LIST (hybrid format)
            'voting_expressions': [str]  # LIST (hybrid format)
        }
    }

    Notes:
    - Backward compatible: handles both old singular and new list formats
    - For lists: compares first item only (configurable via parameters)
    - All text comparisons are case-insensitive after normalization
    """

    def __init__(self):
        """Initialize evaluator."""
        pass

    def _normalize_counting(self, counting: Any) -> List[Dict[str, str]]:
        """
        Normalize counting to list format (backward compatible).

        Args:
            counting: Can be None, dict, or list

        Returns:
            List of counting dicts (empty if None)
        """
        if counting is None:
            return []
        if isinstance(counting, list):
            return counting
        if isinstance(counting, dict):
            return [counting]
        return []

    def _normalize_voting_expressions(self, expressions: Any) -> List[str]:
        """
        Normalize voting expressions to list format (backward compatible).

        Args:
            expressions: Can be None, str, or list

        Returns:
            List of expression strings (empty if None)
        """
        if expressions is None:
            return []
        if isinstance(expressions, list):
            return [str(e) for e in expressions if e]
        if isinstance(expressions, str):
            return [expressions] if expressions else []
        return []

    def extract_participant_vote_pairs(
        self,
        event: Dict[str, Any]
    ) -> Set[Tuple[str, str]]:
        """
        Extract (participant, position) tuples from event.

        Args:
            event: Event dictionary with 'participants' field

        Returns:
            Set of (participant_text, position) tuples (normalized)
        """
        pairs = set()
        participants = event.get('participants', [])
        
        for participant in participants:
            text = participant.get('text', '').strip().lower()
            position = participant.get('position', '').strip().upper()
            if text and position:
                pairs.add((text, position))
        
        return pairs

    def compute_set_based_f1(
        self,
        pred_set: Set[Any],
        gold_set: Set[Any]
    ) -> Dict[str, float]:
        """
        Compute set-based precision, recall, and F1.
        
        Special case: If both sets are empty, this is a perfect match (1.0, 1.0, 1.0).
        This handles events with legitimately no participants.

        Args:
            pred_set: Set of predicted elements
            gold_set: Set of gold elements

        Returns:
            Dictionary with precision, recall, and f1
        """
        # Special case: both empty = perfect match
        if len(pred_set) == 0 and len(gold_set) == 0:
            return {
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0
            }
        
        if len(pred_set) == 0:
            precision = 0.0
        else:
            intersection = pred_set & gold_set
            precision = len(intersection) / len(pred_set)

        if len(gold_set) == 0:
            recall = 0.0
        else:
            intersection = pred_set & gold_set
            recall = len(intersection) / len(gold_set)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def evaluate_event_detection(
        self,
        pred_events: List[Dict[str, Any]],
        gold_events: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Evaluate binary event detection accuracy.

        Args:
            pred_events: List of predicted events (one per segment)
            gold_events: List of gold events (one per segment)

        Returns:
            Dictionary with detection accuracy and confusion matrix counts
        """
        correct = 0
        total = len(gold_events)
        
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0

        for pred_event, gold_event in zip(pred_events, gold_events):
            # Handle None events
            pred_has_event = pred_event.get('has_voting_event', False) if pred_event else False
            gold_has_event = gold_event.get('has_voting_event', False) if gold_event else False

            if pred_has_event == gold_has_event:
                correct += 1
                if gold_has_event:
                    true_positives += 1
                else:
                    true_negatives += 1
            else:
                if pred_has_event and not gold_has_event:
                    false_positives += 1
                else:
                    false_negatives += 1

        accuracy = correct / total if total > 0 else 0.0

        return {
            "event_detection_accuracy": accuracy,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives
        }

    def evaluate_subject_extraction(
        self,
        pred_events: List[Dict[str, Any]],
        gold_events: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Evaluate subject extraction with precision, recall, and F1.
        
        Treats subject extraction as binary classification:
        - True Positive: Subject extracted correctly
        - False Positive: Subject extracted but incorrect
        - False Negative: Subject not extracted but should be

        Args:
            pred_events: List of predicted events
            gold_events: List of gold events

        Returns:
            Dictionary with accuracy, precision, recall, and F1
        """
        correct = 0
        total = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for pred_event, gold_event in zip(pred_events, gold_events):
            # Skip if no event structure (handle None pred_event or gold_event)
            if pred_event is None or gold_event is None:
                continue
                
            if not gold_event.get('has_voting_event', False):
                continue

            total += 1

            # Extract subject (single string) - safely handle None event dicts
            # Note: event can be explicitly None when has_voting_event is False
            pred_event_data = pred_event.get('event') if pred_event else None
            gold_event_data = gold_event.get('event') if gold_event else None
            
            # Handle case where event is None (no voting event)
            pred_subject = pred_event_data.get('subject') if pred_event_data and isinstance(pred_event_data, dict) else None
            gold_subject = gold_event_data.get('subject') if gold_event_data and isinstance(gold_event_data, dict) else None

            # Normalize and compare (handle None values)
            pred_subject_norm = pred_subject.strip().lower() if pred_subject else ''
            gold_subject_norm = gold_subject.strip().lower() if gold_subject else ''

            has_pred = len(pred_subject_norm) > 0
            has_gold = len(gold_subject_norm) > 0

            if pred_subject_norm == gold_subject_norm and has_gold:
                correct += 1
                true_positives += 1
            elif has_pred and not has_gold:
                false_positives += 1
            elif has_pred and has_gold and pred_subject_norm != gold_subject_norm:
                false_positives += 1
                false_negatives += 1
            elif not has_pred and has_gold:
                false_negatives += 1

        accuracy = correct / total if total > 0 else 0.0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "subject_accuracy": accuracy,
            "subject_precision": precision,
            "subject_recall": recall,
            "subject_f1": f1,
            "subject_correct": correct,
            "subject_total": total
        }

    def evaluate_participant_votes(
        self,
        pred_events: List[Dict[str, Any]],
        gold_events: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Evaluate participant-vote extraction with set-based F1.

        Args:
            pred_events: List of predicted events
            gold_events: List of gold events

        Returns:
            Dictionary with aggregated metrics across all events
        """
        all_precisions = []
        all_recalls = []
        all_f1s = []

        for pred_event, gold_event in zip(pred_events, gold_events):
            if pred_event is None or gold_event is None:
                continue
            if not gold_event.get('has_voting_event', False):
                continue

            # Handle None event structures (event can be None when has_voting_event is False)
            pred_event_data = pred_event.get('event') or {}
            gold_event_data = gold_event.get('event') or {}
            
            pred_pairs = self.extract_participant_vote_pairs(pred_event_data)
            gold_pairs = self.extract_participant_vote_pairs(gold_event_data)

            metrics = self.compute_set_based_f1(pred_pairs, gold_pairs)
            all_precisions.append(metrics['precision'])
            all_recalls.append(metrics['recall'])
            all_f1s.append(metrics['f1'])

        if len(all_f1s) == 0:
            return {
                "participant_vote_precision": 0.0,
                "participant_vote_recall": 0.0,
                "participant_vote_f1": 0.0
            }

        return {
            "participant_vote_precision": sum(all_precisions) / len(all_precisions),
            "participant_vote_recall": sum(all_recalls) / len(all_recalls),
            "participant_vote_f1": sum(all_f1s) / len(all_f1s)
        }

    def evaluate_complete_event(
        self,
        pred_events: List[Dict[str, Any]],
        gold_events: List[Dict[str, Any]],
        include_voting_expression: bool = True,
        compare_all_counting: bool = False,
        compare_all_expressions: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate complete event extraction (strict accuracy).

        All components must match exactly:
        - Subject (exact string match)
        - All participant-vote pairs (set match)
        - Counting information (first item or all items, based on compare_all_counting)
        - Voting expression (first item or all items, based on compare_all_expressions)

        Args:
            pred_events: List of predicted events
            gold_events: List of gold events
            include_voting_expression: Whether to require voting expression match
            compare_all_counting: If True, compare all counting items; if False, only first
            compare_all_expressions: If True, compare all expressions; if False, only first

        Returns:
            Dictionary with strict accuracy and detailed counts
        """
        correct = 0
        total = 0
        failures = {
            'subject_mismatch': 0,
            'participants_mismatch': 0,
            'counting_mismatch': 0,
            'expression_mismatch': 0
        }

        for pred_event, gold_event in zip(pred_events, gold_events):
            if pred_event is None or gold_event is None:
                continue
            if not gold_event.get('has_voting_event', False):
                continue

            total += 1

            # Check all components - handle None event structures (event can be None)
            pred_ev = pred_event.get('event') or {}
            gold_ev = gold_event.get('event') or {}

            # 1. Compare subjects (single string)
            pred_subject = pred_ev.get('subject', None)
            gold_subject = gold_ev.get('subject', None)

            pred_subject_norm = pred_subject.strip().lower() if pred_subject else None
            gold_subject_norm = gold_subject.strip().lower() if gold_subject else None

            if pred_subject_norm != gold_subject_norm:
                failures['subject_mismatch'] += 1
                continue

            # 2. Compare participant-vote pairs (set)
            pred_pairs = self.extract_participant_vote_pairs(pred_ev)
            gold_pairs = self.extract_participant_vote_pairs(gold_ev)

            if pred_pairs != gold_pairs:
                failures['participants_mismatch'] += 1
                continue

            # 3. Compare counting (LIST - hybrid format)
            pred_counting_raw = pred_ev.get('counting')
            gold_counting_raw = gold_ev.get('counting')

            # Normalize to lists (backward compatible)
            pred_counting = self._normalize_counting(pred_counting_raw)
            gold_counting = self._normalize_counting(gold_counting_raw)

            if compare_all_counting:
                # Compare all items as sets
                pred_counting_set = {
                    (c.get('text', '').strip().lower(), c.get('type', '').strip().upper())
                    for c in pred_counting
                }
                gold_counting_set = {
                    (c.get('text', '').strip().lower(), c.get('type', '').strip().upper())
                    for c in gold_counting
                }
                counting_match = (pred_counting_set == gold_counting_set)
            else:
                # Compare only first items
                pred_first = pred_counting[0] if pred_counting else None
                gold_first = gold_counting[0] if gold_counting else None

                if pred_first is None and gold_first is None:
                    counting_match = True
                elif pred_first is None or gold_first is None:
                    counting_match = False
                else:
                    pred_text = pred_first.get('text', '').strip().lower()
                    pred_type = pred_first.get('type', '').strip().upper()
                    gold_text = gold_first.get('text', '').strip().lower()
                    gold_type = gold_first.get('type', '').strip().upper()
                    counting_match = (pred_text == gold_text and pred_type == gold_type)

            if not counting_match:
                failures['counting_mismatch'] += 1
                continue

            # 4. Optionally compare voting expressions (LIST - hybrid format)
            if include_voting_expression:
                # Handle both old and new field names
                pred_expr_raw = pred_ev.get('voting_expressions') or pred_ev.get('voting_expression')
                gold_expr_raw = gold_ev.get('voting_expressions') or gold_ev.get('voting_expression')

                # Normalize to lists (backward compatible)
                pred_expressions = self._normalize_voting_expressions(pred_expr_raw)
                gold_expressions = self._normalize_voting_expressions(gold_expr_raw)

                if compare_all_expressions:
                    # Compare all expressions as sets
                    pred_expr_set = {e.strip().lower() for e in pred_expressions}
                    gold_expr_set = {e.strip().lower() for e in gold_expressions}
                    expressions_match = (pred_expr_set == gold_expr_set)
                else:
                    # Compare only first items
                    pred_first = pred_expressions[0].strip().lower() if pred_expressions else ""
                    gold_first = gold_expressions[0].strip().lower() if gold_expressions else ""
                    expressions_match = (pred_first == gold_first)

                if not expressions_match:
                    failures['expression_mismatch'] += 1
                    continue

            # All components match
            correct += 1

        accuracy = correct / total if total > 0 else 0.0

        return {
            "complete_event_accuracy": accuracy,
            "complete_matches": correct,
            "total_events": total,
            "failures": failures
        }

    def compute_metrics(
        self,
        pred_events: List[Dict[str, Any]],
        gold_events: List[Dict[str, Any]],
        include_voting_expression: bool = False
    ) -> Dict[str, Any]:
        """
        Compute all event-level evaluation metrics.

        Args:
            pred_events: List of predicted events
            gold_events: List of gold events
            include_voting_expression: Include voting expression in complete event

        Returns:
            Dictionary with comprehensive event-level metrics
        """
        # Event detection
        detection_metrics = self.evaluate_event_detection(pred_events, gold_events)

        # Subject extraction
        subject_metrics = self.evaluate_subject_extraction(pred_events, gold_events)

        # Participant-vote extraction
        participant_metrics = self.evaluate_participant_votes(pred_events, gold_events)

        # Complete event extraction
        complete_metrics = self.evaluate_complete_event(
            pred_events,
            gold_events,
            include_voting_expression=include_voting_expression
        )

        # Combine all metrics
        return {
            **detection_metrics,
            **subject_metrics,
            **participant_metrics,
            **complete_metrics,
            "evaluation_method": "event_level_structured",
            "included_voting_expression": include_voting_expression
        }


# Convenience function
def compute_event_metrics(
    pred_events: List[Dict[str, Any]],
    gold_events: List[Dict[str, Any]],
    include_voting_expression: bool = False
) -> Dict[str, float]:
    """
    Compute event-level metrics.
    
    Args:
        pred_events: List of predicted events
        gold_events: List of gold events
        include_voting_expression: Whether to include voting expression in
                                  complete event evaluation
    
    Returns:
        Dictionary with event-level metrics
    """
    evaluator = EventLevelEvaluator()
    return evaluator.compute_metrics(
        pred_events,
        gold_events,
        include_voting_expression=include_voting_expression
    )