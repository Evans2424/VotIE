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
        Evaluate binary event detection with F1 score.
        
        As per paper: Binary classification of segments with/without voting events.
        Uses standard precision, recall, and F1 for the positive class (has_voting_event=True).

        Args:
            pred_events: List of predicted events (one per segment)
            gold_events: List of gold events (one per segment)

        Returns:
            Dictionary with F1, precision, recall, and counts
        """
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for pred_event, gold_event in zip(pred_events, gold_events):
            # Handle None events
            pred_has_event = pred_event.get('has_voting_event', False) if pred_event else False
            gold_has_event = gold_event.get('has_voting_event', False) if gold_event else False

            if pred_has_event and gold_has_event:
                true_positives += 1
            elif pred_has_event and not gold_has_event:
                false_positives += 1
            elif not pred_has_event and gold_has_event:
                false_negatives += 1
            # true_negatives not needed for F1 calculation

        # Calculate precision, recall, F1
        if true_positives + false_positives == 0:
            precision = 0.0
        else:
            precision = true_positives / (true_positives + false_positives)

        if true_positives + false_negatives == 0:
            recall = 0.0
        else:
            recall = true_positives / (true_positives + false_negatives)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return {
            "event_detection_f1": f1
        }

    def evaluate_subject_extraction(
        self,
        pred_events: List[Dict[str, Any]],
        gold_events: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Evaluate subject extraction at span level with F1 score.
        
        As per paper: "Subject and counting extraction are evaluated at the span level"
        Computes F1 treating each event with a voting event as a binary classification:
        - TP: Both have subject and they match
        - FP: Predicted subject exists but doesn't match gold (or gold has no subject)
        - FN: Gold subject exists but pred doesn't match (or pred has no subject)

        Args:
            pred_events: List of predicted events
            gold_events: List of gold events

        Returns:
            Dictionary with F1, precision, recall
        """
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for pred_event, gold_event in zip(pred_events, gold_events):
            # Skip if no event structure (handle None pred_event or gold_event)
            if pred_event is None or gold_event is None:
                continue
                
            if not gold_event.get('has_voting_event', False):
                continue

            # Extract subject (single string) - safely handle None event dicts
            pred_event_data = pred_event.get('event') if pred_event else None
            gold_event_data = gold_event.get('event') if gold_event else None
            
            # Handle case where event is None (no voting event)
            pred_subject = pred_event_data.get('subject') if pred_event_data and isinstance(pred_event_data, dict) else None
            gold_subject = gold_event_data.get('subject') if gold_event_data and isinstance(gold_event_data, dict) else None

            # Normalize and compare (handle None values) - span-level matching
            pred_subject_norm = pred_subject.strip().lower() if pred_subject else ''
            gold_subject_norm = gold_subject.strip().lower() if gold_subject else ''

            # Exact match at span level
            has_gold = bool(gold_subject_norm)
            has_pred = bool(pred_subject_norm)
            match = pred_subject_norm == gold_subject_norm

            if has_gold and has_pred and match:
                true_positives += 1
            elif has_pred and not match:
                false_positives += 1
            elif has_gold and not match:
                false_negatives += 1

        # Calculate precision, recall, F1
        if true_positives + false_positives == 0:
            precision = 0.0
        else:
            precision = true_positives / (true_positives + false_positives)

        if true_positives + false_negatives == 0:
            recall = 0.0
        else:
            recall = true_positives / (true_positives + false_negatives)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return {
            "subject_f1": f1
        }

    def evaluate_counting_extraction(
        self,
        pred_events: List[Dict[str, Any]],
        gold_events: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Evaluate counting extraction at span level with F1 score.
        
        As per paper: "Subject and counting extraction are evaluated at the span level"
        Compares the first counting expression (text + type).
        
        Computes F1 treating each event with a voting event as a binary classification:
        - TP: Both have counting and they match (text + type)
        - FP: Predicted counting exists but doesn't match gold (or gold has no counting)
        - FN: Gold counting exists but pred doesn't match (or pred has no counting)

        Args:
            pred_events: List of predicted events
            gold_events: List of gold events

        Returns:
            Dictionary with F1, precision, recall
        """
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for pred_event, gold_event in zip(pred_events, gold_events):
            if pred_event is None or gold_event is None:
                continue
            if not gold_event.get('has_voting_event', False):
                continue

            # Handle None event structures
            pred_ev = pred_event.get('event') or {}
            gold_ev = gold_event.get('event') or {}

            # Get first counting expression
            pred_counting_raw = pred_ev.get('counting')
            gold_counting_raw = gold_ev.get('counting')

            pred_counting = self._normalize_counting(pred_counting_raw)
            gold_counting = self._normalize_counting(gold_counting_raw)

            pred_first = pred_counting[0] if pred_counting else None
            gold_first = gold_counting[0] if gold_counting else None

            # Compare at span level (text + type)
            has_gold = gold_first is not None
            has_pred = pred_first is not None
            match = False

            if pred_first is not None and gold_first is not None:
                pred_text = pred_first.get('text', '').strip().lower()
                pred_type = pred_first.get('type', '').strip().upper()
                gold_text = gold_first.get('text', '').strip().lower()
                gold_type = gold_first.get('type', '').strip().upper()
                match = (pred_text == gold_text and pred_type == gold_type)
            elif pred_first is None and gold_first is None:
                # Both empty - true negative, not counted in F1
                continue

            if has_gold and has_pred and match:
                true_positives += 1
            elif has_pred and not match:
                false_positives += 1
            elif has_gold and not match:
                false_negatives += 1

        # Calculate precision, recall, F1
        if true_positives + false_positives == 0:
            precision = 0.0
        else:
            precision = true_positives / (true_positives + false_positives)

        if true_positives + false_negatives == 0:
            recall = 0.0
        else:
            recall = true_positives / (true_positives + false_negatives)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return {
            "counting_f1": f1
        }

    def evaluate_participant_votes(
        self,
        pred_events: List[Dict[str, Any]],
        gold_events: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Evaluate participant-vote extraction with set-based F1.
        
        As per paper: "participant-vote extraction F1, computed over the 
        set of (participant, position) tuples"

        Args:
            pred_events: List of predicted events
            gold_events: List of gold events

        Returns:
            Dictionary with aggregated F1, precision, and recall across all events
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
            "participant_vote_f1": sum(all_f1s) / len(all_f1s)
        }

    def evaluate_complete_event(
        self,
        pred_events: List[Dict[str, Any]],
        gold_events: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Evaluate complete event extraction with F1 score.
        
        As per paper: "strict event-level accuracy, where an event is correct 
        only if all components match the gold annotation"
        
        Treats each voting event as a binary classification:
        - TP: Event has voting event and ALL components match exactly
        - FP: Predicted as voting event but components don't all match
        - FN: Gold has voting event but prediction doesn't match all components
        
        Required matches:
        - Subject (exact span match)
        - All participant-vote pairs (set match)
        - Counting expression (exact span + type match)

        Args:
            pred_events: List of predicted events
            gold_events: List of gold events

        Returns:
            Dictionary with F1, precision, recall, and detailed failure counts
        """
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        failures = {
            'subject_mismatch': 0,
            'participants_mismatch': 0,
            'counting_mismatch': 0
        }

        for pred_event, gold_event in zip(pred_events, gold_events):
            if pred_event is None or gold_event is None:
                continue
            
            # Get voting event flags
            pred_has_event = pred_event.get('has_voting_event', False)
            gold_has_event = gold_event.get('has_voting_event', False)
            
            # Skip segments without gold voting events
            if not gold_has_event:
                # But count false positives if we predicted an event where there is none
                if pred_has_event:
                    false_positives += 1
                continue

            # Gold has a voting event - check if prediction matches all components
            # Check all components - handle None event structures (event can be None)
            pred_ev = pred_event.get('event') or {}
            gold_ev = gold_event.get('event') or {}

            # If pred doesn't have an event but gold does, it's a false negative
            if not pred_has_event:
                false_negatives += 1
                continue

            # Both have events - check all components
            # 1. Compare subjects (single string)
            pred_subject = pred_ev.get('subject', None)
            gold_subject = gold_ev.get('subject', None)

            pred_subject_norm = pred_subject.strip().lower() if pred_subject else None
            gold_subject_norm = gold_subject.strip().lower() if gold_subject else None

            if pred_subject_norm != gold_subject_norm:
                failures['subject_mismatch'] += 1
                false_negatives += 1
                continue

            # 2. Compare participant-vote pairs (set)
            pred_pairs = self.extract_participant_vote_pairs(pred_ev)
            gold_pairs = self.extract_participant_vote_pairs(gold_ev)

            if pred_pairs != gold_pairs:
                failures['participants_mismatch'] += 1
                false_negatives += 1
                continue

            # 3. Compare counting (span-level: text + type)
            pred_counting_raw = pred_ev.get('counting')
            gold_counting_raw = gold_ev.get('counting')

            # Normalize to lists (backward compatible)
            pred_counting = self._normalize_counting(pred_counting_raw)
            gold_counting = self._normalize_counting(gold_counting_raw)

            # Compare first counting expression (as per paper: span-level matching)
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
                false_negatives += 1
                continue

            # All components match - true positive
            true_positives += 1

        # Calculate precision, recall, F1
        if true_positives + false_positives == 0:
            precision = 0.0
        else:
            precision = true_positives / (true_positives + false_positives)

        if true_positives + false_negatives == 0:
            recall = 0.0
        else:
            recall = true_positives / (true_positives + false_negatives)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return {
            "complete_event_f1": f1
        }

    def compute_metrics(
        self,
        pred_events: List[Dict[str, Any]],
        gold_events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute all event-level evaluation metrics as per paper (F1 scores).
        
        All metrics use F1 score (precision, recall, F1) instead of accuracy:
        1. Event detection F1 (binary classification per segment)
        2. Subject extraction F1 (span-level exact matching)
        3. Counting extraction F1 (span-level exact matching with type)
        4. Participant-vote F1 (set-based over tuples)
        5. Complete event F1 (strict - all components must match)

        Args:
            pred_events: List of predicted events
            gold_events: List of gold events

        Returns:
            Dictionary with comprehensive F1-based metrics
        """
        # Event detection F1
        detection_metrics = self.evaluate_event_detection(pred_events, gold_events)

        # Subject extraction F1 (span-level)
        subject_metrics = self.evaluate_subject_extraction(pred_events, gold_events)

        # Counting extraction F1 (span-level)
        counting_metrics = self.evaluate_counting_extraction(pred_events, gold_events)

        # Participant-vote extraction F1 (set-based)
        participant_metrics = self.evaluate_participant_votes(pred_events, gold_events)

        # Complete event extraction F1 (strict)
        complete_metrics = self.evaluate_complete_event(pred_events, gold_events)

        # Combine all metrics (only 5 key F1 scores)
        return {
            "event_detection_f1": detection_metrics["event_detection_f1"],
            "subject_f1": subject_metrics["subject_f1"],
            "counting_f1": counting_metrics["counting_f1"],
            "participant_vote_f1": participant_metrics["participant_vote_f1"],
            "complete_event_f1": complete_metrics["complete_event_f1"]
        }


# Convenience function
def compute_event_metrics(
    pred_events: List[Dict[str, Any]],
    gold_events: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Compute event-level metrics as per paper.
    
    Args:
        pred_events: List of predicted events
        gold_events: List of gold events
    
    Returns:
        Dictionary with event-level metrics
    """
    evaluator = EventLevelEvaluator()
    return evaluator.compute_metrics(pred_events, gold_events)