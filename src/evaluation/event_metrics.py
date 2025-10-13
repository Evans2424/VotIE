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

    Evaluates structured event extraction with the following event schema:
    {
        'has_voting_event': bool,
        'event': {
            'subject': str,  # Single string, not list
            'participants': [{'text': str, 'position': str}, ...],
            'counting': {'text': str, 'type': str} or None,
            'voting_expression': str or None
        }
    }
    """

    def __init__(self):
        """Initialize evaluator."""
        pass

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
            pred_has_event = pred_event.get('has_voting_event', False)
            gold_has_event = gold_event.get('has_voting_event', False)

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
        Evaluate subject extraction (exact string matching).
        
        Each event has exactly one subject (single string), so we use
        exact matching rather than set-based F1.

        Args:
            pred_events: List of predicted events
            gold_events: List of gold events

        Returns:
            Dictionary with accuracy
        """
        correct = 0
        total = 0

        for pred_event, gold_event in zip(pred_events, gold_events):
            if not gold_event.get('has_voting_event', False):
                continue

            total += 1

            # Extract subject (single string)
            pred_subject = pred_event.get('event', {}).get('subject', '')
            gold_subject = gold_event.get('event', {}).get('subject', '')

            # Normalize and compare
            pred_subject = pred_subject.strip().lower()
            gold_subject = gold_subject.strip().lower()

            if pred_subject == gold_subject:
                correct += 1

        accuracy = correct / total if total > 0 else 0.0

        return {
            "subject_accuracy": accuracy,
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
            if not gold_event.get('has_voting_event', False):
                continue

            pred_pairs = self.extract_participant_vote_pairs(pred_event.get('event', {}))
            gold_pairs = self.extract_participant_vote_pairs(gold_event.get('event', {}))

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

    def compare_counting(self, pred_event: Dict, gold_event: Dict) -> bool:
        """
        Compare counting information (single dictionary or None).
        
        Args:
            pred_event: Predicted event
            gold_event: Gold event
        
        Returns:
            True if counting matches exactly
        """
        pred_counting = pred_event.get('counting')
        gold_counting = gold_event.get('counting')
        
        # Both None = match
        if pred_counting is None and gold_counting is None:
            return True
        
        # One None, one not = no match
        if pred_counting is None or gold_counting is None:
            return False
        
        # Compare text and type
        pred_text = pred_counting.get('text', '').strip().lower()
        gold_text = gold_counting.get('text', '').strip().lower()
        pred_type = pred_counting.get('type', '').strip().upper()
        gold_type = gold_counting.get('type', '').strip().upper()
        
        return pred_text == gold_text and pred_type == gold_type

    def evaluate_complete_event(
        self,
        pred_events: List[Dict[str, Any]],
        gold_events: List[Dict[str, Any]],
        include_voting_expression: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate complete event extraction (strict accuracy).

        All components must match exactly:
        - Subject (exact string match)
        - All participant-vote pairs (set match)
        - Counting information (exact match)
        - Voting expression (exact match, if include_voting_expression=True)

        Args:
            pred_events: List of predicted events
            gold_events: List of gold events
            include_voting_expression: Whether to require voting expression match

        Returns:
            Dictionary with strict accuracy
        """
        correct = 0
        total = 0

        for pred_event, gold_event in zip(pred_events, gold_events):
            if not gold_event.get('has_voting_event', False):
                continue

            total += 1

            pred_ev = pred_event.get('event', {})
            gold_ev = gold_event.get('event', {})

            # Compare subject (single string)
            pred_subject = pred_ev.get('subject', '').strip().lower()
            gold_subject = gold_ev.get('subject', '').strip().lower()
            
            if pred_subject != gold_subject:
                continue

            # Compare participant-vote pairs (set)
            pred_pairs = self.extract_participant_vote_pairs(pred_ev)
            gold_pairs = self.extract_participant_vote_pairs(gold_ev)
            
            if pred_pairs != gold_pairs:
                continue

            # Compare counting (single dict or None)
            if not self.compare_counting(pred_ev, gold_ev):
                continue

            # Optionally compare voting expression
            if include_voting_expression:
                pred_expr = pred_ev.get('voting_expression', '').strip().lower()
                gold_expr = gold_ev.get('voting_expression', '').strip().lower()
                
                if pred_expr != gold_expr:
                    continue

            # All components match
            correct += 1

        accuracy = correct / total if total > 0 else 0.0

        return {
            "complete_event_accuracy": accuracy,
            "complete_matches": correct,
            "total_events": total
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