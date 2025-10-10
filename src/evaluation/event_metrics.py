#!/usr/bin/env python
"""
Event-Level Evaluation for Voting Event Extraction.

This module provides event-level evaluation metrics for structured voting event
extraction, including:
1. Event Detection: Binary accuracy for identifying segments with voting events
2. Participant-Vote Extraction: Set-based F1 over (participant, position) tuples
3. Complete Event Extraction: Strict accuracy where all components must match

Span matching uses text overlap: predicted and gold spans match if their text
is identical and types agree.
"""

import logging
from typing import List, Dict, Any, Set, Tuple, Optional

logger = logging.getLogger(__name__)


class EventLevelEvaluator:
    """
    Event-level evaluation for voting event extraction.

    Evaluates structured event extraction including:
    - Event detection (binary classification)
    - Participant-vote pair extraction (set-based F1)
    - Subject and counting extraction (span-level)
    - Complete event extraction (strict match)
    """

    def __init__(self):
        """Initialize evaluator."""
        pass

    def extract_span_text(self, span: Dict[str, Any]) -> str:
        """
        Extract normalized text from span.

        Args:
            span: Span dictionary with 'text' field

        Returns:
            Normalized span text
        """
        return span.get('text', '').strip().lower()

    def spans_match(self, pred_span: Dict[str, Any], gold_span: Dict[str, Any]) -> bool:
        """
        Check if two spans match (text and type must be identical).

        Args:
            pred_span: Predicted span
            gold_span: Gold span

        Returns:
            True if spans match
        """
        pred_text = self.extract_span_text(pred_span)
        gold_text = self.extract_span_text(gold_span)
        pred_label = pred_span.get('label', '').upper()
        gold_label = gold_span.get('label', '').upper()

        return pred_text == gold_text and pred_label == gold_label

    def extract_participant_vote_pairs(
        self,
        event: Dict[str, Any]
    ) -> Set[Tuple[str, str]]:
        """
        Extract (participant, position) tuples from event.

        Args:
            event: Event dictionary with 'participants' field

        Returns:
            Set of (participant_text, position) tuples
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
        This handles cases where events legitimately have no participants/elements.

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
        
        true_positives = 0  # Correctly detected events
        false_positives = 0  # Predicted event when there was none
        true_negatives = 0  # Correctly identified no event
        false_negatives = 0  # Missed an event

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
            # Only evaluate events that exist in gold
            if not gold_event.get('has_voting_event', False):
                continue

            pred_pairs = self.extract_participant_vote_pairs(pred_event.get('event', {}))
            gold_pairs = self.extract_participant_vote_pairs(gold_event.get('event', {}))

            metrics = self.compute_set_based_f1(pred_pairs, gold_pairs)
            all_precisions.append(metrics['precision'])
            all_recalls.append(metrics['recall'])
            all_f1s.append(metrics['f1'])

        # Aggregate across all events
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

    def evaluate_subject_extraction(
        self,
        pred_events: List[Dict[str, Any]],
        gold_events: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Evaluate subject extraction (span-level exact match).

        Args:
            pred_events: List of predicted events
            gold_events: List of gold events

        Returns:
            Dictionary with precision, recall, and F1
        """
        correct = 0
        pred_count = 0
        gold_count = 0

        for pred_event, gold_event in zip(pred_events, gold_events):
            if not gold_event.get('has_voting_event', False):
                continue

            pred_subjects = pred_event.get('event', {}).get('subjects', [])
            gold_subjects = gold_event.get('event', {}).get('subjects', [])

            gold_count += len(gold_subjects)
            pred_count += len(pred_subjects)

            # Normalize and compare
            gold_subjects_normalized = {s.strip().lower() for s in gold_subjects}
            
            for pred_subj in pred_subjects:
                pred_normalized = pred_subj.strip().lower()
                if pred_normalized in gold_subjects_normalized:
                    correct += 1

        precision = correct / pred_count if pred_count > 0 else 0.0
        recall = correct / gold_count if gold_count > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "subject_precision": precision,
            "subject_recall": recall,
            "subject_f1": f1
        }

    def evaluate_complete_event(
        self,
        pred_events: List[Dict[str, Any]],
        gold_events: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Evaluate complete event extraction (strict accuracy).

        All components must match:
        - Subject
        - All participant-vote pairs
        - Counting information

        Args:
            pred_events: List of predicted events
            gold_events: List of gold events

        Returns:
            Dictionary with strict accuracy
        """
        correct = 0
        total = 0

        for pred_event, gold_event in zip(pred_events, gold_events):
            if not gold_event.get('has_voting_event', False):
                continue

            total += 1

            # Check all components
            pred_ev = pred_event.get('event', {})
            gold_ev = gold_event.get('event', {})

            # Compare subjects
            pred_subjects = set(s.strip().lower() for s in pred_ev.get('subjects', []))
            gold_subjects = set(s.strip().lower() for s in gold_ev.get('subjects', []))
            
            if pred_subjects != gold_subjects:
                continue

            # Compare participant-vote pairs
            pred_pairs = self.extract_participant_vote_pairs(pred_ev)
            gold_pairs = self.extract_participant_vote_pairs(gold_ev)
            
            if pred_pairs != gold_pairs:
                continue

            # Compare counting
            pred_counting = set()
            for c in pred_ev.get('counting', []):
                text = c.get('text', '').strip().lower()
                ctype = c.get('type', '').strip().upper()
                pred_counting.add((text, ctype))
            
            gold_counting = set()
            for c in gold_ev.get('counting', []):
                text = c.get('text', '').strip().lower()
                ctype = c.get('type', '').strip().upper()
                gold_counting.add((text, ctype))
            
            if pred_counting != gold_counting:
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
        gold_events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute all event-level evaluation metrics.

        Args:
            pred_events: List of predicted events
            gold_events: List of gold events

        Returns:
            Dictionary with comprehensive event-level metrics
        """
        # Event detection
        detection_metrics = self.evaluate_event_detection(pred_events, gold_events)

        # Participant-vote extraction
        participant_metrics = self.evaluate_participant_votes(pred_events, gold_events)

        # Subject extraction
        subject_metrics = self.evaluate_subject_extraction(pred_events, gold_events)

        # Complete event extraction
        complete_metrics = self.evaluate_complete_event(pred_events, gold_events)

        # Combine all metrics
        return {
            **detection_metrics,
            **participant_metrics,
            **subject_metrics,
            **complete_metrics,
            "evaluation_method": "event_level_structured"
        }


# Convenience function
def compute_event_metrics(
    pred_events: List[Dict[str, Any]],
    gold_events: List[Dict[str, Any]]
) -> Dict[str, float]:
    """Compute event-level metrics."""
    evaluator = EventLevelEvaluator()
    return evaluator.compute_metrics(pred_events, gold_events)
