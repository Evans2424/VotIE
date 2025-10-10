#!/usr/bin/env python
"""
Unified NER Evaluation Interface.

This module provides a unified evaluation interface that delegates to specialized
evaluation modules:
- entity_metrics: Entity-level evaluation using seqeval
- span_metrics: Span-level evaluation with exact match and macro F1
- event_metrics: Event-level evaluation for structured voting events

This file maintains backward compatibility while the actual evaluation logic
has been separated into dedicated modules.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Import specialized evaluators
from .entity_metrics import EntityLevelEvaluator
from .span_metrics import SpanLevelEvaluator
from .event_metrics import EventLevelEvaluator


class NEREvaluator:
    """
    Unified NER evaluation interface that delegates to specialized evaluators.

    This class maintains backward compatibility while using the new modular
    evaluation structure.
    """

    def __init__(self, id_to_label: Optional[Dict[int, str]] = None):
        """
        Initialize evaluator.

        Args:
            id_to_label: Optional mapping from label IDs to strings
        """
        self.id_to_label = id_to_label or {}
        self.entity_evaluator = EntityLevelEvaluator(id_to_label)
        self.span_evaluator = SpanLevelEvaluator()

    def convert_ids_to_labels(self, predictions: List[List[int]]) -> List[List[str]]:
        """Convert predicted label IDs to label strings."""
        return self.entity_evaluator.convert_ids_to_labels(predictions)

    def compute_metrics(
        self,
        pred_labels: List[List[str]],
        true_labels: List[List[str]],
        tokens: List[List[str]] = None,
        include_span_metrics: bool = True
    ) -> Dict[str, Any]:
        """
        Compute comprehensive evaluation metrics.

        Args:
            pred_labels: Predicted BIO label sequences
            true_labels: Ground truth BIO label sequences
            tokens: Optional token sequences for span evaluation
            include_span_metrics: Whether to include span-level metrics

        Returns:
            Dictionary with comprehensive evaluation metrics
        """
        # Entity-level metrics (always included)
        entity_results = self.entity_evaluator.compute_metrics(pred_labels, true_labels)

        # Span-level metrics (optional)
        if include_span_metrics:
            span_results = self.span_evaluator.compute_metrics(
                pred_labels, true_labels, tokens
            )
            
            # Combine results
            combined_results = {**entity_results, **span_results}
            combined_results["evaluation_method"] = "entity_and_span_level"
        else:
            combined_results = entity_results

        # Map keys to match trainer expectations
        # The trainer expects: entity_f1_strict, precision, recall, accuracy
        # Entity evaluator returns: entity_f1, entity_precision, entity_recall, entity_accuracy
        return {
            'entity_f1_strict': combined_results.get('entity_f1', 0.0),
            'precision': combined_results.get('entity_precision', 0.0),
            'recall': combined_results.get('entity_recall', 0.0),
            'accuracy': combined_results.get('entity_accuracy', 0.0),
            'entity_accuracy': combined_results.get('entity_accuracy', 0.0),
            'evaluation_method': combined_results.get('evaluation_method', 'seqeval'),
            'per_type_metrics': combined_results.get('per_type_metrics', {})
        }


# Legacy support for existing code
def compute_seqeval_metrics(
    true_labels: List[List[str]],
    pred_labels: List[List[str]]
) -> Dict[str, float]:
    """
    Legacy function for backward compatibility.
    
    Uses entity-level evaluation only.
    """
    evaluator = EntityLevelEvaluator()
    results = evaluator.compute_metrics(pred_labels, true_labels)
    return {
        "precision": results["entity_precision"],
        "recall": results["entity_recall"],
        "f1": results["entity_f1"],
        "accuracy": results["entity_accuracy"]
    }