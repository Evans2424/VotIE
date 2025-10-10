#!/usr/bin/env python
"""
Unified NER Evaluation Interface.

This module provides a unified evaluation interface that delegates to specialized
evaluation modules:
- token_metrics: Token-level evaluation using seqeval
- span_metrics: Span-level evaluation with exact match and macro F1
- event_metrics: Event-level evaluation for structured voting events

This file maintains backward compatibility while the actual evaluation logic
has been separated into dedicated modules.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Import specialized evaluators
from .token_metrics import TokenLevelEvaluator
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
        self.token_evaluator = TokenLevelEvaluator(id_to_label)
        self.span_evaluator = SpanLevelEvaluator()

    def convert_ids_to_labels(self, predictions: List[List[int]]) -> List[List[str]]:
        """Convert predicted label IDs to label strings."""
        return self.token_evaluator.convert_ids_to_labels(predictions)

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
        # Token-level metrics (always included)
        token_results = self.token_evaluator.compute_metrics(pred_labels, true_labels)

        # Span-level metrics (optional)
        if include_span_metrics:
            span_results = self.span_evaluator.compute_metrics(
                pred_labels, true_labels, tokens
            )
            
            # Combine results
            combined_results = {**token_results, **span_results}
            combined_results["evaluation_method"] = "token_and_span_level"
        else:
            combined_results = token_results

        return combined_results


# Legacy support for existing code
def compute_seqeval_metrics(
    true_labels: List[List[str]],
    pred_labels: List[List[str]]
) -> Dict[str, float]:
    """
    Legacy function for backward compatibility.
    
    Uses token-level evaluation only.
    """
    evaluator = TokenLevelEvaluator()
    results = evaluator.compute_metrics(pred_labels, true_labels)
    return {
        "precision": results["token_precision"],
        "recall": results["token_recall"],
        "f1": results["token_f1"],
        "accuracy": results["token_accuracy"]
    }