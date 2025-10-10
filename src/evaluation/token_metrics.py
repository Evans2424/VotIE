#!/usr/bin/env python
"""
Token-Level NER Evaluation using seqeval library.

This module provides token-level evaluation metrics for Named Entity Recognition (NER)
using the seqeval library, which is the standard evaluation framework for sequence
labeling tasks in NLP.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Import seqeval for standard NER evaluation
from seqeval.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from seqeval.scheme import IOB2


class TokenLevelEvaluator:
    """
    Token-level NER evaluation using seqeval library.

    Provides strict entity-level evaluation based on exact boundary matching
    using the IOB2 tagging scheme.
    """

    def __init__(self, id_to_label: Optional[Dict[int, str]] = None):
        """
        Initialize evaluator.

        Args:
            id_to_label: Optional mapping from label IDs to strings
        """
        self.id_to_label = id_to_label or {}

    def convert_ids_to_labels(self, predictions: List[List[int]]) -> List[List[str]]:
        """Convert predicted label IDs to label strings."""
        if not self.id_to_label:
            # Assume predictions are already strings
            return predictions

        return [
            [self.id_to_label.get(pred_id, 'O') for pred_id in pred_seq]
            for pred_seq in predictions
        ]

    def compute_metrics(
        self,
        pred_labels: List[List[str]],
        true_labels: List[List[str]]
    ) -> Dict[str, Any]:
        """
        Compute token-level evaluation metrics using seqeval.

        Args:
            pred_labels: Predicted BIO label sequences
            true_labels: Ground truth BIO label sequences

        Returns:
            Dictionary with comprehensive evaluation metrics including:
            - Overall precision, recall, F1, and accuracy
            - Per-entity-type metrics (precision, recall, F1, support)
        """
        try:
            # Primary metrics
            precision = precision_score(true_labels, pred_labels, scheme=IOB2)
            recall = recall_score(true_labels, pred_labels, scheme=IOB2)
            f1 = f1_score(true_labels, pred_labels, scheme=IOB2)
            accuracy = accuracy_score(true_labels, pred_labels)

            # Per-type metrics using classification_report
            report = classification_report(
                true_labels,
                pred_labels,
                scheme=IOB2,
                output_dict=True
            )
            
            per_type_metrics = {}
            for entity_type, metrics in report.items():
                # Skip overall metrics
                if entity_type not in ['micro avg', 'macro avg', 'weighted avg']:
                    if isinstance(metrics, dict) and 'precision' in metrics:
                        per_type_metrics[entity_type] = {
                            'precision': float(metrics['precision']),
                            'recall': float(metrics['recall']),
                            'f1_score': float(metrics['f1-score']),
                            'support': int(metrics['support'])
                        }

            return {
                "token_precision": float(precision),
                "token_recall": float(recall),
                "token_f1": float(f1),
                "token_accuracy": float(accuracy),
                "per_type_metrics": per_type_metrics,
                "evaluation_method": "seqeval_token_level"
            }

        except Exception as e:
            logger.error(f"Error computing token-level metrics: {e}")
            return {
                "token_precision": 0.0,
                "token_recall": 0.0,
                "token_f1": 0.0,
                "token_accuracy": 0.0,
                "per_type_metrics": {},
                "evaluation_method": "seqeval_token_level"
            }


# Legacy support for existing code
def compute_token_metrics(
    true_labels: List[List[str]],
    pred_labels: List[List[str]]
) -> Dict[str, float]:
    """Legacy function for backward compatibility."""
    evaluator = TokenLevelEvaluator()
    results = evaluator.compute_metrics(pred_labels, true_labels)
    return {
        "precision": results["token_precision"],
        "recall": results["token_recall"],
        "f1": results["token_f1"],
        "accuracy": results["token_accuracy"]
    }
