#!/usr/bin/env python
"""
Span-Level NER Evaluation with Exact Match Paradigm.

This module provides span-level evaluation metrics for Named Entity Recognition (NER)
based on exact span matching (both text and entity type must match).
Includes macro-averaged F1 computation across entity types.
"""

import logging
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class SpanLevelEvaluator:
    """
    Span-level NER evaluation using exact match paradigm.

    A predicted span matches a gold span if and only if:
    1. The text is identical (character-level exact match)
    2. The entity type agrees
    """

    def __init__(self):
        """Initialize evaluator."""
        pass

    def extract_spans_from_bio(
        self,
        tokens: List[str],
        labels: List[str]
    ) -> Set[Tuple[int, int, str, str]]:
        """
        Extract spans from BIO-tagged sequence.

        Args:
            tokens: List of tokens
            labels: List of BIO labels

        Returns:
            Set of tuples (start_idx, end_idx, entity_type, text)
        """
        spans = set()
        current_entity = None
        start_idx = None
        entity_tokens = []

        for idx, (token, label) in enumerate(zip(tokens, labels)):
            if label.startswith('B-'):
                # Save previous entity if exists
                if current_entity is not None:
                    text = ' '.join(entity_tokens)
                    spans.add((start_idx, idx - 1, current_entity, text))
                
                # Start new entity
                current_entity = label[2:]
                start_idx = idx
                entity_tokens = [token]
            
            elif label.startswith('I-'):
                entity_type = label[2:]
                if current_entity == entity_type:
                    entity_tokens.append(token)
                else:
                    # Inconsistent tag, treat as new entity
                    if current_entity is not None:
                        text = ' '.join(entity_tokens)
                        spans.add((start_idx, idx - 1, current_entity, text))
                    current_entity = entity_type
                    start_idx = idx
                    entity_tokens = [token]
            
            else:  # 'O' label
                if current_entity is not None:
                    text = ' '.join(entity_tokens)
                    spans.add((start_idx, idx - 1, current_entity, text))
                    current_entity = None
                    start_idx = None
                    entity_tokens = []

        # Don't forget the last entity
        if current_entity is not None:
            text = ' '.join(entity_tokens)
            spans.add((start_idx, len(tokens) - 1, current_entity, text))

        return spans

    def compute_span_metrics(
        self,
        pred_spans: Set[Tuple[int, int, str, str]],
        gold_spans: Set[Tuple[int, int, str, str]]
    ) -> Dict[str, float]:
        """
        Compute precision, recall, and F1 for span sets.

        Args:
            pred_spans: Set of predicted spans
            gold_spans: Set of gold spans

        Returns:
            Dictionary with precision, recall, and f1
        """
        if len(pred_spans) == 0:
            precision = 0.0
        else:
            # Exact match: (start, end, type, text) must all match
            true_positives = len(pred_spans & gold_spans)
            precision = true_positives / len(pred_spans)

        if len(gold_spans) == 0:
            recall = 0.0
        else:
            true_positives = len(pred_spans & gold_spans)
            recall = true_positives / len(gold_spans)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": len(gold_spans)
        }

    def compute_metrics(
        self,
        pred_labels: List[List[str]],
        true_labels: List[List[str]],
        tokens: List[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compute span-level evaluation metrics with macro F1.

        Args:
            pred_labels: Predicted BIO label sequences
            true_labels: Ground truth BIO label sequences
            tokens: List of token sequences (if None, will use indices as tokens)

        Returns:
            Dictionary with comprehensive evaluation metrics including:
            - Overall span-level precision, recall, F1
            - Macro F1 (averaged across entity types)
            - Per-entity-type metrics
        """
        if tokens is None:
            # Generate dummy tokens if not provided
            tokens = [[f"token_{i}" for i in range(len(seq))] for seq in true_labels]

        # Aggregate all spans across all sequences
        all_pred_spans = set()
        all_gold_spans = set()
        
        # Track per-type spans for macro averaging
        per_type_pred_spans = defaultdict(set)
        per_type_gold_spans = defaultdict(set)

        for idx, (token_seq, pred_seq, gold_seq) in enumerate(
            zip(tokens, pred_labels, true_labels)
        ):
            # Extract spans
            pred_spans = self.extract_spans_from_bio(token_seq, pred_seq)
            gold_spans = self.extract_spans_from_bio(token_seq, gold_seq)

            # Add sequence index to make spans unique across sequences
            pred_spans_with_idx = {(idx, *span) for span in pred_spans}
            gold_spans_with_idx = {(idx, *span) for span in gold_spans}

            all_pred_spans.update(pred_spans_with_idx)
            all_gold_spans.update(gold_spans_with_idx)

            # Track per-type for macro averaging
            for span in pred_spans:
                entity_type = span[2]
                per_type_pred_spans[entity_type].add((idx, *span))
            
            for span in gold_spans:
                entity_type = span[2]
                per_type_gold_spans[entity_type].add((idx, *span))

        # Compute overall micro metrics
        overall_metrics = self.compute_span_metrics(all_pred_spans, all_gold_spans)

        # Compute per-type metrics
        all_entity_types = set(per_type_pred_spans.keys()) | set(per_type_gold_spans.keys())
        per_type_metrics = {}
        
        f1_scores = []
        for entity_type in all_entity_types:
            type_pred = per_type_pred_spans.get(entity_type, set())
            type_gold = per_type_gold_spans.get(entity_type, set())
            
            type_metrics = self.compute_span_metrics(type_pred, type_gold)
            per_type_metrics[entity_type] = type_metrics
            
            # Collect F1 for macro averaging
            f1_scores.append(type_metrics["f1"])

        # Compute macro F1 (average of per-type F1 scores)
        macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

        return {
            "span_precision": overall_metrics["precision"],
            "span_recall": overall_metrics["recall"],
            "span_f1": overall_metrics["f1"],
            "span_macro_f1": macro_f1,
            "per_type_metrics": per_type_metrics,
            "total_pred_spans": len(all_pred_spans),
            "total_gold_spans": len(all_gold_spans),
            "evaluation_method": "span_level_exact_match"
        }


# Convenience function
def compute_span_metrics(
    true_labels: List[List[str]],
    pred_labels: List[List[str]],
    tokens: List[List[str]] = None
) -> Dict[str, float]:
    """Compute span-level metrics."""
    evaluator = SpanLevelEvaluator()
    results = evaluator.compute_metrics(pred_labels, true_labels, tokens)
    return {
        "precision": results["span_precision"],
        "recall": results["span_recall"],
        "f1": results["span_f1"],
        "macro_f1": results["span_macro_f1"]
    }
