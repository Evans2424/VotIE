#!/usr/bin/env python3
"""
VotIE Comprehensive Model Evaluation

This module provides end-to-end evaluation of trained Vote Information Extraction (VotIE) models including:
1. Token-level evaluation (using seqeval)
2. Event-level evaluation (structured voting event extraction)

The evaluation pipeline:
    Test Data (JSONL) → Predictions → Token Metrics + Event Construction → Event Metrics → Results

Usage:
    python scripts/evaluate.py --model-path results/bert_crf/best_model --test-data data/votie_bio/test.jsonl
"""

import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import load_jsonl_file
from src.evaluation.token_metrics import TokenLevelEvaluator
from src.evaluation.event_metrics import EventLevelEvaluator
from src.utils.event_constructor import EventConstructor
from scripts.predict import VotIEPredictor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VotIEEvaluator:
    """
    Comprehensive evaluator for Vote Information Extraction (VotIE) models.

    Orchestrates the complete evaluation pipeline:
    1. Load test data
    2. Generate predictions using trained model
    3. Compute token-level metrics
    4. Construct events from predictions
    5. Compute event-level metrics
    6. Save comprehensive results
    """

    def __init__(
        self,
        model_path: str,
        test_data_path: str,
        output_dir: str = None,
        device: str = 'cpu'
    ):
        """
        Initialize evaluator.

        Args:
            model_path: Path to trained model directory
            test_data_path: Path to test JSONL file
            output_dir: Directory to save results (default: model_path/../evaluation)
            device: Device to use ('cpu' or 'cuda')
        """
        self.model_path = Path(model_path)
        self.test_data_path = Path(test_data_path)
        self.device = device

        # Set output directory
        if output_dir is None:
            self.output_dir = self.model_path.parent / 'evaluation'
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        logger.info(f"Loading VotIE model from {self.model_path}")
        self.predictor = VotIEPredictor(str(self.model_path), device=device)

        self.token_evaluator = TokenLevelEvaluator()
        self.event_evaluator = EventLevelEvaluator()
        self.event_constructor = EventConstructor()

        logger.info("Evaluator initialized successfully")

    def load_test_data(self) -> List[Dict[str, Any]]:
        """
        Load test data from JSONL file.

        Returns:
            List of test examples with 'id', 'tokens', and 'labels' fields
        """
        logger.info(f"Loading test data from {self.test_data_path}")
        test_data = load_jsonl_file(self.test_data_path)
        logger.info(f"Loaded {len(test_data)} test examples")
        return test_data

    def generate_predictions(
        self,
        test_data: List[Dict[str, Any]]
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Generate predictions for all test examples.

        Args:
            test_data: List of test examples

        Returns:
            Tuple of (predictions, gold_labels) where each is a list of label sequences
        """
        logger.info(f"Generating predictions for {len(test_data)} test examples")

        predictions = []
        gold_labels = []

        for i, example in enumerate(test_data):
            # Progress logging every 50 examples
            if (i + 1) % 50 == 0:
                logger.info(f"Progress: {i + 1}/{len(test_data)} ({100*(i+1)/len(test_data):.1f}%)")

            tokens = example['tokens']
            labels = example['labels']

            # Generate predictions
            pred_labels = self.predictor.predict(tokens)

            predictions.append(pred_labels)
            gold_labels.append(labels)

        logger.info(f"Completed: {len(predictions)}/{len(test_data)} examples (100.0%)")
        return predictions, gold_labels

    def evaluate_token_level(
        self,
        predictions: List[List[str]],
        gold_labels: List[List[str]]
    ) -> Dict[str, Any]:
        """
        Compute token-level evaluation metrics.

        Args:
            predictions: List of predicted label sequences
            gold_labels: List of gold label sequences

        Returns:
            Dictionary with token-level metrics
        """
        logger.info("Computing token-level metrics")
        token_metrics = self.token_evaluator.compute_metrics(predictions, gold_labels)

        logger.info(f"Token-level F1: {token_metrics['token_f1']:.4f}")
        logger.info(f"Token-level Precision: {token_metrics['token_precision']:.4f}")
        logger.info(f"Token-level Recall: {token_metrics['token_recall']:.4f}")

        return token_metrics

    def construct_events(
        self,
        test_data: List[Dict[str, Any]],
        predictions: List[List[str]],
        gold_labels: List[List[str]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Construct structured events from predictions and gold labels.

        Args:
            test_data: List of test examples (for tokens and IDs)
            predictions: List of predicted label sequences
            gold_labels: List of gold label sequences

        Returns:
            Tuple of (predicted_events, gold_events)
        """
        logger.info("Constructing events from predictions")

        # Prepare examples for event construction
        pred_examples = [
            {'id': ex['id'], 'tokens': ex['tokens']}
            for ex in test_data
        ]
        gold_examples = [
            {'id': ex['id'], 'tokens': ex['tokens']}
            for ex in test_data
        ]

        # Construct events
        pred_events = self.event_constructor.construct_events(pred_examples, predictions)
        gold_events = self.event_constructor.construct_events(gold_examples, gold_labels)

        # Count events
        pred_event_count = sum(1 for e in pred_events if e.get('has_voting_event', False))
        gold_event_count = sum(1 for e in gold_events if e.get('has_voting_event', False))

        logger.info(f"Predicted events: {pred_event_count}/{len(pred_events)}")
        logger.info(f"Gold events: {gold_event_count}/{len(gold_events)}")

        return pred_events, gold_events

    def evaluate_event_level(
        self,
        pred_events: List[Dict[str, Any]],
        gold_events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute event-level evaluation metrics.

        Args:
            pred_events: List of predicted events
            gold_events: List of gold events

        Returns:
            Dictionary with event-level metrics
        """
        logger.info("Computing event-level metrics")
        event_metrics = self.event_evaluator.compute_metrics(pred_events, gold_events)

        logger.info(f"Event detection accuracy: {event_metrics['event_detection_accuracy']:.4f}")
        logger.info(f"Participant-vote F1: {event_metrics['participant_vote_f1']:.4f}")
        logger.info(f"Subject F1: {event_metrics['subject_f1']:.4f}")
        logger.info(f"Complete event accuracy: {event_metrics['complete_event_accuracy']:.4f}")

        return event_metrics

    def save_results(
        self,
        results: Dict[str, Any],
        predictions: List[List[str]],
        pred_events: List[Dict[str, Any]],
        test_data: List[Dict[str, Any]]
    ):
        """
        Save evaluation results to files.

        Args:
            results: Complete evaluation results
            predictions: Token-level predictions
            pred_events: Event-level predictions
            test_data: Original test data
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save comprehensive results
        results_file = self.output_dir / f"evaluation_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved evaluation results to {results_file}")

        # Save predictions for error analysis
        predictions_file = self.output_dir / f"predictions_{timestamp}.jsonl"
        with open(predictions_file, 'w', encoding='utf-8') as f:
            for example, pred_labels in zip(test_data, predictions):
                output = {
                    'id': example['id'],
                    'tokens': example['tokens'],
                    'gold_labels': example['labels'],
                    'pred_labels': pred_labels
                }
                f.write(json.dumps(output, ensure_ascii=False) + '\n')
        logger.info(f"Saved predictions to {predictions_file}")

        # Save structured events for event-level analysis
        events_file = self.output_dir / f"predicted_events_{timestamp}.json"
        with open(events_file, 'w', encoding='utf-8') as f:
            json.dump(pred_events, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved predicted events to {events_file}")

    def evaluate(self) -> Dict[str, Any]:
        """
        Run complete VotIE evaluation pipeline.

        Returns:
            Dictionary with comprehensive evaluation results
        """
        logger.info("=" * 80)
        logger.info("Starting VotIE comprehensive evaluation")
        logger.info("=" * 80)

        # Step 1: Load test data
        test_data = self.load_test_data()

        # Step 2: Generate predictions
        predictions, gold_labels = self.generate_predictions(test_data)

        # Step 3: Token-level evaluation
        token_metrics = self.evaluate_token_level(predictions, gold_labels)

        # Step 4: Construct events
        pred_events, gold_events = self.construct_events(test_data, predictions, gold_labels)

        # Step 5: Event-level evaluation
        event_metrics = self.evaluate_event_level(pred_events, gold_events)

        # Compile complete results
        results = {
            'metadata': {
                'model_path': str(self.model_path),
                'test_data_path': str(self.test_data_path),
                'timestamp': datetime.now().isoformat(),
                'num_test_examples': len(test_data),
                'device': self.device
            },
            'token_level_metrics': token_metrics,
            'event_level_metrics': event_metrics
        }

        # Step 6: Save results
        self.save_results(results, predictions, pred_events, test_data)

        logger.info("=" * 80)
        logger.info("Evaluation complete")
        logger.info("=" * 80)

        return results


def main():
    """Command-line interface for VotIE model evaluation."""
    parser = argparse.ArgumentParser(
        description='VotIE: Comprehensive evaluation with token and event-level metrics'
    )

    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained model directory (e.g., results/bert_crf/best_model)'
    )

    parser.add_argument(
        '--test-data',
        type=str,
        required=True,
        help='Path to test JSONL file (e.g., data/votie_bio/test.jsonl)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save evaluation results (default: model_path/../evaluation)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use for inference (default: cpu)'
    )

    args = parser.parse_args()

    # Create VotIE evaluator
    evaluator = VotIEEvaluator(
        model_path=args.model_path,
        test_data_path=args.test_data,
        output_dir=args.output_dir,
        device=args.device
    )

    # Run evaluation
    results = evaluator.evaluate()

    # Print summary
    print("\n" + "=" * 80)
    print("VotIE EVALUATION SUMMARY")
    print("=" * 80)
    print(f"\nModel: {args.model_path}")
    print(f"Test Data: {args.test_data}")
    print(f"Test Examples: {results['metadata']['num_test_examples']}")

    print("\nToken-Level Metrics:")
    print(f"  Precision: {results['token_level_metrics']['token_precision']:.4f}")
    print(f"  Recall:    {results['token_level_metrics']['token_recall']:.4f}")
    print(f"  F1 Score:  {results['token_level_metrics']['token_f1']:.4f}")
    print(f"  Accuracy:  {results['token_level_metrics']['token_accuracy']:.4f}")

    print("\nEvent-Level Metrics:")
    print(f"  Event Detection Accuracy:   {results['event_level_metrics']['event_detection_accuracy']:.4f}")
    print(f"  Participant-Vote F1:        {results['event_level_metrics']['participant_vote_f1']:.4f}")
    print(f"  Subject Extraction F1:      {results['event_level_metrics']['subject_f1']:.4f}")
    print(f"  Complete Event Accuracy:    {results['event_level_metrics']['complete_event_accuracy']:.4f}")

    print("\n" + "=" * 80)
    print(f"Results saved to: {evaluator.output_dir}")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
