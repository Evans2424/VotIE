#!/usr/bin/env python3
"""
Evaluate Trained Models on English Test Set

Simple evaluation script that uses the SimpleNERPredictor to evaluate
trained models on the English test set and save results to JSON.
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import torch

from simple_predict import SimpleNERPredictor
from core.data.dataset import load_jsonl_file
from core.evaluation.metrics import NEREvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_model(model_path: str, test_file: str, device: str = 'cpu') -> dict:
    """
    Evaluate a single model on the test set.

    Args:
        model_path: Path to trained model
        test_file: Path to test JSONL file
        device: Device to use

    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"Loading model from {model_path}")
    predictor = SimpleNERPredictor(model_path, device=device)

    logger.info(f"Loading test data from {test_file}")
    test_data = load_jsonl_file(Path(test_file))
    logger.info(f"Loaded {len(test_data)} test examples")
    
    # Clear GPU cache if using CUDA
    if device == 'cuda':
        torch.cuda.empty_cache()

    # Run predictions
    all_predictions = []
    all_labels = []

    logger.info("Running predictions...")
    for example in tqdm(test_data, desc="Evaluating"):
        tokens = example['tokens']
        labels = example['labels']

        try:
            predictions = predictor.predict(tokens)
            all_predictions.append(predictions)
            all_labels.append(labels)
        except Exception as e:
            logger.error(f"Error predicting example: {e}")
            # Use all O labels as fallback
            all_predictions.append(['O'] * len(tokens))
            all_labels.append(labels)

    # Compute metrics
    logger.info("Computing metrics...")
    evaluator = NEREvaluator(id_to_label=None)
    metrics = evaluator.compute_metrics(all_predictions, all_labels)

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained model on English test set'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained model directory'
    )
    parser.add_argument(
        '--test_file',
        type=str,
        default='../data/publication_datasets/ner_datasets/en/splits/test.jsonl',
        help='Path to test JSONL file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='english_eval_results.json',
        help='Output JSON file for results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cpu', 'cuda'],
        help='Device to use for evaluation'
    )

    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent
    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = script_dir / model_path

    test_file = Path(args.test_file)
    if not test_file.is_absolute():
        test_file = script_dir / test_file

    output_file = Path(args.output)
    if not output_file.is_absolute():
        output_file = script_dir / output_file

    # Log device being used
    logger.info(f"Using device: {args.device}")
    if args.device == 'cuda' and torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
    
    # Evaluate
    metrics = evaluate_model(str(model_path), str(test_file), args.device)

    # Prepare results
    results = {
        'metadata': {
            'evaluation_date': datetime.now().isoformat(),
            'model_path': str(model_path),
            'test_file': str(test_file),
            'device': args.device
        },
        'metrics': metrics
    }

    # Save results
    logger.info(f"Saving results to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"Model: {model_path.name}")
    print(f"Test file: {test_file.name}")
    print("-"*80)
    print(f"F1 Score:  {metrics.get('entity_f1_strict', 0.0):.4f}")
    print(f"Precision: {metrics.get('precision', 0.0):.4f}")
    print(f"Recall:    {metrics.get('recall', 0.0):.4f}")
    print(f"Accuracy:  {metrics.get('accuracy', 0.0):.4f}")
    print("="*80)
    print(f"\nFull results saved to: {output_file}")


if __name__ == '__main__':
    main()
