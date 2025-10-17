#!/usr/bin/env python3
"""
VotIE - Evaluation Script for DeBERTa-CRF HuggingFace Model

This script loads the published DeBERTa-CRF model from HuggingFace Hub,
runs it on the test set, saves predictions, and computes comprehensive
evaluation metrics (both entity-level and event-level).

Usage:
    # Evaluate on test set (GPU)
    python scripts/evaluate_deberta_hf.py

    # Evaluate on test set (CPU)
    python scripts/evaluate_deberta_hf.py --device cpu

Author: VotIE Team
"""

import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer, AutoModel
from src.data.dataset import load_jsonl_file
from scripts.evaluate import evaluate

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model configuration
DEFAULT_MODEL_NAME = "Anonymous3445/DeBERTa-CRF-VotIE"
DEFAULT_TEST_FILE = "data/votie_bio/test.jsonl"
DEFAULT_PREDICTIONS_FILE = "predictions/deberta_hf_predictions.jsonl"
DEFAULT_EVALUATION_FILE = "evaluation/deberta_hf_results.json"


def predict_and_save(model_name: str, test_file: str, output_file: str, device: str = "cuda"):
    """
    Run predictions using HuggingFace model and save results.

    Args:
        model_name: HuggingFace model identifier
        test_file: Path to test JSONL file
        output_file: Path to save predictions
        device: Device for inference ('cuda' or 'cpu')
    """
    logger.info("="*80)
    logger.info("Loading Model and Running Predictions")
    logger.info("="*80)

    # Load model and tokenizer
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    # Move to device
    if device == "cuda":
        import torch
        if torch.cuda.is_available():
            model = model.to(device)
            logger.info(f"✓ Model loaded on GPU ({torch.cuda.get_device_name(0)})")
        else:
            logger.warning("CUDA not available, using CPU")
            device = "cpu"

    if device == "cpu":
        model = model.to("cpu")
        logger.info("✓ Model loaded on CPU")

    model.eval()

    # Load test data
    logger.info(f"Loading test data from: {test_file}")
    examples = load_jsonl_file(test_file)
    logger.info(f"✓ Loaded {len(examples)} examples")

    # Generate predictions
    logger.info("Generating predictions...")
    predictions = []

    for example in tqdm(examples, desc="Predicting"):
        text = example['text']
        tokens = example['tokens']

        # Tokenize
        inputs = tokenizer(text, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get predictions (returns word-level results)
        word_predictions = model.decode(**inputs, tokenizer=tokenizer, text=text)

        # Extract labels in order (model returns one prediction per word)
        pred_labels = [pred['label'] for pred in word_predictions]

        # Ensure length matches (pad or truncate if needed)
        if len(pred_labels) != len(tokens):
            if len(pred_labels) < len(tokens):
                pred_labels += ['O'] * (len(tokens) - len(pred_labels))
            else:
                pred_labels = pred_labels[:len(tokens)]

        # Create prediction entry
        result = {
            'id': example['id'],
            'tokens': tokens,
            'pred_labels': pred_labels
        }

        # Include gold labels if available
        if 'labels' in example:
            result['gold_labels'] = example['labels']

        predictions.append(result)

    # Save predictions
    logger.info(f"Saving predictions to: {output_file}")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')

    logger.info(f"✓ Predictions saved to {output_file}")
    logger.info(f"✓ Total examples: {len(predictions)}")


def main():
    """Main function to run evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Evaluate DeBERTa-CRF HuggingFace model on VotIE test set"
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default=DEFAULT_MODEL_NAME,
        help='HuggingFace model identifier (default: Anonymous3445/DeBERTa-CRF-VotIE)'
    )
    parser.add_argument(
        '--test-file',
        type=str,
        default=DEFAULT_TEST_FILE,
        help='Path to test JSONL file (default: data/votie_bio/test.jsonl)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=DEFAULT_PREDICTIONS_FILE,
        help='Path to save predictions (default: predictions/deberta_hf_predictions.jsonl)'
    )
    parser.add_argument(
        '--evaluation-file',
        type=str,
        default=DEFAULT_EVALUATION_FILE,
        help='Path to save evaluation results (default: evaluation/deberta_hf_results.json)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device for inference (default: cuda)'
    )
    parser.add_argument(
        '--no-events',
        action='store_true',
        help='Skip event-level evaluation (faster, entity-level only)'
    )

    args = parser.parse_args()

    try:
        # Run predictions
        predict_and_save(
            model_name=args.model_name,
            test_file=args.test_file,
            output_file=args.output,
            device=args.device
        )

        # Run evaluation using existing evaluate function
        logger.info("")
        logger.info("="*80)
        logger.info("Running Evaluation")
        logger.info("="*80)

        evaluate(
            predictions_path=args.output,
            output_path=args.evaluation_file,
            include_events=not args.no_events
        )

        logger.info("")
        logger.info("="*80)
        logger.info("✓ Evaluation Complete!")
        logger.info("="*80)
        logger.info(f"Predictions: {args.output}")
        logger.info(f"Results: {args.evaluation_file}")
        logger.info("="*80)

        return 0

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        logger.exception("Full traceback:")
        return 1


if __name__ == '__main__':
    sys.exit(main())
