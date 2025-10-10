#!/usr/bin/env python3
"""
VotIE Prediction Interface

Provides prediction interface for Vote Information Extraction (VotIE) models.
Uses the same windowing and data handling infrastructure as training for consistency.

Usage:
    # Predict on single example
    predictor = VotIEPredictor(model_path)
    predictions = predictor.predict(tokens)
    
    # Predict on dataset file
    python scripts/predict.py --model results/bert_crf/best_model --input data/votie_bio/test.jsonl --output predictions.jsonl
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import torch
from transformers import AutoTokenizer

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.transformer_models import (
    BertLinearNER, BertCRFNER,
    XLMRLinearNER, XLMRCRFNER,
    DebertaLinearNER, DebertaCRFNER
)
from src.data.dataset import align_tokens_with_subwords

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VotIEPredictor:
    """Vote Information Extraction (VotIE) predictor with automatic windowing support."""

    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Initialize predictor.

        Args:
            model_path: Path to trained model directory
            device: Device to use ('cpu' or 'cuda')
        """
        self.model_path = Path(model_path)
        self.device = device

        # Load config
        with open(self.model_path / 'config.json', 'r') as f:
            self.config = json.load(f)

        # Load label mapping
        self.id_to_label = {int(k): v for k, v in self.config['id2label'].items()}
        self.label_to_id = self.config['label2id']

        # Determine model class
        architecture = self.config.get('architecture', '')
        model_name = self.config.get('base_model', 'neuralmind/bert-base-portuguese-cased')

        model_classes = {
            'BertLinearNER': BertLinearNER,
            'BertCRFNER': BertCRFNER,
            'XLMRLinearNER': XLMRLinearNER,
            'XLMRCRFNER': XLMRCRFNER,
            'DebertaLinearNER': DebertaLinearNER,
            'DebertaCRFNER': DebertaCRFNER,
        }

        if architecture not in model_classes:
            raise ValueError(f"Unknown architecture: {architecture}")

        model_class = model_classes[architecture]
        num_labels = len(self.id_to_label)

        # Load model
        self.model = model_class(model_name, num_labels)
        state_dict = torch.load(self.model_path / 'pytorch_model.bin', map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        logger.info(f"Loaded {architecture} model from {model_path}")

    def predict(self, tokens: List[str]) -> List[str]:
        """
        Predict labels for a list of tokens with automatic windowing.

        Args:
            tokens: List of input tokens

        Returns:
            List of predicted labels (same length as tokens)
        """
        # Use training's windowing logic directly
        from src.data.dataset import create_dynamic_windows

        windows = create_dynamic_windows(
            tokens=tokens,
            labels=['O'] * len(tokens),  # Dummy labels
            tokenizer=self.tokenizer,
            max_length=512,
            overlap_tokens=50,
            example_id="predict"
        )

        if len(windows) == 1:
            return self._predict_single_window(windows[0][0])
        else:
            return self._predict_with_windowing(windows, tokens)

    def _predict_single_window(self, tokens: List[str]) -> List[str]:
        """Predict for a single window (guaranteed to fit in 512 subwords)."""
        dummy_labels = ['O'] * len(tokens)

        # Tokenize (same as training)
        _, input_ids, _, subtoken_mask, _ = align_tokens_with_subwords(
            tokens=tokens,
            labels=dummy_labels,
            tokenizer=self.tokenizer,
            max_length=512,
            example_id="predict",
            allow_truncation=False  # Should never truncate - windowing handles long sequences
        )

        # Pad to 512
        seq_length = len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * (512 - seq_length)
        attention_mask = [1] * seq_length + [0] * (512 - seq_length)
        subtoken_mask += [False] * (512 - seq_length)

        # Convert to tensors
        input_ids = torch.tensor([input_ids]).to(self.device)
        attention_mask = torch.tensor([attention_mask]).to(self.device)
        subtoken_mask = torch.tensor([subtoken_mask]).to(self.device)

        # Predict
        with torch.no_grad():
            predictions = self.model.decode(
                input_ids=input_ids,
                attention_mask=attention_mask,
                subtoken_mask=subtoken_mask,
                apply_bio_validation=False,
                id_to_label=None
            )

        # Convert IDs to labels
        pred_labels = [self.id_to_label.get(pid, 'O') for pid in predictions[0]]
        return pred_labels

    def _predict_with_windowing(self, windows, original_tokens: List[str]) -> List[str]:
        """Predict using windowing - reconstruct original sequence from windows."""
        logger.debug(f"Windowing prediction: {len(windows)} windows for {len(original_tokens)} tokens")

        # Predict on each window
        all_predictions = []
        for window_idx, (window_tokens, _) in enumerate(windows):
            window_preds = self._predict_single_window(window_tokens)
            all_predictions.append(window_preds)
            logger.debug(f"Window {window_idx}: {len(window_tokens)} tokens -> {len(window_preds)} predictions")

        # Reconstruct: track actual positions from create_dynamic_windows logic
        merged_predictions = ['O'] * len(original_tokens)
        coverage_map = {}  # Track which window covers each position

        # Replicate create_dynamic_windows position tracking
        start_idx = 0
        overlap_tokens = 50

        for window_idx, (window_tokens, _) in enumerate(windows):
            window_preds = all_predictions[window_idx]
            window_len = len(window_tokens)

            logger.debug(f"Window {window_idx}: positions {start_idx} to {start_idx + window_len - 1} "
                        f"(window length: {window_len})")

            # Copy predictions to correct position in original sequence
            for i in range(window_len):
                pos = start_idx + i
                if pos < len(original_tokens):
                    # For overlapping regions, prefer the first window's prediction
                    if pos not in coverage_map:
                        merged_predictions[pos] = window_preds[i]
                        coverage_map[pos] = window_idx
                else:
                    logger.warning(f"Window {window_idx} position {pos} exceeds original length {len(original_tokens)}")

            # Update start position for next window (same logic as create_dynamic_windows)
            step_size = max(window_len - overlap_tokens, 1)
            start_idx = start_idx + step_size

        # Check for uncovered positions (can happen if last window was reduced)
        uncovered = [i for i in range(len(original_tokens)) if i not in coverage_map]
        if uncovered:
            logger.warning(f"Found {len(uncovered)} uncovered positions. Recursively predicting.")
            # Recursively predict on uncovered tokens (handles windowing if needed)
            uncovered_start = uncovered[0]
            uncovered_tokens = original_tokens[uncovered_start:]
            uncovered_preds = self.predict(uncovered_tokens)  # Recursive call

            # Fill in uncovered positions
            for i, pred in enumerate(uncovered_preds):
                pos = uncovered_start + i
                if pos < len(original_tokens):
                    merged_predictions[pos] = pred

        logger.debug(f"Merged predictions: {len(merged_predictions)} (expected: {len(original_tokens)})")

        return merged_predictions

    def predict_batch(self, examples: List[Dict[str, Any]], show_progress: bool = True) -> List[Dict[str, Any]]:
        """
        Predict labels for a batch of examples.

        Args:
            examples: List of dicts with 'id', 'tokens', and optionally 'labels'
            show_progress: Whether to show progress bar

        Returns:
            List of dicts with 'id', 'tokens', 'pred_labels', and optionally 'gold_labels'
        """
        results = []
        iterator = tqdm(examples, desc="Predicting") if show_progress else examples

        for example in iterator:
            tokens = example['tokens']
            pred_labels = self.predict(tokens)

            result = {
                'id': example['id'],
                'tokens': tokens,
                'pred_labels': pred_labels
            }

            # Include gold labels if available
            if 'labels' in example:
                result['gold_labels'] = example['labels']

            results.append(result)

        return results

    def predict_file(self, input_file: str, output_file: str, show_progress: bool = True) -> None:
        """
        Predict labels for all examples in a JSONL file and save to output file.

        Args:
            input_file: Path to input JSONL file (with 'id', 'tokens', optionally 'labels')
            output_file: Path to output JSONL file (will contain predictions)
            show_progress: Whether to show progress bar
        """
        from src.data.dataset import load_jsonl_file

        logger.info(f"Loading examples from {input_file}")
        examples = load_jsonl_file(input_file)
        logger.info(f"Loaded {len(examples)} examples")

        logger.info("Generating predictions...")
        predictions = self.predict_batch(examples, show_progress=show_progress)

        logger.info(f"Saving predictions to {output_file}")
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for pred in predictions:
                f.write(json.dumps(pred, ensure_ascii=False) + '\n')

        logger.info(f"✅ Saved {len(predictions)} predictions to {output_file}")


def main():
    """Command-line interface for VotIE predictions."""
    parser = argparse.ArgumentParser(
        description='VotIE: Generate predictions using trained model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict on test set
  python scripts/predict.py --model results/bert_crf/best_model --input data/votie_bio/test.jsonl --output predictions.jsonl

  # Predict on dev set
  python scripts/predict.py --model results/deberta_crf/run_20251010_030123/best_model --input data/votie_bio/dev.jsonl --output dev_predictions.jsonl

  # Use GPU
  python scripts/predict.py --model results/bert_crf/best_model --input data/votie_bio/test.jsonl --output predictions.jsonl --device cuda
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model directory (e.g., results/bert_crf/best_model)'
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input JSONL file (e.g., data/votie_bio/test.jsonl)'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output predictions file (e.g., predictions.jsonl)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use for inference (default: cpu)'
    )

    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable progress bar'
    )

    args = parser.parse_args()

    # Create predictor
    logger.info(f"Loading model from {args.model}")
    predictor = VotIEPredictor(args.model, device=args.device)

    # Generate predictions
    predictor.predict_file(
        input_file=args.input,
        output_file=args.output,
        show_progress=not args.no_progress
    )

    print(f"\n✅ Predictions saved to: {args.output}")


if __name__ == '__main__':
    main()
