#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script for NER Baselines

Automatically detects all trained models in the results/ directory and evaluates
them on the specified test set (default: English test set). Results are saved
in a structured JSON format suitable for publication.

Usage:
    python evaluate_all_models.py --test_file path/to/test.jsonl --output results.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import torch
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core.evaluation.metrics import NEREvaluator
from core.data.dataset import load_jsonl_data

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelDetector:
    """Detects and catalogs all trained models in the results directory."""

    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.models = []

    def detect_models(self) -> List[Dict[str, Any]]:
        """
        Scan the results directory and detect all trained models.

        Returns:
            List of model information dictionaries
        """
        logger.info(f"Scanning for models in {self.results_dir}")

        # Define model type patterns
        model_patterns = {
            'transformers': {
                'bert_linear': 'baselines/transformers/configs/bert_linear*.yaml',
                'bert_crf': 'baselines/transformers/configs/bert_crf*.yaml',
                'xlmr_linear': 'baselines/transformers/configs/xlmr_linear*.yaml',
                'xlmr_crf': 'baselines/transformers/configs/xlmr_crf*.yaml',
                'deberta_linear': 'baselines/transformers/configs/deberta_linear*.yaml',
                'deberta_crf': 'baselines/transformers/configs/deberta_crf*.yaml',
            },
            'traditional': {
                'crf': 'baselines/traditional/crf',
                'bilstm_fasttext': 'baselines/traditional/bilstm_fasttext',
            }
        }

        # Scan results directory for model checkpoints
        for model_type in ['bert_linear', 'bert_crf', 'xlmr_linear', 'xlmr_crf',
                           'deberta_linear', 'deberta_crf', 'bilstm_fasttext', 'crf']:
            model_dir = self.results_dir / model_type

            if not model_dir.exists():
                logger.debug(f"Model directory not found: {model_dir}")
                continue

            # Check for trained models
            if model_type in ['bert_linear', 'bert_crf', 'xlmr_linear', 'xlmr_crf',
                              'deberta_linear', 'deberta_crf']:
                # Transformer models - look for run directories
                self._detect_transformer_models(model_type, model_dir)
            elif model_type == 'bilstm_fasttext':
                # BiLSTM models - look for model files directly or legacy best_model/run directories
                self._detect_bilstm_models(model_type, model_dir)
            elif model_type == 'crf':
                # CRF models - look for run directories
                self._detect_crf_models(model_type, model_dir)

        logger.info(f"Detected {len(self.models)} trained models")
        return self.models

    def _detect_transformer_models(self, model_type: str, model_dir: Path):
        """Detect transformer-based models."""
        # Look for run directories
        for run_dir in sorted(model_dir.glob('run_*')):
            if (run_dir / 'pytorch_model.bin').exists() or (run_dir / 'model.safetensors').exists():
                # Check for config and label mapping
                if (run_dir / 'config.json').exists() and (run_dir / 'label_mapping.json').exists():
                    self.models.append({
                        'model_type': model_type,
                        'model_family': 'transformers',
                        'model_path': run_dir,
                        'run_name': run_dir.name,
                        'checkpoint_type': 'final'
                    })
                    logger.info(f"Found {model_type} model: {run_dir.name}")

    def _detect_bilstm_models(self, model_type: str, model_dir: Path):
        """Detect BiLSTM models."""
        # Check for model files directly in model directory (new format)
        if (model_dir / 'pytorch_model.bin').exists():
            self.models.append({
                'model_type': model_type,
                'model_family': 'traditional',
                'model_path': model_dir,
                'run_name': model_dir.name,
                'checkpoint_type': 'best'
            })
            logger.info(f"Found BiLSTM model: {model_dir.name}")
        
        # Also check for legacy best_model directory format
        best_model_dir = model_dir / 'best_model'
        if best_model_dir.exists() and (best_model_dir / 'pytorch_model.bin').exists():
            self.models.append({
                'model_type': model_type,
                'model_family': 'traditional',
                'model_path': best_model_dir,
                'run_name': 'best_model',
                'checkpoint_type': 'best'
            })
            logger.info(f"Found BiLSTM model: best_model (legacy format)")

        # Also check run directories
        for run_dir in sorted(model_dir.glob('run_*')):
            if (run_dir / 'pytorch_model.bin').exists():
                self.models.append({
                    'model_type': model_type,
                    'model_family': 'traditional',
                    'model_path': run_dir,
                    'run_name': run_dir.name,
                    'checkpoint_type': 'final'
                })
                logger.info(f"Found BiLSTM model: {run_dir.name}")

    def _detect_crf_models(self, model_type: str, model_dir: Path):
        """Detect CRF models."""
        # Look for run directories
        for run_dir in sorted(model_dir.glob('run_*')):
            if (run_dir / 'crf_model.pkl').exists() and (run_dir / 'metadata.json').exists():
                self.models.append({
                    'model_type': model_type,
                    'model_family': 'traditional',
                    'model_path': run_dir,
                    'run_name': run_dir.name,
                    'checkpoint_type': 'final'
                })
                logger.info(f"Found CRF model: {run_dir.name}")


class ModelEvaluator:
    """Evaluates detected models on the test set."""

    def __init__(self, test_data: List[Dict], device: str = 'cuda'):
        self.test_data = test_data
        self.device = device if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")

    def evaluate_all_models(self, models: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate all detected models.

        Args:
            models: List of model information dictionaries

        Returns:
            Dictionary containing evaluation results for all models
        """
        results = {
            'metadata': {
                'evaluation_date': datetime.now().isoformat(),
                'test_set_size': len(self.test_data),
                'num_models_evaluated': len(models),
                'device': self.device
            },
            'models': {}
        }

        for model_info in tqdm(models, desc="Evaluating models"):
            try:
                logger.info(f"\nEvaluating {model_info['model_type']} - {model_info['run_name']}")
                model_results = self._evaluate_single_model(model_info)

                # Store results with unique key
                model_key = f"{model_info['model_type']}_{model_info['run_name']}"
                results['models'][model_key] = {
                    'model_info': model_info,
                    'evaluation_results': model_results
                }

            except Exception as e:
                logger.error(f"Error evaluating {model_info['model_type']}: {e}")
                logger.exception(e)
                results['models'][f"{model_info['model_type']}_{model_info['run_name']}"] = {
                    'model_info': model_info,
                    'evaluation_results': {'error': str(e)}
                }

        return results

    def _evaluate_single_model(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single model."""
        model_family = model_info['model_family']
        model_path = model_info['model_path']

        if model_family == 'transformers':
            return self._evaluate_transformer_model(model_path)
        elif model_info['model_type'] == 'bilstm_fasttext':
            return self._evaluate_bilstm_model(model_path)
        elif model_info['model_type'] == 'crf':
            return self._evaluate_crf_model(model_path)
        else:
            raise ValueError(f"Unknown model family: {model_family}")

    def _evaluate_transformer_model(self, model_path: Path) -> Dict[str, Any]:
        """Evaluate transformer-based model."""
        from baselines.transformers.model import (
            BertLinearNER, BertCRFNER,
            XLMRLinearNER, XLMRCRFNER,
            DebertaLinearNER, DebertaCRFNER
        )
        from transformers import AutoTokenizer
        from torch.utils.data import DataLoader

        # Load label mapping
        with open(model_path / 'label_mapping.json', 'r') as f:
            label_mapping = json.load(f)
        id_to_label = {int(k): v for k, v in label_mapping['id_to_label'].items()}
        label_to_id = label_mapping['label_to_id']

        # Load config
        with open(model_path / 'config.json', 'r') as f:
            config = json.load(f)

        # Determine model class
        model_type = config.get('model_type', '')
        model_name = config.get('_name_or_path', '')

        model_classes = {
            'bert_linear': BertLinearNER,
            'bert_crf': BertCRFNER,
            'xlmr_linear': XLMRLinearNER,
            'xlmr_crf': XLMRCRFNER,
            'deberta_linear': DebertaLinearNER,
            'deberta_crf': DebertaCRFNER,
        }

        # Find matching model class
        model_class = None
        for key, cls in model_classes.items():
            if key in str(model_path).lower() or key in model_type.lower():
                model_class = cls
                break

        if model_class is None:
            raise ValueError(f"Could not determine model class for {model_path}")

        # Load model
        num_labels = len(id_to_label)
        model = model_class(model_name, num_labels)

        # Load weights
        if (model_path / 'pytorch_model.bin').exists():
            state_dict = torch.load(model_path / 'pytorch_model.bin', map_location=self.device)
            model.load_state_dict(state_dict)

        model = model.to(self.device)
        model.eval()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Evaluate
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for example in tqdm(self.test_data, desc="Evaluating", leave=False):
                tokens = example['tokens']
                labels = example['labels']

                # Tokenize
                tokenized = tokenizer(
                    tokens,
                    is_split_into_words=True,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                )

                # Move to device
                input_ids = tokenized['input_ids'].to(self.device)
                attention_mask = tokenized['attention_mask'].to(self.device)

                # Get predictions
                if hasattr(model, 'decode'):
                    predictions = model.decode(
                        input_ids,
                        attention_mask,
                        orig_token_indices=tokenized,
                        apply_bio_validation=True,
                        id_to_label=id_to_label
                    )
                else:
                    outputs = model(input_ids, attention_mask)
                    logits = outputs['logits']
                    pred_ids = torch.argmax(logits, dim=-1)

                    # Postprocess predictions
                    from baselines.transformers.model import postprocess_model_output
                    predictions = postprocess_model_output(pred_ids, tokenized)

                # Convert to labels
                pred_labels = [id_to_label.get(p, 'O') for p in predictions[0]]

                # Ensure same length as original tokens
                if len(pred_labels) > len(tokens):
                    pred_labels = pred_labels[:len(tokens)]
                elif len(pred_labels) < len(tokens):
                    pred_labels.extend(['O'] * (len(tokens) - len(pred_labels)))

                all_predictions.append(pred_labels)
                all_labels.append(labels)

        # Compute metrics
        evaluator = NEREvaluator(id_to_label=None)
        metrics = evaluator.compute_metrics(all_predictions, all_labels)

        return metrics

    def _evaluate_bilstm_model(self, model_path: Path) -> Dict[str, Any]:
        """Evaluate BiLSTM model."""
        from src.models.bilstm_crf import BiLSTMFastNER, FastTextEmbedding

        # Load model
        model = BiLSTMFastNER.from_pretrained(model_path)
        model = model.to(self.device)
        model.eval()

        # Load label mapping
        with open(model_path / 'label_mapping.json', 'r') as f:
            label_mapping = json.load(f)
        id_to_label = {int(k): v for k, v in label_mapping['id_to_label'].items()}

        # Load vocabulary
        vocab_path = model_path.parent if (model_path.parent / 'vocabulary.json').exists() else model_path
        embedding_handler = FastTextEmbedding.load_vocabulary(vocab_path)

        # Evaluate
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for example in tqdm(self.test_data, desc="Evaluating", leave=False):
                tokens = example['tokens']
                labels = example['labels']

                # Encode tokens
                encoded_tokens = embedding_handler.encode_sentences([tokens])
                input_ids = torch.tensor(encoded_tokens, dtype=torch.long).to(self.device)

                # Create attention mask
                attention_mask = (input_ids != 0).long()

                # Get predictions
                predictions = model.decode(input_ids, attention_mask)
                pred_labels = [id_to_label.get(p, 'O') for p in predictions[0]]

                all_predictions.append(pred_labels)
                all_labels.append(labels)

        # Compute metrics
        evaluator = NEREvaluator(id_to_label=None)
        metrics = evaluator.compute_metrics(all_predictions, all_labels)

        return metrics

    def _evaluate_crf_model(self, model_path: Path) -> Dict[str, Any]:
        """Evaluate CRF model."""
        from src.models.crf import TraditionalCRFModel

        # Load model
        model = TraditionalCRFModel.load_model(model_path)

        # Prepare test data in CRF format
        test_data_formatted = [
            {'tokens': ex['tokens'], 'labels': ex['labels']}
            for ex in self.test_data
        ]

        # Get predictions
        predictions = model.predict(test_data_formatted)

        # Extract true labels
        all_labels = [ex['labels'] for ex in test_data_formatted]

        # Compute metrics using seqeval
        evaluator = NEREvaluator(id_to_label=None)
        metrics = evaluator.compute_metrics(predictions, all_labels)

        return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate all trained models on the test set'
    )
    parser.add_argument(
        '--test_file',
        type=str,
        default='../data/publication_datasets/ner_datasets/en/splits/test.jsonl',
        help='Path to test JSONL file'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='results',
        help='Directory containing trained models'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='evaluation_results.json',
        help='Output JSON file for results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for evaluation'
    )

    args = parser.parse_args()

    # Convert paths
    script_dir = Path(__file__).parent
    test_file = Path(args.test_file)
    if not test_file.is_absolute():
        test_file = script_dir / test_file

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = script_dir / results_dir

    output_file = Path(args.output)
    if not output_file.is_absolute():
        output_file = script_dir / output_file

    # Load test data
    logger.info(f"Loading test data from {test_file}")
    test_data = load_jsonl_data(test_file)
    logger.info(f"Loaded {len(test_data)} test examples")

    # Detect models
    detector = ModelDetector(results_dir)
    models = detector.detect_models()

    if not models:
        logger.error("No trained models found!")
        return

    # Evaluate models
    evaluator = ModelEvaluator(test_data, device=args.device)
    results = evaluator.evaluate_all_models(models)

    # Save results
    logger.info(f"Saving results to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Test set size: {len(test_data)}")
    print(f"Models evaluated: {len(models)}")
    print("\nResults by model:")
    print("-"*80)

    for model_key, model_data in results['models'].items():
        if 'error' in model_data['evaluation_results']:
            print(f"{model_key}: ERROR - {model_data['evaluation_results']['error']}")
        else:
            metrics = model_data['evaluation_results']
            f1 = metrics.get('entity_f1_strict', 0.0)
            precision = metrics.get('precision', 0.0)
            recall = metrics.get('recall', 0.0)
            print(f"{model_key}:")
            print(f"  F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

    print("="*80)
    print(f"\nFull results saved to: {output_file}")


if __name__ == '__main__':
    main()
