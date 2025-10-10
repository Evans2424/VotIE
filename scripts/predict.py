#!/usr/bin/env python3
"""
VotIE Prediction Script

Simple, robust prediction interface for Vote Information Extraction (VotIE) models.
Handles windowing automatically for long sequences.

Usage:
    from scripts.predict import predict
    predict(
        model_path='results/bert_crf/best_model',
        test_data_path='data/votie_bio/test.jsonl',
        output_path='predictions/test_predictions.jsonl'
    )
"""

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
from src.models.bilstm_crf import BiLSTMFastTextNER
from src.data.dataset import load_jsonl_file, align_tokens_with_subwords, create_dynamic_windows

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_path: str, device: str = 'cuda'):
    """Load trained VotIE model from checkpoint."""
    model_path = Path(model_path)
    
    # Load config
    with open(model_path / 'config.json', 'r') as f:
        config = json.load(f)
    
    # Load label mapping - handle both formats
    if 'id2label' in config:
        # Transformer models store labels in config
        id_to_label = {int(k): v for k, v in config['id2label'].items()}
    else:
        # BiLSTM models store labels in separate file
        label_mapping_file = model_path / 'label_mapping.json'
        if label_mapping_file.exists():
            with open(label_mapping_file, 'r') as f:
                label_mapping = json.load(f)
            id_to_label = label_mapping['idx2label']
            # Convert string keys to int
            id_to_label = {int(k): v for k, v in id_to_label.items()}
        else:
            raise FileNotFoundError(f"Neither 'id2label' in config nor 'label_mapping.json' found in {model_path}")
    
    # Check if this is a BiLSTM model
    model_type = config.get('model_type', '')
    if model_type == 'bilstm_fasttext':
        # Load BiLSTM model using its from_pretrained method
        model = BiLSTMFastTextNER.from_pretrained(model_path)
        model.to(device)
        model.eval()
        
        # Store model path for vocabulary loading during prediction
        model._model_path = str(model_path)
        
        # BiLSTM models don't use transformers tokenizer
        tokenizer = None
        
        logger.info(f"✓ Loaded BiLSTM model from {model_path}")
        return model, tokenizer, id_to_label, config
    
    # Check if this is a CRF model
    if model_type == 'crf':
        from src.models.crf import TraditionalCRFModel
        
        # Load CRF model using its load_model method
        model = TraditionalCRFModel.load_model(model_path)
        
        # CRF models don't use device or transformers tokenizer
        tokenizer = None
        
        logger.info(f"✓ Loaded CRF model from {model_path}")
        return model, tokenizer, id_to_label, config
    
    # Handle transformer models
    architecture = config.get('architecture', '')
    base_model_name = config.get('base_model', 'neuralmind/bert-base-portuguese-cased')
    
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
    num_labels = len(id_to_label)
    
    # Load model
    model = model_class(base_model_name, num_labels)
    state_dict = torch.load(model_path / 'pytorch_model.bin', map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    logger.info(f"✓ Loaded {architecture} model from {model_path}")
    
    return model, tokenizer, id_to_label, config


def predict_single_window(model, tokenizer, tokens: List[str], device: str = 'cuda') -> List[int]:
    """Predict labels for a single window (guaranteed to fit in max_length)."""
    dummy_labels = ['O'] * len(tokens)
    
    # Tokenize (same as training)
    _, input_ids, _, subtoken_mask, _ = align_tokens_with_subwords(
        tokens=tokens,
        labels=dummy_labels,
        tokenizer=tokenizer,
        max_length=512,
        example_id="predict",
        allow_truncation=False
    )
    
    # Pad to 512
    seq_length = len(input_ids)
    input_ids += [tokenizer.pad_token_id] * (512 - seq_length)
    attention_mask = [1] * seq_length + [0] * (512 - seq_length)
    subtoken_mask += [False] * (512 - seq_length)
    
    # Convert to tensors
    input_ids_tensor = torch.tensor([input_ids]).to(device)
    attention_mask_tensor = torch.tensor([attention_mask]).to(device)
    subtoken_mask_tensor = torch.tensor([subtoken_mask]).to(device)
    
    # Predict
    with torch.no_grad():
        predictions = model.decode(
            input_ids=input_ids_tensor,
            attention_mask=attention_mask_tensor,
            subtoken_mask=subtoken_mask_tensor,
            apply_bio_validation=False,
            id_to_label=None
        )
    
    return predictions[0]


def predict_with_windowing(model, tokenizer, tokens: List[str], device: str = 'cuda') -> List[int]:
    """Predict labels using windowing for long sequences."""
    # Create windows using the same logic as training
    windows = create_dynamic_windows(
        tokens=tokens,
        labels=['O'] * len(tokens),  # Dummy labels
        tokenizer=tokenizer,
        max_length=512,
        overlap_tokens=50,
        example_id="predict"
    )
    
    # Single window - no merging needed
    if len(windows) == 1:
        return predict_single_window(model, tokenizer, windows[0][0], device)
    
    # Multiple windows - predict and merge
    all_predictions = []
    for window_tokens, _ in windows:
        window_preds = predict_single_window(model, tokenizer, window_tokens, device)
        all_predictions.append(window_preds)
    
    # Merge predictions from overlapping windows
    merged_predictions = [0] * len(tokens)
    coverage_map = {}
    
    start_idx = 0
    overlap_tokens = 50
    
    for window_idx, (window_tokens, _) in enumerate(windows):
        window_preds = all_predictions[window_idx]
        window_len = len(window_tokens)
        
        for i in range(window_len):
            pos = start_idx + i
            if pos < len(tokens) and pos not in coverage_map:
                merged_predictions[pos] = window_preds[i]
                coverage_map[pos] = window_idx
        
        step_size = max(window_len - overlap_tokens, 1)
        start_idx = start_idx + step_size
    
    return merged_predictions


def predict_crf(model, tokens: List[str]) -> List[int]:
    """Predict labels for CRF model."""
    # CRF models work directly with tokens - no device or tensor conversion needed
    # Create a test example in the expected format
    test_example = {'tokens': tokens}
    
    # Use the CRF model's predict method
    predictions = model.predict([test_example])
    
    # predictions is a list of lists, get the first (and only) sequence
    pred_labels = predictions[0]
    
    # Convert label strings to IDs using the model's label list
    label_to_id = {label: idx for idx, label in enumerate(model.label_list)}
    pred_ids = [label_to_id.get(label, 0) for label in pred_labels]
    
    return pred_ids


def predict_bilstm(model, tokens: List[str], device: str = 'cuda', model_path: str = None) -> List[int]:
    """Predict labels for BiLSTM model."""
    import json
    
    # Load vocabularies
    if model_path is None:
        raise ValueError("model_path is required for BiLSTM prediction")
    
    model_path = Path(model_path)
    
    # Load word vocabulary
    vocab_path = model_path / 'vocabulary.json' 
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    word2idx = vocab['word2idx']
    
    # Load character vocabulary if exists
    char_vocab_path = model_path / 'char_vocab.json'
    char2idx = None
    if char_vocab_path.exists():
        with open(char_vocab_path, 'r', encoding='utf-8') as f:
            char_vocab = json.load(f)
        char2idx = char_vocab['char2idx']
    
    # Convert tokens to input_ids
    input_ids = []
    for token in tokens:
        token_id = word2idx.get(token, word2idx.get('<UNK>', 0))
        input_ids.append(token_id)
    
    # Create attention mask (all 1s since we don't pad)
    attention_mask = [1] * len(input_ids)
    
    # Convert to tensors
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(device)
    
    # Character encoding if available
    char_ids = None
    if char2idx is not None:
        batch_char_ids = []
        for token in tokens:
            token_char_ids = []
            for char in token:
                char_id = char2idx.get(char, char2idx.get('<UNK>', 0))
                token_char_ids.append(char_id)
            # Pad/truncate to max char length (assume 20)
            if len(token_char_ids) > 20:
                token_char_ids = token_char_ids[:20]
            else:
                token_char_ids.extend([0] * (20 - len(token_char_ids)))
            batch_char_ids.append(token_char_ids)
        
        char_ids = torch.tensor([batch_char_ids], dtype=torch.long).to(device)
    
    # Get predictions
    with torch.no_grad():
        if char_ids is not None:
            # Model expects input_ids, attention_mask, char_ids
            predictions = model.decode(input_ids, attention_mask, char_ids)
        else:
            # Model expects input_ids, attention_mask
            predictions = model.decode(input_ids, attention_mask)
    
    # predictions is a list of lists, get the first (and only) sequence
    return predictions[0]


def predict_example(model, tokenizer, id_to_label: Dict[int, str], example: Dict[str, Any], device: str = 'cuda') -> Dict[str, Any]:
    """Generate predictions for a single example."""
    tokens = example['tokens']
    
    # Check model type for prediction
    if tokenizer is None:
        # Check if this is a CRF model (has label_list attribute)
        if hasattr(model, 'label_list') and hasattr(model, 'crf'):
            # CRF models
            pred_label_ids = predict_crf(model, tokens)
        else:
            # BiLSTM models need model path for vocabulary loading
            pred_label_ids = predict_bilstm(model, tokens, device, getattr(model, '_model_path', None))
    else:
        # Predict label IDs using transformer model
        pred_label_ids = predict_with_windowing(model, tokenizer, tokens, device)
    
    # Convert IDs to labels
    pred_labels = [id_to_label.get(lid, 'O') for lid in pred_label_ids]
    
    result = {
        'id': example['id'],
        'tokens': tokens,
        'pred_labels': pred_labels
    }
    
    # Include gold labels if available
    if 'labels' in example:
        result['gold_labels'] = example['labels']
    
    return result


def predict(model_path: str, test_data_path: str, output_path: str, device: str = 'cuda'):
    """
    Generate predictions for a test dataset and save to file.
    
    Args:
        model_path: Path to trained model directory
        test_data_path: Path to input JSONL file
        output_path: Path to output predictions file
        device: Device to use for inference ('cpu' or 'cuda')
    """
    logger.info("="*80)
    logger.info("VotIE Prediction")
    logger.info("="*80)
    
    # Load model
    logger.info(f"Loading model from: {model_path}")
    model, tokenizer, id_to_label, config = load_model(model_path, device=device)
    
    # Load test data
    logger.info(f"Loading test data from: {test_data_path}")
    examples = load_jsonl_file(test_data_path)
    logger.info(f"✓ Loaded {len(examples)} examples")
    
    # Generate predictions
    logger.info("Generating predictions...")
    predictions = []
    
    for example in tqdm(examples, desc="Predicting"):
        pred = predict_example(model, tokenizer, id_to_label, example, device)
        predictions.append(pred)
    
    # Save predictions
    logger.info(f"Saving predictions to: {output_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')
    
    logger.info("="*80)
    logger.info(f"✓ Predictions saved to {output_path}")
    logger.info(f"✓ Total examples: {len(predictions)}")
    logger.info("="*80)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python scripts/predict.py MODEL_PATH TEST_DATA_PATH OUTPUT_PATH [DEVICE]")
        print("\nExample:")
        print("  python scripts/predict.py results/bert_crf/best_model data/votie_bio/test.jsonl predictions/test_predictions.jsonl")
        print("  python scripts/predict.py results/bert_crf/best_model data/votie_bio/test.jsonl predictions/test_predictions.jsonl cuda")
        sys.exit(1)
    
    model_path = sys.argv[1]
    test_data_path = sys.argv[2]
    output_path = sys.argv[3]
    device = sys.argv[4] if len(sys.argv) > 4 else 'cpu'
    
    predict(model_path, test_data_path, output_path, device)
