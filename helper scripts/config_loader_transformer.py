#!/usr/bin/env python3
"""
Unified configuration loader for transformer baseline models.
Supports BERT, DeBERTa, and XLM-RoBERTa models.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: Optional[str] = None, baseline_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to custom config file (optional)
        baseline_name: Name of baseline (e.g., 'bert_linear', 'deberta_crf') to use default config

    Returns:
        Dictionary containing configuration parameters

    Raises:
        ValueError: If neither config_path nor baseline_name is provided
        FileNotFoundError: If the configuration file doesn't exist
    """
    if config_path:
        config_file = Path(config_path)
    elif baseline_name:
        config_dir = Path(__file__).parent / "configs"
        config_file = config_dir / f"{baseline_name}.yaml"
    else:
        raise ValueError("Either config_path or baseline_name must be provided")

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def extract_training_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract training parameters from configuration.

    Args:
        config: Full configuration dictionary

    Returns:
        Dictionary with training parameters
    """
    training_config = config.get('training', {})
    optimization_config = config.get('optimization', {})
    augmentation_config = config.get('augmentation', {})
    windowing_config = config.get('windowing', {})

    # Merge all relevant training parameters
    params = {
        # Training parameters
        'num_train_epochs': training_config.get('epochs', 5),
        'train_batch_size': training_config.get('batch_size', 16),
        'eval_batch_size': training_config.get('batch_size', 16) * 2,  # Usually 2x for eval
        'learning_rate': float(training_config.get('learning_rate', 2e-5)),
        'warmup_proportion': float(training_config.get('warmup_proportion', 0.1)),
        'weight_decay': float(training_config.get('weight_decay', 0.01)),
        'patience': training_config.get('patience', 3),
        'device': training_config.get('device', 0),  # CUDA device ID or -1 for CPU
        'apply_bio_validation': training_config.get('apply_bio_validation', True),

        # Optimization parameters
        'max_grad_norm': optimization_config.get('grad_clip', 1.0),

        # Data augmentation parameters
        'max_length': augmentation_config.get('max_length', 512),
        'padding': augmentation_config.get('padding', 'max_length'),
        'truncation': augmentation_config.get('truncation', True),

        # Windowing parameters
        'enable_windowing': windowing_config.get('enable_windowing', True),
        'overlap_tokens': windowing_config.get('overlap_tokens', 50),
    }

    return params


def extract_model_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract model parameters from configuration.

    Args:
        config: Full configuration dictionary

    Returns:
        Dictionary with model parameters
    """
    model_config = config.get('model', {})

    params = {
        'model_name': model_config.get('model_name'),
        'architecture': model_config.get('architecture', 'linear'),
        'dropout': model_config.get('dropout', 0.1),
        'name': model_config.get('name'),  # e.g., 'bert_linear', 'deberta_crf'
    }

    return params


def extract_data_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract data parameters from configuration.

    Args:
        config: Full configuration dictionary

    Returns:
        Dictionary with data parameters
    """
    data_config = config.get('data', {})

    params = {
        'data_dir': data_config.get('data_dir', 'data/publication_datasets/ner_datasets/pt/splits'),
        'train_files': data_config.get('train_files', ['train.jsonl']),
        'dev_file': data_config.get('dev_file', 'dev.jsonl'),
        'test_file': data_config.get('test_file', 'test.jsonl'),
    }

    return params
