#!/usr/bin/env python3
"""
Configuration loader for traditional baseline models (CRF, BiLSTM+FastText).
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: Optional[str] = None, baseline_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to custom config file (optional)
        baseline_name: Name of baseline (e.g., 'crf', 'bilstm_fasttext') to use default config

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


def extract_model_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract model parameters from configuration.

    Args:
        config: Full configuration dictionary

    Returns:
        Dictionary with model parameters
    """
    model_config = config.get('model', {})
    return model_config


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


def extract_training_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract training parameters from configuration.

    Args:
        config: Full configuration dictionary

    Returns:
        Dictionary with training parameters
    """
    training_config = config.get('training', {})
    return training_config


def extract_embedding_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract embedding parameters from configuration (for BiLSTM).

    Args:
        config: Full configuration dictionary

    Returns:
        Dictionary with embedding parameters
    """
    embedding_config = config.get('embeddings', {})
    return embedding_config
