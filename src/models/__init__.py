"""
Models package for NER Vote Identification.

This package contains all the transformer-based NER models:
- BERT-based models (BERTimbau)
- XLM-R models 
- DeBERTa models
"""

from .base import BaseNERModel
from .bert_models import BertLinearNER, BertCRFNER
from .xlmr_models import XLMRLinearNER, XLMRCRFNER
from .deberta_models import DebertaLinearNER, DebertaCRFNER

# Import postprocessing utilities from data directory
from src.data.postprocessing import postprocess_model_output, postprocess_labels_to_original

__all__ = [
    # Base class
    'BaseNERModel',
    # BERT models
    'BertLinearNER', 'BertCRFNER',
    # XLM-R models  
    'XLMRLinearNER', 'XLMRCRFNER',
    # DeBERTa models
    'DebertaLinearNER', 'DebertaCRFNER',
    # Utilities
    'postprocess_model_output', 'postprocess_labels_to_original'
]
