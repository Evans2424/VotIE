"""
LangExtract NER - Portuguese Voting Entity Extraction

Simple Gemini 2.0 Flash implementation for Portuguese municipal documents.
"""

from .data_loader import VotingDataLoader, VotingDocument
from .shared.models_spans import GeminiModel
from .shared.models_events import GeminiEventExtractor
from .evaluation_events import evaluate_langextract_events
from .config import get_config

__version__ = "1.0.0"

__all__ = [
    "VotingDataLoader",
    "VotingDocument",
    "GeminiModel",
    "GeminiEventExtractor",
    "evaluate_langextract_events",
    "get_config"
]
