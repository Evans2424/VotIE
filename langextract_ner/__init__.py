"""
LangExtract NER - Portuguese Voting Entity Extraction

Simple Gemini 2.0 Flash implementation for Portuguese municipal documents.
"""

from .data_loader import VotingDataLoader, VotingDocument
from .evaluation import VotingEntityEvaluator, EvaluationReport
from .shared.models import GeminiModel
from .config import get_config

__version__ = "1.0.0"

__all__ = [
    "VotingDataLoader",
    "VotingDocument",
    "VotingEntityEvaluator",
    "EvaluationReport",
    "GeminiModel",
    "get_config"
]
