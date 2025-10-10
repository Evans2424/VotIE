"""
Configuration for LangExtract Portuguese Voting NER

Simple configuration for Gemini 2.0 Flash extraction.
"""

import os
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Config:
    """Main configuration for LangExtract NER"""

    # Gemini Model Configuration
    # Available models:
    # - "gemini-2.0-flash-exp"  (fast, experimental)
    # - "gemini-1.5-flash"      (fast, stable)
    # - "gemini-1.5-pro"        (high quality)
    # - "gemini-2.5-pro"        (latest pro model)
    # gemini-2.5-flash

    api_key: str = None
    model_id: str = "gemini-2.5-flash"  # <<< CHANGE THIS to switch models

    # Language
    language: str = "pt"  # Portuguese only for publication

    # Entity Types (English labels)
    entity_types: tuple = (
        "VOTER-FAVOR",
        "VOTER-AGAINST",
        "VOTER-ABSTENTION",
        "VOTER-ABSENT",
        "SUBJECT",
        "VOTING",
        "COUNTING-MAJORITY",
        "COUNTING-UNANIMITY"
    )

    # Paths
    @property
    def project_root(self) -> Path:
        """Project root directory"""
        return Path(__file__).parent.parent

    @property
    def test_dataset_path(self) -> Path:
        """Test dataset path"""
        return self.project_root / "data/publication_datasets/ner_datasets/pt/splits/test.jsonl"

    @property
    def experiments_dir(self) -> Path:
        """Experiments output directory"""
        path = Path(__file__).parent / "experiments"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def __post_init__(self):
        """Load API key from environment if not set"""
        if not self.api_key:
            self.api_key = os.getenv('LANGEXTRACT_API_KEY') or os.getenv('GOOGLE_API_KEY')


# Global config instance
_config = None


def get_config() -> Config:
    """Get global configuration instance"""
    global _config
    if _config is None:
        _config = Config()
    return _config
