#!/usr/bin/env python3
"""
Deslexicalization system for NER training data.

This module provides deslexicalization capabilities for training data to improve
model generalization by replacing specific named entities with:
- VOTER entities: Fake names (using Faker) or generic tokens
- Municipality names: @MUNICIPALITY placeholder
- Location names: @LOCAL placeholder

Integrates with the existing training pipeline and provides comprehensive logging.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from collections import defaultdict

# Import the existing person name deslexicalization utilities
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from deslexicalization_utils import (
    PersonNameDetector,
    deslexicalize_segment,
    normalize_whitespace,
    normalize_case_to_sentence_case
)

logger = logging.getLogger(__name__)


class DeslexicalizationLogger:
    """Handles logging of deslexicalization operations to a JSONL file."""

    def __init__(self, log_file: Path):
        """
        Initialize the deslexicalization logger.

        Args:
            log_file: Path to the JSONL log file
        """
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Initialize log file with header
        with open(self.log_file, 'w', encoding='utf-8') as f:
            header = {
                'type': 'header',
                'timestamp': datetime.now().isoformat(),
                'description': 'Deslexicalization log file - records all deslexicalization operations'
            }
            f.write(json.dumps(header, ensure_ascii=False) + '\n')

        logger.info(f"Deslexicalization logging initialized: {log_file}")

    def log_operation(self, operation: Dict[str, Any]):
        """
        Log a single deslexicalization operation.

        Args:
            operation: Dictionary containing operation details
        """
        operation['timestamp'] = datetime.now().isoformat()

        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(operation, ensure_ascii=False) + '\n')

    def log_summary(self, summary: Dict[str, Any]):
        """
        Log a summary of deslexicalization operations.

        Args:
            summary: Dictionary containing summary statistics
        """
        summary['type'] = 'summary'
        summary['timestamp'] = datetime.now().isoformat()
        self.log_operation(summary)


class NERDeslexicalizer:
    """
    Main deslexicalization class for NER training data.

    Handles deslexicalization of:
    - VOTER entities (using fake names or generic tokens)
    - Municipality-specific terms (→ @MUNICIPALITY)
    - Location names (→ @LOCAL)
    """

    # Common Portuguese municipality names to replace
    MUNICIPALITY_NAMES = {
        'PORTO', 'LISBOA', 'COIMBRA', 'BRAGA', 'FARO', 'ÉVORA', 'GUIMARÃES',
        'SETÚBAL', 'FUNCHAL', 'AVEIRO', 'VISEU', 'LEIRIA', 'BRAGANÇA',
        'VILA NOVA DE GAIA', 'AMADORA', 'MATOSINHOS', 'GONDOMAR',
        'ALANDROAL', 'CAMPO MAIOR', 'COVILHÃ', 'FUNDÃO',
        # Add variants without accents
        'EVORA', 'GUIMARAES', 'SETUBAL', 'COVILHA', 'FUNDAO', 'CAMPOMAIOR'
    }

    # Common location terms in municipal documents
    LOCATION_TERMS = {
        'PRAÇA', 'RUA', 'AVENIDA', 'LARGO', 'BAIRRO', 'FREGUESIA',
        'JARDIM', 'PARQUE', 'EDIFÍCIO', 'EDIFICIO',
        # Add English variants for multilingual support
        'STREET', 'AVENUE', 'SQUARE', 'PARK', 'BUILDING', 'DISTRICT'
    }

    def __init__(
        self,
        use_fake_names: bool = True,
        faker_locale: str = 'pt_PT',
        replace_municipalities: bool = True,
        replace_locations: bool = True,
        log_file: Optional[Path] = None
    ):
        """
        Initialize the NER deslexicalizer.

        Args:
            use_fake_names: If True, use Faker for VOTER entities, else use generic tokens
            faker_locale: Locale for Faker (default: 'pt_PT')
            replace_municipalities: If True, replace municipality names with @MUNICIPALITY
            replace_locations: If True, replace location names with @LOCAL
            log_file: Path to log file (if None, logging is disabled)
        """
        self.use_fake_names = use_fake_names
        self.faker_locale = faker_locale
        self.replace_municipalities = replace_municipalities
        self.replace_locations = replace_locations

        # Initialize person name detector
        self.person_detector = PersonNameDetector()

        # Initialize logger if log file is provided
        self.logger = DeslexicalizationLogger(log_file) if log_file else None

        # Statistics
        self.stats = defaultdict(int)

        logger.info(f"NER Deslexicalizer initialized:")
        logger.info(f"  - Use fake names: {use_fake_names}")
        logger.info(f"  - Faker locale: {faker_locale}")
        logger.info(f"  - Replace municipalities: {replace_municipalities}")
        logger.info(f"  - Replace locations: {replace_locations}")
        logger.info(f"  - Logging: {'enabled' if log_file else 'disabled'}")

    def _is_municipality_reference(self, token: str, entity_type: str) -> bool:
        """
        Check if a token is a municipality reference.

        Args:
            token: Token to check
            entity_type: Entity type (e.g., 'SUBJECT', 'VOTING')

        Returns:
            True if token is a municipality reference
        """
        token_upper = token.upper()

        # Check if token is in our municipality list
        if token_upper in self.MUNICIPALITY_NAMES:
            return True

        # Check for common municipality patterns
        municipality_patterns = ['CÂMARA', 'CAMARA', 'MUNICÍPIO', 'MUNICIPIO', 'AUTARQUIA']
        if any(pattern in token_upper for pattern in municipality_patterns):
            return True

        return False

    def _is_location_reference(self, token: str, entity_type: str) -> bool:
        """
        Check if a token is a location reference.

        Args:
            token: Token to check
            entity_type: Entity type

        Returns:
            True if token is a location reference
        """
        token_upper = token.upper()

        # Check if token starts with location terms
        for location_term in self.LOCATION_TERMS:
            if token_upper.startswith(location_term):
                return True

        return False

    def _replace_entity_tokens(
        self,
        tokens: List[str],
        labels: List[str],
        entity_type: str,
        replacement_token: str
    ) -> Tuple[List[str], int]:
        """
        Replace all tokens of a specific entity type with a replacement token.

        Args:
            tokens: List of tokens
            labels: List of BIO labels
            entity_type: Entity type to replace (e.g., 'SUBJECT', 'VOTING')
            replacement_token: Token to replace with (e.g., '@MUNICIPALITY', '@LOCAL')

        Returns:
            Tuple of (modified_tokens, count of replacements)
        """
        modified_tokens = tokens.copy()
        replacements = 0

        for i, (token, label) in enumerate(zip(tokens, labels)):
            # Check if this is the target entity type
            if label.startswith('B-' + entity_type) or label.startswith('I-' + entity_type):
                # Only replace if it matches our criteria
                if entity_type in ['SUBJECT', 'VOTING']:
                    if self.replace_municipalities and self._is_municipality_reference(token, entity_type):
                        modified_tokens[i] = replacement_token
                        replacements += 1
                    elif self.replace_locations and self._is_location_reference(token, entity_type):
                        modified_tokens[i] = '@LOCAL'
                        replacements += 1

        return modified_tokens, replacements

    def deslexicalize_example(
        self,
        example: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Apply deslexicalization to a single example.

        Args:
            example: Dictionary containing 'id', 'tokens', 'labels', 'text'

        Returns:
            Tuple of (modified_example, operation_log)
        """
        example_id = example.get('id', 'unknown')
        tokens = example['tokens'].copy()
        labels = example['labels'].copy()
        original_text = example.get('text', ' '.join(tokens))

        operation_log = {
            'type': 'deslexicalization',
            'example_id': example_id,
            'original_tokens': tokens.copy(),
            'original_text': original_text,
            'replacements': []
        }

        # Step 1: Replace VOTER entities using the existing deslexicalization_utils
        voter_modified_tokens, voter_tags, voter_text, voter_stats = deslexicalize_segment(
            tokens=tokens,
            tags=labels,
            text=original_text,
            enable_deslexicalization=True,
            use_fake_names=self.use_fake_names,
            faker_locale=self.faker_locale,
            detector=self.person_detector
        )

        # Update tokens with VOTER replacements
        tokens = voter_modified_tokens

        # Log VOTER replacements
        if voter_stats['names_detected'] > 0:
            for pattern in voter_stats['patterns_found']:
                operation_log['replacements'].append({
                    'entity_type': 'VOTER',
                    'original': pattern['original'],
                    'replacement': 'fake_name' if self.use_fake_names else '[PERSON]',
                    'start_idx': pattern['start_idx'],
                    'end_idx': pattern['end_idx']
                })
                self.stats['voter_replacements'] += 1

        # Step 2: Replace municipality references
        if self.replace_municipalities:
            municipality_tokens, muni_count = self._replace_entity_tokens(
                tokens, labels, 'SUBJECT', '@MUNICIPALITY'
            )
            if muni_count > 0:
                operation_log['replacements'].append({
                    'entity_type': 'MUNICIPALITY',
                    'count': muni_count,
                    'replacement': '@MUNICIPALITY'
                })
                self.stats['municipality_replacements'] += muni_count
                tokens = municipality_tokens

            # Also check VOTING entities for municipalities
            municipality_tokens, muni_count = self._replace_entity_tokens(
                tokens, labels, 'VOTING', '@MUNICIPALITY'
            )
            if muni_count > 0:
                operation_log['replacements'].append({
                    'entity_type': 'MUNICIPALITY',
                    'count': muni_count,
                    'replacement': '@MUNICIPALITY'
                })
                self.stats['municipality_replacements'] += muni_count
                tokens = municipality_tokens

        # Step 3: Replace location references
        if self.replace_locations:
            location_tokens, loc_count = self._replace_entity_tokens(
                tokens, labels, 'SUBJECT', '@LOCAL'
            )
            if loc_count > 0:
                operation_log['replacements'].append({
                    'entity_type': 'LOCATION',
                    'count': loc_count,
                    'replacement': '@LOCAL'
                })
                self.stats['location_replacements'] += loc_count
                tokens = location_tokens

        # Reconstruct text
        modified_text = ' '.join(tokens)

        # Normalize whitespace and case if any replacements were made
        if operation_log['replacements']:
            modified_text = normalize_whitespace(modified_text)
            modified_text = normalize_case_to_sentence_case(modified_text)

        # Create modified example
        modified_example = {
            'id': example_id,
            'tokens': tokens,
            'labels': labels,  # Labels remain unchanged
            'text': modified_text
        }

        # Add original data if replacements were made
        if operation_log['replacements']:
            operation_log['modified_tokens'] = tokens
            operation_log['modified_text'] = modified_text
            self.stats['examples_modified'] += 1
        else:
            self.stats['examples_unmodified'] += 1

        # Log operation
        if self.logger and operation_log['replacements']:
            self.logger.log_operation(operation_log)

        return modified_example, operation_log

    def deslexicalize_dataset(
        self,
        examples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Apply deslexicalization to an entire dataset.

        Args:
            examples: List of example dictionaries

        Returns:
            List of modified examples
        """
        logger.info(f"Starting deslexicalization of {len(examples)} examples...")

        modified_examples = []

        for i, example in enumerate(examples):
            modified_example, operation_log = self.deslexicalize_example(example)
            modified_examples.append(modified_example)

            # Log progress
            if (i + 1) % 100 == 0:
                logger.info(f"  Processed {i + 1}/{len(examples)} examples")

        # Log summary
        logger.info(f"Deslexicalization complete:")
        logger.info(f"  - Examples modified: {self.stats['examples_modified']}")
        logger.info(f"  - Examples unmodified: {self.stats['examples_unmodified']}")
        logger.info(f"  - VOTER replacements: {self.stats['voter_replacements']}")
        logger.info(f"  - Municipality replacements: {self.stats['municipality_replacements']}")
        logger.info(f"  - Location replacements: {self.stats['location_replacements']}")

        # Log summary to file
        if self.logger:
            self.logger.log_summary(dict(self.stats))

        return modified_examples

    def get_stats(self) -> Dict[str, int]:
        """Get deslexicalization statistics."""
        return dict(self.stats)


def apply_deslexicalization(
    examples: List[Dict[str, Any]],
    config: Dict[str, Any],
    log_file: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """
    Apply deslexicalization to examples based on config settings.

    This is the main entry point for deslexicalization in the training pipeline.

    Args:
        examples: List of example dictionaries from JSONL
        config: Deslexicalization configuration dictionary
        log_file: Path to log file (optional)

    Returns:
        List of deslexicalized examples
    """
    if not config.get('enabled', False):
        logger.info("Deslexicalization is disabled")
        return examples

    # Create deslexicalizer
    deslexicalizer = NERDeslexicalizer(
        use_fake_names=config.get('use_fake_names', True),
        faker_locale=config.get('faker_locale', 'pt_PT'),
        replace_municipalities=config.get('replace_municipalities', True),
        replace_locations=config.get('replace_locations', True),
        log_file=log_file
    )

    # Apply deslexicalization
    modified_examples = deslexicalizer.deslexicalize_dataset(examples)

    return modified_examples
