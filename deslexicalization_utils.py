#!/usr/bin/env python3
"""
Utility functions for person name deslexicalization in NER training data.

This module provides functions to detect and replace person names with generic tokens
to improve model generalization and prevent memorization of specific individuals.
"""

import re
from typing import List, Tuple, Set, Optional
import logging
import unicodedata

# Note: Removed spaCy dependency - not needed with intelligent classification

# Try to import faker, but handle gracefully if not available
try:
    from faker import Faker
    FAKER_AVAILABLE = True
except ImportError:
    FAKER_AVAILABLE = False
    Faker = None

logger = logging.getLogger(__name__)

def reconstruct_text_with_proper_spacing(original_text: str, original_tokens: List[str], 
                                        modified_tokens: List[str]) -> str:
    """
    Reconstruct text from modified tokens while preserving original spacing.
    
    Args:
        original_text: The original text string
        original_tokens: List of original tokens
        modified_tokens: List of modified tokens (same length as original_tokens)
        
    Returns:
        Reconstructed text with preserved spacing
    """
    if len(original_tokens) != len(modified_tokens):
        logger.warning(f"Token length mismatch: {len(original_tokens)} vs {len(modified_tokens)}")
        return " ".join(modified_tokens)
    
    if not original_tokens:
        return ""
    
    # Simple approach: try to preserve spacing by finding token positions in original text
    result = original_text
    
    # Replace tokens from right to left to avoid position shifts
    replacements = []
    current_pos = 0
    
    for i, (orig_token, mod_token) in enumerate(zip(original_tokens, modified_tokens)):
        if orig_token != mod_token:
            # Find the position of this token in the remaining text
            token_pos = result.find(orig_token, current_pos)
            if token_pos != -1:
                replacements.append((token_pos, token_pos + len(orig_token), mod_token))
                current_pos = token_pos + len(orig_token)
    
    # Apply replacements from right to left
    for start_pos, end_pos, replacement in reversed(replacements):
        result = result[:start_pos] + replacement + result[end_pos:]
    
    return result

def normalize_whitespace(text: str) -> str:
    """
    Normalize excessive whitespace while preserving single spaces and newlines.
    
    Args:
        text: Input text with potentially excessive whitespace
        
    Returns:
        Text with normalized whitespace
    """
    # Replace multiple spaces (but not newlines) with single spaces
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Replace multiple consecutive newlines with maximum of 2 newlines
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # Remove trailing whitespace from each line
    lines = text.split('\n')
    lines = [line.rstrip() for line in lines]
    text = '\n'.join(lines)
    
    return text

def normalize_accents(text: str) -> str:
    """
    Normalize accents in text for comparison purposes.
    
    Args:
        text: Input text with potential accents
        
    Returns:
        Text with accents normalized
    """
    if not text:
        return text
    
    # Normalize unicode to decomposed form (NFD)
    normalized = unicodedata.normalize('NFD', text)
    
    # Remove combining characters (accents)
    no_accents = ''.join(
        char for char in normalized 
        if unicodedata.category(char) != 'Mn'
    )
    
    return no_accents

def normalize_case_to_sentence_case(text: str) -> str:
    """
    Convert ALL CAPS text to proper sentence case while preserving acronyms and proper nouns.
    
    Args:
        text: Input text that may be in ALL CAPS
        
    Returns:
        Text with proper sentence case
    """
    if not text:
        return text
    
    # Simplified word categories for case normalization
    KEEP_CAPS = {'ID', 'CM', 'PDF', 'URL', 'HTTP', 'HTTPS', 'API', 'GPS', 'USB', 'DVD', 'CD', 'TV', 'PC', 
                 'BE', 'PS', 'PSD', 'CDS', 'PCP', 'PAN', 'IL', 'CH', 'LIVRE', 'PP', 'MPT', 'PTP', 'PDR'}
    MONTHS = {'JANEIRO', 'FEVEREIRO', 'MARÇO', 'ABRIL', 'MAIO', 'JUNHO', 
              'JULHO', 'AGOSTO', 'SETEMBRO', 'OUTUBRO', 'NOVEMBRO', 'DEZEMBRO'}
    ARTICLES_PREPOSITIONS = {'A', 'O', 'AS', 'OS', 'DE', 'DA', 'DO', 'DAS', 'DOS', 'EM', 'NA', 'NO', 
                            'NAS', 'NOS', 'POR', 'PARA', 'COM', 'E', 'OU', 'MAS', 'QUE'}
    
    def process_word(word):
        # Keep punctuation separate
        prefix_punct = ''
        suffix_punct = ''
        clean_word = word
        
        # Extract leading punctuation
        while clean_word and not clean_word[0].isalnum():
            prefix_punct += clean_word[0]
            clean_word = clean_word[1:]
        
        # Extract trailing punctuation
        while clean_word and not clean_word[-1].isalnum():
            suffix_punct = clean_word[-1] + suffix_punct
            clean_word = clean_word[:-1]
        
        if not clean_word:
            return word
        
        # If the word should keep caps (acronyms)
        if clean_word.upper() in KEEP_CAPS:
            return prefix_punct + clean_word.upper() + suffix_punct
        
        # If it's a month, capitalize it
        if clean_word.upper() in MONTHS:
            return prefix_punct + clean_word.capitalize() + suffix_punct
        
        # If it's an article/preposition (but not at start of sentence)
        if clean_word.upper() in ARTICLES_PREPOSITIONS:
            return prefix_punct + clean_word.lower() + suffix_punct
        
        # If it's all caps and more than 1 character, convert to lowercase
        if clean_word.isupper() and len(clean_word) > 1:
            return prefix_punct + clean_word.lower() + suffix_punct
        
        # Otherwise keep as is
        return word
    
    # Split into sentences and process each one
    sentences = re.split(r'([.!?]+)', text)
    result_sentences = []
    
    for i, sentence in enumerate(sentences):
        if not sentence.strip():
            result_sentences.append(sentence)
            continue
            
        # If this is punctuation, keep as is
        if re.match(r'^[.!?]+$', sentence):
            result_sentences.append(sentence)
            continue
        
        # Process words in the sentence
        # Preserve leading and trailing whitespace
        leading_space = len(sentence) - len(sentence.lstrip())
        trailing_space = len(sentence) - len(sentence.rstrip())
        
        stripped_sentence = sentence.strip()
        words = stripped_sentence.split()
        if not words:
            result_sentences.append(sentence)
            continue
            
        processed_words = []
        for j, word in enumerate(words):
            # Process the word normally first
            processed_word = process_word(word)
            
            # For first word of sentence, ensure it's capitalized (overriding lowercase rules)
            if j == 0 and processed_word:
                # Find the first alphabetic character and capitalize it
                for k, char in enumerate(processed_word):
                    if char.isalpha():
                        processed_word = processed_word[:k] + char.upper() + processed_word[k+1:]
                        break
            
            processed_words.append(processed_word)
        
        # Restore leading and trailing whitespace
        processed_sentence = ' '.join(processed_words)
        if leading_space > 0:
            processed_sentence = sentence[:leading_space] + processed_sentence
        if trailing_space > 0:
            processed_sentence = processed_sentence + sentence[-trailing_space:]
        
        result_sentences.append(processed_sentence)
    
    return ''.join(result_sentences)

# Portuguese titles and roles commonly used in municipal documents
PORTUGUESE_TITLES = {
    'Dr.', 'Dra.', 'Sr.', 'Sra.', 'Srª', 'Exmo.', 'Exma.', 'Prof.', 'Profa.',
    'Eng.', 'Arq.', 'DOUTOR', 'DOUTORA', 'SENHOR', 'SENHORA'
}

MUNICIPAL_ROLES = {
    'PRESIDENTE', 'VEREADOR', 'VEREADORA', 'SECRETÁRIO', 'SECRETÁRIA',
    'DIRECTOR', 'DIRECTORA', 'DIRETOR', 'DIRETORA', 'CHEFE', 'COORDENADOR', 
    'COORDENADORA', 'TÉCNICO', 'TÉCNICA', 'ASSISTENTE'
}

# Note: Removed EXCLUSION_WORDS - now handled by intelligent classification in is_likely_person_name()

# Note: Removed PROTECTED_PHRASES - now handled by intelligent classification in is_likely_person_name()

# Note: Removed PROTECTED_INSTITUTIONAL_ENTITIES - now using intelligent pattern-based classification

class PersonNameDetector:
    """Detector for person names in Portuguese municipal documents."""
    
    def __init__(self):
        """Initialize the person name detector."""
        pass  # Simplified - no spaCy dependency needed
    
    def is_capitalized_word(self, token: str) -> bool:
        """Check if token is a capitalized word (likely a proper noun)."""
        return (token.isalpha() and 
                token[0].isupper() and 
                len(token) > 1)
    
    def is_title(self, token: str) -> bool:
        """Check if token is a title or role."""
        return token in PORTUGUESE_TITLES or token in MUNICIPAL_ROLES
    
    # Note: Removed is_protected_phrase method - logic moved to is_likely_person_name()
    
    def is_likely_person_name(self, entity_tokens: List[str]) -> bool:
        """
        Classify if entity tokens represent a person name vs institutional reference.
        
        Uses linguistic patterns to distinguish between:
        - Person names: "Dr. João Silva", "Maria Santos" 
        - Institutional refs: "A eleita pelo PS", "O presidente", "A câmara"
        
        Args:
            entity_tokens: List of tokens forming the entity
            
        Returns:
            True if likely a person name, False if institutional reference
        """
        if not entity_tokens:
            return False
            
        entity_text = ' '.join(entity_tokens).upper()
        
        # 1. Strong institutional indicators - articles at start
        if entity_text.startswith(('A ', 'O ', 'AS ', 'OS ')):
            return False
        
        # 2. Enhanced political party pattern detection
        # Matches: "do BE", "pelo PS", "da CDS", etc.
        political_prepositions = ['DO ', 'DA ', 'DOS ', 'DAS ', 'PELO ', 'PELA ']
        if any(entity_text.startswith(prep) for prep in political_prepositions):
            # Check if what follows the preposition looks like a political party
            for prep in political_prepositions:
                if entity_text.startswith(prep):
                    remainder = entity_text[len(prep):].strip()
                    # Political parties are typically 2-6 character abbreviations or known political words
                    if (len(remainder) <= 6 and remainder.isalpha()) or \
                       any(word in remainder for word in ['BLOCO', 'ESQUERDA', 'SOCIAL', 'DEMOCRATA', 'COMUNISTA', 'CENTRO']):
                        return False
                    break
        
        # 2.5. Detect political party references within longer entities
        # Matches: "Senhor Vereador do BE", "Deputada pelo PS", etc.
        political_party_indicators = ['BE', 'PS', 'PSD', 'CDS', 'PCP', 'PAN', 'IL', 'CH', 'LIVRE', 'CDU']
        if any(party in entity_text for party in political_party_indicators):
            # Check if it's a political reference pattern
            if any(word in entity_text for word in ['VEREADOR', 'DEPUTAD', 'SENHOR', 'SENHORA', 'DO', 'DA', 'PELO', 'PELA']):
                return False
        
        # 3. Enhanced institutional detection with pattern-based approach
        # Normalize entity text for accent-insensitive matching
        normalized_entity_text = normalize_accents(entity_text)
        
        # Pattern: Words ending in institutional suffixes
        institutional_suffixes = ['MUNICIPAL', 'SOCIAL', 'CIVIL', 'NACIONAL', 'VOLUNTARIOS']
        if any(normalized_entity_text.endswith(suffix) for suffix in institutional_suffixes):
            return False
        
        # Pattern: Common institutional word roots (more elegant than full word lists)
        institutional_roots = ['EXECUT', 'CAMAR', 'ASSEMBL', 'COMISS', 'MUNICIP', 'AUTARQU', 
                              'PROTEC', 'BOMBER', 'POLIC', 'GUARD', 'SERVIC', 'CENTR', 'SAUD', 'SEGURANC']
        if any(root in normalized_entity_text for root in institutional_roots):
            return False
        
        # Pattern: Known institutional entities (keep essential ones)
        essential_institutional_words = {'ELEITA', 'ELEITO', 'CRUZ', 'VERMELHA'}
        if any(word in normalized_entity_text for word in essential_institutional_words):
            return False
        
        # 4. Role detection (enhanced to be more precise)
        role_words = {'PRESIDENTE', 'VEREADOR', 'VEREADORA', 'SECRETARIO', 'SECRETARIA'}
        if len(entity_tokens) <= 2 and any(normalize_accents(word.upper()) in role_words for word in entity_tokens):
            return False
        
        # 5. Fixed preposition pattern detection (removes space requirement bug)
        preposition_words = {'PELO', 'PELA', 'DO', 'DA', 'DOS', 'DAS', 'DE'}
        if entity_tokens and entity_tokens[0].upper() in preposition_words:
            return False
        
        # Strong person indicators - titles
        if any(token in PORTUGUESE_TITLES for token in entity_tokens):
            return True
            
        # Additional person indicators - typical name patterns
        # If all tokens are capitalized proper nouns without institutional words
        if (len(entity_tokens) >= 2 and 
            all(self.is_capitalized_word(token) for token in entity_tokens)):
            return True
        
        # Default to person if no strong institutional indicators found
        # This is conservative - prefer false positives over false negatives
        return True
    
    def detect_title_name_patterns(self, tokens: List[str]) -> List[Tuple[int, int]]:
        """
        Detect title + name patterns (e.g., "Dr. João Silva").
        
        Returns:
            List of (start_idx, end_idx) tuples for detected person names
        """
        patterns = []
        i = 0
        
        while i < len(tokens):
            # Look for title followed by capitalized words
            if self.is_title(tokens[i]):
                start_idx = i
                i += 1
                
                # Skip punctuation after title
                while i < len(tokens) and tokens[i] in {'.', ',', ':'}:
                    i += 1
                
                # Collect capitalized words after title
                name_start = i
                while (i < len(tokens) and 
                       self.is_capitalized_word(tokens[i]) and
                       i - name_start < 5):  # Max 5 name tokens
                    i += 1
                
                # If we found name tokens after title, add the pattern
                if i > name_start:
                    # Include the entire title + name span
                    patterns.append((start_idx, i))
                else:
                    i = start_idx + 1
            else:
                i += 1
        
        return patterns
    
    def detect_capitalized_sequences(self, tokens: List[str]) -> List[Tuple[int, int]]:
        """
        Detect sequences of capitalized words that likely represent person names.
        
        Returns:
            List of (start_idx, end_idx) tuples for detected person names
        """
        patterns = []
        i = 0
        
        while i < len(tokens):
            if self.is_capitalized_word(tokens[i]):
                start_idx = i
                
                # Collect consecutive capitalized words
                while (i < len(tokens) and 
                       self.is_capitalized_word(tokens[i]) and
                       i - start_idx < 4):  # Max 4 name tokens
                    i += 1
                
                # Only consider sequences of 2+ words as names, and limit length
                sequence_length = i - start_idx
                if sequence_length >= 2 and sequence_length <= 4:
                    patterns.append((start_idx, i))
                else:
                    i = start_idx + 1
            else:
                i += 1
        
        return patterns
    
    # Note: Removed spaCy validation - our intelligent classification is sufficient
    
    def detect_person_names(self, tokens: List[str], tags: List[str], text: str) -> List[Tuple[int, int]]:
        """
        Detect person names in a token sequence using intelligent classification.
        
        This method prioritizes VOTANTE entities when available, but applies intelligent
        filtering to distinguish actual person names from institutional references.
        
        Args:
            tokens: List of tokens
            tags: List of BIO tags
            text: Original text
            
        Returns:
            List of (start_idx, end_idx) tuples for detected person names
        """
        # Check if we have VOTANTE entities - prioritize them if available
        votante_entity_tokens = set()
        for i, tag in enumerate(tags):
            if tag.startswith('B-VOTANTE') or tag.startswith('I-VOTANTE'):
                votante_entity_tokens.add(i)
        
        if votante_entity_tokens:
            # Find continuous spans of VOTANTE tokens
            votante_spans = []
            current_span_start = None
            
            for i in range(len(tokens)):
                if i in votante_entity_tokens:
                    if current_span_start is None:
                        current_span_start = i
                else:
                    if current_span_start is not None:
                        votante_spans.append((current_span_start, i))
                        current_span_start = None
            
            # Handle case where span continues to the end
            if current_span_start is not None:
                votante_spans.append((current_span_start, len(tokens)))
            
            # Apply intelligent classification to filter spans
            filtered_spans = []
            for start_idx, end_idx in votante_spans:
                entity_tokens = tokens[start_idx:end_idx]
                entity_text = ' '.join(entity_tokens)
                
                if self.is_likely_person_name(entity_tokens):
                    filtered_spans.append((start_idx, end_idx))
                else:
                    logger.debug(f"Protecting institutional entity: '{entity_text}'")
            
            return filtered_spans
        
        # No VOTANTE entities found - return empty list (only replace VOTANTE entities)
        logger.debug("No VOTANTE entities found - no deslexicalization applied")
        return []


def deslexicalize_segment(tokens: List[str], tags: List[str], text: str,
                         enable_deslexicalization: bool = True,
                         deslexicalization_token: str = "[PERSON]",
                         detector: Optional[PersonNameDetector] = None,
                         use_fake_names: bool = False,
                         faker_locale: str = 'pt_PT') -> Tuple[List[str], List[str], str, dict]:
    """
    Apply deslexicalization to a segment by replacing person names with generic tokens or fake names.
    
    Uses intelligent classification to distinguish actual person names from institutional
    references, prioritizing VOTANTE entities when available.
    
    Args:
        tokens: List of tokens
        tags: List of BIO tags  
        text: Original text
        enable_deslexicalization: Whether to apply deslexicalization
        deslexicalization_token: Token to replace person names with (if not using fake names)
        detector: PersonNameDetector instance (will create if None)
        use_fake_names: If True, replace with fake names instead of generic tokens
        faker_locale: Locale for fake name generation (default: 'pt_PT' for Portuguese)
        
    Returns:
        Tuple of (modified_tokens, unchanged_tags, reconstructed_text, stats)
    """
    stats = {
        'names_detected': 0,
        'tokens_replaced': 0,
        'patterns_found': []
    }
    
    if not enable_deslexicalization or not tokens:
        return tokens, tags, text, stats
    
    # Create detector if not provided
    if detector is None:
        detector = PersonNameDetector()
    
    # Initialize Faker if using fake names
    fake = None
    if use_fake_names:
        if not FAKER_AVAILABLE:
            logger.warning("Faker not available, falling back to token replacement")
            use_fake_names = False
        else:
            try:
                fake = Faker(faker_locale)
                logger.debug(f"Initialized Faker with locale {faker_locale}")
            except Exception as e:
                logger.warning(f"Failed to initialize Faker with locale {faker_locale}: {e}. Using default locale.")
                fake = Faker()
    
    # Detect person names
    person_spans = detector.detect_person_names(tokens, tags, text)
    stats['names_detected'] = len(person_spans)
    stats['use_fake_names'] = use_fake_names
    
    if not person_spans:
        # No person names detected - return original text unchanged
        return tokens, tags, text, stats
    
    # Apply normalization only when there are replacements to be made
    normalized_text = normalize_whitespace(text)
    # Apply case normalization to convert ALL CAPS to proper sentence case
    normalized_text = normalize_case_to_sentence_case(normalized_text)
    
    # Create normalized tokens for consistent replacement
    # We need to tokenize the normalized text to match the replacements
    normalized_tokens = []
    temp_text = normalized_text
    for orig_token in tokens:
        # Find the token in the normalized text
        token_pos = temp_text.find(orig_token)
        if token_pos != -1:
            normalized_tokens.append(orig_token)
            temp_text = temp_text[token_pos + len(orig_token):]
        else:
            # If exact match not found, try case-insensitive match
            temp_lower = temp_text.lower()
            orig_lower = orig_token.lower()
            token_pos = temp_lower.find(orig_lower)
            if token_pos != -1:
                # Use the normalized version from the text
                actual_token = temp_text[token_pos:token_pos + len(orig_token)]
                normalized_tokens.append(actual_token)
                temp_text = temp_text[token_pos + len(orig_token):]
            else:
                normalized_tokens.append(orig_token)
    
    # Apply replacements
    modified_tokens = normalized_tokens.copy()
    
    for start_idx, end_idx in person_spans:
        # Record the pattern for statistics
        original_name = " ".join(tokens[start_idx:end_idx])
        span_length = end_idx - start_idx
        
        stats['patterns_found'].append({
            'original': original_name,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'length': span_length
        })
        
        if use_fake_names and fake:
            # Generate fake name with appropriate number of tokens
            fake_name_tokens = _generate_fake_name_tokens(fake, span_length)
            
            # Replace tokens with fake name tokens
            for i, fake_token in enumerate(fake_name_tokens):
                if start_idx + i < end_idx:
                    modified_tokens[start_idx + i] = fake_token
                    stats['tokens_replaced'] += 1
        else:
            # Replace tokens with static token (original behavior)
            for i in range(start_idx, end_idx):
                modified_tokens[i] = deslexicalization_token
                stats['tokens_replaced'] += 1
    
    # Reconstruct text from modified tokens with proper spacing
    modified_text = reconstruct_text_with_proper_spacing(normalized_text, normalized_tokens, modified_tokens)
    
    # Normalize excessive whitespace and case
    modified_text = normalize_whitespace(modified_text)
    modified_text = normalize_case_to_sentence_case(modified_text)
    
    replacement_method = "fake names" if use_fake_names else "generic tokens"
    logger.debug(f"Deslexicalized {stats['names_detected']} names using {replacement_method}, "
                f"replaced {stats['tokens_replaced']} tokens")
    
    return modified_tokens, tags, modified_text, stats


def _generate_fake_name_tokens(fake, num_tokens: int) -> List[str]:
    """
    Generate fake name tokens matching the required number of tokens.
    
    Args:
        fake: Faker instance
        num_tokens: Number of tokens needed
        
    Returns:
        List of fake name tokens
    """
    if num_tokens == 1:
        # Single token - use first name
        return [fake.first_name()]
    elif num_tokens == 2:
        # Two tokens - use first name + last name
        return [fake.first_name(), fake.last_name()]
    elif num_tokens == 3:
        # Three tokens - try to generate first + middle + last
        # Or use prefix + first + last (like "Dr. João Silva")
        return [fake.prefix(), fake.first_name(), fake.last_name()]
    else:
        # Four or more tokens - generate a full name and split/pad as needed
        full_name = fake.name()
        name_parts = full_name.split()
        
        if len(name_parts) >= num_tokens:
            # If we have enough parts, take the first num_tokens
            return name_parts[:num_tokens]
        else:
            # If we need more parts, add middle names
            result = name_parts.copy()
            while len(result) < num_tokens:
                result.insert(-1, fake.first_name())  # Insert middle names
            return result[:num_tokens]


def create_deslexicalization_report(all_stats: List[dict]) -> dict:
    """
    Create a summary report of deslexicalization statistics.
    
    Args:
        all_stats: List of stats dictionaries from deslexicalize_segment calls
        
    Returns:
        Summary statistics dictionary
    """
    total_segments = len(all_stats)
    segments_with_names = sum(1 for s in all_stats if s['names_detected'] > 0)
    total_names = sum(s['names_detected'] for s in all_stats)
    total_tokens_replaced = sum(s['tokens_replaced'] for s in all_stats)
    
    # Collect all patterns
    all_patterns = []
    for stats in all_stats:
        all_patterns.extend(stats['patterns_found'])
    
    # Find most common name patterns
    pattern_counts = {}
    for pattern in all_patterns:
        original = pattern['original']
        pattern_counts[original] = pattern_counts.get(original, 0) + 1
    
    most_common = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return {
        'total_segments': total_segments,
        'segments_with_names': segments_with_names,
        'segments_without_names': total_segments - segments_with_names,
        'total_names_detected': total_names,
        'total_tokens_replaced': total_tokens_replaced,
        'average_names_per_segment': total_names / max(total_segments, 1),
        'average_tokens_per_name': total_tokens_replaced / max(total_names, 1),
        'most_common_patterns': most_common
    }