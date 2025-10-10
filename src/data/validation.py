"""
BIO tagging validation utilities for NER data processing.
"""

from typing import List, Set, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Valid BIO tag transitions for different entity types
VALID_TRANSITIONS = {
    'BIO': {
        'B': ['B', 'I', 'O'],
        'I': ['B', 'I', 'O'],
        'O': ['B', 'O'],
    }
}


def validate_bio_sequence(labels: List[str], scheme: str = 'BIO', example_id: str = None) -> List[str]:
    """
    Validate and correct BIO tagging sequence according to BIO constraints.
    
    Args:
        labels: List of BIO labels
        scheme: Tagging scheme ('BIO')
        example_id: Optional example identifier for logging
        
    Returns:
        Corrected label sequence
    """
    if scheme not in VALID_TRANSITIONS:
        logger.warning(f"Unknown scheme {scheme}, returning original labels")
        return labels
    
    if not labels:
        return labels
    
    corrected_labels = []
    valid_transitions = VALID_TRANSITIONS[scheme]
    corrections_made = 0
    
    for i, current_label in enumerate(labels):
        if i == 0:
            # First label - check if it's a valid starting label
            if current_label.startswith('I-'):
                # Invalid: sequence cannot start with I-
                corrected_label = current_label.replace('I-', 'B-')
                corrected_labels.append(corrected_label)
                corrections_made += 1
                if example_id:
                    logger.debug(f"[{example_id}] Corrected starting I- to B-: {current_label} -> {corrected_label}")
            else:
                corrected_labels.append(current_label)
        else:
            # Check transition from previous label
            prev_label = corrected_labels[-1]
            transition = (prev_label, current_label)
            
            if transition in valid_transitions and valid_transitions[transition]:
                # Valid transition
                corrected_labels.append(current_label)
            else:
                # Invalid transition - apply correction rules
                if current_label.startswith('I-'):
                    # Invalid I- tag
                    entity_type = current_label[2:]  # Extract entity type
                    prev_entity_type = prev_label[2:] if prev_label != 'O' else None
                    
                    if prev_label == 'O':
                        # I- after O -> convert to B-
                        corrected_label = f'B-{entity_type}'
                    elif prev_entity_type != entity_type:
                        # I- with different entity type -> convert to B-
                        corrected_label = f'B-{entity_type}'
                    else:
                        # This shouldn't happen with our rules, but keep original
                        corrected_label = current_label
                    
                    corrected_labels.append(corrected_label)
                    corrections_made += 1
                    
                    if example_id:
                        logger.debug(f"[{example_id}] Corrected invalid I- transition: {prev_label} -> {current_label} became {prev_label} -> {corrected_label}")
                else:
                    # Other invalid transitions (shouldn't occur with our rules)
                    corrected_labels.append(current_label)
    
    if corrections_made > 0 and example_id:
        logger.info(f"[{example_id}] Made {corrections_made} BIO corrections")
    
    return corrected_labels


def validate_bio_batch(label_sequences: List[List[str]], scheme: str = 'BIO') -> Tuple[List[List[str]], Dict[str, Any]]:
    """
    Validate a batch of BIO sequences.
    
    Args:
        label_sequences: List of label sequences
        scheme: Tagging scheme ('BIO')
        
    Returns:
        Tuple of (corrected_sequences, validation_stats)
    """
    corrected_sequences = []
    total_corrections = 0
    sequences_corrected = 0
    
    for i, seq in enumerate(label_sequences):
        original_seq = seq.copy()
        corrected_seq = validate_bio_sequence(seq, scheme, f"batch_{i}")
        corrected_sequences.append(corrected_seq)
        
        # Count corrections
        seq_corrections = sum(1 for orig, corr in zip(original_seq, corrected_seq) if orig != corr)
        if seq_corrections > 0:
            sequences_corrected += 1
            total_corrections += seq_corrections
    
    stats = {
        'total_sequences': len(label_sequences),
        'sequences_corrected': sequences_corrected,
        'total_corrections': total_corrections,
        'correction_rate': sequences_corrected / len(label_sequences) if label_sequences else 0
    }
    
    return corrected_sequences, stats
