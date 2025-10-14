#!/usr/bin/env python
"""
Event Constructor for Voting Event Extraction.

This module constructs structured voting events from BIO-tagged predictions.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class EventConstructor:
    """
    Constructs structured voting events from BIO-tagged predictions.

    Event Structure (HYBRID FORMAT):
    {
        'id': str,
        'has_voting_event': bool,
        'event': {
            'subject': str or None,                       # SINGULAR (first one if multiple)
            'participants': [{'text': str, 'position': str}],  # LIST (multiple allowed)
            'counting': [{'text': str, 'type': str}],     # LIST (multiple allowed)
            'voting_expressions': [str]                    # LIST (multiple allowed)
        }
    }

    Note: Only 'subject' is singular (takes first if multiple found).
    All other fields are lists to handle multiple occurrences.
    """
    
    def __init__(self):
        """Initialize the event constructor."""
        pass
    
    def construct_event(
        self,
        tokens: List[str],
        labels: List[str],
        example_id: str
    ) -> Dict[str, Any]:
        """
        Construct structured event from BIO-tagged sequence.
        
        Args:
            tokens: List of word tokens
            labels: List of BIO labels (e.g., 'B-SUBJECT', 'I-SUBJECT', 'O')
            example_id: Unique identifier for this example
            
        Returns:
            Event dictionary with has_voting_event and structured event
        """
        # Check if this segment contains a voting event
        has_voting = self._has_voting_event(labels)
        
        if not has_voting:
            return {
                'id': example_id,
                'has_voting_event': False,
                'event': None
            }
        
        # Extract all entity spans
        subject = self._extract_subject(tokens, labels)
        participants = self._extract_participants(tokens, labels)
        counting = self._extract_counting_list(tokens, labels)  # Returns LIST
        voting_expressions = self._extract_voting_expressions_list(tokens, labels)  # Returns LIST

        return {
            'id': example_id,
            'has_voting_event': True,
            'event': {
                'subject': subject,
                'participants': participants,
                'counting': counting,
                'voting_expressions': voting_expressions
            }
        }
    
    def construct_events(
        self,
        examples: List[Dict[str, Any]],
        predictions: List[List[str]]
    ) -> List[Dict[str, Any]]:
        """
        Construct events for multiple examples.
        
        Args:
            examples: List of examples with 'tokens' and 'id' fields
            predictions: List of predicted label sequences
            
        Returns:
            List of constructed events
        """
        events = []
        
        for example, pred_labels in zip(examples, predictions):
            event = self.construct_event(
                example['tokens'],
                pred_labels,
                example['id']
            )
            events.append(event)
        
        return events
    
    def _has_voting_event(self, labels: List[str]) -> bool:
        """Check if labels contain a voting event (B-VOTING tag)."""
        return any(label.startswith('B-VOTING') for label in labels)
    
    def _extract_subject(
        self,
        tokens: List[str],
        labels: List[str]
    ) -> Optional[str]:
        """
        Extract subject span (single string).
        
        If multiple SUBJECT spans exist, takes the first one (or could concatenate).
        
        Args:
            tokens: Word tokens
            labels: BIO labels
            
        Returns:
            Subject text string or None
        """
        spans = self._extract_spans_by_type(tokens, labels, 'SUBJECT')
        
        if not spans:
            return None
        
        # Take first subject span
        subject_text = ' '.join(spans[0])
        
        return subject_text
    
    def _extract_participants(
        self,
        tokens: List[str],
        labels: List[str]
    ) -> List[Dict[str, str]]:
        """
        Extract participant spans with positions.
        
        Parses VOTER-* labels to extract position (FAVOR, AGAINST, ABSTENTION, ABSENT).
        
        Args:
            tokens: Word tokens
            labels: BIO labels
            
        Returns:
            List of {'text': str, 'position': str} dictionaries
        """
        participants = []
        
        # Find all VOTER-* spans
        i = 0
        while i < len(labels):
            label = labels[i]
            
            if label.startswith('B-VOTER-'):
                # Extract position from label (e.g., B-VOTER-FAVOR -> FAVOR)
                span_type = label[2:]  # Remove 'B-' prefix (e.g., 'VOTER-FAVOR')
                position = span_type.split('-', 1)[1]  # Extract position part
                
                # Collect span tokens
                span_tokens = [tokens[i]]
                j = i + 1
                
                # Continue with I-VOTER-* tags
                while j < len(labels) and labels[j] == f'I-{span_type}':
                    span_tokens.append(tokens[j])
                    j += 1
                
                participant_text = ' '.join(span_tokens)
                
                participants.append({
                    'text': participant_text,
                    'position': position.capitalize()
                })
                
                i = j
            else:
                i += 1
        
        return participants
    
    def _extract_counting_list(
        self,
        tokens: List[str],
        labels: List[str]
    ) -> List[Dict[str, str]]:
        """
        Extract all counting spans with aggregation types (LIST of dicts).

        Parses COUNTING-* labels to extract type (UNANIMITY, MAJORITY).
        Returns all counting expressions found.

        Args:
            tokens: Word tokens
            labels: BIO labels

        Returns:
            List of {'text': str, 'type': str} dictionaries (empty list if none found)
        """
        counting_list = []
        i = 0

        while i < len(labels):
            label = labels[i]

            if label.startswith('B-COUNTING-'):
                # Extract counting type (e.g., B-COUNTING-UNANIMITY -> UNANIMITY)
                span_type = label[2:]  # Remove 'B-' prefix
                counting_type = span_type.split('-', 1)[1]  # Extract type part

                # Collect span tokens
                span_tokens = [tokens[i]]
                j = i + 1

                # Continue with I-COUNTING-* tags
                while j < len(labels) and labels[j] == f'I-{span_type}':
                    span_tokens.append(tokens[j])
                    j += 1

                counting_text = ' '.join(span_tokens)

                counting_list.append({
                    'text': counting_text,
                    'type': counting_type.capitalize()
                })

                i = j
            else:
                i += 1

        return counting_list
    
    def _extract_voting_expressions_list(
        self,
        tokens: List[str],
        labels: List[str]
    ) -> List[str]:
        """
        Extract all voting expression spans (LIST of strings).

        Returns all VOTING spans found.

        Args:
            tokens: Word tokens
            labels: BIO labels

        Returns:
            List of voting expression text strings (empty list if none found)
        """
        spans = self._extract_spans_by_type(tokens, labels, 'VOTING')

        # Convert all spans to strings and return as list
        return [' '.join(span) for span in spans]
    
    def _extract_spans_by_type(
        self,
        tokens: List[str],
        labels: List[str],
        span_type: str
    ) -> List[List[str]]:
        """
        Extract all spans of a given type.
        
        Args:
            tokens: Word tokens
            labels: BIO labels
            span_type: Type to extract (e.g., 'SUBJECT', 'VOTING')
            
        Returns:
            List of token lists, one per span
        """
        spans = []
        i = 0
        
        while i < len(labels):
            label = labels[i]
            
            if label == f'B-{span_type}':
                # Start of span
                span_tokens = [tokens[i]]
                j = i + 1
                
                # Collect continuation tokens
                while j < len(labels) and labels[j] == f'I-{span_type}':
                    span_tokens.append(tokens[j])
                    j += 1
                
                spans.append(span_tokens)
                i = j
            else:
                i += 1
        
        return spans


# Convenience function
def construct_events_from_predictions(
    examples: List[Dict[str, Any]],
    predictions: List[List[str]]
) -> List[Dict[str, Any]]:
    """
    Construct events from BIO predictions.
    
    Args:
        examples: List of examples with 'tokens' and 'id'
        predictions: List of predicted BIO label sequences
    
    Returns:
        List of structured event dictionaries
    """
    constructor = EventConstructor()
    return constructor.construct_events(examples, predictions)