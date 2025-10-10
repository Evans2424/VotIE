#!/usr/bin/env python
"""
Event Constructor for Voting Event Extraction.

This module constructs structured voting events from BIO-tagged predictions.
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class EventConstructor:
    """
    Constructs structured voting events from BIO-tagged predictions.
    
    Implements the paper's algorithm:
    1. Identify voting segments (containing VOTING span)
    2. Extract subject (SUBJECT spans)
    3. Extract participants (VOTER-* spans for (mention, position) pairs)
    4. Extract counting (COUNTING-* spans for aggregation type)
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
            Event dictionary with structure:
            {
                'id': str,
                'has_voting_event': bool,
                'event': {
                    'subjects': [str],
                    'participants': [{'text': str, 'position': str}],
                    'counting': [{'text': str, 'type': str}],
                    'voting_expressions': [str]
                }
            }
        """
        # Check if this segment contains a voting event
        has_voting = self._has_voting_event(labels)
        
        if not has_voting:
            return {
                'id': example_id,
                'has_voting_event': False,
                'event': {
                    'subjects': [],
                    'participants': [],
                    'counting': [],
                    'voting_expressions': []
                }
            }
        
        # Extract all entity spans
        subjects = self._extract_subjects(tokens, labels)
        participants = self._extract_participants(tokens, labels)
        counting = self._extract_counting(tokens, labels)
        voting_expressions = self._extract_voting_expressions(tokens, labels)
        
        return {
            'id': example_id,
            'has_voting_event': True,
            'event': {
                'subjects': subjects,
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
    
    def _extract_subjects(self, tokens: List[str], labels: List[str]) -> List[str]:
        """
        Extract subject spans from BIO labels.
        
        Args:
            tokens: Word tokens
            labels: BIO labels
            
        Returns:
            List of subject text strings
        """
        subjects = []
        spans = self._extract_spans_by_type(tokens, labels, 'SUBJECT')
        
        for span_tokens in spans:
            subject_text = ' '.join(span_tokens)
            subjects.append(subject_text)
        
        return subjects
    
    def _extract_participants(
        self,
        tokens: List[str],
        labels: List[str]
    ) -> List[Dict[str, str]]:
        """
        Extract participant spans with positions.
        
        Parses VOTER-* labels to extract position (FAVOR, AGAINST, ABSTENTION).
        
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
    
    def _extract_counting(
        self,
        tokens: List[str],
        labels: List[str]
    ) -> List[Dict[str, str]]:
        """
        Extract counting spans with aggregation types.
        
        Parses COUNTING-* labels to extract type (UNANIMITY, MAJORITY).
        
        Args:
            tokens: Word tokens
            labels: BIO labels
            
        Returns:
            List of {'text': str, 'type': str} dictionaries
        """
        counting = []
        
        # Find all COUNTING-* spans
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
                
                counting.append({
                    'text': counting_text,
                    'type': counting_type.capitalize()
                })
                
                i = j
            else:
                i += 1
        
        return counting
    
    def _extract_voting_expressions(
        self,
        tokens: List[str],
        labels: List[str]
    ) -> List[str]:
        """
        Extract voting expression spans.
        
        Args:
            tokens: Word tokens
            labels: BIO labels
            
        Returns:
            List of voting expression text strings
        """
        voting_expressions = []
        spans = self._extract_spans_by_type(tokens, labels, 'VOTING')
        
        for span_tokens in spans:
            voting_text = ' '.join(span_tokens)
            voting_expressions.append(voting_text)
        
        return voting_expressions
    
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
