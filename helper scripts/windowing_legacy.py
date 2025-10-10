#!/usr/bin/env python
"""
Unified Windowing Logic for NER Prediction and Training

This module provides centralized windowing logic that is used both during training
and prediction, ensuring consistency across the entire system.

Based on the dynamic windowing approach used in training, which creates windows
AFTER tokenization using the actual tokenizer to avoid estimation errors.
"""

import logging
from typing import List, Tuple, Dict, Any, Optional
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class WindowingManager:
    """
    Centralized windowing manager that provides consistent windowing logic
    for both training and prediction phases.
    """
    
    def __init__(self, tokenizer: AutoTokenizer, config: Optional[Dict[str, Any]] = None):
        """
        Initialize windowing manager.
        
        Args:
            tokenizer: The tokenizer to use for subword length calculations
            config: Configuration dictionary with windowing parameters
        """
        self.tokenizer = tokenizer
        config = config or {}
        
        # Extract windowing parameters with defaults
        self.max_length = config.get('max_length', 512)
        self.overlap_tokens = config.get('overlap_tokens', 50)
        self.safety_margin = config.get('windowing_safety_margin', 2)
        self.max_windows = config.get('max_windows_per_sequence', 20)
        self.max_window_tokens = config.get('max_window_tokens', 200)  # Max tokens per window

        # Effective max length (reserve space for [CLS] and [SEP])
        self.effective_max_length = self.max_length - self.safety_margin
        
        # Minimum step size to prevent infinite loops on degenerate inputs
        self.min_step_size = max(10, self.overlap_tokens // 5)
        
        logger.info(f"WindowingManager initialized: max_length={self.max_length}, "
                   f"overlap_tokens={self.overlap_tokens}, effective_max_length={self.effective_max_length}, "
                   f"min_step_size={self.min_step_size}")
    
    def needs_windowing(self, tokens: List[str]) -> bool:
        """
        Determine if a token sequence needs windowing using actual tokenizer.
        
        Args:
            tokens: List of word-level tokens
            
        Returns:
            True if windowing is needed, False otherwise
        """
        text = " ".join(tokens)
        try:
            subwords = self.tokenizer.tokenize(text)
            return len(subwords) > self.effective_max_length
        except Exception as e:
            logger.warning(f"Tokenization failed during windowing check: {e}")
            # Conservative fallback: assume 1.5x expansion (Portuguese typical)
            return len(tokens) * 1.5 > self.effective_max_length
    
    def create_windows(
        self, 
        tokens: List[str], 
        labels: Optional[List[str]] = None,
        example_id: str = "unknown"
    ) -> List[Tuple[List[str], Optional[List[str]]]]:
        """
        Create sliding windows from tokens using the same logic as training.
        
        Args:
            tokens: Original word tokens
            labels: BIO labels for tokens (optional, for prediction without labels)
            example_id: Example identifier for logging
            
        Returns:
            List of (window_tokens, window_labels) tuples
            window_labels will be None if labels parameter was None
        """
        if not tokens:
            logger.warning(f"Empty token sequence in {example_id}")
            return []
        
        # Validate labels if provided
        if labels is not None and len(tokens) != len(labels):
            logger.error(f"Token/label mismatch in {example_id}: {len(tokens)} vs {len(labels)}")
            return []
        
        # If no windowing needed, return as-is
        if not self.needs_windowing(tokens):
            return [(tokens, labels)]
        
        # Create sliding windows
        windows = []
        start_idx = 0
        window_id = 0
        
        logger.debug(f"Creating windows for {example_id}: {len(tokens)} tokens")
        
        while start_idx < len(tokens) and window_id < self.max_windows:
            # Calculate end index for this window (use max_window_tokens, not effective_max_length)
            end_idx = min(start_idx + self.max_window_tokens, len(tokens))
            
            # Extract window
            window_tokens = tokens[start_idx:end_idx]
            window_labels = labels[start_idx:end_idx] if labels else None
            
            # Check actual tokenized length and reduce if necessary
            window_tokens, window_labels = self._adjust_window_size(
                window_tokens, window_labels, example_id, window_id
            )
            
            # Avoid cutting entities in half - look for good cut point
            if end_idx < len(tokens) and len(window_tokens) > self.overlap_tokens:
                window_tokens, window_labels = self._find_optimal_cut_point(
                    window_tokens, window_labels
                )
            
            windows.append((window_tokens, window_labels))
            
            # Calculate next start position
            if start_idx + len(window_tokens) >= len(tokens):
                break
            
            # Move forward by window_size - overlap, ensuring progress
            # Use minimum step size to avoid infinite loops on degenerate inputs
            step_size = max(len(window_tokens) - self.overlap_tokens, self.min_step_size)
            start_idx = start_idx + step_size
            window_id += 1
        
        # Safety check
        if window_id >= self.max_windows:
            logger.error(f"Hit maximum window limit ({self.max_windows}) for {example_id}. "
                        f"This may indicate problematic tokens (e.g., long sequences of special characters).")
        
        logger.debug(f"Created {len(windows)} windows for {example_id}")
        return windows
    
    def _adjust_window_size(
        self, 
        window_tokens: List[str], 
        window_labels: Optional[List[str]],
        example_id: str,
        window_id: int
    ) -> Tuple[List[str], Optional[List[str]]]:
        """
        Adjust window size based on actual subword tokenization.
        
        Args:
            window_tokens: Initial window tokens
            window_labels: Initial window labels (or None)
            example_id: Example identifier for logging
            window_id: Window number for logging
            
        Returns:
            Adjusted (window_tokens, window_labels) tuple
        """
        if not window_tokens:
            return window_tokens, window_labels
        
        # Check actual tokenized length
        window_text = " ".join(window_tokens)
        try:
            window_subwords = self.tokenizer.tokenize(window_text)
        except Exception as e:
            logger.warning(f"Tokenization failed for {example_id} window {window_id}: {e}")
            return window_tokens, window_labels
        
        # If still too long, reduce window size iteratively
        reduction_attempts = 0
        max_attempts = 10
        
        while len(window_subwords) > self.effective_max_length and len(window_tokens) > 1:
            if reduction_attempts >= max_attempts:
                logger.warning(f"Max reduction attempts reached for {example_id} window {window_id}")
                break
            
            # Reduce by 10% and try again
            new_size = max(1, int(len(window_tokens) * 0.9))
            window_tokens = window_tokens[:new_size]
            if window_labels:
                window_labels = window_labels[:new_size]
            
            window_text = " ".join(window_tokens)
            try:
                window_subwords = self.tokenizer.tokenize(window_text)
            except Exception:
                break
            
            reduction_attempts += 1
        
        if reduction_attempts > 0:
            logger.debug(f"Reduced window size for {example_id} window {window_id}: "
                        f"{reduction_attempts} reductions, final size: {len(window_tokens)} tokens, "
                        f"{len(window_subwords)} subwords")
        
        return window_tokens, window_labels
    
    def _find_optimal_cut_point(
        self, 
        window_tokens: List[str], 
        window_labels: Optional[List[str]]
    ) -> Tuple[List[str], Optional[List[str]]]:
        """
        Find optimal cut point to avoid splitting entities.
        
        Args:
            window_tokens: Window tokens
            window_labels: Window labels (or None)
            
        Returns:
            Adjusted (window_tokens, window_labels) tuple
        """
        if not window_labels or len(window_tokens) <= self.overlap_tokens:
            return window_tokens, window_labels
        
        # Look backwards from end for a good cut point (preferably at "O" tag)
        cut_point = len(window_tokens)
        search_range = min(20, len(window_tokens) - self.overlap_tokens)
        
        for i in range(len(window_tokens) - 1, max(0, len(window_tokens) - search_range), -1):
            if window_labels[i] == "O":
                cut_point = i + 1
                break
        
        if cut_point < len(window_tokens):
            logger.debug(f"Found optimal cut point at token {cut_point} "
                        f"(was {len(window_tokens)})")
            return window_tokens[:cut_point], window_labels[:cut_point]
        
        return window_tokens, window_labels
    
    def merge_windowed_predictions(
        self, 
        windowed_predictions: List[Tuple[List[str], List[str], List[Dict[str, Any]]]],
        original_tokens: List[str]
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Merge predictions from multiple windows back into a single sequence.
        
        Args:
            windowed_predictions: List of (tokens, predictions, entities) from each window
            original_tokens: Original token sequence before windowing
            
        Returns:
            Tuple of (merged_predictions, merged_entities)
        """
        if not windowed_predictions:
            return ["O"] * len(original_tokens), []
        
        if len(windowed_predictions) == 1:
            # Single window - return as-is but ensure proper length
            tokens, predictions, entities = windowed_predictions[0]
            if len(predictions) == len(original_tokens):
                return predictions, entities
        
        # Multiple windows - need sophisticated merging
        merged_predictions = ["O"] * len(original_tokens)
        all_entities = []
        
        # Track which tokens have been assigned by which window
        token_assignments = {}
        
        window_start = 0
        for window_idx, (window_tokens, window_predictions, window_entities) in enumerate(windowed_predictions):
            # Find the actual start position of this window in original tokens
            actual_start = self._find_window_start_position(
                window_tokens, original_tokens, window_start
            )
            
            # Determine non-overlapping region for this window
            if window_idx == 0:
                # First window: use everything except overlap
                end_pos = min(len(window_predictions), 
                             len(window_predictions) - self.overlap_tokens // 2)
                start_pos = 0
            elif window_idx == len(windowed_predictions) - 1:
                # Last window: skip initial overlap
                start_pos = self.overlap_tokens // 2
                end_pos = len(window_predictions)
            else:
                # Middle window: skip overlap on both sides
                start_pos = self.overlap_tokens // 2
                end_pos = len(window_predictions) - self.overlap_tokens // 2
            
            # Apply predictions from non-overlapping region
            for i in range(start_pos, min(end_pos, len(window_predictions))):
                global_pos = actual_start + i
                if 0 <= global_pos < len(merged_predictions):
                    merged_predictions[global_pos] = window_predictions[i]
                    token_assignments[global_pos] = window_idx
            
            # Adjust entity positions and add to results
            for entity in window_entities:
                adjusted_entity = entity.copy()
                if 'start' in adjusted_entity and 'end' in adjusted_entity:
                    adjusted_entity['start'] += actual_start
                    adjusted_entity['end'] += actual_start
                    
                    # Only include entity if it's in the non-overlapping region
                    entity_start_in_window = adjusted_entity['start'] - actual_start
                    entity_end_in_window = adjusted_entity['end'] - actual_start
                    
                    if (start_pos <= entity_start_in_window < end_pos or
                        start_pos < entity_end_in_window <= end_pos):
                        all_entities.append(adjusted_entity)
            
            # Update window_start for next iteration
            window_start = actual_start + len(window_tokens) - self.overlap_tokens
        
        # Remove duplicate entities
        unique_entities = self._remove_duplicate_entities(all_entities)
        
        logger.debug(f"Merged {len(windowed_predictions)} windows: "
                    f"{len(unique_entities)} unique entities")
        
        return merged_predictions, unique_entities
    
    def _find_window_start_position(
        self, 
        window_tokens: List[str], 
        original_tokens: List[str], 
        hint_start: int
    ) -> int:
        """Find the actual start position of window tokens in original sequence."""
        # Start search from hint position
        search_start = max(0, hint_start)
        search_end = min(len(original_tokens), search_start + len(window_tokens) + self.overlap_tokens)
        
        # Look for exact match
        window_len = len(window_tokens)
        for start in range(search_start, search_end - window_len + 1):
            if original_tokens[start:start + window_len] == window_tokens:
                return start
        
        # Fallback to hint position
        return hint_start
    
    def _remove_duplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate entities from merged results."""
        if not entities:
            return []
        
        # Sort by start position
        sorted_entities = sorted(entities, key=lambda x: (x.get('start', 0), x.get('end', 0)))
        
        unique_entities = []
        for entity in sorted_entities:
            # Check if this entity overlaps significantly with any existing entity
            is_duplicate = False
            
            for existing in unique_entities:
                if (entity.get('type') == existing.get('type') and
                    self._entities_overlap(entity, existing)):
                    # Keep the longer entity
                    entity_len = entity.get('end', 0) - entity.get('start', 0)
                    existing_len = existing.get('end', 0) - existing.get('start', 0)
                    
                    if entity_len > existing_len:
                        # Replace existing with current
                        unique_entities.remove(existing)
                        unique_entities.append(entity)
                    
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_entities.append(entity)
        
        return unique_entities
    
    def _entities_overlap(self, entity1: Dict[str, Any], entity2: Dict[str, Any]) -> bool:
        """Check if two entities overlap significantly."""
        start1, end1 = entity1.get('start', 0), entity1.get('end', 0)
        start2, end2 = entity2.get('start', 0), entity2.get('end', 0)
        
        # Calculate overlap
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        overlap_len = max(0, overlap_end - overlap_start)
        
        # Calculate minimum entity length
        min_len = min(end1 - start1, end2 - start2)
        
        # Consider overlapping if overlap is > 50% of shorter entity
        return overlap_len > min_len * 0.5