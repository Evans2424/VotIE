#!/usr/bin/env python
"""
Unified Windowing Logic for NER Prediction and Training

This module provides centralized windowing logic that is used both during training
and prediction, ensuring consistency across the entire system.

Based on the dynamic windowing approach used in training, which creates windows
AFTER tokenization using the actual tokenizer to avoid estimation errors.

Best Practices Implemented:
1. Tokenizer-based validation: Only sequences exceeding actual subword limit are windowed
2. Conservative window sizing: Start with intelligent estimate, validate with actual tokenizer
3. Entity preservation: Avoid splitting entities across window boundaries
4. Overlap management: Track actual overlaps for accurate merging
5. Coverage verification: Ensure all tokens are covered in merged predictions
"""

import logging
from typing import List, Tuple, Dict, Any, Optional
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class WindowingManager:
    """
    Centralized windowing manager that provides consistent windowing logic
    for both training and prediction phases.

    Key Design Principles:
    - Only window sequences that exceed max_length when tokenized (not estimated)
    - Use actual tokenizer to validate window sizes (no guessing)
    - Preserve entity boundaries when possible
    - Track window metadata for accurate merging
    """

    # Constants for windowing behavior
    PORTUGUESE_SUBWORD_EXPANSION = 1.35  # Conservative estimate: PT tokens → subwords
    WINDOW_REDUCTION_FACTOR = 0.9        # Reduce by 10% when window too large
    MAX_REDUCTION_ATTEMPTS = 10          # Max iterations to fit window
    OPTIMAL_CUT_SEARCH_RANGE = 20        # Tokens to search for entity boundary
    ENTITY_OVERLAP_THRESHOLD = 0.5       # 50% overlap considered duplicate
    DEFAULT_SAFETY_MARGIN = 2            # Reserve for [CLS] and [SEP]
    DEFAULT_MAX_WINDOWS = 20             # Prevent infinite loops
    DEFAULT_OVERLAP_TOKENS = 50          # Default overlap between windows

    def __init__(self, tokenizer: AutoTokenizer, config: Optional[Dict[str, Any]] = None):
        """
        Initialize windowing manager.

        Args:
            tokenizer: The tokenizer to use for subword length calculations
            config: Configuration dictionary with windowing parameters:
                - max_length: Maximum sequence length (default: 512)
                - overlap_tokens: Overlap between windows (default: 50)
                - windowing_safety_margin: Reserve for special tokens (default: 2)
                - max_windows_per_sequence: Safety limit (default: 20)
                - subword_expansion_factor: Language-specific expansion (default: 1.35)

        Raises:
            ValueError: If parameters are invalid
        """
        self.tokenizer = tokenizer
        config = config or {}

        # Extract windowing parameters with defaults
        self.max_length = config.get('max_length', 512)
        self.overlap_tokens = config.get('overlap_tokens', self.DEFAULT_OVERLAP_TOKENS)
        self.safety_margin = config.get('windowing_safety_margin', self.DEFAULT_SAFETY_MARGIN)
        self.max_windows = config.get('max_windows_per_sequence', self.DEFAULT_MAX_WINDOWS)
        self.subword_expansion = config.get('subword_expansion_factor', self.PORTUGUESE_SUBWORD_EXPANSION)

        # Effective max length (reserve space for [CLS] and [SEP])
        self.effective_max_length = self.max_length - self.safety_margin

        # Calculate intelligent initial window size
        # Start conservatively to avoid excessive tokenization calls in _adjust_window_size
        self.initial_window_size = int(self.effective_max_length / self.subword_expansion)

        # Validate parameters
        self._validate_parameters()

        logger.info(f"WindowingManager initialized: max_length={self.max_length}, "
                   f"overlap_tokens={self.overlap_tokens}, effective_max_length={self.effective_max_length}, "
                   f"initial_window_size={self.initial_window_size}")

    def _validate_parameters(self):
        """Validate configuration parameters to catch errors early."""
        if self.overlap_tokens < 0:
            raise ValueError(f"overlap_tokens must be >= 0, got {self.overlap_tokens}")

        if self.overlap_tokens >= self.effective_max_length:
            raise ValueError(
                f"overlap_tokens ({self.overlap_tokens}) must be < effective_max_length ({self.effective_max_length})"
            )

        if self.max_length < 10:
            raise ValueError(f"max_length ({self.max_length}) is too small, must be >= 10")

        if self.initial_window_size < self.overlap_tokens:
            logger.warning(
                f"initial_window_size ({self.initial_window_size}) < overlap_tokens ({self.overlap_tokens}). "
                f"This may cause issues. Consider reducing overlap_tokens or increasing max_length."
            )

        if self.subword_expansion < 1.0:
            logger.warning(f"subword_expansion_factor ({self.subword_expansion}) < 1.0 is unusual")

    def needs_windowing(self, tokens: List[str]) -> bool:
        """
        Determine if a token sequence needs windowing using actual tokenizer.

        This is the core of the tokenizer-based approach: we don't estimate,
        we actually tokenize and check the real subword count.

        Args:
            tokens: List of word-level tokens

        Returns:
            True if windowing is needed, False otherwise
        """
        if not tokens:
            return False

        text = " ".join(tokens)
        try:
            subwords = self.tokenizer.tokenize(text)
            needs_window = len(subwords) > self.effective_max_length

            if needs_window:
                logger.debug(f"Sequence needs windowing: {len(tokens)} tokens → {len(subwords)} subwords "
                           f"(limit: {self.effective_max_length})")

            return needs_window
        except Exception as e:
            logger.warning(f"Tokenization failed during windowing check: {e}")
            # Conservative fallback: estimate based on expansion factor
            estimated_subwords = len(tokens) * self.subword_expansion
            return estimated_subwords > self.effective_max_length

    def create_windows(
        self,
        tokens: List[str],
        labels: Optional[List[str]] = None,
        example_id: str = "unknown"
    ) -> List[Tuple[List[str], Optional[List[str]]]]:
        """
        Create sliding windows from tokens using the same logic as training.

        Strategy:
        1. Start with initial_window_size (intelligent estimate)
        2. Validate with actual tokenizer, reduce if needed
        3. Find optimal cut points to avoid splitting entities
        4. Create overlapping windows for continuity

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

        # If no windowing needed, return as-is (core principle: only window when necessary)
        if not self.needs_windowing(tokens):
            return [(tokens, labels)]

        # Create sliding windows with metadata tracking
        windows = []
        start_idx = 0
        window_id = 0

        logger.debug(f"Creating windows for {example_id}: {len(tokens)} tokens")

        while start_idx < len(tokens) and window_id < self.max_windows:
            # Calculate end index using intelligent initial window size
            end_idx = min(start_idx + self.initial_window_size, len(tokens))

            # Extract window
            window_tokens = tokens[start_idx:end_idx]
            window_labels = labels[start_idx:end_idx] if labels else None

            # Validate and adjust based on actual tokenization
            window_tokens, window_labels = self._adjust_window_size(
                window_tokens, window_labels, example_id, window_id
            )

            if not window_tokens:
                logger.error(f"Empty window created for {example_id} at position {start_idx}")
                break

            # Find optimal cut point to avoid splitting entities
            if end_idx < len(tokens) and len(window_tokens) > self.overlap_tokens:
                window_tokens, window_labels = self._find_optimal_cut_point(
                    window_tokens, window_labels
                )

            windows.append((window_tokens, window_labels))

            # Calculate next start position
            window_size = len(window_tokens)
            if start_idx + window_size >= len(tokens):
                # Reached the end
                break

            # Move forward by window_size - overlap, ensuring progress
            step_size = max(window_size - self.overlap_tokens, 1)
            start_idx = start_idx + step_size
            window_id += 1

        # Verify coverage
        total_covered = sum(len(w[0]) for w in windows) - (len(windows) - 1) * self.overlap_tokens
        if total_covered < len(tokens):
            logger.warning(
                f"Windowing may not cover all tokens for {example_id}: "
                f"{total_covered} covered, {len(tokens)} total. "
                f"Created {len(windows)} windows."
            )

        # Safety check
        if window_id >= self.max_windows:
            logger.error(
                f"Hit maximum window limit ({self.max_windows}) for {example_id}. "
                f"Only {total_covered}/{len(tokens)} tokens covered."
            )

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

        This is where we validate our initial window size against the actual
        tokenizer output, reducing iteratively until it fits.

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
            # Return as-is and hope for the best
            return window_tokens, window_labels

        # If fits, return immediately
        if len(window_subwords) <= self.effective_max_length:
            return window_tokens, window_labels

        # If too long, reduce window size iteratively
        reduction_attempts = 0
        original_size = len(window_tokens)

        while len(window_subwords) > self.effective_max_length and len(window_tokens) > 1:
            if reduction_attempts >= self.MAX_REDUCTION_ATTEMPTS:
                logger.warning(
                    f"Max reduction attempts ({self.MAX_REDUCTION_ATTEMPTS}) reached for "
                    f"{example_id} window {window_id}. Final: {len(window_tokens)} tokens, "
                    f"{len(window_subwords)} subwords (limit: {self.effective_max_length})"
                )
                break

            # Reduce by 10% and try again
            new_size = max(1, int(len(window_tokens) * self.WINDOW_REDUCTION_FACTOR))
            window_tokens = window_tokens[:new_size]
            if window_labels:
                window_labels = window_labels[:new_size]

            window_text = " ".join(window_tokens)
            try:
                window_subwords = self.tokenizer.tokenize(window_text)
            except Exception:
                # If tokenization fails, stop here
                break

            reduction_attempts += 1

        if reduction_attempts > 0:
            logger.debug(
                f"Reduced window size for {example_id} window {window_id}: "
                f"{original_size} → {len(window_tokens)} tokens "
                f"({reduction_attempts} iterations, {len(window_subwords)} subwords)"
            )

        return window_tokens, window_labels

    def _find_optimal_cut_point(
        self,
        window_tokens: List[str],
        window_labels: Optional[List[str]]
    ) -> Tuple[List[str], Optional[List[str]]]:
        """
        Find optimal cut point to avoid splitting entities.

        Look backwards from the end of the window to find a position
        where we can cut without splitting an entity (preferably at "O" tag).

        Args:
            window_tokens: Window tokens
            window_labels: Window labels (or None)

        Returns:
            Adjusted (window_tokens, window_labels) tuple
        """
        if not window_labels or len(window_tokens) <= self.overlap_tokens:
            # No labels or window too small to adjust
            return window_tokens, window_labels

        # Look backwards from end for a good cut point (preferably at "O" tag)
        cut_point = len(window_tokens)
        search_range = min(self.OPTIMAL_CUT_SEARCH_RANGE, len(window_tokens) - self.overlap_tokens)

        for i in range(len(window_tokens) - 1, max(0, len(window_tokens) - search_range - 1), -1):
            if window_labels[i] == "O":
                cut_point = i + 1
                break

        if cut_point < len(window_tokens):
            logger.debug(
                f"Found optimal cut point at token {cut_point}/{len(window_tokens)} "
                f"(saved {len(window_tokens) - cut_point} tokens from entity split)"
            )
            return window_tokens[:cut_point], window_labels[:cut_point]

        return window_tokens, window_labels

    def merge_windowed_predictions(
        self,
        windowed_predictions: List[Tuple[List[str], List[str], List[Dict[str, Any]]]],
        original_tokens: List[str]
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Merge predictions from multiple windows back into a single sequence.

        Strategy:
        1. Find actual position of each window in original sequence
        2. Use non-overlapping regions from each window
        3. First window: use all except last half-overlap
        4. Middle windows: skip first and last half-overlap
        5. Last window: skip first half-overlap
        6. Deduplicate entities in overlapping regions
        7. Verify full coverage

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
            else:
                logger.warning(
                    f"Single window size mismatch: {len(predictions)} predictions, "
                    f"{len(original_tokens)} tokens"
                )
                # Pad or truncate to match
                if len(predictions) < len(original_tokens):
                    predictions = predictions + ["O"] * (len(original_tokens) - len(predictions))
                else:
                    predictions = predictions[:len(original_tokens)]
                return predictions, entities

        # Multiple windows - need sophisticated merging
        merged_predictions = ["O"] * len(original_tokens)
        all_entities = []
        coverage = [False] * len(original_tokens)  # Track coverage

        window_start = 0
        for window_idx, (window_tokens, window_predictions, window_entities) in enumerate(windowed_predictions):
            # Find the actual start position of this window in original tokens
            actual_start = self._find_window_start_position(
                window_tokens, original_tokens, window_start
            )

            if actual_start < 0:
                logger.warning(f"Could not locate window {window_idx} in original sequence")
                continue

            # Determine non-overlapping region for this window
            # Strategy: each window "owns" its middle portion
            if window_idx == 0:
                # First window: use everything except last half-overlap
                start_pos = 0
                end_pos = len(window_predictions) - self.overlap_tokens // 2
            elif window_idx == len(windowed_predictions) - 1:
                # Last window: skip first half-overlap, use rest
                start_pos = self.overlap_tokens // 2
                end_pos = len(window_predictions)
            else:
                # Middle window: skip half-overlap on both sides
                start_pos = self.overlap_tokens // 2
                end_pos = len(window_predictions) - self.overlap_tokens // 2

            # Clamp to valid range
            start_pos = max(0, start_pos)
            end_pos = min(len(window_predictions), end_pos)

            # Apply predictions from non-overlapping region
            for i in range(start_pos, end_pos):
                global_pos = actual_start + i
                if 0 <= global_pos < len(merged_predictions):
                    merged_predictions[global_pos] = window_predictions[i]
                    coverage[global_pos] = True

            # Adjust entity positions and add to results
            for entity in window_entities:
                adjusted_entity = entity.copy()
                if 'start' in adjusted_entity and 'end' in adjusted_entity:
                    adjusted_entity['start'] += actual_start
                    adjusted_entity['end'] += actual_start

                    # Only include entity if it's primarily in the non-overlapping region
                    entity_start_in_window = entity['start']
                    entity_end_in_window = entity['end']

                    # Check if entity's midpoint is in the non-overlapping region
                    entity_midpoint = (entity_start_in_window + entity_end_in_window) // 2

                    if start_pos <= entity_midpoint < end_pos:
                        all_entities.append(adjusted_entity)

            # Update window_start hint for next iteration
            window_start = actual_start + len(window_tokens) - self.overlap_tokens

        # Verify coverage
        uncovered_count = sum(1 for c in coverage if not c)
        if uncovered_count > 0:
            uncovered_positions = [i for i, c in enumerate(coverage) if not c]
            logger.warning(
                f"Incomplete coverage: {uncovered_count}/{len(original_tokens)} tokens not covered. "
                f"Positions: {uncovered_positions[:20]}{'...' if len(uncovered_positions) > 20 else ''}"
            )

        # Remove duplicate entities
        unique_entities = self._remove_duplicate_entities(all_entities)

        logger.debug(
            f"Merged {len(windowed_predictions)} windows: "
            f"{len(all_entities)} total entities → {len(unique_entities)} unique entities"
        )

        return merged_predictions, unique_entities

    def _find_window_start_position(
        self,
        window_tokens: List[str],
        original_tokens: List[str],
        hint_start: int
    ) -> int:
        """
        Find the actual start position of window tokens in original sequence.

        Uses exact matching to locate the window in the original sequence,
        starting from a hint position for efficiency.

        Args:
            window_tokens: Tokens from the window
            original_tokens: Original full token sequence
            hint_start: Suggested start position (from previous window)

        Returns:
            Start index of window in original_tokens, or hint_start if not found
        """
        # Start search from hint position
        search_start = max(0, hint_start)
        # Don't search too far (windows should be somewhat sequential)
        search_end = min(len(original_tokens), search_start + len(window_tokens) + self.overlap_tokens * 2)

        # Look for exact match
        window_len = len(window_tokens)
        for start in range(search_start, max(search_start, search_end - window_len + 1)):
            if original_tokens[start:start + window_len] == window_tokens:
                return start

        # Fallback: search more broadly if not found near hint
        if search_start > 0 or search_end < len(original_tokens):
            for start in range(0, len(original_tokens) - window_len + 1):
                if original_tokens[start:start + window_len] == window_tokens:
                    logger.warning(
                        f"Window found at position {start}, far from hint {hint_start}. "
                        f"This may indicate windowing issues."
                    )
                    return start

        # Could not locate window
        logger.warning(
            f"Could not locate window in original sequence. "
            f"Window: {len(window_tokens)} tokens, Original: {len(original_tokens)} tokens, "
            f"Hint: {hint_start}. Using hint as fallback."
        )
        return hint_start

    def _remove_duplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate entities from merged results, keeping the longest version.

        Fixed version that doesn't modify list during iteration.

        Args:
            entities: List of entity dictionaries

        Returns:
            List of unique entities
        """
        if not entities:
            return []

        # Sort by start position for stable processing
        sorted_entities = sorted(entities, key=lambda x: (x.get('start', 0), x.get('end', 0)))

        unique_entities = []
        for entity in sorted_entities:
            # Find if this entity overlaps with any existing unique entity
            overlapping_idx = None

            for i, existing in enumerate(unique_entities):
                if (entity.get('type') == existing.get('type') and
                    self._entities_overlap(entity, existing)):
                    overlapping_idx = i
                    break

            if overlapping_idx is not None:
                # Found overlapping entity - keep the longer one
                existing = unique_entities[overlapping_idx]
                entity_len = entity.get('end', 0) - entity.get('start', 0)
                existing_len = existing.get('end', 0) - existing.get('start', 0)

                if entity_len > existing_len:
                    # Replace with longer entity
                    unique_entities[overlapping_idx] = entity
                    logger.debug(
                        f"Replaced duplicate entity at {existing.get('start')}-{existing.get('end')} "
                        f"with {entity.get('start')}-{entity.get('end')} (longer)"
                    )
                # else: keep existing (it's longer or equal)
            else:
                # No overlap - add as new entity
                unique_entities.append(entity)

        return unique_entities

    def _entities_overlap(self, entity1: Dict[str, Any], entity2: Dict[str, Any]) -> bool:
        """
        Check if two entities overlap significantly.

        Considers entities overlapping if they share > 50% of the shorter entity's span.

        Args:
            entity1: First entity dictionary with 'start' and 'end' keys
            entity2: Second entity dictionary with 'start' and 'end' keys

        Returns:
            True if entities overlap significantly, False otherwise
        """
        start1, end1 = entity1.get('start', 0), entity1.get('end', 0)
        start2, end2 = entity2.get('start', 0), entity2.get('end', 0)

        # Calculate overlap
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        overlap_len = max(0, overlap_end - overlap_start)

        if overlap_len == 0:
            return False

        # Calculate minimum entity length
        len1 = end1 - start1
        len2 = end2 - start2
        min_len = min(len1, len2)

        # Consider overlapping if overlap is > 50% of shorter entity
        return overlap_len > min_len * self.ENTITY_OVERLAP_THRESHOLD
