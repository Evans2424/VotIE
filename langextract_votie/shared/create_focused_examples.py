#!/usr/bin/env python3
"""
Extract focused examples from selected_event_examples.json.

This script creates concise examples by extracting only the text segments
that contain voting entities, reducing token usage while maintaining real examples.
"""

import json
from pathlib import Path
from typing import List, Dict, Any


def extract_context_window(text: str, start: int, end: int, window: int = 100) -> tuple:
    """
    Extract text around a span with context window.
    
    Args:
        text: Full text
        start: Span start position
        end: Span end position
        window: Characters to include before/after
        
    Returns:
        (extracted_text, new_start, new_end)
    """
    # Expand to sentence boundaries if possible
    context_start = max(0, start - window)
    context_end = min(len(text), end + window)
    
    # Try to find sentence boundaries
    while context_start > 0 and text[context_start] not in '.!?\n':
        context_start -= 1
        if start - context_start > window * 2:  # Don't go too far
            context_start = max(0, start - window)
            break
    
    if context_start > 0 and text[context_start] in '.!?\n':
        context_start += 1
        
    while context_end < len(text) and text[context_end - 1] not in '.!?\n':
        context_end += 1
        if context_end - end > window * 2:  # Don't go too far
            context_end = min(len(text), end + window)
            break
    
    extracted = text[context_start:context_end].strip()
    new_start = start - context_start
    new_end = end - context_start
    
    return extracted, new_start, new_end


def create_focused_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a focused version of an example by extracting voting context.
    
    Strategy:
    1. Find all entity spans
    2. Determine the minimal text range that covers all entities
    3. Expand slightly for context
    4. Update all offsets
    """
    text = example['text']
    spans = example.get('spans', [])
    
    if not spans:
        return None
    
    # Find min/max positions
    min_start = min(span['start'] for span in spans)
    max_end = max(span['end'] for span in spans)
    
    # Extract with context (200 chars on each side)
    focused_text, offset, _ = extract_context_window(text, min_start, max_end, window=200)
    
    # Update span offsets
    new_spans = []
    for span in spans:
        new_span = span.copy()
        new_span['start'] = span['start'] - offset
        new_span['end'] = span['end'] - offset
        
        # Validate
        extracted = focused_text[new_span['start']:new_span['end']]
        if extracted != span['text']:
            print(f"Warning: Span mismatch in {example['id']}: '{extracted}' vs '{span['text']}'")
            continue
            
        new_spans.append(new_span)
    
    # Create focused example
    focused_example = {
        'id': example['id'],
        'text': focused_text,
        'municipality': example['municipality'],
        'document_id': example['document_id'],
        'segment_id': example['segment_id'],
        'has_voting_event': example['has_voting_event'],
        'event': example['event'],
        'spans': new_spans
    }
    
    return focused_example


def main():
    # Load selected examples
    input_file = Path(__file__).parent / 'selected_event_examples.json'
    output_file = Path(__file__).parent / 'short_examples.json'
    
    print(f"Loading examples from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        examples = json.load(f)
    
    print(f"Processing {len(examples)} examples...")
    
    focused_examples = []
    for i, example in enumerate(examples, 1):
        print(f"  Processing example {i}/{len(examples)}: {example['id']}")
        
        focused = create_focused_example(example)
        if focused:
            focused_examples.append(focused)
            
            # Print stats
            orig_len = len(example['text'])
            focused_len = len(focused['text'])
            reduction = (1 - focused_len / orig_len) * 100
            print(f"    Reduced from {orig_len} to {focused_len} chars ({reduction:.1f}% reduction)")
    
    print(f"\nCreated {len(focused_examples)} focused examples")
    
    # Calculate total stats
    total_orig = sum(len(ex['text']) for ex in examples)
    total_focused = sum(len(ex['text']) for ex in focused_examples)
    total_reduction = (1 - total_focused / total_orig) * 100
    
    print(f"Total reduction: {total_orig} -> {total_focused} chars ({total_reduction:.1f}% reduction)")
    
    # Save
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(focused_examples, f, indent=2, ensure_ascii=False)
    
    print("Done!")


if __name__ == '__main__':
    main()
