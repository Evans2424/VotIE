#!/usr/bin/env python3
"""
Convert Event-Based Format to Span-Based Format

This script converts the event-based JSONL format to a simpler span-based format
that focuses only on entity spans without the structured event information.

Input: data/votie_events/{train,dev,test}.jsonl
Output: data/votie_spans/{train,dev,test}.json
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_event_to_span(event_example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert event-based example to span-based format.

    Removes the event structure, keeping only the flat span annotations.
    """
    span_example = {
        "id": event_example["id"],
        "text": event_example["text"],
        "municipality": event_example["municipality"],
        "document_id": event_example["document_id"],
        "segment_id": event_example["segment_id"],
        "tokens": event_example["tokens"],
        "spans": event_example["spans"]
    }

    return span_example


def convert_split(input_file: Path, output_file: Path) -> Dict[str, Any]:
    """
    Convert a single split file from events to spans format.

    Returns:
        Statistics dictionary
    """
    logger.info(f"Converting {input_file.name}...")

    examples = []
    span_count = 0
    entity_label_counts = Counter()

    # Read input
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            event_example = json.loads(line.strip())
            span_example = convert_event_to_span(event_example)
            examples.append(span_example)

            # Track statistics
            span_count += len(span_example["spans"])
            for span in span_example["spans"]:
                entity_label_counts[span["label"]] += 1

    # Write output as JSON array
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)

    logger.info(f"  Converted {len(examples)} examples with {span_count} total spans")

    return {
        "num_examples": len(examples),
        "total_spans": span_count,
        "entity_label_counts": dict(entity_label_counts)
    }


def validate_conversion(event_dir: Path, span_dir: Path) -> bool:
    """
    Validate that span conversion preserved all information correctly.

    Returns:
        True if validation passed
    """
    logger.info("Validating conversion...")

    validation_passed = True

    for split in ["train", "dev", "test"]:
        event_file = event_dir / f"{split}.jsonl"
        span_file = span_dir / f"{split}.json"

        # Load both formats
        event_examples = []
        with open(event_file, 'r', encoding='utf-8') as f:
            for line in f:
                event_examples.append(json.loads(line.strip()))

        # Load JSON array
        with open(span_file, 'r', encoding='utf-8') as f:
            span_examples = json.load(f)

        # Check counts match
        if len(event_examples) != len(span_examples):
            logger.error(f"  {split}: Example count mismatch! "
                        f"{len(event_examples)} events vs {len(span_examples)} spans")
            validation_passed = False
            continue

        # Check each example
        for i, (event_ex, span_ex) in enumerate(zip(event_examples, span_examples)):
            # Check IDs match
            if event_ex["id"] != span_ex["id"]:
                logger.error(f"  {split}: ID mismatch at index {i}")
                validation_passed = False

            # Check span counts match
            if len(event_ex["spans"]) != len(span_ex["spans"]):
                logger.error(f"  {split}: Span count mismatch for {event_ex['id']}")
                validation_passed = False

            # Check tokens match
            if event_ex["tokens"] != span_ex["tokens"]:
                logger.error(f"  {split}: Token mismatch for {event_ex['id']}")
                validation_passed = False

        logger.info(f"  {split}: {'✓ PASS' if validation_passed else '✗ FAIL'}")

    return validation_passed


def main():
    """Main conversion pipeline."""
    logger.info("=" * 80)
    logger.info("EVENT TO SPAN FORMAT CONVERSION")
    logger.info("=" * 80)

    # Paths
    input_dir = Path("data/votie_events")
    output_dir = Path("data/votie_spans")

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        logger.error("Please run convert_citilink_to_events.py first")
        return 1

    # Convert each split
    all_stats = {}
    for split in ["train", "dev", "test"]:
        input_file = input_dir / f"{split}.jsonl"
        output_file = output_dir / f"{split}.json"

        if not input_file.exists():
            logger.warning(f"Split file not found: {input_file}")
            continue

        stats = convert_split(input_file, output_file)
        all_stats[split] = stats

    # Save statistics
    stats_file = output_dir / "statistics.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved statistics to {stats_file}")

    # Validate conversion
    validation_passed = validate_conversion(input_dir, output_dir)

    # Print summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("CONVERSION COMPLETE")
    logger.info("=" * 80)

    total_examples = sum(stats["num_examples"] for stats in all_stats.values())
    total_spans = sum(stats["total_spans"] for stats in all_stats.values())

    logger.info(f"Total examples: {total_examples}")
    logger.info(f"Total spans: {total_spans}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Validation: {'PASSED' if validation_passed else 'FAILED'}")
    logger.info("=" * 80)

    return 0 if validation_passed else 1


if __name__ == "__main__":
    sys.exit(main())
