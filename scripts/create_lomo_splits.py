#!/usr/bin/env python3
"""
Create Leave-One-Municipality-Out (LOMO) Splits

This script creates LOMO splits for cross-domain evaluation. For each municipality,
it creates a split where that municipality is the test set, and the other 5
municipalities form the train/dev sets.

Input: data/votie_events/{train,dev,test}.jsonl (combined)
Output: data/votie_lomo/M0X_name/
  - train.jsonl (BIO format)
  - dev.jsonl (BIO format)
  - test.jsonl (BIO format)
  - statistics.json
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict

import numpy as np
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Municipality mapping
MUNICIPALITIES = {
    "M01": "alandroal",
    "M02": "campomaior",
    "M03": "covilha",
    "M04": "fundao",
    "M05": "guimaraes",
    "M06": "porto"
}

RANDOM_SEED = 42
TRAIN_RATIO = 0.85  # Of the 5 municipalities
DEV_RATIO = 0.15


# ============================================================================
# SPLITTING LOGIC
# ============================================================================

def generate_stratify_key(example: Dict) -> str:
    """Generate stratification key for an example (for dev split within LOMO)."""
    municipality = example["municipality"]

    # Classify by entity pattern
    labels = set(span["label"] for span in example["spans"])

    if any(l in labels for l in ["VOTER-FAVOR", "COUNTING-UNANIMITY"]):
        pattern = "positive_vote"
    elif any(l in labels for l in ["VOTER-AGAINST", "VOTER-ABSENT"]):
        pattern = "negative_vote"
    elif "VOTER-ABSTENTION" in labels:
        pattern = "abstention"
    elif "COUNTING-MAJORITY" in labels:
        pattern = "majority_decision"
    elif "SUBJECT" in labels:
        pattern = "has_subject"
    else:
        pattern = "no_entities"

    return f"{municipality}_{pattern}"


def adaptive_stratify(examples: List[Dict], stratify_keys: List[str]) -> List[str]:
    """Ensure minimum 2 examples per stratum."""
    key_counts = Counter(stratify_keys)

    simplified_keys = []
    for i, key in enumerate(stratify_keys):
        if key_counts[key] >= 2:
            simplified_keys.append(key)
        else:
            municipality = examples[i]["municipality"]
            simplified_keys.append(f"{municipality}_consolidated")

    simplified_counts = Counter(simplified_keys)
    remaining_problems = [k for k, count in simplified_counts.items() if count < 2]

    if remaining_problems:
        final_keys = []
        for key in simplified_keys:
            if simplified_counts[key] >= 2:
                final_keys.append(key)
            else:
                final_keys.append("consolidated_mixed")
        return final_keys

    return simplified_keys


def create_lomo_split(all_examples: List[Dict], target_municipality: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Create LOMO split for a target municipality.

    Returns:
        (train_examples, dev_examples, test_examples)
    """
    logger.info(f"Creating LOMO split for {target_municipality} ({MUNICIPALITIES[target_municipality]})")

    # Test set = target municipality only
    test_examples = [ex for ex in all_examples if ex["municipality"] == target_municipality]

    # Train pool = all other municipalities
    train_pool = [ex for ex in all_examples if ex["municipality"] != target_municipality]

    logger.info(f"  Test set: {len(test_examples)} examples from {target_municipality}")
    logger.info(f"  Train pool: {len(train_pool)} examples from 5 other municipalities")

    # Split train pool into train/dev (85/15, stratified)
    stratify_keys = [generate_stratify_key(ex) for ex in train_pool]
    final_stratify_keys = adaptive_stratify(train_pool, stratify_keys)

    indices = np.arange(len(train_pool))

    try:
        train_indices, dev_indices = train_test_split(
            indices,
            stratify=[final_stratify_keys[i] for i in indices],
            test_size=DEV_RATIO,
            random_state=RANDOM_SEED
        )
    except ValueError as e:
        logger.warning(f"  Stratified split failed: {e}, using random split")
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(indices)

        dev_size = int(len(indices) * DEV_RATIO)
        dev_indices = indices[:dev_size]
        train_indices = indices[dev_size:]

    train_examples = [train_pool[i] for i in train_indices]
    dev_examples = [train_pool[i] for i in dev_indices]

    logger.info(f"  Final split: {len(train_examples)} train, {len(dev_examples)} dev, {len(test_examples)} test")

    return train_examples, dev_examples, test_examples


# ============================================================================
# FORMAT CONVERSION
# ============================================================================

def spans_to_bio_tags(tokens: List[str], spans: List[Dict]) -> List[str]:
    """Convert spans to BIO tags."""
    bio_tags = ['O'] * len(tokens)
    sorted_spans = sorted(spans, key=lambda s: (s["token_start"], s["token_end"]))
    assigned_tokens = set()

    for span in sorted_spans:
        token_start = span["token_start"]
        token_end = span["token_end"]
        label = span["label"]

        span_tokens = range(token_start, token_end + 1)
        if any(t in assigned_tokens for t in span_tokens):
            continue

        if token_start <= token_end < len(tokens):
            bio_tags[token_start] = f"B-{label}"
            for i in range(token_start + 1, token_end + 1):
                if i < len(tokens):
                    bio_tags[i] = f"I-{label}"
                    assigned_tokens.add(i)
            assigned_tokens.add(token_start)

    return bio_tags


def convert_to_bio_jsonl(examples: List[Dict]) -> List[Dict]:
    """Convert examples to BIO JSONL format."""
    bio_examples = []
    
    for example in examples:
        tokens = example["tokens"]
        spans = example["spans"]
        bio_tags = spans_to_bio_tags(tokens, spans)
        
        bio_example = {
            "id": example["id"],
            "text": example["text"],
            "tokens": tokens,
            "labels": bio_tags,
            "municipality": example["municipality"],
            "document_id": example["document_id"],
            "segment_id": example["segment_id"]
        }
        bio_examples.append(bio_example)
    
    return bio_examples


# ============================================================================
# SAVING
# ============================================================================

def save_lomo_split(municipality_id: str, train_ex: List[Dict],
                    dev_ex: List[Dict], test_ex: List[Dict], output_dir: Path):
    """Save LOMO split in BIO JSONL format (compatible with training script)."""
    municipality_name = MUNICIPALITIES[municipality_id]
    lomo_dir = output_dir / f"{municipality_id}_{municipality_name}"

    logger.info(f"Saving LOMO split for {municipality_id} to {lomo_dir}")

    # Create directory
    lomo_dir.mkdir(parents=True, exist_ok=True)

    # Save BIO JSONL format (main format for training)
    for split_name, examples in [("train", train_ex), ("dev", dev_ex), ("test", test_ex)]:
        bio_examples = convert_to_bio_jsonl(examples)
        with open(lomo_dir / f"{split_name}.jsonl", 'w', encoding='utf-8') as f:
            for ex in bio_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    logger.info(f"  Saved BIO JSONL format to {lomo_dir}")


def save_lomo_statistics(municipality_id: str, train_ex: List[Dict],
                         dev_ex: List[Dict], test_ex: List[Dict], output_dir: Path):
    """Generate and save statistics for LOMO split."""

    def analyze_split(examples: List[Dict]) -> Dict:
        entity_counts = Counter()
        municipality_counts = Counter()
        total_tokens = 0

        for ex in examples:
            for span in ex["spans"]:
                entity_counts[span["label"]] += 1
            municipality_counts[ex["municipality"]] += 1
            total_tokens += len(ex["tokens"])

        return {
            "num_examples": len(examples),
            "total_tokens": total_tokens,
            "avg_tokens_per_example": total_tokens / len(examples) if examples else 0,
            "entity_counts": dict(entity_counts),
            "municipality_distribution": dict(municipality_counts)
        }

    stats = {
        "target_municipality": municipality_id,
        "municipality_name": MUNICIPALITIES[municipality_id],
        "train": analyze_split(train_ex),
        "dev": analyze_split(dev_ex),
        "test": analyze_split(test_ex)
    }

    municipality_name = MUNICIPALITIES[municipality_id]
    stats_file = output_dir / f"{municipality_id}_{municipality_name}" / "statistics.json"

    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info(f"  Saved statistics to {stats_file}")


# ============================================================================
# VALIDATION
# ============================================================================

def validate_lomo_split(municipality_id: str, train_ex: List[Dict],
                       dev_ex: List[Dict], test_ex: List[Dict]) -> bool:
    """Validate LOMO split correctness."""
    validation_passed = True

    # 1. Verify test set contains ONLY target municipality
    test_municipalities = set(ex["municipality"] for ex in test_ex)
    if test_municipalities != {municipality_id}:
        logger.error(f"  Test set contamination! Contains: {test_municipalities}")
        validation_passed = False

    # 2. Verify train/dev contain NO target municipality
    train_municipalities = set(ex["municipality"] for ex in train_ex)
    dev_municipalities = set(ex["municipality"] for ex in dev_ex)

    if municipality_id in train_municipalities:
        logger.error(f"  Target municipality found in train set!")
        validation_passed = False

    if municipality_id in dev_municipalities:
        logger.error(f"  Target municipality found in dev set!")
        validation_passed = False

    # 3. Verify all other municipalities present in train
    expected_municipalities = set(MUNICIPALITIES.keys()) - {municipality_id}
    if train_municipalities != expected_municipalities:
        logger.warning(f"  Expected municipalities {expected_municipalities}, "
                      f"got {train_municipalities}")

    # 4. Verify no data leakage between splits
    train_ids = set(ex["id"] for ex in train_ex)
    dev_ids = set(ex["id"] for ex in dev_ex)
    test_ids = set(ex["id"] for ex in test_ex)

    if train_ids & dev_ids:
        logger.error(f"  Train/Dev overlap: {len(train_ids & dev_ids)} examples!")
        validation_passed = False

    if train_ids & test_ids:
        logger.error(f"  Train/Test overlap: {len(train_ids & test_ids)} examples!")
        validation_passed = False

    if dev_ids & test_ids:
        logger.error(f"  Dev/Test overlap: {len(dev_ids & test_ids)} examples!")
        validation_passed = False

    if validation_passed:
        logger.info(f"  ✓ Validation passed for {municipality_id}")
    else:
        logger.error(f"  ✗ Validation failed for {municipality_id}")

    return validation_passed


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main LOMO splits creation pipeline."""
    logger.info("=" * 80)
    logger.info("LEAVE-ONE-MUNICIPALITY-OUT (LOMO) SPLITS CREATION")
    logger.info("=" * 80)

    # Paths
    events_dir = Path("data/votie_events")
    output_dir = Path("data/votie_lomo")

    if not events_dir.exists():
        logger.error(f"Events directory not found: {events_dir}")
        logger.error("Please run convert_citilink_to_events.py first")
        return 1

    # Load all examples from main splits
    logger.info("Loading all examples from main splits...")
    all_examples = []

    for split in ["train", "dev", "test"]:
        split_file = events_dir / f"{split}.jsonl"
        if not split_file.exists():
            logger.warning(f"Split file not found: {split_file}")
            continue

        with open(split_file, 'r', encoding='utf-8') as f:
            for line in f:
                all_examples.append(json.loads(line.strip()))

    logger.info(f"Loaded {len(all_examples)} total examples")

    # Verify municipality distribution
    municipality_counts = Counter(ex["municipality"] for ex in all_examples)
    logger.info("Municipality distribution:")
    for muni_id in sorted(MUNICIPALITIES.keys()):
        count = municipality_counts.get(muni_id, 0)
        logger.info(f"  {muni_id} ({MUNICIPALITIES[muni_id]:12}): {count:4d} examples")
    logger.info("")

    # Create LOMO split for each municipality
    all_validation_passed = True

    for municipality_id in sorted(MUNICIPALITIES.keys()):
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing {municipality_id} - {MUNICIPALITIES[municipality_id]}")
        logger.info(f"{'='*80}")

        # Create split
        train_ex, dev_ex, test_ex = create_lomo_split(all_examples, municipality_id)

        # Validate
        validation_passed = validate_lomo_split(municipality_id, train_ex, dev_ex, test_ex)
        all_validation_passed = all_validation_passed and validation_passed

        # Save
        save_lomo_split(municipality_id, train_ex, dev_ex, test_ex, output_dir)

        # Save statistics
        save_lomo_statistics(municipality_id, train_ex, dev_ex, test_ex, output_dir)

    # Print final summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("LOMO SPLITS CREATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Created LOMO splits for {len(MUNICIPALITIES)} municipalities")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Validation: {'ALL PASSED' if all_validation_passed else 'SOME FAILED'}")
    logger.info("")
    logger.info("Directory structure:")
    logger.info("  votie_lomo/")
    for muni_id in sorted(MUNICIPALITIES.keys()):
        logger.info(f"    {muni_id}_{MUNICIPALITIES[muni_id]}/")
        logger.info(f"      train.jsonl  (BIO format)")
        logger.info(f"      dev.jsonl    (BIO format)")
        logger.info(f"      test.jsonl   (BIO format)")
        logger.info(f"      statistics.json")
    logger.info("=" * 80)

    return 0 if all_validation_passed else 1


if __name__ == "__main__":
    sys.exit(main())
