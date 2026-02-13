#!/usr/bin/env python3
"""
Prepare data splits for incremental learning experiments v2.

This script creates data splits for:
- Fine-tuning from LOMO checkpoints vs training from scratch
- Three document placement scenarios (A: train only, B: dev only, C: split)
- All 6 municipalities × 6 steps (0-5 documents added)

Usage:
    python scripts/prepare_incremental_v2_data.py --municipalities M01 M02 --scenarios A B
    python scripts/prepare_incremental_v2_data.py --all  # All municipalities, all scenarios
"""

import json
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from collections import defaultdict


def load_jsonl(path: Path) -> List[Dict]:
    """Load JSONL file."""
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(data: List[Dict], path: Path):
    """Save data to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def save_json(data: Dict, path: Path):
    """Save data to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def group_by_document(examples: List[Dict]) -> Dict[str, List[Dict]]:
    """Group segments by document ID."""
    groups = defaultdict(list)
    for ex in examples:
        doc_id = ex.get('document_id', ex['id'].rsplit('_seg', 1)[0])
        groups[doc_id].append(ex)
    return dict(groups)


def prepare_incremental_experiments_v2(
    municipality: str,
    seed: int = 42,
    num_test_docs: int = 10,
    num_add_docs: int = 5,
    scenarios: List[str] = ['A', 'B', 'C']
) -> Dict[str, List[Dict]]:
    """
    Prepare data splits for all experiment types and scenarios.

    Args:
        municipality: Target municipality (e.g., 'M01')
        seed: Random seed for document shuffling
        num_test_docs: Number of documents in fixed test set
        num_add_docs: Maximum number of documents to add incrementally
        scenarios: List of scenarios to prepare ('A', 'B', and/or 'C')

    Returns:
        Dictionary mapping 'finetune_A', 'fromscratch_A', etc. to step info lists
    """
    print(f"\n{'='*80}")
    print(f"Preparing data for municipality: {municipality}")
    print(f"{'='*80}")

    # Load LOMO base data (from 5 other municipalities)
    base_dir = Path(f"data/citilink_bio_municipality_splits/{municipality}")

    if not base_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {base_dir}")

    lomo_train = load_jsonl(base_dir / "train.jsonl")
    lomo_dev = load_jsonl(base_dir / "dev.jsonl")
    lomo_test = load_jsonl(base_dir / "test.jsonl")

    print(f"Base LOMO data loaded:")
    print(f"  Train: {len(lomo_train)} segments (from other municipalities)")
    print(f"  Dev:   {len(lomo_dev)} segments (from other municipalities)")
    print(f"  Test:  {len(lomo_test)} segments (from {municipality})")

    # Group test set documents and shuffle with fixed seed
    doc_groups = group_by_document(lomo_test)
    doc_ids = sorted(doc_groups.keys())

    print(f"  Test documents: {len(doc_ids)}")

    # IMPORTANT: Always allocate maximum pool size (for scenario C)
    # This ensures test set is identical regardless of which scenarios are prepared
    pool_multiplier = 2  # Always allocate for scenario C (even if not preparing it)
    required_docs = num_add_docs * pool_multiplier + num_test_docs

    if len(doc_ids) < required_docs:
        raise ValueError(
            f"{municipality} has only {len(doc_ids)} documents, "
            f"but need {required_docs} ({num_add_docs}*2 for scenario C support + {num_test_docs} for test). "
            f"Note: Pool is always sized for scenario C to ensure consistent test sets."
        )

    rng = random.Random(seed)
    rng.shuffle(doc_ids)

    # Split into incremental pool and fixed test set
    # ALWAYS allocate full pool size (for scenario C) to ensure test set consistency
    pool_size = num_add_docs * 2  # Always 10 docs for pool (5 for A/B, or 5+5 for C)
    add_pool_docs = doc_ids[:pool_size]
    test_docs = doc_ids[pool_size:pool_size + num_test_docs]

    # Fixed test segments (identical across all experiments)
    test_segments = [seg for doc_id in test_docs for seg in doc_groups[doc_id]]

    print(f"\nDocument allocation (seed={seed}):")
    print(f"  Incremental pool: {pool_size} documents")
    print(f"  Fixed test set:   {num_test_docs} documents ({len(test_segments)} segments)")

    results = {}

    for exp_type in ['finetune', 'fromscratch']:
        for scenario in scenarios:
            print(f"\n  Processing {exp_type} - Scenario {scenario}...")
            steps = []

            for k in range(num_add_docs + 1):  # 0 through 5
                step_dir = Path(
                    f"data/incremental_experiments_v2/{exp_type}/scenario_{scenario}/"
                    f"{municipality}/step_{k:02d}"
                )
                step_dir.mkdir(parents=True, exist_ok=True)

                # Determine document allocation based on scenario
                if scenario == 'A':  # Train only
                    # Use first 5 docs from pool (regardless of pool size)
                    added_train_docs = add_pool_docs[:num_add_docs][:k]
                    added_dev_docs = []
                elif scenario == 'B':  # Dev only
                    # Use first 5 docs from pool (regardless of pool size)
                    added_train_docs = []
                    added_dev_docs = add_pool_docs[:num_add_docs][:k]
                else:  # Scenario C: Split pool (first half to train, second half to dev)
                    # Pool is always 10 docs, split into two halves of 5 each
                    train_pool = add_pool_docs[:num_add_docs]  # First 5
                    dev_pool = add_pool_docs[num_add_docs:num_add_docs*2]  # Second 5
                    # Use up to k documents from each pool
                    added_train_docs = train_pool[:k]
                    added_dev_docs = dev_pool[:k]

                # Build train/dev sets
                train_segs = lomo_train + [
                    seg for doc_id in added_train_docs for seg in doc_groups[doc_id]
                ]
                dev_segs = lomo_dev + [
                    seg for doc_id in added_dev_docs for seg in doc_groups[doc_id]
                ]

                # Save files
                save_jsonl(train_segs, step_dir / "train.jsonl")
                save_jsonl(dev_segs, step_dir / "dev.jsonl")
                save_jsonl(test_segments, step_dir / "test.jsonl")

                # Save metadata
                metadata = {
                    "experiment_type": exp_type,
                    "scenario": scenario,
                    "municipality": municipality,
                    "step": k,
                    "documents_added": {
                        "train": added_train_docs,
                        "dev": added_dev_docs,
                        "test": []
                    },
                    "num_segments": {
                        "train": len(train_segs),
                        "dev": len(dev_segs),
                        "test": len(test_segments)
                    },
                    "created_at": datetime.now().isoformat(),
                    "data_source": {
                        "lomo_base": str(base_dir),
                        "incremental_pool_seed": seed
                    }
                }

                save_json(metadata, step_dir / "metadata.json")

                steps.append({
                    'step': k,
                    'data_dir': str(step_dir),
                    'metadata': metadata
                })

            results[f"{exp_type}_{scenario}"] = steps
            print(f"    Created {len(steps)} steps")

    print(f"\n✅ Data preparation complete for {municipality}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Prepare data splits for incremental learning experiments v2"
    )
    parser.add_argument(
        '--municipalities', nargs='+',
        default=['M01', 'M02', 'M03', 'M04', 'M05', 'M06'],
        help='List of municipalities to process'
    )
    parser.add_argument(
        '--scenarios', nargs='+', choices=['A', 'B', 'C'],
        default=['A', 'B', 'C'],
        help='Scenarios to prepare (A: train only, B: dev only, C: split)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for document shuffling'
    )
    parser.add_argument(
        '--num-test-docs', type=int, default=10,
        help='Number of documents in fixed test set'
    )
    parser.add_argument(
        '--num-add-docs', type=int, default=5,
        help='Maximum number of documents to add incrementally'
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Prepare data for all municipalities and scenarios'
    )

    args = parser.parse_args()

    if args.all:
        municipalities = ['M01', 'M02', 'M03', 'M04', 'M05', 'M06']
        scenarios = ['A', 'B', 'C']
    else:
        municipalities = args.municipalities
        scenarios = args.scenarios

    print("="*80)
    print("INCREMENTAL LEARNING EXPERIMENTS V2 - DATA PREPARATION")
    print("="*80)
    print(f"Municipalities: {municipalities}")
    print(f"Scenarios: {scenarios}")
    print(f"Seed: {args.seed}")
    print(f"Test docs: {args.num_test_docs}")
    print(f"Max add docs: {args.num_add_docs}")
    print("="*80)

    all_results = {}

    for municipality in municipalities:
        try:
            results = prepare_incremental_experiments_v2(
                municipality=municipality,
                seed=args.seed,
                num_test_docs=args.num_test_docs,
                num_add_docs=args.num_add_docs,
                scenarios=scenarios
            )
            all_results[municipality] = results
        except Exception as e:
            print(f"\n❌ Error processing {municipality}: {e}")
            continue

    # Save summary
    summary_path = Path("data/incremental_experiments_v2/preparation_summary.json")
    summary = {
        "timestamp": datetime.now().isoformat(),
        "municipalities": municipalities,
        "scenarios": scenarios,
        "seed": args.seed,
        "num_test_docs": args.num_test_docs,
        "num_add_docs": args.num_add_docs,
        "results": {
            mun: {
                config: [{"step": s['step'], "data_dir": s['data_dir']}
                         for s in steps]
                for config, steps in mun_results.items()
            }
            for mun, mun_results in all_results.items()
        }
    }
    save_json(summary, summary_path)

    print(f"\n{'='*80}")
    print("✅ DATA PREPARATION COMPLETE")
    print(f"{'='*80}")
    print(f"Summary saved to: {summary_path}")
    print(f"\nNext steps:")
    print(f"  1. Validate data: python scripts/validate_incremental_v2_data.py")
    print(f"  2. Run experiments: python scripts/run_incremental_v2_experiments.py --gpus 0 1 2 3")


if __name__ == '__main__':
    main()
