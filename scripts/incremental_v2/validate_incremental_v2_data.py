#!/usr/bin/env python3
"""
Validate data splits for incremental learning experiments v2.

Performs comprehensive validation checks:
- Test set consistency across all configurations
- Scenario-specific constraints (A: train only, B: dev only, C: split)
- Metadata accuracy
- No data leakage between splits
- Document allocation correctness

Usage:
    python scripts/validate_incremental_v2_data.py
    python scripts/validate_incremental_v2_data.py --municipalities M01 M02
    python scripts/validate_incremental_v2_data.py --scenarios A B
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Set
from collections import defaultdict


def load_jsonl(path: Path) -> List[Dict]:
    """Load JSONL file."""
    if not path.exists():
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def load_json(path: Path) -> Dict:
    """Load JSON file."""
    if not path.exists():
        return {}
    with open(path, 'r') as f:
        return json.load(f)


def get_example_ids(examples: List[Dict]) -> Set[str]:
    """Extract example IDs from list."""
    return set(ex['id'] for ex in examples)


def get_document_ids(examples: List[Dict]) -> Set[str]:
    """Extract document IDs from list."""
    docs = set()
    for ex in examples:
        doc_id = ex.get('document_id', ex['id'].rsplit('_seg', 1)[0])
        docs.add(doc_id)
    return docs


# ── Validation Checks ───────────────────────────────────────────────────────


def check_test_set_consistency() -> bool:
    """Verify test set is identical across all experiments for same municipality."""
    print("\n" + "="*80)
    print("CHECK 1: Test Set Consistency")
    print("="*80)
    print("Verifying test sets are identical across all configurations...")

    municipalities = ['M01', 'M02', 'M03', 'M04', 'M05', 'M06']
    all_passed = True

    for mun in municipalities:
        test_sets = []
        configs = []

        for exp_type in ['finetune', 'fromscratch']:
            for scenario in ['A', 'B', 'C']:
                for step in range(6):
                    test_path = Path(
                        f"data/incremental_experiments_v2/{exp_type}/"
                        f"scenario_{scenario}/{mun}/step_{step:02d}/test.jsonl"
                    )

                    if test_path.exists():
                        test_data = load_jsonl(test_path)
                        test_ids = get_example_ids(test_data)
                        test_sets.append(test_ids)
                        configs.append(f"{exp_type}_scenario{scenario}_step{step:02d}")

        if not test_sets:
            print(f"  ⚠️  {mun}: No data found (skipping)")
            continue

        # All test sets should be identical
        first_set = test_sets[0]
        all_identical = all(test_set == first_set for test_set in test_sets)

        if all_identical:
            print(f"  ✅ {mun}: All {len(test_sets)} test sets identical ({len(first_set)} examples)")
        else:
            print(f"  ❌ {mun}: Test sets differ across configurations")
            # Find differences
            for i, (test_set, config) in enumerate(zip(test_sets, configs)):
                if test_set != first_set:
                    diff = test_set.symmetric_difference(first_set)
                    print(f"      {config}: {len(diff)} differences")
            all_passed = False

    return all_passed


def check_scenario_A_train_only() -> bool:
    """Verify Scenario A only adds documents to training set."""
    print("\n" + "="*80)
    print("CHECK 2: Scenario A - Train Only")
    print("="*80)
    print("Verifying Scenario A only modifies training set...")

    municipalities = ['M01', 'M02', 'M03', 'M04', 'M05', 'M06']
    all_passed = True

    for mun in municipalities:
        # Get baseline dev set (step 0)
        baseline_dev_path = Path(
            f"data/incremental_experiments_v2/finetune/scenario_A/"
            f"{mun}/step_00/dev.jsonl"
        )

        if not baseline_dev_path.exists():
            continue

        baseline_dev = get_example_ids(load_jsonl(baseline_dev_path))

        # Check all steps
        mun_passed = True
        for exp_type in ['finetune', 'fromscratch']:
            for step in range(6):
                dev_path = Path(
                    f"data/incremental_experiments_v2/{exp_type}/"
                    f"scenario_A/{mun}/step_{step:02d}/dev.jsonl"
                )

                if dev_path.exists():
                    dev_set = get_example_ids(load_jsonl(dev_path))

                    if dev_set != baseline_dev:
                        print(f"  ❌ {mun} {exp_type} step {step}: Dev set modified")
                        mun_passed = False
                        all_passed = False

        if mun_passed:
            print(f"  ✅ {mun}: Dev sets unchanged across all steps")

    return all_passed


def check_scenario_B_dev_only() -> bool:
    """Verify Scenario B only adds documents to dev set."""
    print("\n" + "="*80)
    print("CHECK 3: Scenario B - Dev Only")
    print("="*80)
    print("Verifying Scenario B only modifies dev set...")

    municipalities = ['M01', 'M02', 'M03', 'M04', 'M05', 'M06']
    all_passed = True

    for mun in municipalities:
        # Get baseline train set (step 0)
        baseline_train_path = Path(
            f"data/incremental_experiments_v2/finetune/scenario_B/"
            f"{mun}/step_00/train.jsonl"
        )

        if not baseline_train_path.exists():
            continue

        baseline_train = get_example_ids(load_jsonl(baseline_train_path))

        # Check all steps
        mun_passed = True
        for exp_type in ['finetune', 'fromscratch']:
            for step in range(6):
                train_path = Path(
                    f"data/incremental_experiments_v2/{exp_type}/"
                    f"scenario_B/{mun}/step_{step:02d}/train.jsonl"
                )

                if train_path.exists():
                    train_set = get_example_ids(load_jsonl(train_path))

                    if train_set != baseline_train:
                        print(f"  ❌ {mun} {exp_type} step {step}: Train set modified")
                        mun_passed = False
                        all_passed = False

        if mun_passed:
            print(f"  ✅ {mun}: Train sets unchanged across all steps")

    return all_passed


def check_scenario_C_different_docs() -> bool:
    """Verify Scenario C uses different documents for train vs dev."""
    print("\n" + "="*80)
    print("CHECK 4: Scenario C - Different Documents for Train/Dev")
    print("="*80)
    print("Verifying Scenario C uses different documents for train and dev...")

    municipalities = ['M01', 'M02', 'M03', 'M04', 'M05', 'M06']
    all_passed = True

    for mun in municipalities:
        mun_passed = True

        for exp_type in ['finetune', 'fromscratch']:
            for step in range(1, 6):  # Skip step 0 (no added docs)
                metadata_path = Path(
                    f"data/incremental_experiments_v2/{exp_type}/"
                    f"scenario_C/{mun}/step_{step:02d}/metadata.json"
                )

                if metadata_path.exists():
                    metadata = load_json(metadata_path)

                    train_docs = set(metadata['documents_added']['train'])
                    dev_docs = set(metadata['documents_added']['dev'])

                    # Should have no overlap
                    overlap = train_docs & dev_docs

                    if overlap:
                        print(
                            f"  ❌ {mun} {exp_type} step {step}: "
                            f"{len(overlap)} overlapping documents"
                        )
                        mun_passed = False
                        all_passed = False

        if mun_passed:
            print(f"  ✅ {mun}: No overlapping documents in train/dev")

    return all_passed


def check_metadata_accuracy() -> bool:
    """Verify metadata matches actual data."""
    print("\n" + "="*80)
    print("CHECK 5: Metadata Accuracy")
    print("="*80)
    print("Verifying metadata matches actual data files...")

    base_dir = Path("data/incremental_experiments_v2")
    all_passed = True

    # Find all metadata files
    metadata_files = list(base_dir.rglob("metadata.json"))

    print(f"Found {len(metadata_files)} metadata files to check")

    errors = 0

    for metadata_path in metadata_files:
        metadata = load_json(metadata_path)
        data_dir = metadata_path.parent

        # Check train
        train_path = data_dir / "train.jsonl"
        if train_path.exists():
            actual_train = len(load_jsonl(train_path))
            expected_train = metadata['num_segments']['train']

            if actual_train != expected_train:
                print(f"  ❌ {data_dir.relative_to(base_dir)}: Train count mismatch")
                print(f"      Expected {expected_train}, found {actual_train}")
                errors += 1
                all_passed = False

        # Check dev
        dev_path = data_dir / "dev.jsonl"
        if dev_path.exists():
            actual_dev = len(load_jsonl(dev_path))
            expected_dev = metadata['num_segments']['dev']

            if actual_dev != expected_dev:
                print(f"  ❌ {data_dir.relative_to(base_dir)}: Dev count mismatch")
                print(f"      Expected {expected_dev}, found {actual_dev}")
                errors += 1
                all_passed = False

        # Check test
        test_path = data_dir / "test.jsonl"
        if test_path.exists():
            actual_test = len(load_jsonl(test_path))
            expected_test = metadata['num_segments']['test']

            if actual_test != expected_test:
                print(f"  ❌ {data_dir.relative_to(base_dir)}: Test count mismatch")
                print(f"      Expected {expected_test}, found {actual_test}")
                errors += 1
                all_passed = False

    if errors == 0:
        print(f"  ✅ All {len(metadata_files)} metadata files accurate")
    else:
        print(f"  ❌ Found {errors} metadata mismatches")

    return all_passed


def check_no_data_leakage() -> bool:
    """Verify no data leakage between train/dev/test."""
    print("\n" + "="*80)
    print("CHECK 6: No Data Leakage")
    print("="*80)
    print("Verifying no overlap between train/dev/test sets...")

    base_dir = Path("data/incremental_experiments_v2")
    all_passed = True

    # Check all data directories
    data_dirs = []
    for exp_type in ['finetune', 'fromscratch']:
        for scenario in ['A', 'B', 'C']:
            for mun in ['M01', 'M02', 'M03', 'M04', 'M05', 'M06']:
                for step in range(6):
                    data_dir = (
                        base_dir / exp_type / f"scenario_{scenario}" /
                        mun / f"step_{step:02d}"
                    )
                    if data_dir.exists():
                        data_dirs.append(data_dir)

    errors = 0

    for data_dir in data_dirs:
        train_ids = get_example_ids(load_jsonl(data_dir / "train.jsonl"))
        dev_ids = get_example_ids(load_jsonl(data_dir / "dev.jsonl"))
        test_ids = get_example_ids(load_jsonl(data_dir / "test.jsonl"))

        # Check for overlaps
        train_dev_overlap = train_ids & dev_ids
        train_test_overlap = train_ids & test_ids
        dev_test_overlap = dev_ids & test_ids

        if train_dev_overlap or train_test_overlap or dev_test_overlap:
            rel_path = data_dir.relative_to(base_dir)
            print(f"  ❌ {rel_path}: Data leakage detected")

            if train_dev_overlap:
                print(f"      Train/Dev overlap: {len(train_dev_overlap)} examples")
            if train_test_overlap:
                print(f"      Train/Test overlap: {len(train_test_overlap)} examples")
            if dev_test_overlap:
                print(f"      Dev/Test overlap: {len(dev_test_overlap)} examples")

            errors += 1
            all_passed = False

    if errors == 0:
        print(f"  ✅ No data leakage in {len(data_dirs)} data directories")
    else:
        print(f"  ❌ Found data leakage in {errors} directories")

    return all_passed


def check_incremental_growth() -> bool:
    """Verify data grows incrementally as expected."""
    print("\n" + "="*80)
    print("CHECK 7: Incremental Growth")
    print("="*80)
    print("Verifying data grows incrementally with each step...")

    municipalities = ['M01', 'M02', 'M03', 'M04', 'M05', 'M06']
    all_passed = True

    for mun in municipalities:
        for exp_type in ['finetune', 'fromscratch']:
            for scenario in ['A', 'B', 'C']:
                prev_train_count = None
                prev_dev_count = None

                for step in range(6):
                    data_dir = Path(
                        f"data/incremental_experiments_v2/{exp_type}/"
                        f"scenario_{scenario}/{mun}/step_{step:02d}"
                    )

                    if not data_dir.exists():
                        continue

                    train_count = len(load_jsonl(data_dir / "train.jsonl"))
                    dev_count = len(load_jsonl(data_dir / "dev.jsonl"))

                    # Read lomo_fallback flag — finetune step 0 uses LOMO data as a
                    # fallback, so the step 0→1 transition is a deliberate size drop
                    # (from LOMO fallback back to real M01 docs) and should not be
                    # treated as a validation failure.
                    metadata_path = data_dir / "metadata.json"
                    prev_was_fallback_train = False
                    prev_was_fallback_dev = False
                    if step > 0 and metadata_path.exists():
                        prev_dir = Path(
                            f"data/incremental_experiments_v2/{exp_type}/"
                            f"scenario_{scenario}/{mun}/step_{(step-1):02d}"
                        )
                        prev_meta_path = prev_dir / "metadata.json"
                        if prev_meta_path.exists():
                            with open(prev_meta_path) as f:
                                prev_meta = json.load(f)
                            fallback = prev_meta.get('lomo_fallback', {})
                            prev_was_fallback_train = fallback.get('train', False)
                            prev_was_fallback_dev = fallback.get('dev', False)

                    # Check growth based on scenario
                    if scenario == 'A' and prev_train_count is not None:
                        if step > 0 and not prev_was_fallback_train and train_count <= prev_train_count:
                            print(
                                f"  ❌ {mun} {exp_type} scenario A step {step}: "
                                f"Train not growing"
                            )
                            all_passed = False

                    elif scenario == 'B' and prev_dev_count is not None:
                        if step > 0 and not prev_was_fallback_dev and dev_count <= prev_dev_count:
                            print(
                                f"  ❌ {mun} {exp_type} scenario B step {step}: "
                                f"Dev not growing"
                            )
                            all_passed = False

                    elif scenario == 'C':
                        if prev_train_count is not None and step > 0:
                            if not prev_was_fallback_train and train_count <= prev_train_count:
                                print(
                                    f"  ❌ {mun} {exp_type} scenario C step {step}: "
                                    f"Train not growing"
                                )
                                all_passed = False
                        if prev_dev_count is not None and step > 0:
                            if not prev_was_fallback_dev and dev_count <= prev_dev_count:
                                print(
                                    f"  ❌ {mun} {exp_type} scenario C step {step}: "
                                    f"Dev not growing"
                                )
                                all_passed = False

                    prev_train_count = train_count
                    prev_dev_count = dev_count

    if all_passed:
        print(f"  ✅ All data grows incrementally as expected")

    return all_passed


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description='Validate incremental learning experiment data'
    )
    parser.add_argument(
        '--municipalities', nargs='+',
        help='Specific municipalities to validate (default: all)'
    )
    parser.add_argument(
        '--scenarios', nargs='+', choices=['A', 'B', 'C'],
        help='Specific scenarios to validate (default: all)'
    )

    args = parser.parse_args()

    print("="*80)
    print("INCREMENTAL LEARNING EXPERIMENTS V2 - DATA VALIDATION")
    print("="*80)

    # Run all validation checks
    checks = [
        ("Test set consistency", check_test_set_consistency),
        ("Scenario A: train only", check_scenario_A_train_only),
        ("Scenario B: dev only", check_scenario_B_dev_only),
        ("Scenario C: different docs", check_scenario_C_different_docs),
        ("Metadata accuracy", check_metadata_accuracy),
        ("No data leakage", check_no_data_leakage),
        ("Incremental growth", check_incremental_growth),
    ]

    results = {}
    all_passed = True

    for name, check_fn in checks:
        try:
            result = check_fn()
            results[name] = result
            if not result:
                all_passed = False
        except Exception as e:
            print(f"\n  ❌ Error in {name}: {e}")
            results[name] = False
            all_passed = False

    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name}: {status}")

    print("="*80)

    if all_passed:
        print("\n✅ ALL VALIDATION CHECKS PASSED")
        print("\nData is ready for experiments. Next steps:")
        print("  python scripts/run_incremental_v2_experiments.py --gpus 0 1 2 3 --scenario A")
        return 0
    else:
        print("\n❌ SOME VALIDATION CHECKS FAILED")
        print("\nPlease fix data issues before running experiments.")
        return 1


if __name__ == '__main__':
    exit(main())
