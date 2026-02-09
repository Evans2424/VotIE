#!/usr/bin/env python3
"""
Run incremental document addition experiments.

Starting from a LOMO baseline (trained on all municipalities except the target),
incrementally adds documents from the target municipality to training data
and measures how performance improves.

This helps answer: "How many annotated documents from a new municipality
do I need to significantly improve extraction performance?"

Usage:
    python scripts/run_incremental_document_experiment.py
    python scripts/run_incremental_document_experiment.py --municipality M02
    python scripts/run_incremental_document_experiment.py --steps 0 1 2 3  # Run specific steps only
    python scripts/run_incremental_document_experiment.py --skip-training   # Reuse existing models
"""

import sys
import json
import random
import subprocess
import yaml
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from collections import defaultdict
import argparse


def run_command(cmd: List[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'=' * 80}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 80}\n")

    try:
        subprocess.run(cmd, check=True)
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n✗ {description} failed with error: {e}")
        return False


def load_jsonl(path: Path) -> List[Dict]:
    """Load a JSONL file."""
    examples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def save_jsonl(examples: List[Dict], path: Path):
    """Save examples to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')


def group_by_document(examples: List[Dict]) -> Dict[str, List[Dict]]:
    """Group segments by document_id."""
    groups = defaultdict(list)
    for ex in examples:
        doc_id = ex.get('document_id', ex['id'].rsplit('_seg', 1)[0])
        groups[doc_id].append(ex)
    return dict(groups)


def create_config(base_config_path: Path, data_dir: str, output_path: Path):
    """Create a config YAML pointing to the given data directory."""
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['data']['data_dir'] = data_dir
    config['data']['train_file'] = 'train.jsonl'
    config['data']['dev_file'] = 'dev.jsonl'
    config['data']['test_file'] = 'test.jsonl'

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def prepare_data_splits(
    municipality: str,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Prepare incremental data splits.

    Returns a list of step info dicts with keys:
        step, num_docs_added, doc_ids_added, num_train_segments, num_test_segments
    """
    base_dir = Path(f"data/citilink_spans_municipality_splits/{municipality}")

    train_data = load_jsonl(base_dir / "train.jsonl")
    dev_data = load_jsonl(base_dir / "dev.jsonl")
    test_data = load_jsonl(base_dir / "test.jsonl")

    print(f"Base LOMO data for {municipality}:")
    print(f"  Train: {len(train_data)} segments (from other municipalities)")
    print(f"  Dev:   {len(dev_data)} segments")
    print(f"  Test:  {len(test_data)} segments (all from {municipality})")

    # Group test segments by document
    doc_groups = group_by_document(test_data)
    doc_ids = sorted(doc_groups.keys())

    print(f"  Unique documents in test: {len(doc_ids)}")
    for doc_id in doc_ids:
        print(f"    {doc_id}: {len(doc_groups[doc_id])} segments")

    # Shuffle document order
    rng = random.Random(seed)
    rng.shuffle(doc_ids)

    print(f"\nShuffled document order (seed={seed}):")
    for i, doc_id in enumerate(doc_ids):
        print(f"  {i}: {doc_id} ({len(doc_groups[doc_id])} segments)")

    # Create incremental splits
    steps = []
    output_base = Path(f"data/incremental_experiments/{municipality}")

    for k in range(len(doc_ids)):  # 0 to N-1 (skip last since test would be empty)
        step_dir = output_base / f"step_{k:02d}"

        # Documents added to training (first k)
        added_doc_ids = doc_ids[:k]
        added_segments = []
        for doc_id in added_doc_ids:
            added_segments.extend(doc_groups[doc_id])

        # Remaining documents for testing
        remaining_doc_ids = doc_ids[k:]
        test_segments = []
        for doc_id in remaining_doc_ids:
            test_segments.extend(doc_groups[doc_id])

        # Write data files
        save_jsonl(train_data + added_segments, step_dir / "train.jsonl")
        save_jsonl(dev_data, step_dir / "dev.jsonl")
        save_jsonl(test_segments, step_dir / "test.jsonl")

        steps.append({
            'step': k,
            'num_docs_added': k,
            'doc_ids_added': added_doc_ids,
            'num_train_segments': len(train_data) + len(added_segments),
            'num_test_segments': len(test_segments),
            'data_dir': str(step_dir),
        })

    return steps


def main():
    parser = argparse.ArgumentParser(
        description="Run incremental document addition experiments"
    )
    parser.add_argument(
        '--municipality',
        default='M01',
        help='Target municipality to incrementally add documents from (default: M01)'
    )
    parser.add_argument(
        '--steps',
        nargs='+',
        type=int,
        default=None,
        help='Specific steps to run (default: all). E.g., --steps 0 1 2 3'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=None,
        help='Maximum number of steps to run'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip training, only predict/evaluate'
    )
    parser.add_argument(
        '--skip-prediction',
        action='store_true',
        help='Skip prediction, only evaluate'
    )
    parser.add_argument(
        '--skip-evaluation',
        action='store_true',
        help='Skip evaluation'
    )
    parser.add_argument(
        '--skip-data-prep',
        action='store_true',
        help='Skip data preparation (reuse existing splits)'
    )
    parser.add_argument(
        '--device',
        default='auto',
        choices=['cuda', 'mps', 'cpu', 'auto'],
        help='Device for training/inference (default: auto)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for document shuffling (default: 42)'
    )

    args = parser.parse_args()
    mun = args.municipality

    base_config_path = Path(f"configs/municipality_experiments/xlmr_crf_{mun}.yaml")
    if not base_config_path.exists():
        print(f"Error: Base config not found: {base_config_path}")
        sys.exit(1)

    base_data_dir = Path(f"data/citilink_spans_municipality_splits/{mun}")
    if not base_data_dir.exists():
        print(f"Error: Data directory not found: {base_data_dir}")
        sys.exit(1)

    print("=" * 80)
    print(f"Incremental Document Addition Experiment - {mun}")
    print("=" * 80)

    # Step 1: Prepare data splits
    if not args.skip_data_prep:
        print("\n--- Preparing incremental data splits ---")
        all_steps = prepare_data_splits(mun, seed=args.seed)
    else:
        # Reconstruct step info from existing directories
        step_dirs = sorted(Path(f"data/incremental_experiments/{mun}").glob("step_*"))
        all_steps = []
        for step_dir in step_dirs:
            k = int(step_dir.name.split('_')[1])
            train_count = sum(1 for _ in open(step_dir / "train.jsonl"))
            test_count = sum(1 for _ in open(step_dir / "test.jsonl"))
            all_steps.append({
                'step': k,
                'num_docs_added': k,
                'doc_ids_added': [],
                'num_train_segments': train_count,
                'num_test_segments': test_count,
                'data_dir': str(step_dir),
            })

    # Filter steps
    if args.steps is not None:
        all_steps = [s for s in all_steps if s['step'] in args.steps]
    if args.max_steps is not None:
        all_steps = all_steps[:args.max_steps]

    print(f"\nWill run {len(all_steps)} steps:")
    for s in all_steps:
        print(f"  Step {s['step']:2d}: +{s['num_docs_added']} docs, "
              f"train={s['num_train_segments']} segments, "
              f"test={s['num_test_segments']} segments")

    # Create output directories
    Path("incremental_experiments/predictions").mkdir(parents=True, exist_ok=True)
    Path("incremental_experiments/evaluation").mkdir(parents=True, exist_ok=True)
    Path("configs/incremental_experiments").mkdir(parents=True, exist_ok=True)

    # Step 2: Run experiments
    results = []

    for s in all_steps:
        k = s['step']
        print(f"\n{'#' * 80}")
        print(f"# Step {k}: {s['num_docs_added']} documents added "
              f"(train={s['num_train_segments']}, test={s['num_test_segments']})")
        print(f"{'#' * 80}")

        step_label = f"{mun}_step_{k:02d}"
        experiment_name = f"incremental_{step_label}"
        config_path = Path(f"configs/incremental_experiments/xlmr_crf_{step_label}.yaml")

        # Create config
        create_config(base_config_path, s['data_dir'], config_path)

        success = True

        # Training
        if not args.skip_training:
            cmd = [
                sys.executable, "scripts/train.py",
                "--config", str(config_path),
                "--experiment-name", experiment_name
            ]
            if not run_command(cmd, f"Training step {k} (+{s['num_docs_added']} docs)"):
                print(f"✗ Training failed for step {k}")
                success = False

        # Prediction
        if success and not args.skip_prediction:
            model_path = f"models/xlmr_crf/{experiment_name}"
            test_data_path = f"{s['data_dir']}/test.jsonl"
            predictions_path = f"incremental_experiments/predictions/{step_label}_predictions.jsonl"

            cmd = [
                sys.executable, "scripts/predict.py",
                model_path, test_data_path, predictions_path, args.device
            ]
            if not run_command(cmd, f"Predicting step {k}"):
                print(f"✗ Prediction failed for step {k}")
                success = False

        # Evaluation
        if success and not args.skip_evaluation:
            predictions_path = f"incremental_experiments/predictions/{step_label}_predictions.jsonl"
            eval_path = f"incremental_experiments/evaluation/{step_label}_results.json"

            cmd = [
                sys.executable, "scripts/evaluate.py",
                predictions_path, eval_path
            ]
            if not run_command(cmd, f"Evaluating step {k}"):
                print(f"✗ Evaluation failed for step {k}")
                success = False

        # Collect result
        eval_file = Path(f"incremental_experiments/evaluation/{step_label}_results.json")
        step_result = {
            'step': k,
            'num_docs_added': s['num_docs_added'],
            'num_train_segments': s['num_train_segments'],
            'num_test_segments': s['num_test_segments'],
            'success': success,
        }

        if eval_file.exists():
            with open(eval_file, 'r') as f:
                eval_data = json.load(f)
            entity_metrics = eval_data.get('entity_level_metrics', {})
            step_result['entity_f1'] = entity_metrics.get('entity_f1', None)
            step_result['entity_precision'] = entity_metrics.get('entity_precision', None)
            step_result['entity_recall'] = entity_metrics.get('entity_recall', None)

        results.append(step_result)

    # Step 3: Save and print summary
    summary = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'municipality': mun,
            'seed': args.seed,
            'num_steps': len(results),
        },
        'results': results,
    }

    summary_path = Path("incremental_experiments/results_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Results saved to {summary_path}")

    # Print learning curve table
    print("\n" + "=" * 80)
    print("INCREMENTAL DOCUMENT ADDITION - LEARNING CURVE")
    print("=" * 80)
    print(f"Municipality: {mun}")
    print(f"{'Step':>5} {'Docs Added':>11} {'Train Segs':>11} {'Test Segs':>10} {'F1':>8} {'Precision':>10} {'Recall':>8}")
    print("-" * 80)

    for r in results:
        f1 = f"{r['entity_f1']:.4f}" if r.get('entity_f1') is not None else "N/A"
        prec = f"{r['entity_precision']:.4f}" if r.get('entity_precision') is not None else "N/A"
        rec = f"{r['entity_recall']:.4f}" if r.get('entity_recall') is not None else "N/A"
        print(f"{r['step']:>5} {r['num_docs_added']:>11} {r['num_train_segments']:>11} "
              f"{r['num_test_segments']:>10} {f1:>8} {prec:>10} {rec:>8}")

    print("=" * 80)


if __name__ == '__main__':
    main()
