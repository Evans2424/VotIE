#!/usr/bin/env python3
"""
Run incremental document addition experiments.

Starting from a LOMO baseline (trained on all municipalities except the target),
incrementally adds documents from the target municipality to training data
and measures how performance improves.

Usage:
    python scripts/run_incremental_document_experiment.py --gpus 1 3
    python scripts/run_incremental_document_experiment.py --gpus 1 3 --background
    python scripts/run_incremental_document_experiment.py --gpus 1 3 --steps 0 1 2 3
    python scripts/run_incremental_document_experiment.py --gpus 1 3 --skip-training
"""

import os
import sys
import json
import random
import signal
import subprocess
import yaml
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from collections import defaultdict
import argparse

# ── Helpers ──────────────────────────────────────────────────────────────────


def load_jsonl(path: Path) -> List[Dict]:
    examples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def save_jsonl(examples: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')


def group_by_document(examples: List[Dict]) -> Dict[str, List[Dict]]:
    groups = defaultdict(list)
    for ex in examples:
        doc_id = ex.get('document_id', ex['id'].rsplit('_seg', 1)[0])
        groups[doc_id].append(ex)
    return dict(groups)


def create_config(base_config_path: Path, data_dir: str, gpu_id: int, output_path: Path):
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['data']['data_dir'] = data_dir
    config['data']['train_file'] = 'train.jsonl'
    config['data']['dev_file'] = 'dev.jsonl'
    config['data']['test_file'] = 'test.jsonl'
    config['training']['device'] = gpu_id

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"    Config {output_path.name}: device={gpu_id}")


# ── Data preparation ─────────────────────────────────────────────────────────


def prepare_data_splits(municipality: str, seed: int = 42) -> List[Dict[str, Any]]:
    """Read LOMO data, shuffle test documents, write incremental splits."""
    base_dir = Path(f"data/citilink_spans_municipality_splits/{municipality}")

    train_data = load_jsonl(base_dir / "train.jsonl")
    dev_data = load_jsonl(base_dir / "dev.jsonl")
    test_data = load_jsonl(base_dir / "test.jsonl")

    print(f"Base LOMO data for {municipality}:")
    print(f"  Train: {len(train_data)} segments (from other municipalities)")
    print(f"  Dev:   {len(dev_data)} segments")
    print(f"  Test:  {len(test_data)} segments (all from {municipality})")

    doc_groups = group_by_document(test_data)
    doc_ids = sorted(doc_groups.keys())

    print(f"  Unique documents in test: {len(doc_ids)}")
    for doc_id in doc_ids:
        print(f"    {doc_id}: {len(doc_groups[doc_id])} segments")

    rng = random.Random(seed)
    rng.shuffle(doc_ids)

    print(f"\nShuffled document order (seed={seed}):")
    for i, doc_id in enumerate(doc_ids):
        print(f"  {i}: {doc_id} ({len(doc_groups[doc_id])} segments)")

    steps = []
    output_base = Path(f"data/incremental_experiments/{municipality}")

    for k in range(len(doc_ids)):  # 0 to N-1 (last step would have empty test)
        step_dir = output_base / f"step_{k:02d}"

        added_segments = [seg for d in doc_ids[:k] for seg in doc_groups[d]]
        test_segments = [seg for d in doc_ids[k:] for seg in doc_groups[d]]

        save_jsonl(train_data + added_segments, step_dir / "train.jsonl")
        save_jsonl(dev_data, step_dir / "dev.jsonl")
        save_jsonl(test_segments, step_dir / "test.jsonl")

        steps.append({
            'step': k,
            'num_docs_added': k,
            'num_train_segments': len(train_data) + len(added_segments),
            'num_test_segments': len(test_segments),
            'data_dir': str(step_dir),
        })

    return steps


def load_existing_steps(municipality: str) -> List[Dict[str, Any]]:
    """Reconstruct step info from existing data directories."""
    step_dirs = sorted(Path(f"data/incremental_experiments/{municipality}").glob("step_*"))
    steps = []
    for step_dir in step_dirs:
        k = int(step_dir.name.split('_')[1])
        train_count = sum(1 for _ in open(step_dir / "train.jsonl"))
        test_count = sum(1 for _ in open(step_dir / "test.jsonl"))
        steps.append({
            'step': k,
            'num_docs_added': k,
            'num_train_segments': train_count,
            'num_test_segments': test_count,
            'data_dir': str(step_dir),
        })
    return steps


# ── Process management ───────────────────────────────────────────────────────

# All child processes inherit the parent's process group.
# This means:
#   - Foreground: Ctrl+C sends SIGINT to the entire group, killing everything.
#   - Background: `kill -TERM -PGID` kills the main script + all children.
# We do NOT use os.setsid on children — that would isolate them from the group.

# Active child processes (for cleanup on signal)
_active_procs: List[Dict] = []


def cleanup():
    """Terminate all active child processes."""
    for entry in _active_procs:
        proc = entry['proc']
        if proc.poll() is None:
            proc.terminate()
        entry['log_file'].close()
    _active_procs.clear()


def on_signal(signum, _frame):
    print(f"\nReceived signal {signum}, stopping...")
    cleanup()
    sys.exit(1)


def launch_step(step_info: Dict, mun: str, base_config_path: Path,
                gpu_id: int, skip_training: bool, skip_prediction: bool,
                skip_evaluation: bool) -> Dict:
    """Launch train->predict->evaluate for one step. Returns process info dict."""
    k = step_info['step']
    step_label = f"{mun}_step_{k:02d}"
    experiment_name = f"incremental_{step_label}"
    config_path = Path(f"configs/incremental_experiments/xlmr_crf_{step_label}.yaml")
    log_path = Path(f"incremental_experiments/logs/{step_label}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    create_config(base_config_path, step_info['data_dir'], gpu_id, config_path)

    # Build sequential command chain for this step
    cmds = []

    if not skip_training:
        cmds.append(
            f"{sys.executable} scripts/train.py"
            f" --config {config_path}"
            f" --experiment-name {experiment_name}"
        )

    if not skip_prediction:
        model_path = f"models/xlmr_crf/{experiment_name}"
        test_data_path = f"{step_info['data_dir']}/test.jsonl"
        predictions_path = f"incremental_experiments/predictions/{step_label}_predictions.jsonl"
        cmds.append(
            f"{sys.executable} scripts/predict.py"
            f" {model_path} {test_data_path} {predictions_path} cuda:{gpu_id}"
        )

    if not skip_evaluation:
        predictions_path = f"incremental_experiments/predictions/{step_label}_predictions.jsonl"
        eval_path = f"incremental_experiments/evaluation/{step_label}_results.json"
        cmds.append(
            f"{sys.executable} scripts/evaluate.py"
            f" {predictions_path} {eval_path}"
        )

    if not cmds:
        return None

    log_file = open(log_path, 'w')
    proc = subprocess.Popen(
        " && ".join(cmds), shell=True,
        stdout=log_file, stderr=subprocess.STDOUT
    )

    print(f"  [GPU {gpu_id}] Step {k} (+{step_info['num_docs_added']} docs) -> {log_path}")

    entry = {'proc': proc, 'log_file': log_file,
             'step': k, 'gpu_id': gpu_id, 'log_path': log_path}
    _active_procs.append(entry)
    return entry


def wait_batch(entries: List[Dict]):
    """Wait for all processes in a batch to finish."""
    for entry in entries:
        entry['proc'].wait()
        entry['log_file'].close()
        rc = entry['proc'].returncode
        status = "OK" if rc == 0 else f"FAILED (code {rc})"
        print(f"  [GPU {entry['gpu_id']}] Step {entry['step']} finished: {status}")
        if rc != 0:
            print(f"    See log: {entry['log_path']}")
    _active_procs.clear()


def collect_result(step_info: Dict, mun: str) -> Dict[str, Any]:
    k = step_info['step']
    step_label = f"{mun}_step_{k:02d}"
    eval_file = Path(f"incremental_experiments/evaluation/{step_label}_results.json")

    result = {
        'step': k,
        'num_docs_added': step_info['num_docs_added'],
        'num_train_segments': step_info['num_train_segments'],
        'num_test_segments': step_info['num_test_segments'],
    }

    if eval_file.exists():
        with open(eval_file, 'r') as f:
            metrics = json.load(f).get('entity_level_metrics', {})
        result['entity_f1'] = metrics.get('entity_f1')
        result['entity_precision'] = metrics.get('entity_precision')
        result['entity_recall'] = metrics.get('entity_recall')
        result['success'] = True
    else:
        result['success'] = False

    return result


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Run incremental document addition experiments"
    )
    parser.add_argument('--municipality', default='M01',
                        help='Target municipality (default: M01)')
    parser.add_argument('--gpus', nargs='+', type=int, required=True,
                        help='CUDA GPU IDs (e.g., --gpus 1 3)')
    parser.add_argument('--steps', nargs='+', type=int, default=None,
                        help='Specific steps to run (e.g., --steps 0 1 2 3)')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Maximum number of steps to run')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training, only predict/evaluate')
    parser.add_argument('--skip-prediction', action='store_true',
                        help='Skip prediction, only evaluate')
    parser.add_argument('--skip-evaluation', action='store_true',
                        help='Skip evaluation')
    parser.add_argument('--skip-data-prep', action='store_true',
                        help='Skip data preparation (reuse existing splits)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for document shuffling (default: 42)')
    parser.add_argument('--background', action='store_true',
                        help='Detach from terminal. Logs to incremental_experiments/main.log')

    args = parser.parse_args()

    # ── Background mode: re-launch detached, then exit ──
    if args.background:
        Path("incremental_experiments").mkdir(parents=True, exist_ok=True)
        log_path = Path("incremental_experiments/main.log")
        new_argv = [a for a in sys.argv if a != '--background']
        log_file = open(log_path, 'w')
        proc = subprocess.Popen(
            [sys.executable] + new_argv,
            stdout=log_file, stderr=subprocess.STDOUT,
            start_new_session=True  # new session + process group
        )
        pgid = proc.pid  # session leader's PID == PGID
        Path("incremental_experiments/main.pid").write_text(str(pgid))
        print(f"Running in background (PID/PGID: {pgid})")
        print(f"  tail -f {log_path}")
        print(f"  kill -- -{pgid}")
        sys.exit(0)

    # ── Signal handlers for graceful cleanup ──
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    mun = args.municipality
    gpus = args.gpus
    num_gpus = len(gpus)

    base_config_path = Path(f"configs/municipality_experiments/xlmr_crf_{mun}.yaml")
    if not base_config_path.exists():
        print(f"Error: Base config not found: {base_config_path}")
        sys.exit(1)

    if not Path(f"data/citilink_spans_municipality_splits/{mun}").exists():
        print(f"Error: Data directory not found for {mun}")
        sys.exit(1)

    print("=" * 80)
    print(f"Incremental Document Addition Experiment - {mun}")
    print(f"GPUs: {gpus} ({num_gpus} parallel jobs)")
    print("=" * 80)

    # ── Prepare data splits ──
    if args.skip_data_prep:
        all_steps = load_existing_steps(mun)
    else:
        print("\n--- Preparing incremental data splits ---")
        all_steps = prepare_data_splits(mun, seed=args.seed)

    if args.steps is not None:
        all_steps = [s for s in all_steps if s['step'] in args.steps]
    if args.max_steps is not None:
        all_steps = all_steps[:args.max_steps]

    print(f"\nWill run {len(all_steps)} steps across {num_gpus} GPUs:")
    for s in all_steps:
        print(f"  Step {s['step']:2d}: +{s['num_docs_added']} docs, "
              f"train={s['num_train_segments']}, test={s['num_test_segments']}")

    for d in ["incremental_experiments/predictions",
              "incremental_experiments/evaluation",
              "incremental_experiments/logs",
              "configs/incremental_experiments"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # ── Run experiments in batches of num_gpus ──
    results = []

    for batch_start in range(0, len(all_steps), num_gpus):
        batch = all_steps[batch_start:batch_start + num_gpus]

        print(f"\n{'#' * 80}")
        print(f"# Batch: steps {[s['step'] for s in batch]} "
              f"({batch_start + 1}-{batch_start + len(batch)} of {len(all_steps)})")
        print(f"{'#' * 80}")

        # Launch one process per GPU
        entries = []
        for i, step_info in enumerate(batch):
            entry = launch_step(
                step_info, mun, base_config_path, gpus[i],
                args.skip_training, args.skip_prediction, args.skip_evaluation
            )
            if entry is not None:
                entries.append(entry)

        # Wait for batch to complete
        wait_batch(entries)

        for step_info in batch:
            results.append(collect_result(step_info, mun))

    # ── Summary ──
    results.sort(key=lambda r: r['step'])

    summary = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'municipality': mun,
            'seed': args.seed,
            'gpus': gpus,
            'num_steps': len(results),
        },
        'results': results,
    }

    summary_path = Path("incremental_experiments/results_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n Results saved to {summary_path}")
    print("\n" + "=" * 80)
    print("INCREMENTAL DOCUMENT ADDITION - LEARNING CURVE")
    print("=" * 80)
    print(f"Municipality: {mun}")
    print(f"{'Step':>5} {'Docs':>5} {'Train':>7} {'Test':>6} {'F1':>8} {'Prec':>8} {'Rec':>8}")
    print("-" * 55)

    for r in results:
        f1 = f"{r['entity_f1']:.4f}" if r.get('entity_f1') is not None else "N/A"
        prec = f"{r['entity_precision']:.4f}" if r.get('entity_precision') is not None else "N/A"
        rec = f"{r['entity_recall']:.4f}" if r.get('entity_recall') is not None else "N/A"
        print(f"{r['step']:>5} {r['num_docs_added']:>5} {r['num_train_segments']:>7} "
              f"{r['num_test_segments']:>6} {f1:>8} {prec:>8} {rec:>8}")

    print("=" * 55)


if __name__ == '__main__':
    main()
