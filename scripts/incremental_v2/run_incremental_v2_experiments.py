#!/usr/bin/env python3
"""
Run incremental learning experiments v2.

Orchestrates the complete experimental pipeline:
- Data preparation
- LOMO baseline training (if needed)
- Fine-tuning and from-scratch experiments
- Parallel execution across multiple GPUs

Usage:
    # Scenario A only (recommended first)
    python scripts/run_incremental_v2_experiments.py --gpus 0 1 2 3 --scenario A

    # All scenarios
    python scripts/run_incremental_v2_experiments.py --gpus 0 1 2 3 --scenario all

    # Specific municipality and steps
    python scripts/run_incremental_v2_experiments.py --gpus 0 1 --municipalities M01 --steps 0 1 2

    # Prepare data only
    python scripts/run_incremental_v2_experiments.py --prepare-data-only --scenario A
"""

import sys
import time
import subprocess
import argparse
import signal
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import data preparation and config generation
from scripts.incremental_v2.prepare_incremental_v2_data import prepare_incremental_experiments_v2
from scripts.incremental_v2.generate_incremental_v2_configs import generate_config


# ── Process Management ──────────────────────────────────────────────────────

_active_procs: List[Dict] = []


def cleanup():
    """Terminate all active child processes."""
    for entry in _active_procs:
        proc = entry['proc']
        if proc.poll() is None:
            print(f"  Terminating {entry['id']}...")
            proc.terminate()
        if entry['log_file']:
            entry['log_file'].close()
    _active_procs.clear()


def on_signal(signum, frame):
    """Handle termination signals."""
    del frame  # Unused but required by signal handler signature
    print(f"\n\nReceived signal {signum}, stopping all experiments...")
    cleanup()
    sys.exit(1)


signal.signal(signal.SIGTERM, on_signal)
signal.signal(signal.SIGINT, on_signal)


# ── LOMO Baseline Training ──────────────────────────────────────────────────


def train_lomo_baseline(municipality: str, output_dir: Path, gpu_id: int) -> bool:
    """
    Train LOMO baseline model for a municipality.

    Args:
        municipality: Municipality code (e.g., 'M01')
        output_dir: Directory to save the trained model
        gpu_id: CUDA device ID

    Returns:
        True if training succeeded, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"Training LOMO Baseline: {municipality}")
    print(f"{'='*80}")

    # Use the base municipality config
    base_config = Path(f"configs/municipality_experiments/deberta_crf_{municipality}.yaml")

    if not base_config.exists():
        print(f"  ❌ Base config not found: {base_config}")
        return False

    # Load config and set output directory
    with open(base_config, 'r') as f:
        config = yaml.safe_load(f)

    config['training']['device'] = gpu_id

    # Create temporary config for LOMO baseline
    lomo_config_path = Path(f"configs/incremental_v2/lomo_baseline_{municipality}.yaml")
    lomo_config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(lomo_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    experiment_name = f"lomo_baseline_{municipality}"

    # Train model
    log_path = output_dir / "training.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = (
        f"{sys.executable} scripts/train.py "
        f"--config {lomo_config_path} "
        f"--experiment-name {experiment_name} "
        f"--output-dir {output_dir}"
    )

    print(f"  Running: {cmd}")
    print(f"  Log: {log_path}")

    with open(log_path, 'w') as log_file:
        proc = subprocess.Popen(
            cmd, shell=True,
            stdout=log_file, stderr=subprocess.STDOUT
        )
        proc.wait()

    if proc.returncode == 0:
        # Verify checkpoint exists
        checkpoint_path = output_dir / "best_model" / "checkpoint.pt"
        if checkpoint_path.exists():
            print(f"  ✅ LOMO baseline trained successfully")
            return True
        else:
            print(f"  ❌ Training completed but checkpoint not found: {checkpoint_path}")
            return False
    else:
        print(f"  ❌ Training failed (exit code {proc.returncode})")
        print(f"  See log: {log_path}")
        return False


def train_lomo_baselines_parallel(
    checkpoint_dir: Path,
    municipalities: List[str],
    gpus: List[int]
) -> bool:
    """
    Train LOMO baselines for all municipalities in parallel.

    Args:
        checkpoint_dir: Base directory for LOMO checkpoints
        municipalities: List of municipality codes
        gpus: List of GPU IDs to use in parallel

    Returns:
        True if all baselines trained successfully
    """
    print(f"\n{'='*80}")
    print(f"Training LOMO Baselines in Parallel")
    print(f"{'='*80}")
    print(f"Municipalities: {municipalities}")
    print(f"GPUs: {gpus}")
    print(f"{'='*80}\n")

    # Check which need training
    to_train = []
    for mun in municipalities:
        checkpoint_path = checkpoint_dir / mun / "best_model" / "checkpoint.pt"
        if not checkpoint_path.exists():
            to_train.append(mun)
        else:
            print(f"✅ {mun}: LOMO baseline already exists")

    if not to_train:
        print("\n✅ All LOMO baselines already exist")
        return True

    print(f"\n📋 Need to train: {to_train}")

    num_gpus = len(gpus)

    # Train in batches across GPUs
    for batch_start in range(0, len(to_train), num_gpus):
        batch = to_train[batch_start:batch_start + num_gpus]

        print(f"\n{'#'*80}")
        print(f"# Training batch: {batch} ({batch_start + 1}-{batch_start + len(batch)} of {len(to_train)})")
        print(f"{'#'*80}\n")

        # Launch training for each municipality in batch
        entries = []
        for i, mun in enumerate(batch):
            output_dir = checkpoint_dir / mun

            # Build training command
            base_config = Path(f"configs/municipality_experiments/deberta_crf_{mun}.yaml")
            if not base_config.exists():
                print(f"⚠️  Base config not found: {base_config}, skipping {mun}")
                continue

            # Load and modify config
            import yaml
            with open(base_config, 'r') as f:
                config = yaml.safe_load(f)

            config['training']['device'] = gpus[i]

            # Create temporary config
            lomo_config_path = Path(f"configs/incremental_v2/lomo_baseline_{mun}.yaml")
            lomo_config_path.parent.mkdir(parents=True, exist_ok=True)

            with open(lomo_config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

            experiment_name = f"lomo_baseline_{mun}"

            cmd = (
                f"{sys.executable} scripts/train.py "
                f"--config {lomo_config_path} "
                f"--experiment-name {experiment_name} "
                f"--output-dir {output_dir}"
            )

            log_path = output_dir / "training.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)

            log_file = open(log_path, 'w')
            proc = subprocess.Popen(
                cmd, shell=True,
                stdout=log_file, stderr=subprocess.STDOUT
            )

            entries.append({
                'proc': proc,
                'log_file': log_file,
                'municipality': mun,
                'gpu_id': gpus[i],
                'log_path': log_path,
                'output_dir': output_dir
            })

            print(f"  [GPU {gpus[i]}] Training {mun} -> {log_path}")

        # Wait for batch to complete
        all_success = True
        for entry in entries:
            entry['proc'].wait()
            entry['log_file'].close()

            rc = entry['proc'].returncode
            status = "✅ OK" if rc == 0 else f"❌ FAILED (code {rc})"

            print(f"  [GPU {entry['gpu_id']}] {entry['municipality']}: {status}")

            if rc != 0:
                print(f"    See log: {entry['log_path']}")
                all_success = False
            else:
                # Verify checkpoint exists
                checkpoint_path = entry['output_dir'] / "best_model" / "checkpoint.pt"
                if not checkpoint_path.exists():
                    print(f"    ⚠️  Checkpoint not found: {checkpoint_path}")
                    all_success = False

        if not all_success:
            print("\n❌ Some LOMO baselines failed to train")
            return False

    print(f"\n✅ All LOMO baselines trained successfully")
    return True


def ensure_lomo_checkpoints(
    checkpoint_dir: Path,
    municipalities: List[str],
    gpus: List[int]
) -> bool:
    """
    Ensure LOMO checkpoints exist for all municipalities.

    Args:
        checkpoint_dir: Base directory for LOMO checkpoints
        municipalities: List of municipality codes
        gpus: List of GPU IDs (will use first GPU for sequential training)

    Returns:
        True if all checkpoints exist or were created successfully
    """
    missing_checkpoints = []

    for mun in municipalities:
        checkpoint_path = checkpoint_dir / mun / "best_model" / "checkpoint.pt"
        if not checkpoint_path.exists():
            missing_checkpoints.append(mun)

    if not missing_checkpoints:
        print("\n✅ All LOMO checkpoints already exist")
        return True

    print(f"\n⚠️  Missing LOMO checkpoints for: {missing_checkpoints}")
    print(f"Will train {len(missing_checkpoints)} LOMO baselines...")

    gpu_id = gpus[0]  # Use first GPU for sequential training

    for mun in missing_checkpoints:
        output_dir = checkpoint_dir / mun
        success = train_lomo_baseline(mun, output_dir, gpu_id)

        if not success:
            print(f"\n❌ Failed to train LOMO baseline for {mun}")
            return False

    print(f"\n✅ All LOMO checkpoints ready")
    return True


# ── Experiment Management ───────────────────────────────────────────────────


def build_experiment_matrix(
    municipalities: List[str],
    experiment_types: List[str],
    scenarios: List[str],
    steps: List[int]
) -> List[Dict[str, Any]]:
    """Build list of all experiments to run."""
    matrix = []

    for mun in municipalities:
        for exp_type in experiment_types:
            for scenario in scenarios:
                for step in steps:
                    matrix.append({
                        'municipality': mun,
                        'type': exp_type,
                        'scenario': scenario,
                        'step': step,
                        'id': f"{exp_type}_scenario{scenario}_{mun}_step{step:02d}"
                    })

    return matrix


def launch_experiment(
    exp_info: Dict[str, Any],
    base_config: Path,
    gpu_id: int,
    lomo_checkpoint_dir: Path
) -> Optional[Dict]:
    """
    Launch single experiment (train → predict → evaluate).

    Args:
        exp_info: Experiment information dict
        base_config: Base config template path
        gpu_id: CUDA device ID
        lomo_checkpoint_dir: Directory containing LOMO checkpoints

    Returns:
        Process info dict or None if launch failed
    """
    # Generate config
    lomo_ckpt = None
    if exp_info['type'] == 'finetune':
        lomo_ckpt = (
            lomo_checkpoint_dir / exp_info['municipality'] /
            "best_model" / "checkpoint.pt"
        )

        if not lomo_ckpt.exists():
            print(f"  ❌ LOMO checkpoint not found: {lomo_ckpt}")
            return None

    config_path = generate_config(
        experiment_type=exp_info['type'],
        scenario=exp_info['scenario'],
        municipality=exp_info['municipality'],
        step=exp_info['step'],
        base_config_template=base_config,
        gpu_id=gpu_id,
        lomo_checkpoint_path=lomo_ckpt
    )

    # Build command chain
    experiment_name = exp_info['id']
    model_dir = f"models/incremental_v2/{experiment_name}"
    data_dir = (
        f"data/incremental_experiments_v2/{exp_info['type']}/"
        f"scenario_{exp_info['scenario']}/{exp_info['municipality']}/"
        f"step_{exp_info['step']:02d}"
    )
    predictions_path = f"incremental_experiments_v2/predictions/{experiment_name}.jsonl"
    eval_path = f"incremental_experiments_v2/evaluation/{experiment_name}.json"

    cmds = [
        f"{sys.executable} scripts/train.py --config {config_path} --experiment-name {experiment_name}",
        f"{sys.executable} scripts/predict.py {model_dir} {data_dir}/test.jsonl {predictions_path} cuda:{gpu_id}",
        f"{sys.executable} scripts/evaluate.py {predictions_path} {eval_path}"
    ]

    # Launch process
    log_path = Path(f"incremental_experiments_v2/logs/{experiment_name}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_file = open(log_path, 'w')
    proc = subprocess.Popen(
        " && ".join(cmds), shell=True,
        stdout=log_file, stderr=subprocess.STDOUT
    )

    entry = {
        'proc': proc,
        'log_file': log_file,
        'log_path': log_path,
        'id': experiment_name,
        'gpu_id': gpu_id,
        'exp_info': exp_info
    }

    _active_procs.append(entry)

    print(f"  [GPU {gpu_id}] {experiment_name} -> {log_path}")

    return entry


def wait_batch(entries: List[Dict]):
    """Wait for all processes in a batch to finish."""
    for entry in entries:
        entry['proc'].wait()
        if entry['log_file']:
            entry['log_file'].close()

        rc = entry['proc'].returncode
        status = "✅ OK" if rc == 0 else f"❌ FAILED (code {rc})"

        print(f"  [GPU {entry['gpu_id']}] {entry['id']}: {status}")

        if rc != 0:
            print(f"    See log: {entry['log_path']}")

    _active_procs.clear()


def run_experiment_batches(
    experiment_matrix: List[Dict],
    gpus: List[int],
    base_config: Path,
    lomo_checkpoint_dir: Path
):
    """Run experiments in parallel batches."""
    num_gpus = len(gpus)
    total = len(experiment_matrix)

    print(f"\n{'='*80}")
    print(f"Running {total} experiments across {num_gpus} GPUs")
    print(f"{'='*80}\n")

    for batch_start in range(0, total, num_gpus):
        batch = experiment_matrix[batch_start:batch_start + num_gpus]

        print(f"\n{'#'*80}")
        print(f"# Batch: {batch_start + 1}-{batch_start + len(batch)} of {total}")
        print(f"# Experiments: {[e['id'] for e in batch]}")
        print(f"{'#'*80}\n")

        # Launch experiments
        entries = []
        for i, exp_info in enumerate(batch):
            entry = launch_experiment(
                exp_info,
                base_config,
                gpus[i],
                lomo_checkpoint_dir
            )
            if entry:
                entries.append(entry)

        # Wait for batch to complete
        if entries:
            wait_batch(entries)

        print(f"\n  Batch {batch_start + 1}-{batch_start + len(batch)} complete\n")


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description='Run incremental learning experiments v2'
    )
    parser.add_argument(
        '--municipalities', nargs='+',
        default=['M01', 'M02', 'M03', 'M04', 'M05', 'M06'],
        help='Municipalities to process'
    )
    parser.add_argument(
        '--experiment-type', choices=['finetune', 'fromscratch', 'both'],
        default='both',
        help='Experiment type to run'
    )
    parser.add_argument(
        '--scenario', choices=['A', 'B', 'C', 'all'],
        default='A',
        help='Scenario to run (A: train only, B: dev only, C: split)'
    )
    parser.add_argument(
        '--steps', nargs='+', type=int,
        help='Specific steps to run (default: 0-5)'
    )
    parser.add_argument(
        '--gpus', nargs='+', type=int, required=True,
        help='GPU IDs to use'
    )
    parser.add_argument(
        '--prepare-data-only', action='store_true',
        help='Only prepare data, do not run experiments'
    )
    parser.add_argument(
        '--skip-data-prep', action='store_true',
        help='Skip data preparation (use existing data)'
    )
    parser.add_argument(
        '--lomo-checkpoint-dir',
        default='data/incremental_experiments_v2/lomo_checkpoints',
        help='Directory for LOMO checkpoints'
    )
    parser.add_argument(
        '--base-config',
        default='configs/municipality_experiments/deberta_crf_M01.yaml',
        help='Base config template'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for data preparation'
    )
    parser.add_argument(
        '--train-lomo-only', action='store_true',
        help='Only train LOMO baselines (in parallel), then exit'
    )

    args = parser.parse_args()

    print("="*80)
    print("INCREMENTAL LEARNING EXPERIMENTS V2")
    print("="*80)
    print(f"Municipalities: {args.municipalities}")
    print(f"Experiment types: {args.experiment_type}")
    print(f"Scenario: {args.scenario}")
    print(f"GPUs: {args.gpus}")
    print("="*80)

    # Step 1: Prepare data
    if not args.skip_data_prep:
        print("\n--- Data Preparation ---")
        scenarios = ['A', 'B', 'C'] if args.scenario == 'all' else [args.scenario]

        for mun in args.municipalities:
            try:
                prepare_incremental_experiments_v2(
                    municipality=mun,
                    seed=args.seed,
                    scenarios=scenarios
                )
            except Exception as e:
                print(f"❌ Error preparing data for {mun}: {e}")
                if not args.prepare_data_only:
                    sys.exit(1)

    if args.prepare_data_only:
        print("\n✅ Data preparation complete")
        return

    # Handle train-lomo-only mode
    if args.train_lomo_only:
        print("\n--- Training LOMO Baselines Only ---")
        lomo_checkpoint_dir = Path(args.lomo_checkpoint_dir)
        success = train_lomo_baselines_parallel(
            lomo_checkpoint_dir,
            args.municipalities,
            args.gpus
        )

        if success:
            print(f"\n{'='*80}")
            print("✅ ALL LOMO BASELINES TRAINED")
            print(f"{'='*80}")
            print(f"Checkpoints saved to: {lomo_checkpoint_dir}")
            print("\nNext steps:")
            print("  Run incremental experiments:")
            print(f"    python scripts/incremental_v2/run_incremental_v2_experiments.py \\")
            print(f"        --gpus {' '.join(map(str, args.gpus))} \\")
            print(f"        --scenario A")
        else:
            print("\n❌ Failed to train all LOMO baselines")
            sys.exit(1)

        return

    # Step 2: Build experiment matrix
    experiment_types = (
        ['finetune', 'fromscratch'] if args.experiment_type == 'both'
        else [args.experiment_type]
    )
    scenarios = ['A', 'B', 'C'] if args.scenario == 'all' else [args.scenario]
    steps = args.steps if args.steps else list(range(6))

    experiment_matrix = build_experiment_matrix(
        municipalities=args.municipalities,
        experiment_types=experiment_types,
        scenarios=scenarios,
        steps=steps
    )

    print(f"\nTotal experiments to run: {len(experiment_matrix)}")

    # Step 3: Ensure LOMO checkpoints exist for fine-tuning
    if any(e['type'] == 'finetune' for e in experiment_matrix):
        lomo_checkpoint_dir = Path(args.lomo_checkpoint_dir)
        success = ensure_lomo_checkpoints(
            lomo_checkpoint_dir,
            args.municipalities,
            args.gpus
        )

        if not success:
            print("\n❌ Failed to prepare LOMO checkpoints")
            sys.exit(1)

    # Step 4: Run experiments
    base_config = Path(args.base_config)
    if not base_config.exists():
        # Try to find config for first municipality
        base_config = Path(
            f"configs/municipality_experiments/deberta_crf_{args.municipalities[0]}.yaml"
        )

    if not base_config.exists():
        print(f"\n❌ Base config not found: {base_config}")
        sys.exit(1)

    start_time = time.time()

    try:
        run_experiment_batches(
            experiment_matrix,
            gpus=args.gpus,
            base_config=base_config,
            lomo_checkpoint_dir=Path(args.lomo_checkpoint_dir)
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        cleanup()
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        cleanup()
        sys.exit(1)

    elapsed = time.time() - start_time

    print(f"\n{'='*80}")
    print("✅ ALL EXPERIMENTS COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {elapsed/3600:.1f} hours")
    print(f"Experiments run: {len(experiment_matrix)}")
    print(f"\nNext steps:")
    print(f"  Aggregate results: python scripts/aggregate_incremental_v2_results.py")


if __name__ == '__main__':
    main()
