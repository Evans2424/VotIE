#!/usr/bin/env python3
"""
Generate experiment configs for incremental learning experiments v2.

Creates YAML config files dynamically for each experiment, avoiding the need
to maintain hundreds of static config files.

Usage:
    from generate_incremental_v2_configs import generate_config
    config_path = generate_config('finetune', 'A', 'M01', 1, base_template, 0, lomo_ckpt)
"""

import yaml
from pathlib import Path
from typing import Optional


def generate_config(
    experiment_type: str,
    scenario: str,
    municipality: str,
    step: int,
    base_config_template: Path,
    gpu_id: int,
    lomo_checkpoint_path: Optional[Path] = None
) -> Path:
    """
    Generate experiment config dynamically.

    Args:
        experiment_type: 'finetune' or 'fromscratch'
        scenario: 'A', 'B', or 'C'
        municipality: 'M01' through 'M06'
        step: 0 through 5
        base_config_template: Template config file (e.g., deberta_crf_M01.yaml)
        gpu_id: CUDA device ID
        lomo_checkpoint_path: For fine-tuning experiments only

    Returns:
        Path to generated config file
    """
    # Load template
    with open(base_config_template, 'r') as f:
        config = yaml.safe_load(f)

    # Update data paths
    data_dir = (
        f"data/incremental_experiments_v2/{experiment_type}/"
        f"scenario_{scenario}/{municipality}/step_{step:02d}"
    )
    config['data']['data_dir'] = data_dir
    config['data']['train_file'] = 'train.jsonl'
    config['data']['dev_file'] = 'dev.jsonl'
    config['data']['test_file'] = 'test.jsonl'

    # Set device
    config['training']['device'] = gpu_id

    # Add experiment metadata
    config['experiment'] = {
        'type': experiment_type,
        'scenario': scenario,
        'municipality': municipality,
        'step': step,
        'description': (
            f"{experiment_type.capitalize()} - "
            f"Scenario {scenario} - {municipality} - Step {step}"
        )
    }

    # For fine-tuning: add checkpoint resumption parameters and override
    # hyperparameters to suit small target-domain datasets.
    #
    # The base config is calibrated for full LOMO training (2000+ segments).
    # Fine-tuning datasets are tiny (5–50 segments), so we need:
    #   - Lower LR  : prevents catastrophic forgetting of LOMO representations
    #   - Smaller batch: more gradient updates per epoch
    #   - More epochs : each epoch is cheap, but the model needs more iterations
    #   - Higher patience: dev signal is noisier on tiny data
    if experiment_type == 'finetune' and lomo_checkpoint_path:
        config['training']['resume_from_checkpoint'] = str(lomo_checkpoint_path)
        config['training']['learning_rate'] = 2e-5   # was 5e-5
        config['training']['batch_size'] = 4          # was 16; more updates/epoch
        config['training']['epochs'] = 30             # was 10; each epoch is cheap
        config['training']['patience'] = 7            # was 3; noisier dev signal

    # Save config
    output_path = Path(
        f"configs/incremental_v2/"
        f"{experiment_type}_scenario{scenario}_{municipality}_step{step:02d}.yaml"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    return output_path


def main():
    """Test config generation."""
    import argparse

    parser = argparse.ArgumentParser(description='Test config generation')
    parser.add_argument('--municipality', default='M01')
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--experiment-type', choices=['finetune', 'fromscratch'], default='finetune')
    parser.add_argument('--scenario', choices=['A', 'B', 'C'], default='A')
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    base_config = Path(f"configs/municipality_experiments/deberta_crf_{args.municipality}.yaml")

    if not base_config.exists():
        print(f"Error: Base config not found: {base_config}")
        return

    # Test config generation
    lomo_ckpt = None
    if args.experiment_type == 'finetune':
        lomo_ckpt = Path(
            f"data/incremental_experiments_v2/lomo_checkpoints/"
            f"{args.municipality}/best_model/checkpoint.pt"
        )

    config_path = generate_config(
        experiment_type=args.experiment_type,
        scenario=args.scenario,
        municipality=args.municipality,
        step=args.step,
        base_config_template=base_config,
        gpu_id=args.gpu,
        lomo_checkpoint_path=lomo_ckpt
    )

    print(f"✅ Config generated: {config_path}")

    # Display config
    with open(config_path, 'r') as f:
        print("\nGenerated config:")
        print("="*80)
        print(f.read())


if __name__ == '__main__':
    main()
