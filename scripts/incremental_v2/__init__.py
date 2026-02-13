"""
Incremental Learning Experiments V2

This package contains scripts for running comprehensive incremental learning experiments
to study fine-tuning vs from-scratch training and document placement strategies.

Main Scripts:
    - prepare_incremental_v2_data.py: Data preparation for all scenarios
    - run_incremental_v2_experiments.py: Main orchestration script
    - validate_incremental_v2_data.py: Data validation suite
    - aggregate_incremental_v2_results.py: Results aggregation and analysis
    - generate_incremental_v2_configs.py: Dynamic config generation
    - test_incremental_v2_setup.py: Integration testing

Usage:
    All scripts should be run from the project root directory:

    python scripts/incremental_v2/prepare_incremental_v2_data.py --scenarios A
    python scripts/incremental_v2/run_incremental_v2_experiments.py --gpus 0 1 2 3 --scenario A

See INCREMENTAL_EXPERIMENTS_V2_README.md for detailed documentation.
"""

__version__ = "2.0.0"
