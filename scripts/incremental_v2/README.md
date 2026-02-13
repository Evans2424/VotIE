# Incremental Learning Experiments V2 - Scripts

This directory contains all scripts for the comprehensive incremental learning experimental framework.

## Scripts Overview

### 1. Data Preparation
**`prepare_incremental_v2_data.py`**
- Prepares data splits for all experiment configurations
- Supports 3 scenarios (A: train only, B: dev only, C: split)
- Creates fixed test sets and metadata files

```bash
python scripts/incremental_v2/prepare_incremental_v2_data.py --scenarios A
```

### 2. Main Orchestration
**`run_incremental_v2_experiments.py`**
- Main script to run all experiments
- Handles LOMO baseline training
- Parallel execution across GPUs
- Complete pipeline: train → predict → evaluate

```bash
python scripts/incremental_v2/run_incremental_v2_experiments.py --gpus 0 1 2 3 --scenario A
```

### 3. Data Validation
**`validate_incremental_v2_data.py`**
- 7 comprehensive validation checks
- Verifies data integrity before experiments
- Checks test set consistency, no data leakage, etc.

```bash
python scripts/incremental_v2/validate_incremental_v2_data.py
```

### 4. Results Aggregation
**`aggregate_incremental_v2_results.py`**
- Collects evaluation results
- Computes summary statistics
- Compares fine-tuning vs from-scratch
- Generates LaTeX tables

```bash
python scripts/incremental_v2/aggregate_incremental_v2_results.py --scenario A
```

### 5. Config Generation
**`generate_incremental_v2_configs.py`**
- Dynamically generates YAML configs for experiments
- Avoids maintaining 216 static config files
- Used internally by orchestration script

```bash
python scripts/incremental_v2/generate_incremental_v2_configs.py --help
```

### 6. Integration Testing
**`test_incremental_v2_setup.py`**
- Tests complete pipeline with M01
- Verifies all components work together
- Run before full experiments

```bash
python scripts/incremental_v2/test_incremental_v2_setup.py --gpu 0
```

## Quick Start

```bash
# 1. Test setup
python scripts/incremental_v2/test_incremental_v2_setup.py --gpu 0

# 2. Prepare data
python scripts/incremental_v2/prepare_incremental_v2_data.py --scenarios A

# 3. Validate data
python scripts/incremental_v2/validate_incremental_v2_data.py

# 4. Run experiments
python scripts/incremental_v2/run_incremental_v2_experiments.py \
    --gpus 0 1 2 3 --scenario A

# 5. Aggregate results
python scripts/incremental_v2/aggregate_incremental_v2_results.py --scenario A
```

## Documentation

- **Main Guide**: `../../INCREMENTAL_EXPERIMENTS_V2_README.md`
- **Implementation Summary**: `../../IMPLEMENTATION_SUMMARY.md`
- **Quick Reference**: `../../INCREMENTAL_V2_QUICK_REFERENCE.md`

## Important Notes

- **Run from project root**: All scripts assume they're run from the VotIE - ACL directory
- **Import paths**: Scripts use `scripts.incremental_v2.*` imports
- **GPU requirements**: Most experiments need CUDA-capable GPUs
- **Disk space**: Allow ~100GB for full experiment matrix

## Experiment Matrix

- **Scenario A** (Train only): 72 experiments
- **Scenario B** (Dev only): 72 experiments
- **Scenario C** (Split): 72 experiments
- **Total**: 216 experiments

Each experiment configuration:
- 2 types: fine-tuning, from-scratch
- 6 municipalities: M01-M06
- 6 steps: 0-5 documents added

## Support

For issues or questions:
```bash
# Get help for any script
python scripts/incremental_v2/<script_name>.py --help

# Check logs
tail -f incremental_experiments_v2/logs/*.log
```
