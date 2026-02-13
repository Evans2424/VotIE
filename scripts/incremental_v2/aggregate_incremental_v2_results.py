#!/usr/bin/env python3
"""
Aggregate results from incremental learning experiments v2.

Collects evaluation results, generates summaries, and creates visualizations
showing learning curves and comparisons between fine-tuning and from-scratch.

Usage:
    python scripts/aggregate_incremental_v2_results.py --scenario A
    python scripts/aggregate_incremental_v2_results.py --scenario all
    python scripts/aggregate_incremental_v2_results.py --scenario A --output-dir results/incremental_v2
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
from datetime import datetime


def load_json(path: Path) -> Dict:
    """Load JSON file."""
    if not path.exists():
        return None
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data: Dict, path: Path):
    """Save JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def collect_results(scenario: str) -> Dict[str, Any]:
    """
    Collect all evaluation results for a scenario.

    Returns:
        Dictionary with results organized by municipality and experiment type
    """
    print(f"\n{'='*80}")
    print(f"Collecting results for Scenario {scenario}")
    print(f"{'='*80}")

    eval_dir = Path("incremental_experiments_v2/evaluation")

    if not eval_dir.exists():
        print(f"  ⚠️  Evaluation directory not found: {eval_dir}")
        return {}

    results = defaultdict(lambda: defaultdict(list))

    municipalities = ['M01', 'M02', 'M03', 'M04', 'M05', 'M06']

    for mun in municipalities:
        for exp_type in ['finetune', 'fromscratch']:
            for step in range(6):
                eval_file = eval_dir / f"{exp_type}_scenario{scenario}_{mun}_step{step:02d}.json"

                if eval_file.exists():
                    eval_data = load_json(eval_file)

                    if eval_data:
                        # Extract key metrics
                        entity_metrics = eval_data.get('entity_level_metrics', {})

                        result_entry = {
                            'step': step,
                            'municipality': mun,
                            'experiment_type': exp_type,
                            'scenario': scenario,
                            'entity_f1': entity_metrics.get('entity_f1'),
                            'entity_precision': entity_metrics.get('entity_precision'),
                            'entity_recall': entity_metrics.get('entity_recall'),
                            'entity_accuracy': entity_metrics.get('entity_accuracy'),
                            'num_examples': eval_data.get('num_examples'),
                            'per_type_metrics': entity_metrics.get('per_type_metrics', {}),
                        }

                        results[mun][exp_type].append(result_entry)

    # Sort by step
    for mun in results:
        for exp_type in results[mun]:
            results[mun][exp_type].sort(key=lambda x: x['step'])

    print(f"\nCollected results:")
    for mun in municipalities:
        if mun in results:
            finetune_count = len(results[mun].get('finetune', []))
            fromscratch_count = len(results[mun].get('fromscratch', []))
            print(f"  {mun}: {finetune_count} finetune, {fromscratch_count} fromscratch")

    return dict(results)


def compute_summary_statistics(results: Dict[str, Any], scenario: str) -> Dict[str, Any]:
    """Compute summary statistics across all municipalities."""
    print(f"\nComputing summary statistics for Scenario {scenario}...")

    summary = {
        'scenario': scenario,
        'timestamp': datetime.now().isoformat(),
        'num_municipalities': len(results),
        'by_municipality': {},
        'aggregated': {
            'finetune': [],
            'fromscratch': [],
        }
    }

    # Per-municipality summaries
    for mun, mun_results in results.items():
        mun_summary = {}

        for exp_type in ['finetune', 'fromscratch']:
            if exp_type in mun_results and mun_results[exp_type]:
                steps = mun_results[exp_type]

                # Compute improvement from step 0 to step 5
                step_0 = next((s for s in steps if s['step'] == 0), None)
                step_5 = next((s for s in steps if s['step'] == 5), None)

                if step_0 and step_5:
                    f1_0 = step_0['entity_f1']
                    f1_5 = step_5['entity_f1']

                    if f1_0 and f1_5:
                        absolute_gain = f1_5 - f1_0
                        relative_gain = (f1_5 - f1_0) / f1_0 * 100 if f1_0 > 0 else 0

                        mun_summary[exp_type] = {
                            'step_0_f1': f1_0,
                            'step_5_f1': f1_5,
                            'absolute_gain': absolute_gain,
                            'relative_gain_pct': relative_gain,
                            'all_steps': [
                                {'step': s['step'], 'f1': s['entity_f1']}
                                for s in steps if s['entity_f1'] is not None
                            ]
                        }

        summary['by_municipality'][mun] = mun_summary

    # Aggregated statistics across municipalities
    for exp_type in ['finetune', 'fromscratch']:
        step_f1s = defaultdict(list)

        for mun, mun_results in results.items():
            if exp_type in mun_results:
                for step_result in mun_results[exp_type]:
                    step = step_result['step']
                    f1 = step_result['entity_f1']
                    if f1 is not None:
                        step_f1s[step].append(f1)

        # Compute mean F1 per step
        aggregated_steps = []
        for step in sorted(step_f1s.keys()):
            f1_scores = step_f1s[step]
            if f1_scores:
                aggregated_steps.append({
                    'step': step,
                    'mean_f1': sum(f1_scores) / len(f1_scores),
                    'min_f1': min(f1_scores),
                    'max_f1': max(f1_scores),
                    'std_f1': (
                        (sum((f1 - sum(f1_scores)/len(f1_scores))**2 for f1 in f1_scores) / len(f1_scores))**0.5
                        if len(f1_scores) > 1 else 0
                    ),
                    'num_municipalities': len(f1_scores)
                })

        summary['aggregated'][exp_type] = aggregated_steps

    return summary


def compare_experiment_types(results: Dict[str, Any], scenario: str) -> Dict[str, Any]:
    """Compare fine-tuning vs from-scratch."""
    print(f"\nComparing fine-tuning vs from-scratch for Scenario {scenario}...")

    comparison = {
        'scenario': scenario,
        'timestamp': datetime.now().isoformat(),
        'by_step': [],
        'overall': {}
    }

    # Per-step comparison
    for step in range(6):
        finetune_f1s = []
        fromscratch_f1s = []

        for mun, mun_results in results.items():
            # Fine-tuning
            if 'finetune' in mun_results:
                step_result = next(
                    (s for s in mun_results['finetune'] if s['step'] == step),
                    None
                )
                if step_result and step_result['entity_f1'] is not None:
                    finetune_f1s.append(step_result['entity_f1'])

            # From scratch
            if 'fromscratch' in mun_results:
                step_result = next(
                    (s for s in mun_results['fromscratch'] if s['step'] == step),
                    None
                )
                if step_result and step_result['entity_f1'] is not None:
                    fromscratch_f1s.append(step_result['entity_f1'])

        if finetune_f1s and fromscratch_f1s:
            finetune_mean = sum(finetune_f1s) / len(finetune_f1s)
            fromscratch_mean = sum(fromscratch_f1s) / len(fromscratch_f1s)

            comparison['by_step'].append({
                'step': step,
                'finetune_mean_f1': finetune_mean,
                'fromscratch_mean_f1': fromscratch_mean,
                'difference': finetune_mean - fromscratch_mean,
                'finetune_better': finetune_mean > fromscratch_mean,
                'num_municipalities': min(len(finetune_f1s), len(fromscratch_f1s))
            })

    # Overall comparison
    if comparison['by_step']:
        finetune_wins = sum(1 for s in comparison['by_step'] if s['finetune_better'])
        fromscratch_wins = len(comparison['by_step']) - finetune_wins

        comparison['overall'] = {
            'finetune_wins': finetune_wins,
            'fromscratch_wins': fromscratch_wins,
            'avg_difference': sum(s['difference'] for s in comparison['by_step']) / len(comparison['by_step']),
            'max_difference': max(s['difference'] for s in comparison['by_step']),
            'min_difference': min(s['difference'] for s in comparison['by_step']),
        }

    return comparison


def generate_latex_table(results: Dict[str, Any], scenario: str, output_path: Path):
    """Generate LaTeX table for paper."""
    print(f"\nGenerating LaTeX table for Scenario {scenario}...")

    # Create a table showing F1 scores for all municipalities and steps
    latex = []
    latex.append("% Incremental Learning Results - Scenario " + scenario)
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Incremental Learning Results for Scenario " + scenario + "}")
    latex.append("\\begin{tabular}{l" + "c" * 6 + "}")
    latex.append("\\toprule")
    latex.append("Municipality & Step 0 & Step 1 & Step 2 & Step 3 & Step 4 & Step 5 \\\\")
    latex.append("\\midrule")

    for mun in sorted(results.keys()):
        for exp_type in ['finetune', 'fromscratch']:
            if exp_type in results[mun]:
                row = [f"{mun} ({exp_type[:2]})"]

                for step in range(6):
                    step_result = next(
                        (s for s in results[mun][exp_type] if s['step'] == step),
                        None
                    )

                    if step_result and step_result['entity_f1'] is not None:
                        f1 = step_result['entity_f1']
                        row.append(f"{f1:.3f}")
                    else:
                        row.append("-")

                latex.append(" & ".join(row) + " \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    latex_content = "\n".join(latex)

    with open(output_path, 'w') as f:
        f.write(latex_content)

    print(f"  ✅ LaTeX table saved to: {output_path}")


def print_summary(summary: Dict[str, Any], comparison: Dict[str, Any]):
    """Print human-readable summary to console."""
    print("\n" + "="*80)
    print(f"SUMMARY - Scenario {summary['scenario']}")
    print("="*80)

    print("\nAggregated Results (Mean F1 across municipalities):")
    print(f"{'Step':>6} {'Fine-tune':>12} {'From-scratch':>14} {'Difference':>12}")
    print("-" * 50)

    finetune_steps = {s['step']: s for s in summary['aggregated']['finetune']}
    fromscratch_steps = {s['step']: s for s in summary['aggregated']['fromscratch']}

    for step in range(6):
        ft = finetune_steps.get(step, {}).get('mean_f1', 0)
        fs = fromscratch_steps.get(step, {}).get('mean_f1', 0)
        diff = ft - fs if (ft and fs) else 0

        print(f"{step:>6} {ft:>12.4f} {fs:>14.4f} {diff:>12.4f}")

    print("\nPer-Municipality Improvements (Step 0 → Step 5):")
    print(f"{'Municipality':>12} {'Type':>12} {'Step 0':>10} {'Step 5':>10} {'Abs Gain':>10} {'Rel Gain %':>12}")
    print("-" * 70)

    for mun in sorted(summary['by_municipality'].keys()):
        mun_data = summary['by_municipality'][mun]

        for exp_type in ['finetune', 'fromscratch']:
            if exp_type in mun_data:
                data = mun_data[exp_type]
                print(
                    f"{mun:>12} {exp_type:>12} "
                    f"{data['step_0_f1']:>10.4f} {data['step_5_f1']:>10.4f} "
                    f"{data['absolute_gain']:>10.4f} {data['relative_gain_pct']:>12.2f}"
                )

    if comparison['overall']:
        print("\nOverall Comparison:")
        print(f"  Fine-tuning better: {comparison['overall']['finetune_wins']}/6 steps")
        print(f"  From-scratch better: {comparison['overall']['fromscratch_wins']}/6 steps")
        print(f"  Average difference: {comparison['overall']['avg_difference']:.4f}")


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate results from incremental learning experiments v2'
    )
    parser.add_argument(
        '--scenario', choices=['A', 'B', 'C', 'all'], default='A',
        help='Scenario to aggregate'
    )
    parser.add_argument(
        '--output-dir', default='incremental_experiments_v2/summaries',
        help='Output directory for summaries'
    )

    args = parser.parse_args()

    print("="*80)
    print("INCREMENTAL LEARNING EXPERIMENTS V2 - RESULTS AGGREGATION")
    print("="*80)

    scenarios = ['A', 'B', 'C'] if args.scenario == 'all' else [args.scenario]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for scenario in scenarios:
        # Collect results
        results = collect_results(scenario)

        if not results:
            print(f"\n⚠️  No results found for Scenario {scenario}")
            continue

        # Compute summary statistics
        summary = compute_summary_statistics(results, scenario)

        # Compare experiment types
        comparison = compare_experiment_types(results, scenario)

        # Save results
        save_json(summary, output_dir / f"scenario_{scenario}_summary.json")
        save_json(comparison, output_dir / f"scenario_{scenario}_comparison.json")
        save_json(results, output_dir / f"scenario_{scenario}_raw_results.json")

        # Generate LaTeX table
        generate_latex_table(
            results, scenario,
            output_dir / f"scenario_{scenario}_table.tex"
        )

        # Print summary
        print_summary(summary, comparison)

    print("\n" + "="*80)
    print("✅ AGGREGATION COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  - scenario_*_summary.json: Summary statistics")
    print(f"  - scenario_*_comparison.json: Fine-tuning vs from-scratch comparison")
    print(f"  - scenario_*_raw_results.json: Raw results")
    print(f"  - scenario_*_table.tex: LaTeX table for paper")


if __name__ == '__main__':
    main()
