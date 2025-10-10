#!/usr/bin/env python3
"""
Format LOMO cross-municipality evaluation results into publication-ready tables.
Outputs per-entity metrics for each municipality tested in LaTeX and CSV formats.
"""

import json
import os
from pathlib import Path
from typing import Dict, List
import pandas as pd


def load_lomo_results(results_dir: str) -> Dict[str, dict]:
    """Load all LOMO results from JSON files."""
    results = {}
    results_path = Path(results_dir)

    for json_file in results_path.glob("training_results_*.json"):
        municipality = json_file.stem.replace("training_results_", "")
        with open(json_file, 'r') as f:
            results[municipality] = json.load(f)

    return results


def extract_entity_metrics(results: Dict[str, dict]) -> pd.DataFrame:
    """Extract per-entity metrics from all municipalities."""
    rows = []

    for municipality, data in sorted(results.items()):
        test_results = data.get('test_results', {})
        per_type = test_results.get('per_type_metrics', {})

        for entity_type, metrics in sorted(per_type.items()):
            rows.append({
                'Municipality': municipality.title(),
                'Entity': entity_type,
                'Precision': metrics['precision'] * 100,
                'Recall': metrics['recall'] * 100,
                'F1': metrics['f1_score'] * 100,
                'Support': metrics['support']
            })

    return pd.DataFrame(rows)


def extract_overall_metrics(results: Dict[str, dict]) -> pd.DataFrame:
    """Extract overall metrics for each municipality."""
    rows = []

    for municipality, data in sorted(results.items()):
        test_results = data.get('test_results', {})

        rows.append({
            'Municipality': municipality.title(),
            'Precision': test_results.get('entity_precision', 0) * 100,
            'Recall': test_results.get('entity_recall', 0) * 100,
            'F1': test_results.get('entity_f1_strict', 0) * 100,
            'Accuracy': test_results.get('entity_accuracy', 0) * 100,
            'Support': test_results.get('num_examples', 0)
        })

    return pd.DataFrame(rows)


def format_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    """Format DataFrame as a LaTeX table."""
    # Create a copy to avoid modifying original
    df_copy = df.copy()

    # Format numeric columns
    for col in ['Precision', 'Recall', 'F1', 'Accuracy']:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].apply(lambda x: f"{x:.2f}")

    # Convert to LaTeX
    latex = df_copy.to_latex(
        index=False,
        escape=False,
        column_format='l' + 'r' * (len(df_copy.columns) - 1),
        float_format="%.2f"
    )

    # Add caption and label
    latex = latex.replace(
        '\\begin{tabular}',
        f'\\begin{{table}}[htbp]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\begin{{tabular}}'
    )
    latex = latex.replace('\\end{tabular}', '\\end{tabular}\n\\end{table}')

    return latex


def create_entity_pivot_table(df: pd.DataFrame, metric: str = 'F1') -> pd.DataFrame:
    """Create a pivot table with entities as rows and municipalities as columns."""
    pivot = df.pivot(index='Entity', columns='Municipality', values=metric)

    # Add average column
    pivot['Average'] = pivot.mean(axis=1)

    # Sort by average descending
    pivot = pivot.sort_values('Average', ascending=False)

    return pivot


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Format LOMO results into publication-ready tables'
    )
    parser.add_argument(
        'results_dir',
        help='Path to LOMO results directory'
    )
    parser.add_argument(
        '--output-dir',
        default='.',
        help='Output directory for tables (default: current directory)'
    )
    parser.add_argument(
        '--format',
        choices=['latex', 'csv', 'both'],
        default='both',
        help='Output format (default: both)'
    )

    args = parser.parse_args()

    # Load results
    print(f"Loading LOMO results from {args.results_dir}...")
    results = load_lomo_results(args.results_dir)
    print(f"Found {len(results)} municipalities: {', '.join(sorted(results.keys()))}\n")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Extract metrics
    entity_df = extract_entity_metrics(results)
    overall_df = extract_overall_metrics(results)

    # Create pivot tables for each metric
    f1_pivot = create_entity_pivot_table(entity_df, 'F1')
    precision_pivot = create_entity_pivot_table(entity_df, 'Precision')
    recall_pivot = create_entity_pivot_table(entity_df, 'Recall')

    # Display overall results
    print("=" * 80)
    print("OVERALL RESULTS BY MUNICIPALITY")
    print("=" * 80)
    print(overall_df.to_string(index=False))
    print()

    # Display F1 scores by entity
    print("=" * 80)
    print("F1 SCORES BY ENTITY AND MUNICIPALITY (%)")
    print("=" * 80)
    print(f1_pivot.to_string(float_format=lambda x: f"{x:.2f}"))
    print()

    # Display precision scores
    print("=" * 80)
    print("PRECISION BY ENTITY AND MUNICIPALITY (%)")
    print("=" * 80)
    print(precision_pivot.to_string(float_format=lambda x: f"{x:.2f}"))
    print()

    # Display recall scores
    print("=" * 80)
    print("RECALL BY ENTITY AND MUNICIPALITY (%)")
    print("=" * 80)
    print(recall_pivot.to_string(float_format=lambda x: f"{x:.2f}"))
    print()

    # Save outputs
    if args.format in ['csv', 'both']:
        print("Saving CSV files...")
        overall_df.to_csv(output_dir / 'lomo_overall_results.csv', index=False)
        entity_df.to_csv(output_dir / 'lomo_entity_results.csv', index=False)
        f1_pivot.to_csv(output_dir / 'lomo_f1_pivot.csv')
        precision_pivot.to_csv(output_dir / 'lomo_precision_pivot.csv')
        recall_pivot.to_csv(output_dir / 'lomo_recall_pivot.csv')
        print(f"  ✓ CSV files saved to {output_dir}/")

    if args.format in ['latex', 'both']:
        print("Generating LaTeX tables...")

        # Overall results table
        latex_overall = format_latex_table(
            overall_df,
            caption="LOMO Cross-Municipality Evaluation: Overall Results",
            label="tab:lomo_overall"
        )

        # F1 pivot table
        f1_for_latex = f1_pivot.reset_index()
        f1_for_latex.columns.name = None
        latex_f1 = format_latex_table(
            f1_for_latex,
            caption="LOMO Cross-Municipality Evaluation: F1 Scores by Entity Type (\\%)",
            label="tab:lomo_f1"
        )

        # Precision pivot table
        precision_for_latex = precision_pivot.reset_index()
        precision_for_latex.columns.name = None
        latex_precision = format_latex_table(
            precision_for_latex,
            caption="LOMO Cross-Municipality Evaluation: Precision by Entity Type (\\%)",
            label="tab:lomo_precision"
        )

        # Recall pivot table
        recall_for_latex = recall_pivot.reset_index()
        recall_for_latex.columns.name = None
        latex_recall = format_latex_table(
            recall_for_latex,
            caption="LOMO Cross-Municipality Evaluation: Recall by Entity Type (\\%)",
            label="tab:lomo_recall"
        )

        # Save LaTeX files
        with open(output_dir / 'lomo_overall_table.tex', 'w') as f:
            f.write(latex_overall)
        with open(output_dir / 'lomo_f1_table.tex', 'w') as f:
            f.write(latex_f1)
        with open(output_dir / 'lomo_precision_table.tex', 'w') as f:
            f.write(latex_precision)
        with open(output_dir / 'lomo_recall_table.tex', 'w') as f:
            f.write(latex_recall)

        print(f"  ✓ LaTeX files saved to {output_dir}/")
        print()
        print("LaTeX tables:")
        print("  - lomo_overall_table.tex")
        print("  - lomo_f1_table.tex")
        print("  - lomo_precision_table.tex")
        print("  - lomo_recall_table.tex")

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total municipalities evaluated: {len(results)}")
    print(f"Average F1 score: {overall_df['F1'].mean():.2f}%")
    print(f"Average Precision: {overall_df['Precision'].mean():.2f}%")
    print(f"Average Recall: {overall_df['Recall'].mean():.2f}%")
    print(f"Total test examples: {overall_df['Support'].sum()}")
    print()
    print(f"Best performing municipality: {overall_df.loc[overall_df['F1'].idxmax(), 'Municipality']} "
          f"(F1: {overall_df['F1'].max():.2f}%)")
    print(f"Entity types evaluated: {entity_df['Entity'].nunique()}")
    print()


if __name__ == '__main__':
    main()
