#!/usr/bin/env python3
"""
Analyze evaluation results from all model experiments.

This script parses all evaluation_results.json files from the models directory
and generates comprehensive CSV reports and LaTeX tables including:
- Overall entity-level metrics (macro F1, precision, recall)
- Per entity type metrics
- Event level evaluation metrics
- Formatted tables ready for paper inclusion
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import sys


def extract_baseline_name(directory_name: str) -> str:
    """
    Extract a clean baseline name from directory name.

    Examples:
        bert_crf_20251010_070842 -> bert_crf
        crf_full -> crf
    """
    # Remove timestamp suffix if present
    parts = directory_name.split('_')

    # Check if last part looks like a timestamp (8 digits)
    if len(parts) > 1 and parts[-1].isdigit() and len(parts[-1]) >= 6:
        # Remove last two parts (date_time)
        baseline = '_'.join(parts[:-2])
    else:
        # Remove '_full' suffix for legacy experiments
        baseline = directory_name.replace('_full', '')

    return baseline


def load_evaluation_results(exp_dir: Path) -> List[Dict[str, Any]]:
    """
    Load all evaluation_results.json files from experiment directory.

    Args:
        exp_dir: Path to experiment directory

    Returns:
        List of dictionaries containing baseline name and results
    """
    results = []

    # Find all evaluation_results.json files
    for eval_file in exp_dir.glob("*/evaluation_results.json"):
        baseline_dir = eval_file.parent.name
        baseline_name = extract_baseline_name(baseline_dir)

        try:
            with open(eval_file, 'r') as f:
                data = json.load(f)
                results.append({
                    'baseline': baseline_name,
                    'baseline_dir': baseline_dir,
                    'data': data
                })
                print(f"✓ Loaded: {baseline_name} ({baseline_dir})")
        except Exception as e:
            print(f"✗ Error loading {eval_file}: {e}")

    return results


def create_overall_metrics_df(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create DataFrame with overall entity-level metrics for all baselines.

    Args:
        results: List of result dictionaries

    Returns:
        DataFrame with columns: baseline, precision, recall, f1, accuracy
    """
    rows = []

    for result in results:
        baseline = result['baseline']
        entity_metrics = result['data'].get('entity_level_metrics', {})

        rows.append({
            'baseline': baseline,
            'precision': entity_metrics.get('entity_precision', 0.0),
            'recall': entity_metrics.get('entity_recall', 0.0),
            'f1': entity_metrics.get('entity_f1', 0.0),
            'accuracy': entity_metrics.get('entity_accuracy', 0.0),
            'num_examples': result['data']['metadata'].get('num_examples', 0)
        })

    df = pd.DataFrame(rows)
    # Sort by F1 score descending
    df = df.sort_values('f1', ascending=False)

    return df


def create_per_entity_metrics_df(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create DataFrame with per-entity-type metrics for all baselines.

    Args:
        results: List of result dictionaries

    Returns:
        DataFrame with columns: baseline, entity_type, precision, recall, f1, support
    """
    rows = []

    for result in results:
        baseline = result['baseline']
        entity_metrics = result['data'].get('entity_level_metrics', {})
        per_type = entity_metrics.get('per_type_metrics', {})

        for entity_type, metrics in per_type.items():
            rows.append({
                'baseline': baseline,
                'entity_type': entity_type,
                'precision': metrics.get('precision', 0.0),
                'recall': metrics.get('recall', 0.0),
                'f1': metrics.get('f1_score', 0.0),
                'support': metrics.get('support', 0)
            })

    df = pd.DataFrame(rows)
    # Sort by entity type, then baseline
    df = df.sort_values(['entity_type', 'f1'], ascending=[True, False])

    return df


def create_event_level_metrics_df(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create DataFrame with event-level metrics for all baselines.

    Args:
        results: List of result dictionaries

    Returns:
        DataFrame with event-level evaluation metrics
    """
    rows = []

    for result in results:
        baseline = result['baseline']
        event_metrics = result['data'].get('event_level_metrics', {})

        if not event_metrics:
            continue

        rows.append({
            'baseline': baseline,
            'event_detection_accuracy': event_metrics.get('event_detection_accuracy', 0.0),
            'complete_event_accuracy': event_metrics.get('complete_event_accuracy', 0.0),
            'participant_vote_precision': event_metrics.get('participant_vote_precision', 0.0),
            'participant_vote_recall': event_metrics.get('participant_vote_recall', 0.0),
            'participant_vote_f1': event_metrics.get('participant_vote_f1', 0.0),
            'subject_precision': event_metrics.get('subject_precision', 0.0),
            'subject_recall': event_metrics.get('subject_recall', 0.0),
            'subject_f1': event_metrics.get('subject_f1', 0.0),
            'true_positives': event_metrics.get('true_positives', 0),
            'false_positives': event_metrics.get('false_positives', 0),
            'true_negatives': event_metrics.get('true_negatives', 0),
            'false_negatives': event_metrics.get('false_negatives', 0),
            'complete_matches': event_metrics.get('complete_matches', 0),
            'total_events': event_metrics.get('total_events', 0)
        })

    df = pd.DataFrame(rows)
    # Sort by complete event accuracy descending
    df = df.sort_values('complete_event_accuracy', ascending=False)

    return df


def create_pivot_tables(per_entity_df: pd.DataFrame, output_dir: Path):
    """
    Create pivot tables for easy comparison across baselines.

    Args:
        per_entity_df: DataFrame with per-entity metrics
        output_dir: Directory to save pivot tables
    """
    # Pivot for F1 scores
    f1_pivot = per_entity_df.pivot(
        index='entity_type',
        columns='baseline',
        values='f1'
    )
    f1_pivot.to_csv(output_dir / 'per_entity_f1_pivot.csv')

    # Pivot for Precision
    precision_pivot = per_entity_df.pivot(
        index='entity_type',
        columns='baseline',
        values='precision'
    )
    precision_pivot.to_csv(output_dir / 'per_entity_precision_pivot.csv')

    # Pivot for Recall
    recall_pivot = per_entity_df.pivot(
        index='entity_type',
        columns='baseline',
        values='recall'
    )
    recall_pivot.to_csv(output_dir / 'per_entity_recall_pivot.csv')

    print(f"✓ Created pivot tables in {output_dir}")


def create_latex_table_overall(overall_df: pd.DataFrame, output_dir: Path):
    """
    Create LaTeX table for overall span-level metrics.
    
    Args:
        overall_df: DataFrame with overall metrics
        output_dir: Directory to save LaTeX table
    """
    # Find best values for each metric
    best_precision = overall_df['precision'].max()
    best_recall = overall_df['recall'].max()
    best_f1 = overall_df['f1'].max()
    
    latex_lines = []
    latex_lines.append("\\begin{table}[ht]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Span-level Performance of All Baselines}")
    latex_lines.append("\\label{tab:span_results}")
    latex_lines.append("\\begin{tabular}{lccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("\\textbf{Model} & \\textbf{Precision (\\%)} & \\textbf{Recall (\\%)} & \\textbf{F1 (\\%)} \\\\")
    latex_lines.append("\\midrule")
    
    for _, row in overall_df.iterrows():
        baseline = row['baseline'].replace('_', '\\_')
        
        # Format with bold if best value
        prec_str = f"{row['precision']*100:.1f}"
        if abs(row['precision'] - best_precision) < 0.0001:
            prec_str = f"\\textbf{{{prec_str}}}"
            
        rec_str = f"{row['recall']*100:.1f}"
        if abs(row['recall'] - best_recall) < 0.0001:
            rec_str = f"\\textbf{{{rec_str}}}"
            
        f1_str = f"{row['f1']*100:.1f}"
        if abs(row['f1'] - best_f1) < 0.0001:
            f1_str = f"\\textbf{{{f1_str}}}"
        
        latex_lines.append(f"{baseline} & {prec_str} & {rec_str} & {f1_str} \\\\")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    latex_file = output_dir / 'table_span_results.tex'
    latex_file.write_text('\n'.join(latex_lines))
    print(f"✓ Created LaTeX table: {latex_file}")


def create_latex_table_events(event_df: pd.DataFrame, output_dir: Path):
    """
    Create LaTeX table for event-level metrics.
    
    Args:
        event_df: DataFrame with event metrics
        output_dir: Directory to save LaTeX table
    """
    if event_df.empty:
        return
    
    # Find best values for each metric
    best_event_det = event_df['event_detection_accuracy'].max()
    best_complete = event_df['complete_event_accuracy'].max()
    best_subject = event_df['subject_f1'].max()
    best_part_vote = event_df['participant_vote_f1'].max()
    
    latex_lines = []
    latex_lines.append("\\begin{table}[ht]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Event-level Performance of All Baselines}")
    latex_lines.append("\\label{tab:event_results}")
    latex_lines.append("\\begin{tabular}{lcccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("\\textbf{Model} & \\textbf{Event Det. (\\%)} & \\textbf{Complete (\\%)} & \\textbf{Subject F1 (\\%)} & \\textbf{Part.-Vote F1 (\\%)} \\\\")
    latex_lines.append("\\midrule")
    
    for _, row in event_df.iterrows():
        baseline = row['baseline'].replace('_', '\\_')
        
        # Format with bold if best value
        event_det_str = f"{row['event_detection_accuracy']*100:.1f}"
        if abs(row['event_detection_accuracy'] - best_event_det) < 0.0001:
            event_det_str = f"\\textbf{{{event_det_str}}}"
            
        complete_str = f"{row['complete_event_accuracy']*100:.1f}"
        if abs(row['complete_event_accuracy'] - best_complete) < 0.0001:
            complete_str = f"\\textbf{{{complete_str}}}"
            
        subject_str = f"{row['subject_f1']*100:.1f}"
        if abs(row['subject_f1'] - best_subject) < 0.0001:
            subject_str = f"\\textbf{{{subject_str}}}"
            
        part_vote_str = f"{row['participant_vote_f1']*100:.1f}"
        if abs(row['participant_vote_f1'] - best_part_vote) < 0.0001:
            part_vote_str = f"\\textbf{{{part_vote_str}}}"
        
        latex_lines.append(f"{baseline} & {event_det_str} & {complete_str} & {subject_str} & {part_vote_str} \\\\")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    latex_file = output_dir / 'table_event_results.tex'
    latex_file.write_text('\n'.join(latex_lines))
    print(f"✓ Created LaTeX table: {latex_file}")


def run_error_analysis_for_deberta(exp_dir: Path, output_dir: Path):
    """
    Run event-level error analysis for DeBERTa-CRF model.
    
    Args:
        exp_dir: Experiment directory
        output_dir: Output directory for error analysis
    """
    deberta_dir = exp_dir / 'deberta_crf'
    if not deberta_dir.exists():
        print("\n⚠ DeBERTa-CRF model directory not found, skipping error analysis")
        return
    
    predictions_file = deberta_dir / 'predictions.jsonl'
    if not predictions_file.exists():
        print("\n⚠ DeBERTa-CRF predictions not found, skipping error analysis")
        return
    
    # Find the test data
    test_data_paths = [
        exp_dir.parent.parent / 'data' / 'votie_events' / 'test.json',
        exp_dir.parent / 'data' / 'votie_events' / 'test.json',
    ]
    
    test_file = None
    for path in test_data_paths:
        if path.exists():
            test_file = path
            break
    
    if test_file is None:
        print("\n⚠ Test data file not found, skipping error analysis")
        return
    
    print("\n" + "="*70)
    print("Running Event-Level Error Analysis for DeBERTa-CRF")
    print("="*70)
    
    # Import and run error analysis
    error_analysis_script = exp_dir.parent / 'experiments' / 'pt_stratified_split' / 'analyze_event_errors.py'
    if not error_analysis_script.exists():
        # Try alternative location
        error_analysis_script = exp_dir.parent / 'analyze_event_errors.py'
    
    if not error_analysis_script.exists():
        print("⚠ Error analysis script not found, skipping...")
        return
    
    # Run error analysis as subprocess
    import subprocess
    error_output = output_dir / 'deberta_crf_error_analysis.json'
    
    try:
        result = subprocess.run([
            sys.executable,
            str(error_analysis_script),
            '--predictions', str(predictions_file),
            '--gold', str(test_file),
            '--output', str(error_output)
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(result.stdout)
            print(f"\n✓ Error analysis saved to: {error_output}")
        else:
            print(f"⚠ Error analysis failed: {result.stderr}")
    except Exception as e:
        print(f"⚠ Could not run error analysis: {e}")


def main():
    """Main analysis pipeline."""
    # Set up paths
    script_dir = Path(__file__).parent
    exp_dir = script_dir
    output_dir = script_dir / 'analysis_results'
    output_dir.mkdir(exist_ok=True)

    print("="*70)
    print("VotIE - Model Results Analysis")
    print("="*70)
    print(f"\nModels directory: {exp_dir}")
    print(f"Output directory: {output_dir}\n")

    # Load all evaluation results
    print("Loading evaluation results...")
    results = load_evaluation_results(exp_dir)

    if not results:
        print("No evaluation results found!")
        return

    print(f"\n✓ Loaded {len(results)} model results\n")

    # =================================================================
    # SECTION 1: SPAN-LEVEL RESULTS
    # =================================================================
    print("\n" + "="*70)
    print("SECTION 1: SPAN-LEVEL PERFORMANCE")
    print("="*70)
    
    # Create overall metrics CSV
    print("\nGenerating overall span-level metrics...")
    overall_df = create_overall_metrics_df(results)
    overall_csv = output_dir / 'span_level_overall.csv'
    overall_df.to_csv(overall_csv, index=False, float_format='%.4f')
    print(f"✓ Saved CSV: {overall_csv}")
    
    # Create LaTeX table for span-level results
    create_latex_table_overall(overall_df, output_dir)

    # Display overall metrics
    print("\nSpan-Level Metrics (sorted by F1):")
    print(overall_df.to_string(index=False))

    # Create per-entity metrics CSV
    print("\n\nGenerating per-entity type metrics...")
    per_entity_df = create_per_entity_metrics_df(results)
    per_entity_csv = output_dir / 'span_level_per_entity.csv'
    per_entity_df.to_csv(per_entity_csv, index=False, float_format='%.4f')
    print(f"✓ Saved CSV: {per_entity_csv}")

    # Create pivot tables
    print("\nGenerating pivot tables...")
    create_pivot_tables(per_entity_df, output_dir)

    # Best performing models for span-level
    print("\nBest Span-Level F1 per entity type:")
    for entity_type in sorted(per_entity_df['entity_type'].unique()):
        entity_data = per_entity_df[per_entity_df['entity_type'] == entity_type]
        best = entity_data.loc[entity_data['f1'].idxmax()]
        print(f"  {entity_type:25s}: {best['baseline']:15s} (F1: {best['f1']:.4f})")

    # =================================================================
    # SECTION 2: EVENT-LEVEL RESULTS
    # =================================================================
    print("\n\n" + "="*70)
    print("SECTION 2: EVENT-LEVEL PERFORMANCE")
    print("="*70)
    
    # Create event-level metrics CSV
    print("\nGenerating event-level metrics...")
    event_df = create_event_level_metrics_df(results)
    
    if not event_df.empty:
        event_csv = output_dir / 'event_level_metrics.csv'
        event_df.to_csv(event_csv, index=False, float_format='%.4f')
        print(f"✓ Saved CSV: {event_csv}")
        
        # Create LaTeX table for event-level results
        create_latex_table_events(event_df, output_dir)

        # Display event-level metrics
        print("\nEvent-Level Metrics (sorted by complete event accuracy):")
        display_cols = [
            'baseline',
            'event_detection_accuracy',
            'complete_event_accuracy',
            'subject_f1',
            'participant_vote_f1'
        ]
        print(event_df[display_cols].to_string(index=False))
    else:
        print("\n⚠ No event-level metrics available")

    # =================================================================
    # SECTION 3: ERROR ANALYSIS FOR DEBERTA-CRF
    # =================================================================
    print("\n\n" + "="*70)
    print("SECTION 3: ERROR ANALYSIS - DeBERTa-CRF")
    print("="*70)
    
    run_error_analysis_for_deberta(exp_dir, output_dir)

    # =================================================================
    # SUMMARY
    # =================================================================
    print("\n\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total models analyzed: {len(results)}")
    print(f"Total entity types: {per_entity_df['entity_type'].nunique()}")
    print(f"Entity types: {', '.join(sorted(per_entity_df['entity_type'].unique()))}")

    # Best performing models overall
    best_overall = overall_df.iloc[0]
    print(f"\nBest Overall Span-Level F1: {best_overall['baseline']} ({best_overall['f1']:.4f})")

    if not event_df.empty:
        best_event = event_df.iloc[0]
        print(f"Best Complete Event Accuracy: {best_event['baseline']} ({best_event['complete_event_accuracy']:.4f})")

    print("\n" + "="*70)
    print(f"✓ Analysis complete! Results saved to: {output_dir}")
    print("="*70)

    # List generated files
    print("\nGenerated files:")
    for file in sorted(output_dir.iterdir()):
        if file.is_file():
            print(f"  - {file.name}")


if __name__ == '__main__':
    main()
