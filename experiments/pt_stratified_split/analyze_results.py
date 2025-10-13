#!/usr/bin/env python3
"""
Analyze evaluation results from Portuguese stratified split experiments.

This script parses all evaluation_results.json files from the experiment directory
and generates comprehensive CSV reports including:
- Overall entity-level metrics (macro F1, precision, recall)
- Per entity type metrics
- Event level evaluation metrics
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any


def extract_baseline_name(directory_name: str) -> str:
    """
    Extract a clean baseline name from directory name.

    Examples:
        bert_crf_20251010_070842 -> bert_crf
        crf_full -> crf
        bilstm_fastext -> bilstm_fastext
        lomo_deberta_crf_test_campomaior -> lomo_deberta_crf_test_campomaior
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


def main():
    """Main analysis pipeline."""
    # Set up paths
    script_dir = Path(__file__).parent
    exp_dir = script_dir
    output_dir = script_dir / 'analysis_results'
    output_dir.mkdir(exist_ok=True)

    print("="*70)
    print("Portuguese Stratified Split - Results Analysis")
    print("="*70)
    print(f"\nExperiment directory: {exp_dir}")
    print(f"Output directory: {output_dir}\n")

    # Load all evaluation results
    print("Loading evaluation results...")
    results = load_evaluation_results(exp_dir)

    if not results:
        print("No evaluation results found!")
        return

    print(f"\n✓ Loaded {len(results)} baseline results\n")

    # Create overall metrics CSV
    print("Generating overall metrics CSV...")
    overall_df = create_overall_metrics_df(results)
    overall_csv = output_dir / 'overall_metrics.csv'
    overall_df.to_csv(overall_csv, index=False, float_format='%.4f')
    print(f"✓ Saved: {overall_csv}")

    # Display overall metrics
    print("\n" + "="*70)
    print("Overall Entity-Level Metrics (sorted by F1)")
    print("="*70)
    print(overall_df.to_string(index=False))

    # Create per-entity metrics CSV
    print("\n\nGenerating per-entity metrics CSV...")
    per_entity_df = create_per_entity_metrics_df(results)
    per_entity_csv = output_dir / 'per_entity_metrics.csv'
    per_entity_df.to_csv(per_entity_csv, index=False, float_format='%.4f')
    print(f"✓ Saved: {per_entity_csv}")

    # Create pivot tables
    print("\nGenerating pivot tables...")
    create_pivot_tables(per_entity_df, output_dir)

    # Create event-level metrics CSV
    print("\nGenerating event-level metrics CSV...")
    event_df = create_event_level_metrics_df(results)
    event_csv = output_dir / 'event_level_metrics.csv'
    event_df.to_csv(event_csv, index=False, float_format='%.4f')
    print(f"✓ Saved: {event_csv}")

    # Display event-level metrics
    print("\n" + "="*70)
    print("Event-Level Metrics (sorted by complete event accuracy)")
    print("="*70)
    # Select key columns for display
    display_cols = [
        'baseline',
        'event_detection_accuracy',
        'complete_event_accuracy',
        'participant_vote_f1',
        'subject_f1'
    ]
    print(event_df[display_cols].to_string(index=False))

    # Summary statistics
    print("\n" + "="*70)
    print("Summary Statistics")
    print("="*70)
    print(f"Total baselines analyzed: {len(results)}")
    print(f"Total entity types: {per_entity_df['entity_type'].nunique()}")
    print(f"Entity types: {', '.join(sorted(per_entity_df['entity_type'].unique()))}")

    # Best performing models
    print("\n" + "="*70)
    print("Best Performing Models")
    print("="*70)
    best_overall = overall_df.iloc[0]
    print(f"Best Overall F1: {best_overall['baseline']} ({best_overall['f1']:.4f})")

    if not event_df.empty:
        best_event = event_df.iloc[0]
        print(f"Best Complete Event Accuracy: {best_event['baseline']} ({best_event['complete_event_accuracy']:.4f})")

    # Per entity type best performers
    print("\nBest F1 per entity type:")
    for entity_type in sorted(per_entity_df['entity_type'].unique()):
        entity_data = per_entity_df[per_entity_df['entity_type'] == entity_type]
        best = entity_data.loc[entity_data['f1'].idxmax()]
        print(f"  {entity_type:25s}: {best['baseline']:15s} (F1: {best['f1']:.4f})")

    print("\n" + "="*70)
    print(f"✓ Analysis complete! Results saved to: {output_dir}")
    print("="*70)

    # List generated files
    print("\nGenerated files:")
    for csv_file in sorted(output_dir.glob('*.csv')):
        print(f"  - {csv_file.name}")


if __name__ == '__main__':
    main()
