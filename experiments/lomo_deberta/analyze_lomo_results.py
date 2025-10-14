#!/usr/bin/env python3
"""
Analyze LOMO (Leave-One-Municipality-Out) evaluation results.

This script parses all evaluation_results.json files from the LOMO experiment directory
and generates comprehensive analysis including:
- Cross-municipality performance matrix (F1 scores per model per municipality)
- Per-municipality statistics
- Model performance comparison across municipalities
- Event-level metrics comparison
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import numpy as np


def extract_model_and_municipality(directory_name: str) -> tuple[str, str]:
    """
    Extract model name and test municipality from LOMO directory name.

    Examples:
        lomo_deberta_crf_test_alandroal -> (deberta_crf, alandroal)
        lomo_bert_crf_test_porto -> (bert_crf, porto)
    
    Args:
        directory_name: LOMO experiment directory name
    
    Returns:
        Tuple of (model_name, municipality_name)
    """
    # Expected format: lomo_{model}_test_{municipality}
    parts = directory_name.split('_')
    
    # Find 'test' keyword to split model and municipality
    if 'test' in parts:
        test_idx = parts.index('test')
        # Model is everything between 'lomo' and 'test'
        model = '_'.join(parts[1:test_idx])
        # Municipality is everything after 'test'
        municipality = '_'.join(parts[test_idx+1:])
        return model, municipality
    
    # Fallback: return original name if pattern doesn't match
    return directory_name, "unknown"


def load_lomo_results(exp_dir: Path) -> List[Dict[str, Any]]:
    """
    Load all evaluation_results.json files from LOMO experiment directory.

    Args:
        exp_dir: Path to LOMO experiment directory

    Returns:
        List of dictionaries containing model, municipality, and results
    """
    results = []

    # Find all evaluation_results.json files
    for eval_file in sorted(exp_dir.glob("*/evaluation_results.json")):
        exp_dir_name = eval_file.parent.name
        model, municipality = extract_model_and_municipality(exp_dir_name)

        try:
            with open(eval_file, 'r') as f:
                data = json.load(f)
                results.append({
                    'model': model,
                    'municipality': municipality,
                    'exp_dir': exp_dir_name,
                    'data': data
                })
                print(f"✓ Loaded: {model:20s} on {municipality:15s}")
        except Exception as e:
            print(f"✗ Error loading {eval_file}: {e}")

    return results


def create_municipality_f1_matrix(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a matrix of F1 scores with municipalities as rows and models as columns.

    Args:
        results: List of result dictionaries

    Returns:
        DataFrame with municipalities as index and models as columns
    """
    rows = []
    
    for result in results:
        municipality = result['municipality']
        model = result['model']
        entity_metrics = result['data'].get('entity_level_metrics', {})
        f1 = entity_metrics.get('entity_f1', 0.0)
        
        rows.append({
            'municipality': municipality,
            'model': model,
            'f1': f1
        })
    
    df = pd.DataFrame(rows)
    
    # Pivot to create municipality x model matrix
    pivot_df = df.pivot(index='municipality', columns='model', values='f1')
    
    # Sort municipalities alphabetically
    pivot_df = pivot_df.sort_index()
    
    # Add mean row and column
    pivot_df.loc['Mean'] = pivot_df.mean()
    pivot_df['Mean'] = pivot_df.mean(axis=1)
    
    return pivot_df


def create_comprehensive_metrics_matrix(results: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    """
    Create matrices for all key metrics (F1, Precision, Recall, Event Detection, etc.)

    Args:
        results: List of result dictionaries

    Returns:
        Dictionary of DataFrames, one per metric
    """
    metrics = {
        'f1': [],
        'precision': [],
        'recall': [],
        'accuracy': [],
        'event_detection': [],
        'complete_event_accuracy': [],
        'participant_vote_f1': [],
        'subject_f1': []
    }
    
    for result in results:
        municipality = result['municipality']
        model = result['model']
        entity_metrics = result['data'].get('entity_level_metrics', {})
        event_metrics = result['data'].get('event_level_metrics', {})
        
        base_row = {
            'municipality': municipality,
            'model': model
        }
        
        metrics['f1'].append({**base_row, 'value': entity_metrics.get('entity_f1', 0.0)})
        metrics['precision'].append({**base_row, 'value': entity_metrics.get('entity_precision', 0.0)})
        metrics['recall'].append({**base_row, 'value': entity_metrics.get('entity_recall', 0.0)})
        metrics['accuracy'].append({**base_row, 'value': entity_metrics.get('entity_accuracy', 0.0)})
        metrics['event_detection'].append({**base_row, 'value': event_metrics.get('event_detection_accuracy', 0.0)})
        metrics['complete_event_accuracy'].append({**base_row, 'value': event_metrics.get('complete_event_accuracy', 0.0)})
        metrics['participant_vote_f1'].append({**base_row, 'value': event_metrics.get('participant_vote_f1', 0.0)})
        metrics['subject_f1'].append({**base_row, 'value': event_metrics.get('subject_f1', 0.0)})
    
    # Convert to pivot tables
    pivot_tables = {}
    for metric_name, metric_data in metrics.items():
        df = pd.DataFrame(metric_data)
        pivot_df = df.pivot(index='municipality', columns='model', values='value')
        pivot_df = pivot_df.sort_index()
        
        # Add statistics
        pivot_df.loc['Mean'] = pivot_df.mean()
        pivot_df.loc['Std'] = pivot_df.std()
        pivot_df.loc['Min'] = pivot_df.min()
        pivot_df.loc['Max'] = pivot_df.max()
        pivot_df['Mean'] = pivot_df.mean(axis=1)
        
        pivot_tables[metric_name] = pivot_df
    
    return pivot_tables


def create_per_entity_type_analysis(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create detailed per-entity-type performance across models and municipalities.

    Args:
        results: List of result dictionaries

    Returns:
        DataFrame with entity type performance
    """
    rows = []
    
    for result in results:
        municipality = result['municipality']
        model = result['model']
        entity_metrics = result['data'].get('entity_level_metrics', {})
        per_type = entity_metrics.get('per_type_metrics', {})
        
        for entity_type, metrics in per_type.items():
            rows.append({
                'model': model,
                'municipality': municipality,
                'entity_type': entity_type,
                'precision': metrics.get('precision', 0.0),
                'recall': metrics.get('recall', 0.0),
                'f1': metrics.get('f1_score', 0.0),
                'support': metrics.get('support', 0)
            })
    
    df = pd.DataFrame(rows)
    return df


def create_model_summary(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create summary statistics for each model across all municipalities.

    Args:
        results: List of result dictionaries

    Returns:
        DataFrame with model summary statistics
    """
    model_stats = {}
    
    for result in results:
        model = result['model']
        entity_metrics = result['data'].get('entity_level_metrics', {})
        event_metrics = result['data'].get('event_level_metrics', {})
        
        if model not in model_stats:
            model_stats[model] = {
                'f1_scores': [],
                'precision_scores': [],
                'recall_scores': [],
                'event_detection_scores': [],
                'complete_event_scores': []
            }
        
        model_stats[model]['f1_scores'].append(entity_metrics.get('entity_f1', 0.0))
        model_stats[model]['precision_scores'].append(entity_metrics.get('entity_precision', 0.0))
        model_stats[model]['recall_scores'].append(entity_metrics.get('entity_recall', 0.0))
        model_stats[model]['event_detection_scores'].append(event_metrics.get('event_detection_accuracy', 0.0))
        model_stats[model]['complete_event_scores'].append(event_metrics.get('complete_event_accuracy', 0.0))
    
    summary_rows = []
    for model, stats in model_stats.items():
        summary_rows.append({
            'model': model,
            'mean_f1': np.mean(stats['f1_scores']),
            'std_f1': np.std(stats['f1_scores']),
            'min_f1': np.min(stats['f1_scores']),
            'max_f1': np.max(stats['f1_scores']),
            'mean_precision': np.mean(stats['precision_scores']),
            'mean_recall': np.mean(stats['recall_scores']),
            'mean_event_detection': np.mean(stats['event_detection_scores']),
            'mean_complete_event': np.mean(stats['complete_event_scores']),
            'num_municipalities': len(stats['f1_scores'])
        })
    
    df = pd.DataFrame(summary_rows)
    df = df.sort_values('mean_f1', ascending=False)
    return df


def create_municipality_summary(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create summary statistics for each municipality across all models.

    Args:
        results: List of result dictionaries

    Returns:
        DataFrame with municipality summary statistics
    """
    municipality_stats = {}
    
    for result in results:
        municipality = result['municipality']
        entity_metrics = result['data'].get('entity_level_metrics', {})
        
        if municipality not in municipality_stats:
            municipality_stats[municipality] = {
                'f1_scores': [],
                'num_examples': result['data']['metadata'].get('num_examples', 0)
            }
        
        municipality_stats[municipality]['f1_scores'].append(entity_metrics.get('entity_f1', 0.0))
    
    summary_rows = []
    for municipality, stats in municipality_stats.items():
        summary_rows.append({
            'municipality': municipality,
            'mean_f1': np.mean(stats['f1_scores']),
            'std_f1': np.std(stats['f1_scores']),
            'min_f1': np.min(stats['f1_scores']),
            'max_f1': np.max(stats['f1_scores']),
            'num_examples': stats['num_examples'],
            'num_models': len(stats['f1_scores'])
        })
    
    df = pd.DataFrame(summary_rows)
    df = df.sort_values('mean_f1', ascending=False)
    return df


def main():
    """Main analysis pipeline for LOMO experiments."""
    # Set up paths
    script_dir = Path(__file__).parent
    exp_dir = script_dir
    output_dir = script_dir / 'lomo_analysis'
    output_dir.mkdir(exist_ok=True)

    print("="*80)
    print("LOMO (Leave-One-Municipality-Out) - Results Analysis")
    print("="*80)
    print(f"\nExperiment directory: {exp_dir}")
    print(f"Output directory: {output_dir}\n")

    # Load all LOMO evaluation results
    print("Loading LOMO evaluation results...")
    results = load_lomo_results(exp_dir)

    if not results:
        print("No LOMO evaluation results found!")
        return

    print(f"\n✓ Loaded {len(results)} LOMO experiment results\n")

    # Get unique models and municipalities
    models = sorted(set(r['model'] for r in results))
    municipalities = sorted(set(r['municipality'] for r in results))
    
    print(f"Models: {', '.join(models)}")
    print(f"Municipalities: {', '.join(municipalities)}")
    print(f"Total combinations: {len(models)} models × {len(municipalities)} municipalities = {len(results)} experiments\n")

    # Create F1 matrix (main table requested)
    print("="*80)
    print("Generating Municipality × Model F1 Score Matrix...")
    print("="*80)
    f1_matrix = create_municipality_f1_matrix(results)
    f1_matrix_csv = output_dir / 'municipality_model_f1_matrix.csv'
    f1_matrix.to_csv(f1_matrix_csv, float_format='%.4f')
    print(f"✓ Saved: {f1_matrix_csv}\n")
    
    # Display the matrix
    print(f1_matrix.to_string(float_format=lambda x: f"{x:.4f}"))
    print()

    # Create comprehensive metrics matrices
    print("\nGenerating comprehensive metrics matrices...")
    metrics_matrices = create_comprehensive_metrics_matrix(results)
    
    for metric_name, matrix in metrics_matrices.items():
        csv_file = output_dir / f'municipality_model_{metric_name}_matrix.csv'
        matrix.to_csv(csv_file, float_format='%.4f')
        print(f"✓ Saved: {csv_file}")

    # Model summary
    print("\n" + "="*80)
    print("Model Summary (aggregated across municipalities)")
    print("="*80)
    model_summary = create_model_summary(results)
    model_summary_csv = output_dir / 'model_summary.csv'
    model_summary.to_csv(model_summary_csv, index=False, float_format='%.4f')
    print(f"✓ Saved: {model_summary_csv}\n")
    print(model_summary.to_string(index=False))

    # Municipality summary
    print("\n" + "="*80)
    print("Municipality Summary (aggregated across models)")
    print("="*80)
    municipality_summary = create_municipality_summary(results)
    municipality_summary_csv = output_dir / 'municipality_summary.csv'
    municipality_summary.to_csv(municipality_summary_csv, index=False, float_format='%.4f')
    print(f"✓ Saved: {municipality_summary_csv}\n")
    print(municipality_summary.to_string(index=False))

    # Per-entity-type analysis
    print("\nGenerating per-entity-type analysis...")
    per_entity_df = create_per_entity_type_analysis(results)
    per_entity_csv = output_dir / 'per_entity_type_metrics.csv'
    per_entity_df.to_csv(per_entity_csv, index=False, float_format='%.4f')
    print(f"✓ Saved: {per_entity_csv}")

    # Best and worst performers
    print("\n" + "="*80)
    print("Best & Worst Performing Combinations")
    print("="*80)
    
    # Create a simple list for sorting
    performance_list = []
    for result in results:
        f1 = result['data'].get('entity_level_metrics', {}).get('entity_f1', 0.0)
        performance_list.append({
            'model': result['model'],
            'municipality': result['municipality'],
            'f1': f1
        })
    
    perf_df = pd.DataFrame(performance_list).sort_values('f1', ascending=False)
    
    print("\nTop 5 Best Performances:")
    print(perf_df.head(5).to_string(index=False))
    
    print("\nTop 5 Worst Performances:")
    print(perf_df.tail(5).to_string(index=False))

    # Summary statistics
    print("\n" + "="*80)
    print("Overall Statistics")
    print("="*80)
    all_f1_scores = [r['data'].get('entity_level_metrics', {}).get('entity_f1', 0.0) for r in results]
    print(f"Overall Mean F1: {np.mean(all_f1_scores):.4f}")
    print(f"Overall Std F1:  {np.std(all_f1_scores):.4f}")
    print(f"Overall Min F1:  {np.min(all_f1_scores):.4f}")
    print(f"Overall Max F1:  {np.max(all_f1_scores):.4f}")

    print("\n" + "="*80)
    print(f"✓ Analysis complete! Results saved to: {output_dir}")
    print("="*80)

    # List generated files
    print("\nGenerated files:")
    for csv_file in sorted(output_dir.glob('*.csv')):
        print(f"  - {csv_file.name}")


if __name__ == '__main__':
    main()
