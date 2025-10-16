#!/usr/bin/env python3
"""
Evaluate Gemini 2.0 Event Extraction Predictions

This script evaluates the predictions from the Gemini 2.0 model (LangExtract)
against the gold annotations using the event-level metrics defined in the paper.

Metrics computed:
1. Event Detection F1: Binary classification of segments with/without voting events
2. Subject Extraction F1: Span-level exact matching
3. Counting Extraction F1: Span-level exact matching with type
4. Participant-Vote F1: Set-based over (participant, position) tuples
5. Complete Event F1: Strict - all components must match

Usage:
    python evaluate_gemini_events.py
    python evaluate_gemini_events.py --predictions path/to/predictions.jsonl --gold path/to/gold.json
"""

import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.evaluation.event_metrics import EventLevelEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Mapping from municipality names to codes
MUNICIPALITY_MAPPING = {
    'Alandroal': 'M01',
    'Campomaior': 'M02',
    'Covilha': 'M03',
    'Fundao': 'M04',
    'Guimaraes': 'M05',
    'Porto': 'M06'
}


def normalize_segment_id(segment_id: str) -> str:
    """
    Normalize segment ID to match gold format.
    
    Convert: Covilha_cm_002_2023-02-03_seg045
    To:      M03_cm_002_2023-02-03_seg045
    
    Args:
        segment_id: Original segment ID
        
    Returns:
        Normalized segment ID with municipality code
    """
    for muni_name, muni_code in MUNICIPALITY_MAPPING.items():
        if segment_id.startswith(muni_name + '_'):
            return segment_id.replace(muni_name + '_', muni_code + '_', 1)
    
    # If already in correct format or unknown municipality, return as-is
    return segment_id


def load_predictions(predictions_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load predictions from JSONL file.
    
    Args:
        predictions_path: Path to predictions JSONL file
        
    Returns:
        Dictionary mapping segment_id -> prediction
    """
    predictions = {}
    
    with open(predictions_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                pred = json.loads(line)
                segment_id = pred.get('id')
                
                if not segment_id:
                    logger.warning(f"Line {line_num}: Missing 'id' field")
                    continue
                
                # Normalize the segment ID to match gold format
                normalized_id = normalize_segment_id(segment_id)
                pred['id'] = normalized_id  # Update the ID in the prediction
                predictions[normalized_id] = pred
                
            except json.JSONDecodeError as e:
                logger.error(f"Line {line_num}: JSON decode error: {e}")
                continue
    
    logger.info(f"Loaded {len(predictions)} predictions from {predictions_path}")
    return predictions


def load_gold_events(gold_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load gold annotations from JSON file.
    
    Args:
        gold_path: Path to gold JSON file
        
    Returns:
        Dictionary mapping segment_id -> gold annotation
    """
    with open(gold_path, 'r', encoding='utf-8') as f:
        gold_data = json.load(f)
    
    gold_events = {}
    for item in gold_data:
        segment_id = item.get('id')
        if not segment_id:
            logger.warning(f"Gold item missing 'id': {item.keys()}")
            continue
        
        # Convert to the format expected by evaluator
        gold_event = {
            'id': segment_id,
            'has_voting_event': item.get('has_voting_event', False),
            'event': item.get('event')
        }
        
        gold_events[segment_id] = gold_event
    
    logger.info(f"Loaded {len(gold_events)} gold annotations from {gold_path}")
    return gold_events


def align_predictions_and_gold(
    predictions: Dict[str, Dict[str, Any]],
    gold_events: Dict[str, Dict[str, Any]]
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Align predictions with gold annotations by segment ID.
    
    Args:
        predictions: Dictionary of predictions
        gold_events: Dictionary of gold annotations
        
    Returns:
        Tuple of (aligned_predictions, aligned_gold) lists
    """
    aligned_predictions = []
    aligned_gold = []
    
    # Get all segment IDs from gold (gold is the reference)
    gold_ids = sorted(gold_events.keys())
    
    missing_predictions = []
    
    for segment_id in gold_ids:
        gold_event = gold_events[segment_id]
        pred_event = predictions.get(segment_id)
        
        if pred_event is None:
            logger.warning(f"Missing prediction for segment: {segment_id}")
            missing_predictions.append(segment_id)
            # Use default: no voting event
            pred_event = {
                'id': segment_id,
                'has_voting_event': False,
                'event': None
            }
        
        aligned_gold.append(gold_event)
        aligned_predictions.append(pred_event)
    
    if missing_predictions:
        logger.warning(f"Missing predictions for {len(missing_predictions)} segments")
    
    logger.info(f"Aligned {len(aligned_predictions)} prediction-gold pairs")
    return aligned_predictions, aligned_gold


def print_metrics_table(metrics: Dict[str, float]):
    """Print metrics in a formatted table."""
    print("\n" + "="*70)
    print("EVENT-LEVEL EVALUATION METRICS (Gemini 2.0 Flash)")
    print("="*70)
    print(f"{'Metric':<35} {'F1 Score':>12}")
    print("-"*70)
    
    metric_names = {
        "event_detection_f1": "Event Detection F1",
        "subject_f1": "Subject Extraction F1",
        "counting_f1": "Counting Extraction F1",
        "participant_vote_f1": "Participant-Vote F1",
        "complete_event_f1": "Complete Event F1 (Strict)"
    }
    
    for key in ["event_detection_f1", "subject_f1", "counting_f1", 
                "participant_vote_f1", "complete_event_f1"]:
        if key in metrics:
            name = metric_names.get(key, key)
            value = metrics[key]
            print(f"{name:<35} {value:>11.4f}")
    
    print("="*70)
    
    # Calculate average F1 (macro-average across all 5 metrics)
    avg_f1 = sum(metrics[k] for k in metric_names.keys() if k in metrics) / 5
    print(f"{'Macro-Average F1':<35} {avg_f1:>11.4f}")
    print("="*70 + "\n")


def save_metrics(metrics: Dict[str, float], output_path: Path):
    """Save metrics to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Metrics saved to {output_path}")


def print_error_analysis(
    predictions: List[Dict[str, Any]],
    gold_events: List[Dict[str, Any]],
    max_examples: int = 5
):
    """Print some error examples for analysis."""
    print("\n" + "="*70)
    print("ERROR ANALYSIS (Sample)")
    print("="*70)
    
    errors = []
    
    for pred, gold in zip(predictions, gold_events):
        pred_has = pred.get('has_voting_event', False)
        gold_has = gold.get('has_voting_event', False)
        
        if pred_has != gold_has:
            errors.append({
                'id': gold.get('id'),
                'type': 'detection',
                'pred_has': pred_has,
                'gold_has': gold_has
            })
    
    if errors:
        print(f"\nEvent Detection Errors: {len(errors)}")
        for i, error in enumerate(errors[:max_examples], 1):
            print(f"\n  {i}. Segment: {error['id']}")
            print(f"     Predicted: {'HAS EVENT' if error['pred_has'] else 'NO EVENT'}")
            print(f"     Gold:      {'HAS EVENT' if error['gold_has'] else 'NO EVENT'}")
    
    print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Gemini 2.0 event extraction predictions'
    )
    parser.add_argument(
        '--predictions',
        type=Path,
        default=Path('langextract_votie/experiments/langextract_events_20251015_024557/predictions_events.jsonl'),
        help='Path to predictions JSONL file'
    )
    parser.add_argument(
        '--gold',
        type=Path,
        default=Path('data/votie_events/test.json'),
        help='Path to gold JSON file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('langextract_votie/experiments/langextract_events_20251015_024557/evaluation_metrics.json'),
        help='Path to save evaluation metrics'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if files exist
    if not args.predictions.exists():
        logger.error(f"Predictions file not found: {args.predictions}")
        sys.exit(1)
    
    if not args.gold.exists():
        logger.error(f"Gold file not found: {args.gold}")
        sys.exit(1)
    
    # Load data
    logger.info("Loading predictions and gold annotations...")
    predictions = load_predictions(args.predictions)
    gold_events = load_gold_events(args.gold)
    
    # Align predictions with gold
    logger.info("Aligning predictions with gold annotations...")
    aligned_predictions, aligned_gold = align_predictions_and_gold(
        predictions, gold_events
    )
    
    # Compute metrics
    logger.info("Computing event-level metrics...")
    evaluator = EventLevelEvaluator()
    metrics = evaluator.compute_metrics(aligned_predictions, aligned_gold)
    
    # Print results
    print_metrics_table(metrics)
    
    # Save metrics
    save_metrics(metrics, args.output)
    
    # Error analysis
    if args.verbose:
        print_error_analysis(aligned_predictions, aligned_gold)
    
    logger.info("Evaluation complete!")


if __name__ == '__main__':
    main()
