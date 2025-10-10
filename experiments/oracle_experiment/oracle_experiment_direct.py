#!/usr/bin/env python
"""
Oracle Experiment for Event Constructor Validation - Direct Spans Version.

This script tests the EventConstructor by directly using the span annotations
from data/votie_spans/test.json to construct events, without going through BIO labels.

The experiment:
1. Loads a random sample of 200 segments from data/votie_spans/test.json
2. Directly constructs events from the span annotations (text + label)
3. Compares with ground truth from data/votie_events/test.jsonl
4. Evaluates using src/evaluation/event_metrics.py

Expected result: Near-perfect scores since we're using the exact span annotations
that were used to create the gold events.
"""

import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.evaluation.event_metrics import EventLevelEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_json_file(filepath: Path) -> List[Dict[str, Any]]:
    """Load data from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_jsonl(filepath: Path) -> List[Dict[str, Any]]:
    """Load data from JSONL file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[str, Any]], filepath: Path):
    """Save data to JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def construct_event_from_spans(segment_id: str, spans: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Construct structured event directly from span annotations.
    
    Args:
        segment_id: Unique identifier for the segment
        spans: List of span dictionaries with 'text' and 'label' fields
        
    Returns:
        Event dictionary with structured components
    """
    # Check if this segment has a voting event
    has_voting = any(span['label'] == 'VOTING' for span in spans)
    
    if not has_voting:
        return {
            'id': segment_id,
            'has_voting_event': False,
            'event': {
                'subjects': [],
                'participants': [],
                'counting': [],
                'voting_expressions': []
            }
        }
    
    # Extract components
    subjects = []
    participants = []
    counting = []
    voting_expressions = []
    
    for span in spans:
        label = span['label']
        text = span['text']
        
        if label == 'SUBJECT':
            subjects.append(text)
        
        elif label == 'VOTING':
            voting_expressions.append(text)
        
        elif label.startswith('VOTER-'):
            # Extract position from label (e.g., VOTER-FAVOR -> FAVOR)
            position = label.split('-', 1)[1]
            participants.append({
                'text': text,
                'position': position.capitalize()
            })
        
        elif label.startswith('COUNTING-'):
            # Extract type from label (e.g., COUNTING-UNANIMITY -> UNANIMITY)
            counting_type = label.split('-', 1)[1]
            counting.append({
                'text': text,
                'type': counting_type.capitalize()
            })
    
    return {
        'id': segment_id,
        'has_voting_event': True,
        'event': {
            'subjects': subjects,
            'participants': participants,
            'counting': counting,
            'voting_expressions': voting_expressions
        }
    }


def main():
    """Run oracle experiment with direct span-to-event construction."""
    logger.info("=" * 80)
    logger.info("ORACLE EXPERIMENT: Direct Span-to-Event Construction")
    logger.info("=" * 80)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Define paths relative to project root
    project_root = Path(__file__).parent.parent.parent
    spans_test_path = project_root / "data/votie_spans/test.json"
    events_test_path = project_root / "data/votie_events/test.jsonl"
    
    # Output to the oracle_experiment directory
    output_dir = Path(__file__).parent
    output_path = output_dir / "oracle_experiment_direct_predictions.jsonl"
    
    # Check if files exist
    if not spans_test_path.exists():
        logger.error(f"Spans test file not found: {spans_test_path}")
        return
    if not events_test_path.exists():
        logger.error(f"Events test file not found: {events_test_path}")
        return
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info(f"Loading spans test data from {spans_test_path}")
    spans_test = load_json_file(spans_test_path)
    logger.info(f"Loaded {len(spans_test)} segments from spans test set")
    
    logger.info(f"Loading event test data from {events_test_path}")
    events_test = load_jsonl(events_test_path)
    logger.info(f"Loaded {len(events_test)} segments from events test set")
    
    # Sample 200 segments randomly
    sample_size = min(200, len(spans_test))
    logger.info(f"Sampling {sample_size} segments randomly (seed=42)")
    sampled_indices = random.sample(range(len(spans_test)), sample_size)
    sampled_spans = [spans_test[i] for i in sampled_indices]
    
    # Create ID to event mapping for gold events
    gold_events_dict = {event['id']: event for event in events_test}
    
    # Filter gold events to match sampled segments
    sampled_gold_events = []
    for span_sample in sampled_spans:
        segment_id = span_sample['id']
        if segment_id in gold_events_dict:
            sampled_gold_events.append(gold_events_dict[segment_id])
        else:
            logger.warning(f"No gold event found for segment {segment_id}")
    
    logger.info(f"Found {len(sampled_gold_events)} matching gold events")
    
    # Construct events directly from spans
    logger.info("Constructing events directly from span annotations")
    predicted_events = []
    
    for span_sample in sampled_spans:
        segment_id = span_sample['id']
        spans = span_sample.get('spans', [])
        
        # Construct event directly from spans
        event = construct_event_from_spans(segment_id, spans)
        predicted_events.append(event)
    
    logger.info(f"Constructed {len(predicted_events)} events")
    
    # Save predictions
    logger.info(f"Saving predictions to {output_path}")
    save_jsonl(predicted_events, output_path)
    
    # Evaluate
    logger.info("=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    
    evaluator = EventLevelEvaluator()
    
    # Compute all metrics
    all_metrics = evaluator.compute_metrics(predicted_events, sampled_gold_events)
    
    # 1. Event Detection
    logger.info("\n1. EVENT DETECTION (Binary Classification)")
    logger.info("-" * 80)
    logger.info(f"Accuracy: {all_metrics['event_detection_accuracy']:.4f}")
    logger.info(f"True Positives: {all_metrics['true_positives']}")
    logger.info(f"False Positives: {all_metrics['false_positives']}")
    logger.info(f"True Negatives: {all_metrics['true_negatives']}")
    logger.info(f"False Negatives: {all_metrics['false_negatives']}")
    
    # Calculate precision, recall, F1 for event detection
    tp = all_metrics['true_positives']
    fp = all_metrics['false_positives']
    fn = all_metrics['false_negatives']
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")
    
    # 2. Participant-Vote Extraction
    logger.info("\n2. PARTICIPANT-VOTE PAIR EXTRACTION")
    logger.info("-" * 80)
    logger.info(f"Precision: {all_metrics['participant_vote_precision']:.4f}")
    logger.info(f"Recall: {all_metrics['participant_vote_recall']:.4f}")
    logger.info(f"F1-Score: {all_metrics['participant_vote_f1']:.4f}")
    
    # 3. Subject Extraction
    logger.info("\n3. SUBJECT EXTRACTION")
    logger.info("-" * 80)
    logger.info(f"Precision: {all_metrics['subject_precision']:.4f}")
    logger.info(f"Recall: {all_metrics['subject_recall']:.4f}")
    logger.info(f"F1-Score: {all_metrics['subject_f1']:.4f}")
    
    # 4. Complete Event Extraction
    logger.info("\n4. COMPLETE EVENT EXTRACTION (Strict Match)")
    logger.info("-" * 80)
    logger.info(f"Accuracy: {all_metrics['complete_event_accuracy']:.4f}")
    logger.info(f"Correct Events: {all_metrics['complete_matches']}/{all_metrics['total_events']}")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Sample Size: {sample_size} segments")
    logger.info(f"Input: Direct span annotations from {spans_test_path}")
    logger.info(f"Gold Events: {events_test_path}")
    logger.info(f"Predictions Saved: {output_path}")
    logger.info("")
    logger.info("This experiment directly constructs events from span annotations,")
    logger.info("bypassing BIO label conversion entirely. Results should be near-perfect")
    logger.info("since we're using the exact spans from the gold annotations.")
    logger.info("=" * 80)
    
    # Write results summary to file
    results_summary = {
        "experiment": "oracle_direct_span_to_event",
        "sample_size": sample_size,
        "random_seed": 42,
        "input_source": "direct_span_annotations",
        "method": "spans_to_event_without_bio_conversion",
        "results": all_metrics
    }
    
    results_file = output_dir / "oracle_experiment_direct_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nDetailed results saved to: {results_file}")
    logger.info("Experiment completed successfully!")
    
    # Show a sample comparison
    logger.info("\n" + "=" * 80)
    logger.info("SAMPLE COMPARISON (First voting event)")
    logger.info("=" * 80)
    
    for pred in predicted_events[:10]:
        if pred['has_voting_event']:
            gold = gold_events_dict[pred['id']]
            logger.info(f"\nSegment ID: {pred['id']}")
            logger.info("\nPredicted Event:")
            logger.info(json.dumps(pred['event'], indent=2, ensure_ascii=False))
            logger.info("\nGold Event:")
            logger.info(json.dumps(gold['event'], indent=2, ensure_ascii=False))
            break


if __name__ == "__main__":
    main()
