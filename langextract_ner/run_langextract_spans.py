#!/usr/bin/env python
"""
LangExtract Span Labeling - Event-Based Evaluation

Run span extraction on Portuguese test dataset, convert to events, and evaluate.

This script:
1. Extracts spans using LangExtract (VOTER-FAVOR, SUBJECT, etc.)
2. Converts spans to structured events using event_constructor
3. Evaluates using event-level metrics

Usage:
    python run_langextract_spans.py
    python run_langextract_spans.py --test  # Test mode (10 documents)
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import time
import sys

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from langextract_ner.shared.models_spans import GeminiModel
from langextract_ner.shared.prompts_spans import get_voting_extraction_config
from langextract_ner.config import get_config
from src.utils.event_constructor import EventConstructor
from src.evaluation.event_metrics import EventLevelEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Silence noisy loggers
logging.getLogger('google_genai').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('google.auth').setLevel(logging.WARNING)


def load_events_dataset(events_file: Path) -> List[Dict[str, Any]]:
    """
    Load dataset from votie_events JSON or JSONL file

    Args:
        events_file: Path to events .json or .jsonl file (train/dev/test)

    Returns:
        List of event dictionaries
    """
    logger.info(f"Loading events dataset from: {events_file}")

    # Handle both .json and .jsonl formats
    if events_file.suffix == '.json':
        # Load as JSON array
        with open(events_file, 'r', encoding='utf-8') as f:
            events = json.load(f)
    else:
        # Load as JSONL (line-by-line)
        events = []
        with open(events_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    event = json.loads(line.strip())
                    events.append(event)

    logger.info(f"Loaded {len(events)} events")
    return events


def extract_spans_with_langextract(documents: List[Dict[str, Any]],
                                    model: GeminiModel,
                                    config: Dict[str, Any],
                                    max_documents: int = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract entity spans from documents using LangExtract

    Args:
        documents: List of documents with text
        model: GeminiModel instance
        config: Extraction configuration with prompt and examples
        max_documents: Maximum number of documents to process

    Returns:
        Dictionary mapping document IDs to extracted entity spans
    """
    logger.info(f"Starting span extraction with {model.model_id}")

    predictions = {}
    total_docs = len(documents) if max_documents is None else min(max_documents, len(documents))

    start_time = time.time()

    for i, doc in enumerate(documents[:total_docs]):
        doc_id = doc['id']
        text = doc['text']

        logger.info(f"Processing document {i+1}/{total_docs}: {doc_id}")

        # Extract entities using LangExtract
        result = model.extract(
            text=text,
            prompt=config['prompt'],
            examples=config['examples'],
            document_id=doc_id
        )

        if result.success:
            # Convert extractions to entity format
            entities = []
            for extraction in result.extractions:
                entity = {
                    'start': extraction.char_interval.start_pos,
                    'end': extraction.char_interval.end_pos,
                    'text': extraction.text,
                    'label': extraction.extraction_class,
                    'type': extraction.extraction_class
                }
                entities.append(entity)

            predictions[doc_id] = entities
            logger.info(f"✓ Extracted {len(entities)} entities from {doc_id}")
        else:
            logger.error(f"✗ Failed to extract from {doc_id}: {result.error}")
            predictions[doc_id] = []

        # Rate limiting - small delay between requests
        if i < total_docs - 1:
            time.sleep(1)

    elapsed_time = time.time() - start_time
    docs_per_sec = total_docs / elapsed_time if elapsed_time > 0 else 0

    logger.info(f"Extraction completed: {total_docs} documents in {elapsed_time:.2f}s ({docs_per_sec:.2f} docs/sec)")

    return predictions


def convert_spans_to_bio(text: str, spans: List[Dict[str, Any]]) -> tuple:
    """
    Convert character-offset spans to BIO tags

    Args:
        text: Document text
        spans: List of entity spans with start, end, label

    Returns:
        Tuple of (tokens, labels)
    """
    import spacy

    # Use spacy tokenizer
    nlp = spacy.blank("pt")
    doc = nlp(text)

    tokens = [token.text for token in doc]
    labels = ['O'] * len(tokens)

    # Sort spans by start position
    sorted_spans = sorted(spans, key=lambda x: x['start'])

    # Map character positions to token indices
    for span in sorted_spans:
        span_start = span['start']
        span_end = span['end']
        label = span['label']

        # Find tokens that overlap with this span
        first_token_idx = None
        last_token_idx = None

        for i, token in enumerate(doc):
            if token.idx < span_end and token.idx + len(token.text) > span_start:
                if first_token_idx is None:
                    first_token_idx = i
                last_token_idx = i

        # Apply BIO tags
        if first_token_idx is not None:
            labels[first_token_idx] = f'B-{label}'
            for i in range(first_token_idx + 1, last_token_idx + 1):
                labels[i] = f'I-{label}'

    return tokens, labels


def convert_predictions_to_events(predictions: Dict[str, List[Dict[str, Any]]],
                                   test_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert span predictions to structured events

    Args:
        predictions: Dictionary of doc_id -> entity spans
        test_events: Original test events (for text)

    Returns:
        List of structured event dictionaries
    """
    logger.info("Converting span predictions to structured events...")

    constructor = EventConstructor()
    predicted_events = []

    for event_data in test_events:
        doc_id = event_data['id']
        text = event_data['text']

        # Get predicted spans for this document
        spans = predictions.get(doc_id, [])

        # Convert spans to BIO tags
        tokens, labels = convert_spans_to_bio(text, spans)

        # Construct event from BIO tags
        predicted_event = constructor.construct_event(
            tokens=tokens,
            labels=labels,
            example_id=doc_id
        )

        predicted_events.append(predicted_event)

    return predicted_events


def save_results(predictions: Dict[str, List[Dict[str, Any]]],
                predicted_events: List[Dict[str, Any]],
                evaluation_results: Dict[str, Any],
                output_dir: Path,
                model_id: str,
                extraction_time: float,
                num_docs: int):
    """
    Save predictions and evaluation results

    Args:
        predictions: Span predictions
        predicted_events: Structured event predictions
        evaluation_results: Evaluation metrics
        output_dir: Output directory
        model_id: Model identifier
        extraction_time: Total extraction time
        num_docs: Number of documents processed
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save span predictions
    spans_path = output_dir / "predictions_spans.jsonl"
    with open(spans_path, 'w', encoding='utf-8') as f:
        for doc_id, spans in predictions.items():
            f.write(json.dumps({'id': doc_id, 'spans': spans}, ensure_ascii=False) + '\n')
    logger.info(f"Saved span predictions to: {spans_path}")

    # Save event predictions
    events_path = output_dir / "predictions_events.jsonl"
    with open(events_path, 'w', encoding='utf-8') as f:
        for event in predicted_events:
            f.write(json.dumps(event, ensure_ascii=False) + '\n')
    logger.info(f"Saved event predictions to: {events_path}")

    # Save evaluation results
    eval_path = output_dir / "evaluation_results.json"
    results_dict = {
        "metadata": {
            "model": model_id,
            "timestamp": datetime.now().isoformat(),
            "total_documents": num_docs,
            "evaluation_method": "span_to_event_extraction",
            "extraction_time_seconds": round(extraction_time, 2),
            "avg_time_per_document": round(extraction_time / num_docs, 2)
        },
        "event_level_metrics": evaluation_results
    }

    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved evaluation results to: {eval_path}")

    # Save markdown summary
    md_path = output_dir / "evaluation_summary.md"
    summary = f"""# LangExtract Span-to-Event Extraction Evaluation Report

## Model Information
- **Model**: {model_id}
- **Language**: Portuguese
- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Evaluation Method**: Span extraction → Event construction

## Extraction Performance
- **Total Extraction Time**: {extraction_time:.2f}s
- **Avg Time per Document**: {extraction_time / num_docs:.2f}s
- **Documents Processed**: {num_docs}

## Event-Level Performance

### Event Detection
| Metric | Value |
|--------|-------|
| **Accuracy** | {evaluation_results['event_detection_accuracy']:.4f} |
| **True Positives** | {evaluation_results['true_positives']} |
| **False Positives** | {evaluation_results['false_positives']} |
| **True Negatives** | {evaluation_results['true_negatives']} |
| **False Negatives** | {evaluation_results['false_negatives']} |

### Subject Extraction
| Metric | Value |
|--------|-------|
| **Accuracy** | {evaluation_results['subject_accuracy']:.4f} |
| **Correct** | {evaluation_results['subject_correct']} |
| **Total** | {evaluation_results['subject_total']} |

### Participant-Vote Extraction
| Metric | Value |
|--------|-------|
| **Precision** | {evaluation_results['participant_vote_precision']:.4f} |
| **Recall** | {evaluation_results['participant_vote_recall']:.4f} |
| **F1-Score** | {evaluation_results['participant_vote_f1']:.4f} |

### Complete Event Extraction
| Metric | Value |
|--------|-------|
| **Accuracy** | {evaluation_results['complete_event_accuracy']:.4f} |
| **Complete Matches** | {evaluation_results['complete_matches']} |
| **Total Events** | {evaluation_results['total_events']} |

## Interpretation

- **Event Detection**: How well the model identifies which segments contain voting events
- **Subject Extraction**: Accuracy of extracting what is being voted on
- **Participant-Vote**: F1 for extracting who voted and their positions
- **Complete Event**: Strict accuracy where all components must match exactly
"""

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    logger.info(f"Saved summary to: {md_path}")


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description='LangExtract Span-to-Event Extraction for Portuguese Voting')
    parser.add_argument('--test', action='store_true', help='Run in test mode (10 documents only)')
    parser.add_argument('--max-docs', type=int, default=None, help='Maximum number of documents to process')
    args = parser.parse_args()

    # Setup paths
    project_root = Path(__file__).parent.parent
    
    # Try both .json and .jsonl formats
    test_file_json = project_root / "data/votie_events/test.json"
    test_file_jsonl = project_root / "data/votie_events/test.jsonl"
    
    if test_file_json.exists():
        test_file = test_file_json
    elif test_file_jsonl.exists():
        test_file = test_file_jsonl
    else:
        logger.error(f"Test file not found: tried {test_file_json} and {test_file_jsonl}")
        return
    
    output_dir = Path(__file__).parent / "experiments" / f"langextract_spans_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Load test dataset
    test_events = load_events_dataset(test_file)

    # Determine number of documents to process
    if args.test:
        max_docs = 10
        logger.info("Running in TEST MODE - processing 10 documents")
    elif args.max_docs:
        max_docs = args.max_docs
        logger.info(f"Processing {max_docs} documents")
    else:
        max_docs = len(test_events)
        logger.info(f"Running FULL EVALUATION - processing all {max_docs} documents")

    test_events = test_events[:max_docs]

    # Load configuration
    cfg = get_config()

    if not cfg.api_key:
        logger.error("No API key found. Set GOOGLE_API_KEY or LANGEXTRACT_API_KEY environment variable.")
        return

    # Create model
    logger.info(f"Initializing Gemini model: {cfg.model_id}")
    model = GeminiModel(model_id=cfg.model_id, api_key=cfg.api_key)

    # Get extraction configuration
    config = get_voting_extraction_config()

    # Extract spans
    extraction_start_time = time.time()
    predictions = extract_spans_with_langextract(test_events, model, config, max_documents=max_docs)
    total_extraction_time = time.time() - extraction_start_time

    # Convert spans to events
    predicted_events = convert_predictions_to_events(predictions, test_events)

    # Prepare gold events
    gold_events = [
        {
            "has_voting_event": event.get("has_voting_event", False),
            "event": event.get("event", None)
        }
        for event in test_events
    ]

    # Evaluate
    logger.info("Evaluating predictions...")
    evaluator = EventLevelEvaluator()
    evaluation_results = evaluator.compute_metrics(
        pred_events=predicted_events,
        gold_events=gold_events,
        include_voting_expression=False  # Don't require exact voting expression match
    )

    # Print results
    logger.info("=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Model: {cfg.model_id}")
    logger.info(f"Documents processed: {max_docs}")
    logger.info(f"Total extraction time: {total_extraction_time:.2f}s")
    logger.info(f"Avg time per document: {total_extraction_time / max_docs:.2f}s")
    logger.info("=" * 80)
    logger.info("EVENT DETECTION:")
    logger.info(f"  Accuracy:  {evaluation_results['event_detection_accuracy']:.4f}")
    logger.info("=" * 80)
    logger.info("SUBJECT EXTRACTION:")
    logger.info(f"  Accuracy:  {evaluation_results['subject_accuracy']:.4f}")
    logger.info("=" * 80)
    logger.info("PARTICIPANT-VOTE EXTRACTION:")
    logger.info(f"  Precision: {evaluation_results['participant_vote_precision']:.4f}")
    logger.info(f"  Recall:    {evaluation_results['participant_vote_recall']:.4f}")
    logger.info(f"  F1-Score:  {evaluation_results['participant_vote_f1']:.4f}")
    logger.info("=" * 80)
    logger.info("COMPLETE EVENT:")
    logger.info(f"  Accuracy:  {evaluation_results['complete_event_accuracy']:.4f}")
    logger.info(f"  Matches:   {evaluation_results['complete_matches']}/{evaluation_results['total_events']}")
    logger.info("=" * 80)

    # Save results
    save_results(predictions, predicted_events, evaluation_results, output_dir,
                cfg.model_id, total_extraction_time, max_docs)

    logger.info(f"\nEvaluation complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
