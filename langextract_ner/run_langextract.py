#!/usr/bin/env python
"""
LangExtract NER Evaluation - Simple Entry Point

Run LangExtract voting entity extraction on Portuguese test dataset
and evaluate with NER baseline-compatible metrics.

Usage:
    python run_langextract.py
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import time

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use system env vars

# LangExtract imports
import sys
from pathlib import Path

# Add current directory to path for direct execution
sys.path.insert(0, str(Path(__file__).parent))

from shared.models import GeminiModel
from shared.prompts import get_voting_extraction_config
from evaluation import VotingEntityEvaluator
from data_loader import VotingDocument
from config import get_config

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


def load_test_dataset(test_file: Path) -> List[Dict[str, Any]]:
    """
    Load test dataset from JSONL file

    Args:
        test_file: Path to test.jsonl file

    Returns:
        List of test documents with tokens and labels
    """
    logger.info(f"Loading test dataset from: {test_file}")

    documents = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line.strip())
            documents.append(doc)

    logger.info(f"Loaded {len(documents)} test documents")
    return documents


def reconstruct_text_from_tokens(tokens: List[str]) -> str:
    """
    Reconstruct text from tokenized tokens

    Args:
        tokens: List of token strings

    Returns:
        Reconstructed text string
    """
    # Simple reconstruction - join with spaces
    # This assumes spacy tokenization which generally splits on spaces
    return " ".join(tokens)


def convert_bio_to_entities(tokens: List[str], labels: List[str]) -> List[Dict[str, Any]]:
    """
    Convert BIO labels to entity list

    Args:
        tokens: List of tokens
        labels: List of BIO labels

    Returns:
        List of entity dictionaries with start, end, text, label
    """
    entities = []
    current_entity = None
    char_offset = 0

    for token, label in zip(tokens, labels):
        token_start = char_offset
        token_end = char_offset + len(token)

        if label.startswith('B-'):
            # Start new entity
            if current_entity:
                entities.append(current_entity)

            entity_type = label[2:]  # Remove 'B-' prefix
            current_entity = {
                'start': token_start,
                'end': token_end,
                'text': token,
                'label': entity_type,
                'type': entity_type
            }
        elif label.startswith('I-'):
            # Continue current entity
            if current_entity:
                entity_type = label[2:]
                if current_entity['label'] == entity_type:
                    # Extend current entity
                    current_entity['end'] = token_end
                    current_entity['text'] += ' ' + token
                else:
                    # Label mismatch, start new entity
                    entities.append(current_entity)
                    current_entity = {
                        'start': token_start,
                        'end': token_end,
                        'text': token,
                        'label': entity_type,
                        'type': entity_type
                    }
        else:
            # 'O' label - outside entity
            if current_entity:
                entities.append(current_entity)
                current_entity = None

        # Update char offset (token + space)
        char_offset = token_end + 1

    # Don't forget last entity
    if current_entity:
        entities.append(current_entity)

    return entities


def extract_with_langextract(documents: List[Dict[str, Any]],
                             model: GeminiModel,
                             config: Dict[str, Any],
                             max_documents: int = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract entities from documents using LangExtract

    Args:
        documents: List of test documents
        model: GeminiModel instance
        config: Extraction configuration with prompt and examples
        max_documents: Maximum number of documents to process (for testing)

    Returns:
        Dictionary mapping document IDs to extracted entities
    """
    logger.info(f"Starting LangExtract extraction with {model.model_id}")

    predictions = {}
    total_docs = len(documents) if max_documents is None else min(max_documents, len(documents))

    start_time = time.time()

    for i, doc in enumerate(documents[:total_docs]):
        doc_id = doc['id']
        tokens = doc['tokens']

        # Reconstruct text from tokens
        text = reconstruct_text_from_tokens(tokens)

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

    elapsed_time = time.time() - start_time
    docs_per_sec = total_docs / elapsed_time if elapsed_time > 0 else 0

    logger.info(f"Extraction completed: {total_docs} documents in {elapsed_time:.2f}s ({docs_per_sec:.2f} docs/sec)")

    return predictions


def convert_test_docs_to_voting_documents(test_docs: List[Dict[str, Any]]) -> List[VotingDocument]:
    """
    Convert test documents from JSONL format to VotingDocument objects.

    Reconstructs text from tokens and converts BIO labels to entity spans.

    Args:
        test_docs: List of test documents with tokens and BIO labels

    Returns:
        List of VotingDocument objects for evaluation
    """
    documents = []

    for doc in test_docs:
        doc_id = doc['id']
        tokens = doc['tokens']
        labels = doc['labels']

        # Reconstruct text from tokens
        text = reconstruct_text_from_tokens(tokens)

        # Convert BIO labels to entity spans
        entities = convert_bio_to_entities(tokens, labels)

        # Create VotingDocument with ground truth entities
        voting_doc = VotingDocument(
            id=doc_id,
            text=text,
            entities=entities,  # Ground truth entities
            metadata={
                'municipality': doc.get('municipality_id', 'unknown'),
                'language': 'pt'
            }
        )

        documents.append(voting_doc)

    return documents


def save_evaluation_results(report: Any, output_dir: Path, model_id: str, predictions: Dict[str, List[Dict[str, Any]]] = None):
    """
    Save evaluation results in NER baseline-compatible format

    Args:
        report: EvaluationReport object
        output_dir: Output directory for results
        model_id: Model identifier from config
        predictions: Optional predictions dictionary to save
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions if provided
    if predictions:
        predictions_path = output_dir / "predictions.jsonl"
        with open(predictions_path, 'w', encoding='utf-8') as f:
            for doc_id, entities in predictions.items():
                pred_data = {
                    'document_id': doc_id,
                    'predictions': entities,
                    'num_entities': len(entities)
                }
                f.write(json.dumps(pred_data, ensure_ascii=False) + '\n')
        logger.info(f"Saved predictions to: {predictions_path}")

    # Save JSON report
    json_path = output_dir / "evaluation_report.json"

    # Extract timing info from model_info
    extraction_time = report.model_info.get('total_extraction_time_seconds', 0)
    avg_time_per_doc = report.model_info.get('avg_time_per_document_seconds', 0)

    report_dict = {
        "metadata": {
            "model": model_id,
            "language": report.language,
            "timestamp": report.timestamp,
            "total_documents": report.total_documents,
            "evaluation_method": "langextract_ner",
            "evaluation_schemes": report.evaluation_schemes or ["seqeval_iob2"],
            "extraction_time_seconds": extraction_time,
            "avg_time_per_document_seconds": avg_time_per_doc
        },
        "overall_metrics": {
            "precision": report.overall_precision,
            "recall": report.overall_recall,
            "entity_f1_strict": report.overall_f1,
            "total_predicted_entities": report.total_predicted_entities,
            "total_ground_truth_entities": report.total_ground_truth_entities
        },
        "per_type_metrics": report.entity_level_metrics,
        "nervaluate_metrics": report.nervaluate_metrics or {},
        "error_analysis": report.error_analysis
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved evaluation report to: {json_path}")

    # Save markdown summary
    md_path = output_dir / "evaluation_summary.md"

    summary = f"""# LangExtract NER Evaluation Report

## Model Information
- **Model**: {model_id}
- **Language**: Portuguese
- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Evaluation Method**: {', '.join(report.evaluation_schemes or ['seqeval_iob2'])}

## Extraction Performance
- **Total Extraction Time**: {extraction_time:.2f}s
- **Avg Time per Document**: {avg_time_per_doc:.2f}s
- **Documents Processed**: {report.total_documents}

## Overall Performance

| Metric | Value |
|--------|-------|
| **Precision** | {report.overall_precision:.4f} |
| **Recall** | {report.overall_recall:.4f} |
| **F1-Score** | {report.overall_f1:.4f} |
| **Predicted Entities** | {report.total_predicted_entities} |
| **Ground Truth Entities** | {report.total_ground_truth_entities} |

## Per-Entity Type Performance

| Entity Type | Precision | Recall | F1-Score |
|-------------|-----------|--------|----------|
"""

    for entity_type, metrics in report.entity_level_metrics.items():
        summary += f"| {entity_type} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} |\n"

    # Add nervaluate metrics if available
    if report.nervaluate_metrics:
        summary += "\n## Enhanced Metrics (Nervaluate)\n\n"
        summary += "*Note: Strict/Exact/Partial metrics may be identical when entity boundaries match exactly.*\n\n"
        summary += "| Scheme | Precision | Recall | F1-Score | Correct | Partial | Missed | Spurious |\n"
        summary += "|--------|-----------|--------|----------|---------|---------|--------|----------|\n"

        for scheme in ['strict', 'exact', 'partial', 'type']:
            if scheme in report.nervaluate_metrics and 'overall' in report.nervaluate_metrics[scheme]:
                metrics = report.nervaluate_metrics[scheme]['overall']
                p = metrics.get('precision', 0.0)
                r = metrics.get('recall', 0.0)
                f1 = metrics.get('f1', 0.0)
                correct = metrics.get('correct', 0)
                partial = metrics.get('partial', 0)
                missed = metrics.get('missed', 0)
                spurious = metrics.get('spurious', 0)
                summary += f"| {scheme.title()} | {p:.4f} | {r:.4f} | {f1:.4f} | {correct} | {partial} | {missed} | {spurious} |\n"

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(summary)

    logger.info(f"Saved summary to: {md_path}")


def main():
    """Main execution function"""
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='LangExtract NER Evaluation for Portuguese Voting Entities')
    parser.add_argument('--test', action='store_true', help='Run in test mode (10 documents only)')
    parser.add_argument('--max-docs', type=int, default=None, help='Maximum number of documents to process')
    args = parser.parse_args()

    # Setup paths
    project_root = Path(__file__).parent.parent
    test_file = project_root / "data/publication_datasets/ner_datasets/pt/splits/test.jsonl"
    output_dir = Path(__file__).parent / "experiments" / f"langextract_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Check if test file exists
    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        return

    # Load test dataset
    test_docs = load_test_dataset(test_file)

    # Determine number of documents to process
    if args.test:
        max_docs = 10
        logger.info("Running in TEST MODE - processing 10 documents")
    elif args.max_docs:
        max_docs = args.max_docs
        logger.info(f"Processing {max_docs} documents")
    else:
        max_docs = len(test_docs)
        logger.info(f"Running FULL EVALUATION - processing all {max_docs} documents")

    test_docs = test_docs[:max_docs]

    # Load configuration
    cfg = get_config()

    if not cfg.api_key:
        logger.error("No API key found. Set LANGEXTRACT_API_KEY or GOOGLE_API_KEY environment variable.")
        return

    # Create model from config
    logger.info(f"Initializing Gemini model: {cfg.model_id}")
    model = GeminiModel(model_id=cfg.model_id, api_key=cfg.api_key)

    # Get extraction configuration (Portuguese only)
    config = get_voting_extraction_config()

    # Extract entities and track time
    extraction_start_time = time.time()
    predictions = extract_with_langextract(test_docs, model, config, max_documents=max_docs)
    total_extraction_time = time.time() - extraction_start_time

    # Convert test documents to VotingDocument format with ground truth entities
    logger.info("Converting test documents to evaluation format...")
    ground_truth_documents = convert_test_docs_to_voting_documents(test_docs)

    # Create a simple data loader that returns our pre-loaded documents
    class TestDataLoader:
        """Data loader for test documents - returns pre-loaded ground truth"""
        def __init__(self, documents):
            self.documents = documents

        def load_municipality_data(self, language, municipality):
            return self.documents

    # Initialize evaluator with ground truth documents
    data_loader = TestDataLoader(ground_truth_documents)
    evaluator = VotingEntityEvaluator(data_loader)

    # Run evaluation using VotingEntityEvaluator
    # This properly handles: spacy tokenization, BIO conversion, seqeval metrics, nervaluate
    logger.info("Evaluating predictions with VotingEntityEvaluator...")
    report = evaluator.evaluate_extraction_results(
        predictions=predictions,
        language='pt',
        municipalities=['test_set']  # Dummy value - our loader returns all docs
    )

    # Update model info from config and add extraction time
    report.model_info = {
        'model_id': cfg.model_id,
        'type': 'langextract',
        'total_extraction_time_seconds': round(total_extraction_time, 2),
        'avg_time_per_document_seconds': round(total_extraction_time / max_docs, 2)
    }

    # Print results
    logger.info("=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Model: {cfg.model_id}")
    logger.info(f"Documents processed: {max_docs}")
    logger.info(f"Total extraction time: {total_extraction_time:.2f}s")
    logger.info(f"Avg time per document: {total_extraction_time / max_docs:.2f}s")
    logger.info("=" * 80)
    logger.info(f"Overall Precision: {report.overall_precision:.4f}")
    logger.info(f"Overall Recall:    {report.overall_recall:.4f}")
    logger.info(f"Overall F1-Score:  {report.overall_f1:.4f}")
    logger.info("=" * 80)

    logger.info("\nPer-Entity Type Results:")
    for entity_type, metrics in report.entity_level_metrics.items():
        logger.info(f"  {entity_type:30s} P: {metrics['precision']:.4f}  R: {metrics['recall']:.4f}  F1: {metrics['f1']:.4f}")

    # Save results including predictions
    save_evaluation_results(report, output_dir, cfg.model_id, predictions)

    logger.info(f"\nEvaluation complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
