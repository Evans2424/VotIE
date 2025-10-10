"""
Simplified Evaluation System for LangExtract NER

Direct comparison with JSONL voting data using seqeval standards.
Focused, publication-ready evaluation metrics.

Improvements:
- spaCy tokenization matching the dataset converter
- One-to-one entity matching (greedy with GT removal)
- Correct IOB2 conversion with proper token alignment
"""

import json
import logging
from typing import List, Dict, Any, Tuple, Optional, Set
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

from seqeval.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)
from seqeval.scheme import IOB2

# spaCy for tokenization (matching converter)
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available. Using whitespace tokenization fallback.")

# Optional: Enhanced evaluation with nervaluate
try:
    from nervaluate import Evaluator
    NERVALUATE_AVAILABLE = True
except ImportError:
    NERVALUATE_AVAILABLE = False
    logging.warning("nervaluate not available. Install with: pip install nervaluate")

try:
    from .data_loader import VotingDataLoader, VotingDocument
except ImportError:
    from data_loader import VotingDataLoader, VotingDocument

logger = logging.getLogger(__name__)

# Cache for spaCy models
_SPACY_MODELS = {}


def load_spacy_model(language: str):
    """
    Load spaCy model for tokenization (matching converter logic).
    
    Args:
        language: Language code ("pt" or "en")
    
    Returns:
        spaCy nlp object or None if unavailable
    """
    global _SPACY_MODELS
    
    if not SPACY_AVAILABLE:
        return None
    
    if language in _SPACY_MODELS:
        return _SPACY_MODELS[language]
    
    model_name = "pt_core_news_sm" if language == "pt" else "en_core_web_sm"
    
    try:
        nlp = spacy.load(model_name)
        _SPACY_MODELS[language] = nlp
        logger.info(f"✓ Loaded spaCy {model_name} model for tokenization")
        return nlp
    except OSError:
        logger.warning(f"Could not load spaCy {model_name} model. Using whitespace tokenization.")
        return None


def tokenize_text(text: str, language: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    """
    Tokenize text using spaCy (matching converter) or fallback to whitespace.
    
    Args:
        text: Text to tokenize
        language: Language code ("pt" or "en")
    
    Returns:
        Tuple of (tokens, char_spans) where char_spans are (start, end) tuples
    """
    nlp = load_spacy_model(language)
    
    if nlp:
        doc = nlp(text)
        tokens = [token.text for token in doc]
        char_spans = [(token.idx, token.idx + len(token.text)) for token in doc]
        return tokens, char_spans
    else:
        # Fallback: whitespace tokenization with character position tracking
        tokens = text.split()
        char_spans = []
        current_pos = 0
        for token in tokens:
            start_pos = text.find(token, current_pos)
            if start_pos == -1:
                # Fallback to naive position
                start_pos = current_pos
            end_pos = start_pos + len(token)
            char_spans.append((start_pos, end_pos))
            current_pos = end_pos
        return tokens, char_spans


@dataclass
class EntityMatch:
    """Represents a matched entity for evaluation"""
    predicted_text: str
    predicted_label: str
    predicted_start: int
    predicted_end: int
    ground_truth_text: str
    ground_truth_label: str
    ground_truth_start: int
    ground_truth_end: int
    match_type: str  # "exact", "partial", "label_only", "no_match"


@dataclass
class DocumentEvaluation:
    """Evaluation results for a single document"""
    document_id: str
    predicted_entities: List[Dict[str, Any]]
    ground_truth_entities: List[Dict[str, Any]]
    entity_matches: List[EntityMatch]
    exact_matches: int
    partial_matches: int
    label_matches: int
    precision: float
    recall: float
    f1: float


@dataclass
class EvaluationReport:
    """Complete evaluation report"""
    language: str
    total_documents: int
    total_predicted_entities: int
    total_ground_truth_entities: int
    overall_precision: float
    overall_recall: float
    overall_f1: float
    entity_level_metrics: Dict[str, Dict[str, float]]
    document_evaluations: List[DocumentEvaluation]
    timestamp: str
    model_info: Dict[str, Any]
    error_analysis: Dict[str, Any]
    # Enhanced metrics with nervaluate
    nervaluate_metrics: Optional[Dict[str, Any]] = None
    evaluation_schemes: List[str] = None


class VotingEntityEvaluator:
    """Simplified evaluator for voting entity extraction"""

    def __init__(self, data_loader: VotingDataLoader):
        """
        Initialize evaluator

        Args:
            data_loader: VotingDataLoader instance for ground truth
        """
        self.data_loader = data_loader
        self.entity_types = [
            "VOTER-FAVOR", "VOTER-AGAINST", "VOTER-ABSTENTION",
            "VOTER-ABSENT", "SUBJECT", "VOTING",
            "COUNTING-MAJORITY", "COUNTING-UNANIMITY"
        ]

    def convert_to_bio_tags(self, text: str, entities: List[Dict[str, Any]], language: str = "pt") -> List[str]:
        """
        Convert entities to BIO tags for seqeval evaluation using spaCy tokenization.
        
        This now matches the tokenization used in the dataset converter for proper alignment.

        Args:
            text: Document text
            entities: List of entity dictionaries
            language: Language code ("pt" or "en")

        Returns:
            List of BIO tags (IOB2 scheme)
        """
        # Use spaCy tokenization matching converter
        tokens, token_positions = tokenize_text(text, language)
        tags = ["O"] * len(tokens)

        # Assign entity tags with proper IOB2 scheme
        for entity in entities:
            start_char = entity.get('start', entity.get('begin', 0))
            end_char = entity.get('end', start_char + len(entity.get('text', '')))
            label = entity.get('label', entity.get('type', 'UNKNOWN'))

            # Find tokens that overlap with entity span
            entity_tokens = []
            for i, (token_start, token_end) in enumerate(token_positions):
                # Token overlaps if it has any intersection with entity span
                if (token_start < end_char and token_end > start_char):
                    entity_tokens.append(i)

            # Assign BIO tags following IOB2 scheme
            if entity_tokens:
                # First token gets B- prefix
                tags[entity_tokens[0]] = f"B-{label}"
                # Remaining tokens get I- prefix
                for token_idx in entity_tokens[1:]:
                    tags[token_idx] = f"I-{label}"

        return tags

    def calculate_entity_matches(self, predicted_entities: List[Dict[str, Any]],
                                ground_truth_entities: List[Dict[str, Any]]) -> List[EntityMatch]:
        """
        Calculate matches between predicted and ground truth entities with one-to-one assignment.
        
        Uses greedy matching with GT removal to ensure each GT entity is matched at most once.

        Args:
            predicted_entities: Predicted entity list
            ground_truth_entities: Ground truth entity list

        Returns:
            List of EntityMatch objects (one per predicted entity)
        """
        matches = []
        # Track which GT entities have been matched (for one-to-one constraint)
        matched_gt_indices: Set[int] = set()

        for pred_entity in predicted_entities:
            pred_start = pred_entity.get('start', pred_entity.get('begin', 0))
            pred_end = pred_entity.get('end', pred_start + len(pred_entity.get('text', '')))
            pred_label = pred_entity.get('label', pred_entity.get('type', 'UNKNOWN'))
            pred_text = pred_entity.get('text', '')

            best_match = None
            best_match_idx = None
            best_match_type = "no_match"
            best_overlap = 0

            # Find best unmatched GT entity
            for gt_idx, gt_entity in enumerate(ground_truth_entities):
                # Skip already matched GT entities (one-to-one constraint)
                if gt_idx in matched_gt_indices:
                    continue

                gt_start = gt_entity.get('start', gt_entity.get('begin', 0))
                gt_end = gt_entity.get('end', gt_start + len(gt_entity.get('text', '')))
                gt_label = gt_entity.get('label', gt_entity.get('type', 'UNKNOWN'))
                gt_text = gt_entity.get('text', '')

                # Calculate overlap (intersection length)
                overlap_start = max(pred_start, gt_start)
                overlap_end = min(pred_end, gt_end)
                overlap = max(0, overlap_end - overlap_start)

                # Update best match if this has more overlap
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = gt_entity
                    best_match_idx = gt_idx

                    # Determine match type
                    if pred_start == gt_start and pred_end == gt_end and pred_label == gt_label:
                        best_match_type = "exact"
                    elif pred_label == gt_label and overlap > 0:
                        best_match_type = "partial"
                    elif pred_label == gt_label:
                        best_match_type = "label_only"
                    else:
                        best_match_type = "no_match"

            # Record match and mark GT as used
            if best_match:
                gt_start = best_match.get('start', best_match.get('begin', 0))
                gt_end = best_match.get('end', gt_start + len(best_match.get('text', '')))
                gt_label = best_match.get('label', best_match.get('type', 'UNKNOWN'))
                gt_text = best_match.get('text', '')

                match = EntityMatch(
                    predicted_text=pred_text,
                    predicted_label=pred_label,
                    predicted_start=pred_start,
                    predicted_end=pred_end,
                    ground_truth_text=gt_text,
                    ground_truth_label=gt_label,
                    ground_truth_start=gt_start,
                    ground_truth_end=gt_end,
                    match_type=best_match_type
                )
                matches.append(match)
                
                # Mark this GT as matched (enforce one-to-one)
                if best_match_idx is not None:
                    matched_gt_indices.add(best_match_idx)

        return matches

    def convert_to_nervaluate_format(self, documents: List[VotingDocument],
                                   predictions: Dict[str, List[Dict[str, Any]]]) -> Tuple[List[List[Dict]], List[List[Dict]]]:
        """
        Convert predictions and ground truth to nervaluate format

        Args:
            documents: List of VotingDocument objects
            predictions: Dictionary mapping document IDs to predicted entities

        Returns:
            Tuple of (predicted_entities, true_entities) in nervaluate format
        """
        true_entities = []
        pred_entities = []

        for document in documents:
            doc_id = document.id

            # Convert ground truth entities
            doc_true = []
            for entity in document.entities:
                doc_true.append({
                    'label': entity.get('label', entity.get('type', 'UNKNOWN')),
                    'start': entity.get('start', entity.get('begin', 0)),
                    'end': entity.get('end', entity.get('start', 0) + len(entity.get('text', '')))
                })
            true_entities.append(doc_true)

            # Convert predicted entities
            doc_pred = []
            if doc_id in predictions:
                for entity in predictions[doc_id]:
                    doc_pred.append({
                        'label': entity.get('label', entity.get('type', 'UNKNOWN')),
                        'start': entity.get('start', entity.get('begin', 0)),
                        'end': entity.get('end', entity.get('start', 0) + len(entity.get('text', '')))
                    })
            pred_entities.append(doc_pred)

        return pred_entities, true_entities

    def evaluate_with_nervaluate(self, documents: List[VotingDocument],
                                predictions: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Evaluate using nervaluate for comprehensive NER metrics

        Args:
            documents: List of VotingDocument objects
            predictions: Dictionary mapping document IDs to predicted entities

        Returns:
            Comprehensive nervaluate metrics
        """
        if not NERVALUATE_AVAILABLE:
            logger.warning("nervaluate not available - skipping enhanced evaluation")
            return {}

        logger.info("Running enhanced evaluation with nervaluate...")

        # Convert to nervaluate format
        pred_entities, true_entities = self.convert_to_nervaluate_format(documents, predictions)

        logger.info(f"Converted data for nervaluate:")
        logger.info(f"  - Documents: {len(documents)}")
        logger.info(f"  - True entities lists: {len(true_entities)}")
        logger.info(f"  - Pred entities lists: {len(pred_entities)}")
        logger.info(f"  - Sample true entities: {true_entities[:2] if true_entities else 'None'}")
        logger.info(f"  - Sample pred entities: {pred_entities[:2] if pred_entities else 'None'}")
        logger.info(f"  - Entity types: {self.entity_types}")

        # Check if we have any entities to evaluate
        total_true_entities = sum(len(doc_entities) for doc_entities in true_entities)
        total_pred_entities = sum(len(doc_entities) for doc_entities in pred_entities)

        if total_true_entities == 0 or total_pred_entities == 0:
            logger.warning(f"Insufficient entities for nervaluate: true={total_true_entities}, pred={total_pred_entities}")
            return {}

        # Create nervaluate evaluator
        try:
            evaluator = Evaluator(true_entities, pred_entities, tags=self.entity_types)
            logger.info("✅ nervaluate Evaluator created successfully")
        except Exception as e:
            logger.error(f"Failed to create nervaluate Evaluator: {e}")
            return {}

        # Evaluate with nervaluate (evaluates all schemes at once)
        try:
            raw_results = evaluator.evaluate()
            logger.info("✅ nervaluate evaluation completed with all schemes")
            logger.info(f"Raw results keys: {list(raw_results.keys()) if raw_results else 'Empty results'}")

            # Convert nervaluate results to our expected format
            results = {}
            if 'overall' in raw_results:
                # Extract metrics for each scheme
                scheme_mapping = {
                    'strict': 'strict',
                    'exact': 'exact',
                    'partial': 'partial',
                    'ent_type': 'type'  # nervaluate uses 'ent_type' but we call it 'type'
                }

                for nerv_scheme, our_scheme in scheme_mapping.items():
                    if nerv_scheme in raw_results['overall']:
                        eval_result = raw_results['overall'][nerv_scheme]
                        results[our_scheme] = {
                            'overall': {
                                'precision': eval_result.precision,
                                'recall': eval_result.recall,
                                'f1': eval_result.f1,
                                'correct': eval_result.correct,
                                'incorrect': eval_result.incorrect,
                                'partial': eval_result.partial,
                                'missed': eval_result.missed,
                                'spurious': eval_result.spurious
                            }
                        }

                        # Add per-entity metrics if available
                        if 'entities' in raw_results:
                            results[our_scheme]['entities'] = {}
                            for entity_type in self.entity_types:
                                if entity_type in raw_results['entities'] and nerv_scheme in raw_results['entities'][entity_type]:
                                    entity_eval = raw_results['entities'][entity_type][nerv_scheme]
                                    results[our_scheme]['entities'][entity_type] = {
                                        'precision': entity_eval.precision,
                                        'recall': entity_eval.recall,
                                        'f1': entity_eval.f1,
                                        'correct': entity_eval.correct,
                                        'incorrect': entity_eval.incorrect,
                                        'partial': entity_eval.partial,
                                        'missed': entity_eval.missed,
                                        'spurious': entity_eval.spurious
                                    }

            logger.info(f"Converted results schemes: {list(results.keys())}")

        except Exception as e:
            logger.error(f"Failed to evaluate with nervaluate: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            results = {}

        # Add summary metrics
        if 'strict' in results and results['strict']:
            strict_overall = results['strict'].get('overall', {})
            results['summary'] = {
                'strict_f1': strict_overall.get('f1', 0.0),
                'strict_precision': strict_overall.get('precision', 0.0),
                'strict_recall': strict_overall.get('recall', 0.0),
                'evaluation_schemes_used': list(results.keys())
            }

        return results

    def evaluate_document(self, document_id: str, predicted_entities: List[Dict[str, Any]],
                         document: VotingDocument) -> DocumentEvaluation:
        """
        Evaluate predictions for a single document

        Args:
            document_id: Document identifier
            predicted_entities: Predicted entities
            document: Ground truth VotingDocument

        Returns:
            DocumentEvaluation object
        """
        ground_truth_entities = document.entities

        # Calculate entity matches
        entity_matches = self.calculate_entity_matches(predicted_entities, ground_truth_entities)

        # Count match types
        exact_matches = sum(1 for m in entity_matches if m.match_type == "exact")
        partial_matches = sum(1 for m in entity_matches if m.match_type == "partial")
        label_matches = sum(1 for m in entity_matches if m.match_type == "label_only")

        # Calculate metrics
        precision = exact_matches / len(predicted_entities) if predicted_entities else 0.0
        recall = exact_matches / len(ground_truth_entities) if ground_truth_entities else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return DocumentEvaluation(
            document_id=document_id,
            predicted_entities=predicted_entities,
            ground_truth_entities=ground_truth_entities,
            entity_matches=entity_matches,
            exact_matches=exact_matches,
            partial_matches=partial_matches,
            label_matches=label_matches,
            precision=precision,
            recall=recall,
            f1=f1
        )

    def evaluate_extraction_results(self, predictions: Dict[str, List[Dict[str, Any]]],
                                   language: str, municipalities: List[str]) -> EvaluationReport:
        """
        Evaluate extraction results against ground truth

        Args:
            predictions: Dictionary mapping document IDs to predicted entities
            language: Language code ("pt" or "en")
            municipalities: List of municipalities to evaluate

        Returns:
            Complete evaluation report
        """
        logger.info(f"Starting evaluation for {language} language, municipalities: {municipalities}")

        document_evaluations = []
        total_predicted_entities = 0
        total_ground_truth_entities = 0
        total_exact_matches = 0

        # Collect all predicted and ground truth BIO sequences for seqeval
        all_pred_tags = []
        all_true_tags = []
        entity_type_predictions = {entity_type: {"pred": [], "true": []} for entity_type in self.entity_types}

        for municipality in municipalities:
            documents = self.data_loader.load_municipality_data(language, municipality)

            for document in documents:
                doc_id = document.id

                if doc_id in predictions:
                    predicted_entities = predictions[doc_id]

                    # Evaluate this document
                    doc_eval = self.evaluate_document(doc_id, predicted_entities, document)
                    document_evaluations.append(doc_eval)

                    total_predicted_entities += len(predicted_entities)
                    total_ground_truth_entities += len(document.entities)
                    total_exact_matches += doc_eval.exact_matches

                    # Convert to BIO tags for seqeval (using spaCy tokenization)
                    pred_tags = self.convert_to_bio_tags(document.text, predicted_entities, language)
                    true_tags = self.convert_to_bio_tags(document.text, document.entities, language)

                    all_pred_tags.append(pred_tags)
                    all_true_tags.append(true_tags)

                    # Collect entity-type specific predictions
                    for entity_type in self.entity_types:
                        type_pred_tags = [tag if tag.endswith(entity_type) else "O" for tag in pred_tags]
                        type_true_tags = [tag if tag.endswith(entity_type) else "O" for tag in true_tags]
                        entity_type_predictions[entity_type]["pred"].append(type_pred_tags)
                        entity_type_predictions[entity_type]["true"].append(type_true_tags)

        # Calculate overall metrics using seqeval
        overall_precision = precision_score(all_true_tags, all_pred_tags, scheme=IOB2)
        overall_recall = recall_score(all_true_tags, all_pred_tags, scheme=IOB2)
        overall_f1 = f1_score(all_true_tags, all_pred_tags, scheme=IOB2)

        # Calculate entity-level metrics
        entity_level_metrics = {}
        for entity_type, tags in entity_type_predictions.items():
            if any(any(tag != "O" for tag in seq) for seq in tags["true"]):  # Only if entity type exists in ground truth
                precision = precision_score(tags["true"], tags["pred"], scheme=IOB2)
                recall = recall_score(tags["true"], tags["pred"], scheme=IOB2)
                f1 = f1_score(tags["true"], tags["pred"], scheme=IOB2)
                entity_level_metrics[entity_type] = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1
                }

        # Error analysis
        error_analysis = self._perform_error_analysis(document_evaluations)

        # Enhanced evaluation with nervaluate (if available)
        nervaluate_metrics = None
        evaluation_schemes = ["seqeval_iob2"]

        if NERVALUATE_AVAILABLE:
            try:
                # Collect all documents for nervaluate evaluation
                all_documents = []
                for municipality in municipalities:
                    docs = self.data_loader.load_municipality_data(language, municipality)
                    # Only include documents that have predictions
                    for doc in docs:
                        if doc.id in predictions:
                            all_documents.append(doc)

                if all_documents:
                    nervaluate_metrics = self.evaluate_with_nervaluate(all_documents, predictions)
                    evaluation_schemes.extend(['strict', 'exact', 'partial', 'type'])
                    logger.info("✅ Enhanced nervaluate evaluation completed")
                else:
                    logger.warning("No documents with predictions found for nervaluate evaluation")
            except Exception as e:
                logger.error(f"nervaluate evaluation failed: {e}")
                nervaluate_metrics = {}

        # Create evaluation report
        report = EvaluationReport(
            language=language,
            total_documents=len(document_evaluations),
            total_predicted_entities=total_predicted_entities,
            total_ground_truth_entities=total_ground_truth_entities,
            overall_precision=overall_precision,
            overall_recall=overall_recall,
            overall_f1=overall_f1,
            entity_level_metrics=entity_level_metrics,
            document_evaluations=document_evaluations,
            timestamp=datetime.now().isoformat(),
            model_info={},  # To be filled by caller
            error_analysis=error_analysis,
            nervaluate_metrics=nervaluate_metrics,
            evaluation_schemes=evaluation_schemes
        )

        logger.info(f"Evaluation completed: F1={overall_f1:.3f}, P={overall_precision:.3f}, R={overall_recall:.3f}")
        return report

    def _perform_error_analysis(self, document_evaluations: List[DocumentEvaluation]) -> Dict[str, Any]:
        """
        Perform error analysis on evaluation results

        Args:
            document_evaluations: List of document evaluations

        Returns:
            Error analysis dictionary
        """
        error_patterns = {
            "common_false_positives": {},
            "common_false_negatives": {},
            "label_confusion_matrix": {},
            "length_based_errors": {"short": 0, "medium": 0, "long": 0}
        }

        for doc_eval in document_evaluations:
            # Analyze false positives (predicted but not in ground truth)
            for pred_entity in doc_eval.predicted_entities:
                pred_label = pred_entity.get('label', pred_entity.get('type', 'UNKNOWN'))
                if not any(m.match_type == "exact" for m in doc_eval.entity_matches
                          if m.predicted_label == pred_label):
                    error_patterns["common_false_positives"][pred_label] = \
                        error_patterns["common_false_positives"].get(pred_label, 0) + 1

            # Analyze false negatives (in ground truth but not predicted)
            for gt_entity in doc_eval.ground_truth_entities:
                gt_label = gt_entity.get('label', gt_entity.get('type', 'UNKNOWN'))
                if not any(m.match_type == "exact" for m in doc_eval.entity_matches
                          if m.ground_truth_label == gt_label):
                    error_patterns["common_false_negatives"][gt_label] = \
                        error_patterns["common_false_negatives"].get(gt_label, 0) + 1

        return error_patterns


def save_evaluation_report(report: EvaluationReport, output_path: Path) -> None:
    """
    Save evaluation report to JSON file

    Args:
        report: EvaluationReport to save
        output_path: Output file path
    """
    # Convert dataclasses to dictionaries for JSON serialization
    report_dict = {
        "summary": {
            "language": report.language,
            "total_documents": report.total_documents,
            "total_predicted_entities": report.total_predicted_entities,
            "total_ground_truth_entities": report.total_ground_truth_entities,
            "overall_precision": report.overall_precision,
            "overall_recall": report.overall_recall,
            "overall_f1": report.overall_f1,
            "timestamp": report.timestamp,
            "model_info": report.model_info,
            "evaluation_schemes": report.evaluation_schemes or ["seqeval_iob2"]
        },
        "seqeval_metrics": {
            "precision": report.overall_precision,
            "recall": report.overall_recall,
            "f1": report.overall_f1,
            "entity_level": report.entity_level_metrics
        },
        "nervaluate_metrics": report.nervaluate_metrics or {},
        "nervaluate_summary": {},
        "error_analysis": report.error_analysis,
        "document_level_results": [
            {
                "document_id": doc.document_id,
                "predicted_entities_count": len(doc.predicted_entities),
                "ground_truth_entities_count": len(doc.ground_truth_entities),
                "exact_matches": doc.exact_matches,
                "partial_matches": doc.partial_matches,
                "precision": doc.precision,
                "recall": doc.recall,
                "f1": doc.f1
            }
            for doc in report.document_evaluations
        ]
    }

    # Add explicit nervaluate summary if available
    if report.nervaluate_metrics:
        nervaluate_summary = {}
        for scheme in ['strict', 'exact', 'partial', 'type']:
            if scheme in report.nervaluate_metrics:
                scheme_metrics = report.nervaluate_metrics[scheme]
                if 'overall' in scheme_metrics:
                    nervaluate_summary[f"{scheme}_metrics"] = {
                        "precision": scheme_metrics['overall'].get('precision', 0.0),
                        "recall": scheme_metrics['overall'].get('recall', 0.0),
                        "f1": scheme_metrics['overall'].get('f1', 0.0),
                        "correct": scheme_metrics['overall'].get('correct', 0),
                        "incorrect": scheme_metrics['overall'].get('incorrect', 0),
                        "partial": scheme_metrics['overall'].get('partial', 0),
                        "missed": scheme_metrics['overall'].get('missed', 0),
                        "spurious": scheme_metrics['overall'].get('spurious', 0)
                    }

        report_dict["nervaluate_summary"] = nervaluate_summary

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)

    logger.info(f"Evaluation report saved to: {output_path}")


def create_evaluation_summary(report: EvaluationReport) -> str:
    """
    Create a human-readable evaluation summary

    Args:
        report: EvaluationReport

    Returns:
        Formatted summary string
    """
    summary = f"""
# LangExtract Voting Entity Extraction Evaluation Report

## Overall Performance
- **Language**: {report.language.upper()}
- **Total Documents**: {report.total_documents}
- **Total Predicted Entities**: {report.total_predicted_entities}
- **Total Ground Truth Entities**: {report.total_ground_truth_entities}
- **Evaluation Schemes**: {', '.join(report.evaluation_schemes or ['seqeval_iob2'])}

## Performance Summary

### Seqeval IOB2 Metrics (Traditional NER Evaluation)
- **Precision**: {report.overall_precision:.3f}
- **Recall**: {report.overall_recall:.3f}
- **F1-Score**: {report.overall_f1:.3f}
"""

    # Add nervaluate metrics if available
    if report.nervaluate_metrics:
        summary += """
### Nervaluate Enhanced Metrics (Multi-Scheme Evaluation)

| Evaluation Scheme | Precision | Recall | F1-Score | Description |
|-------------------|-----------|--------|----------|-------------|
"""
        scheme_descriptions = {
            'strict': 'Exact boundary + exact entity type',
            'exact': 'Exact boundary, any type',
            'partial': 'Overlapping boundaries, any type',
            'type': 'Any overlap + correct type'
        }

        for scheme in ['strict', 'exact', 'partial', 'type']:
            if scheme in report.nervaluate_metrics and 'overall' in report.nervaluate_metrics[scheme]:
                metrics = report.nervaluate_metrics[scheme]['overall']
                precision = metrics.get('precision', 0.0)
                recall = metrics.get('recall', 0.0)
                f1 = metrics.get('f1', 0.0)
                description = scheme_descriptions.get(scheme, '')
                summary += f"| **{scheme.title()}** | {precision:.3f} | {recall:.3f} | {f1:.3f} | {description} |\n"

        # Add interpretation
        if 'strict' in report.nervaluate_metrics and 'partial' in report.nervaluate_metrics:
            strict_f1 = report.nervaluate_metrics['strict'].get('overall', {}).get('f1', 0)
            partial_f1 = report.nervaluate_metrics['partial'].get('overall', {}).get('f1', 0)
            boundary_tolerance = partial_f1 - strict_f1
            summary += f"""
#### Key Insights:
- **Boundary Precision**: {boundary_tolerance:.3f} F1 improvement from strict to partial evaluation
- **Type Classification**: Compare strict vs exact schemes for entity type accuracy
- **Entity Detection**: Partial scheme shows overall entity detection capability
"""

    summary += """
## Entity-Level Performance (Seqeval)
"""

    for entity_type, metrics in report.entity_level_metrics.items():
        summary += f"""
### {entity_type}
- Precision: {metrics['precision']:.3f}
- Recall: {metrics['recall']:.3f}
- F1-Score: {metrics['f1']:.3f}
"""

    summary += f"""
## Error Analysis Summary
- Common False Positives: {dict(list(report.error_analysis.get('common_false_positives', {}).items())[:5])}
- Common False Negatives: {dict(list(report.error_analysis.get('common_false_negatives', {}).items())[:5])}

Generated: {report.timestamp}
"""

    return summary