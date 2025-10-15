#!/usr/bin/env python3
"""
Event-Level Error Analysis Script

Analyzes event extraction errors and categorizes them into:
- Subject errors (missing, boundary, incorrect)
- Participant-vote errors (missing voters, incorrect positions)
- Counting errors (missing, incorrect type)
- Voting expression errors (missing, incorrect)
- Multiple simultaneous errors

Usage:
    python scripts/analyze_event_errors.py --predictions experiments/.../predictions.jsonl --gold data/votie_events/test.json
    python scripts/analyze_event_errors.py --predictions experiments/.../predictions.jsonl --gold data/votie_events/test.json --output error_analysis.json
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import event constructor
try:
    from src.utils.event_constructor import EventConstructor
    EVENT_CONSTRUCTOR_AVAILABLE = True
except ImportError:
    print("Warning: EventConstructor not available. Will expect event-formatted predictions.")
    EVENT_CONSTRUCTOR_AVAILABLE = False


@dataclass
class ErrorStats:
    """Statistics for a specific error type"""
    count: int = 0
    percentage: float = 0.0
    examples: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.examples is None:
            self.examples = []


class EventErrorAnalyzer:
    """Analyzes event extraction errors"""

    def __init__(self):
        self.error_categories = {
            'subject_missing': ErrorStats(),
            'subject_boundary': ErrorStats(),
            'subject_incorrect': ErrorStats(),
            'participant_missing': ErrorStats(),
            'participant_extra': ErrorStats(),
            'participant_position_wrong': ErrorStats(),
            'counting_missing': ErrorStats(),
            'counting_type_wrong': ErrorStats(),
            'counting_incorrect': ErrorStats(),
            'voting_expression_missing': ErrorStats(),
            'voting_expression_incorrect': ErrorStats(),
            'multiple_errors': ErrorStats(),
            'event_detection_false_positive': ErrorStats(),
            'event_detection_false_negative': ErrorStats()
        }
        self.total_errors = 0
        self.total_correct = 0
        self.total_events = 0

    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        if not text:
            return ""
        return text.lower().strip()

    def has_text_overlap(self, text1: str, text2: str, threshold: float = 0.5) -> bool:
        """Check if two texts have significant word overlap"""
        if not text1 or not text2:
            return False

        words1 = set(self.normalize_text(text1).split())
        words2 = set(self.normalize_text(text2).split())

        if not words1 or not words2:
            return False

        overlap = len(words1 & words2)
        min_length = min(len(words1), len(words2))

        return overlap / min_length >= threshold

    def analyze_subject(self, pred_subject: str, gold_subject: str, example_id: str) -> List[str]:
        """Analyze subject errors"""
        errors = []

        if not gold_subject:
            # No gold subject, nothing to check
            return errors

        if not pred_subject:
            # Missing subject
            errors.append('subject_missing')
            self.error_categories['subject_missing'].examples.append({
                'id': example_id,
                'gold': gold_subject,
                'predicted': None
            })
        elif self.normalize_text(pred_subject) == self.normalize_text(gold_subject):
            # Perfect match
            pass
        elif self.has_text_overlap(pred_subject, gold_subject, threshold=0.5):
            # Boundary error (partial match)
            errors.append('subject_boundary')
            self.error_categories['subject_boundary'].examples.append({
                'id': example_id,
                'gold': gold_subject,
                'predicted': pred_subject
            })
        else:
            # Completely incorrect
            errors.append('subject_incorrect')
            self.error_categories['subject_incorrect'].examples.append({
                'id': example_id,
                'gold': gold_subject,
                'predicted': pred_subject
            })

        return errors

    def analyze_participants(self, pred_participants: List[Dict], gold_participants: List[Dict], example_id: str) -> List[str]:
        """Analyze participant-vote errors"""
        errors = []

        # Convert to comparable format: (normalized_text, position)
        def normalize_participant(p):
            return (self.normalize_text(p.get('text', '')), p.get('position', '').upper())

        pred_set = set(normalize_participant(p) for p in pred_participants)
        gold_set = set(normalize_participant(p) for p in gold_participants)

        missing = gold_set - pred_set
        extra = pred_set - gold_set

        if missing:
            errors.append('participant_missing')
            self.error_categories['participant_missing'].examples.append({
                'id': example_id,
                'missing': list(missing),
                'count': len(missing)
            })

        if extra:
            errors.append('participant_extra')
            self.error_categories['participant_extra'].examples.append({
                'id': example_id,
                'extra': list(extra),
                'count': len(extra)
            })

        # Check for position errors (same text, wrong position)
        pred_texts = {self.normalize_text(p.get('text', '')): p.get('position', '').upper() for p in pred_participants}
        gold_texts = {self.normalize_text(p.get('text', '')): p.get('position', '').upper() for p in gold_participants}

        for text in pred_texts.keys() & gold_texts.keys():
            if pred_texts[text] != gold_texts[text]:
                errors.append('participant_position_wrong')
                self.error_categories['participant_position_wrong'].examples.append({
                    'id': example_id,
                    'text': text,
                    'gold_position': gold_texts[text],
                    'pred_position': pred_texts[text]
                })

        return errors

    def normalize_counting(self, counting: Any) -> List[Dict]:
        """Normalize counting to list format"""
        if not counting:
            return []
        if isinstance(counting, list):
            return counting
        if isinstance(counting, dict):
            return [counting]
        # If it's a string or other type, skip
        return []

    def analyze_counting(self, pred_counting: Any, gold_counting: Any, example_id: str) -> List[str]:
        """Analyze counting errors"""
        errors = []

        # Normalize to list format
        pred_counting = self.normalize_counting(pred_counting)
        gold_counting = self.normalize_counting(gold_counting)

        # If gold has counting but prediction doesn't
        if gold_counting and not pred_counting:
            errors.append('counting_missing')
            self.error_categories['counting_missing'].examples.append({
                'id': example_id,
                'gold': gold_counting
            })
            return errors

        # If no gold counting, no error
        if not gold_counting:
            return errors

        # Check counting types
        pred_types = {c.get('type', '').upper() for c in pred_counting if isinstance(c, dict)}
        gold_types = {c.get('type', '').upper() for c in gold_counting if isinstance(c, dict)}

        if pred_types != gold_types:
            errors.append('counting_type_wrong')
            self.error_categories['counting_type_wrong'].examples.append({
                'id': example_id,
                'gold_types': list(gold_types),
                'pred_types': list(pred_types)
            })

        # Check counting text overlap
        pred_texts = [self.normalize_text(c.get('text', '')) for c in pred_counting]
        gold_texts = [self.normalize_text(c.get('text', '')) for c in gold_counting]

        has_text_match = any(
            any(self.has_text_overlap(pt, gt, 0.3) for gt in gold_texts)
            for pt in pred_texts
        )

        if not has_text_match and gold_counting:
            errors.append('counting_incorrect')
            self.error_categories['counting_incorrect'].examples.append({
                'id': example_id,
                'gold': gold_texts,
                'predicted': pred_texts
            })

        return errors

    def normalize_expressions(self, expressions: Any) -> List[str]:
        """Normalize voting expressions to list format"""
        if not expressions:
            return []
        if isinstance(expressions, list):
            return [str(e) for e in expressions if e]
        if isinstance(expressions, str):
            return [expressions]
        return []

    def analyze_voting_expressions(self, pred_expressions: Any, gold_expressions: Any, example_id: str) -> List[str]:
        """Analyze voting expression errors"""
        errors = []

        # Normalize to list format
        pred_expressions = self.normalize_expressions(pred_expressions)
        gold_expressions = self.normalize_expressions(gold_expressions)

        # If gold has expressions but prediction doesn't
        if gold_expressions and not pred_expressions:
            errors.append('voting_expression_missing')
            self.error_categories['voting_expression_missing'].examples.append({
                'id': example_id,
                'gold': gold_expressions
            })
            return errors

        # If no gold expressions, no error
        if not gold_expressions:
            return errors

        # Check if any predicted expression overlaps with gold
        pred_norm = [self.normalize_text(e) for e in pred_expressions]
        gold_norm = [self.normalize_text(e) for e in gold_expressions]

        has_match = any(
            any(self.has_text_overlap(pe, ge, 0.3) for ge in gold_norm)
            for pe in pred_norm
        )

        if not has_match:
            errors.append('voting_expression_incorrect')
            self.error_categories['voting_expression_incorrect'].examples.append({
                'id': example_id,
                'gold': gold_expressions,
                'predicted': pred_expressions
            })

        return errors

    def analyze_event(self, pred_event: Dict[str, Any], gold_event: Dict[str, Any], example_id: str) -> List[str]:
        """Analyze a single event and return list of error types"""
        errors = []

        pred_has_event = pred_event.get('has_voting_event', False)
        gold_has_event = gold_event.get('has_voting_event', False)

        # Event detection errors
        if not gold_has_event and pred_has_event:
            errors.append('event_detection_false_positive')
            self.error_categories['event_detection_false_positive'].examples.append({
                'id': example_id
            })
            return errors

        if gold_has_event and not pred_has_event:
            errors.append('event_detection_false_negative')
            self.error_categories['event_detection_false_negative'].examples.append({
                'id': example_id
            })
            return errors

        # If no event in gold, no further analysis needed
        if not gold_has_event:
            return errors

        # Extract event details
        pred_ev = pred_event.get('event', {})
        gold_ev = gold_event.get('event', {})

        # Analyze each component
        errors.extend(self.analyze_subject(
            pred_ev.get('subject'),
            gold_ev.get('subject'),
            example_id
        ))

        errors.extend(self.analyze_participants(
            pred_ev.get('participants', []),
            gold_ev.get('participants', []),
            example_id
        ))

        errors.extend(self.analyze_counting(
            pred_ev.get('counting', []),
            gold_ev.get('counting', []),
            example_id
        ))

        errors.extend(self.analyze_voting_expressions(
            pred_ev.get('voting_expressions', []),
            gold_ev.get('voting_expressions', []),
            example_id
        ))

        return errors

    def normalize_id(self, id_str: str) -> str:
        """
        Normalize ID by removing municipality prefix if present.
        Converts 'M03_document_seg001' to 'document_seg001'
        """
        # Check if ID starts with municipality pattern (M##_)
        if id_str and len(id_str) > 4 and id_str[:3].startswith('M') and id_str[3] == '_':
            # Remove M##_ prefix
            return id_str[4:]
        return id_str

    def analyze_predictions(self, predictions: List[Dict], gold_data: List[Dict]) -> Dict[str, Any]:
        """Analyze all predictions and return comprehensive statistics"""

        # Create lookup by ID (with normalized IDs)
        gold_by_id = {}
        for item in gold_data:
            normalized_id = self.normalize_id(item['id'])
            gold_by_id[normalized_id] = item
            # Also store with original ID for compatibility
            gold_by_id[item['id']] = item

        for pred in predictions:
            example_id = pred.get('id')
            normalized_id = self.normalize_id(example_id)

            # Try both normalized and original ID
            gold = gold_by_id.get(normalized_id) or gold_by_id.get(example_id)

            if not gold:
                print(f"Warning: No gold data found for {example_id} (normalized: {normalized_id})")
                continue

            self.total_events += 1

            # Analyze this event
            errors = self.analyze_event(pred, gold, example_id)

            if not errors:
                self.total_correct += 1
            else:
                self.total_errors += 1

                # Count error types
                for error_type in set(errors):  # Use set to avoid double-counting
                    self.error_categories[error_type].count += 1

                # Track multiple errors
                if len(set(errors)) > 1:
                    self.error_categories['multiple_errors'].count += 1
                    self.error_categories['multiple_errors'].examples.append({
                        'id': example_id,
                        'error_types': list(set(errors)),
                        'error_count': len(set(errors))
                    })

        # Calculate percentages
        for category in self.error_categories.values():
            if self.total_errors > 0:
                category.percentage = (category.count / self.total_errors) * 100

        return self.get_summary()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""

        # Group errors by type
        subject_errors = (
            self.error_categories['subject_missing'].count +
            self.error_categories['subject_boundary'].count +
            self.error_categories['subject_incorrect'].count
        )

        participant_errors = (
            self.error_categories['participant_missing'].count +
            self.error_categories['participant_extra'].count +
            self.error_categories['participant_position_wrong'].count
        )

        counting_errors = (
            self.error_categories['counting_missing'].count +
            self.error_categories['counting_type_wrong'].count +
            self.error_categories['counting_incorrect'].count
        )

        voting_expression_errors = (
            self.error_categories['voting_expression_missing'].count +
            self.error_categories['voting_expression_incorrect'].count
        )

        event_detection_errors = (
            self.error_categories['event_detection_false_positive'].count +
            self.error_categories['event_detection_false_negative'].count
        )

        return {
            'summary': {
                'total_events': self.total_events,
                'correct_events': self.total_correct,
                'failed_events': self.total_errors,
                'accuracy': (self.total_correct / self.total_events * 100) if self.total_events > 0 else 0
            },
            'error_distribution': {
                'subject_errors': {
                    'count': subject_errors,
                    'percentage': (subject_errors / self.total_errors * 100) if self.total_errors > 0 else 0,
                    'breakdown': {
                        'missing': self.error_categories['subject_missing'].count,
                        'boundary': self.error_categories['subject_boundary'].count,
                        'incorrect': self.error_categories['subject_incorrect'].count
                    }
                },
                'participant_errors': {
                    'count': participant_errors,
                    'percentage': (participant_errors / self.total_errors * 100) if self.total_errors > 0 else 0,
                    'breakdown': {
                        'missing': self.error_categories['participant_missing'].count,
                        'extra': self.error_categories['participant_extra'].count,
                        'position_wrong': self.error_categories['participant_position_wrong'].count
                    }
                },
                'counting_errors': {
                    'count': counting_errors,
                    'percentage': (counting_errors / self.total_errors * 100) if self.total_errors > 0 else 0,
                    'breakdown': {
                        'missing': self.error_categories['counting_missing'].count,
                        'type_wrong': self.error_categories['counting_type_wrong'].count,
                        'incorrect': self.error_categories['counting_incorrect'].count
                    }
                },
                'voting_expression_errors': {
                    'count': voting_expression_errors,
                    'percentage': (voting_expression_errors / self.total_errors * 100) if self.total_errors > 0 else 0,
                    'breakdown': {
                        'missing': self.error_categories['voting_expression_missing'].count,
                        'incorrect': self.error_categories['voting_expression_incorrect'].count
                    }
                },
                'event_detection_errors': {
                    'count': event_detection_errors,
                    'percentage': (event_detection_errors / self.total_errors * 100) if self.total_errors > 0 else 0,
                    'breakdown': {
                        'false_positives': self.error_categories['event_detection_false_positive'].count,
                        'false_negatives': self.error_categories['event_detection_false_negative'].count
                    }
                },
                'multiple_errors': {
                    'count': self.error_categories['multiple_errors'].count,
                    'percentage': (self.error_categories['multiple_errors'].count / self.total_errors * 100) if self.total_errors > 0 else 0
                }
            },
            'detailed_errors': {
                category: {
                    'count': stats.count,
                    'percentage': stats.percentage,
                    'examples': stats.examples[:5]  # Include up to 5 examples
                }
                for category, stats in self.error_categories.items()
                if stats.count > 0
            }
        }

    def print_summary(self, summary: Dict[str, Any]):
        """Print human-readable summary"""
        print("=" * 80)
        print("EVENT-LEVEL ERROR ANALYSIS")
        print("=" * 80)
        print()

        # Overall statistics
        s = summary['summary']
        print(f"Total Events:    {s['total_events']}")
        print(f"Correct Events:  {s['correct_events']} ({s['accuracy']:.1f}%)")
        print(f"Failed Events:   {s['failed_events']}")
        print()

        # Error distribution
        print("=" * 80)
        print("ERROR DISTRIBUTION")
        print("=" * 80)

        dist = summary['error_distribution']

        print(f"\nSubject Errors: {dist['subject_errors']['count']} ({dist['subject_errors']['percentage']:.1f}%)")
        print(f"  - Missing:    {dist['subject_errors']['breakdown']['missing']}")
        print(f"  - Boundary:   {dist['subject_errors']['breakdown']['boundary']}")
        print(f"  - Incorrect:  {dist['subject_errors']['breakdown']['incorrect']}")

        print(f"\nParticipant-Vote Errors: {dist['participant_errors']['count']} ({dist['participant_errors']['percentage']:.1f}%)")
        print(f"  - Missing:        {dist['participant_errors']['breakdown']['missing']}")
        print(f"  - Extra:          {dist['participant_errors']['breakdown']['extra']}")
        print(f"  - Position Wrong: {dist['participant_errors']['breakdown']['position_wrong']}")

        print(f"\nCounting Errors: {dist['counting_errors']['count']} ({dist['counting_errors']['percentage']:.1f}%)")
        print(f"  - Missing:    {dist['counting_errors']['breakdown']['missing']}")
        print(f"  - Type Wrong: {dist['counting_errors']['breakdown']['type_wrong']}")
        print(f"  - Incorrect:  {dist['counting_errors']['breakdown']['incorrect']}")

        print(f"\nVoting Expression Errors: {dist['voting_expression_errors']['count']} ({dist['voting_expression_errors']['percentage']:.1f}%)")
        print(f"  - Missing:    {dist['voting_expression_errors']['breakdown']['missing']}")
        print(f"  - Incorrect:  {dist['voting_expression_errors']['breakdown']['incorrect']}")

        print(f"\nEvent Detection Errors: {dist['event_detection_errors']['count']} ({dist['event_detection_errors']['percentage']:.1f}%)")
        print(f"  - False Positives: {dist['event_detection_errors']['breakdown']['false_positives']}")
        print(f"  - False Negatives: {dist['event_detection_errors']['breakdown']['false_negatives']}")

        print(f"\nMultiple Simultaneous Errors: {dist['multiple_errors']['count']} ({dist['multiple_errors']['percentage']:.1f}%)")

        print()
        print("=" * 80)
        print("SUMMARY FOR PAPER")
        print("=" * 80)
        print(f"\nError analysis of {s['failed_events']} failed events reveals:")
        print(f"{dist['subject_errors']['percentage']:.0f}% from subject errors, ", end="")
        print(f"{dist['participant_errors']['percentage']:.0f}% from participant-vote errors, ", end="")
        print(f"{dist['counting_errors']['percentage']:.0f}% from counting errors, ", end="")
        print(f"{dist['voting_expression_errors']['percentage']:.0f}% from voting expression errors, ", end="")
        print(f"{dist['event_detection_errors']['percentage']:.0f}% from event detection errors, and ", end="")
        print(f"{dist['multiple_errors']['percentage']:.0f}% from multiple simultaneous errors.")
        print()
        
    def generate_latex_error_table(self, summary: Dict[str, Any], output_file: Path):
        """Generate LaTeX table for error analysis"""
        dist = summary['error_distribution']
        
        latex_lines = []
        latex_lines.append("\\begin{table}[ht]")
        latex_lines.append("\\centering")
        latex_lines.append("\\caption{Event-Level Error Analysis for DeBERTa-CRF}")
        latex_lines.append("\\label{tab:error_analysis}")
        latex_lines.append("\\begin{tabular}{lcc}")
        latex_lines.append("\\toprule")
        latex_lines.append("\\textbf{Error Category} & \\textbf{Count} & \\textbf{\\%} \\\\")
        latex_lines.append("\\midrule")
        
        # Subject errors
        latex_lines.append(f"Subject Errors & {dist['subject_errors']['count']} & "
                         f"{dist['subject_errors']['percentage']:.1f}\\% \\\\")
        latex_lines.append(f"\\quad Missing & {dist['subject_errors']['breakdown']['missing']} & \\\\")
        latex_lines.append(f"\\quad Boundary & {dist['subject_errors']['breakdown']['boundary']} & \\\\")
        latex_lines.append(f"\\quad Incorrect & {dist['subject_errors']['breakdown']['incorrect']} & \\\\")
        
        # Participant errors
        latex_lines.append(f"Participant-Vote Errors & {dist['participant_errors']['count']} & "
                         f"{dist['participant_errors']['percentage']:.1f}\\% \\\\")
        latex_lines.append(f"\\quad Missing & {dist['participant_errors']['breakdown']['missing']} & \\\\")
        latex_lines.append(f"\\quad Extra & {dist['participant_errors']['breakdown']['extra']} & \\\\")
        latex_lines.append(f"\\quad Position Wrong & {dist['participant_errors']['breakdown']['position_wrong']} & \\\\")
        
        # Counting errors
        latex_lines.append(f"Counting Errors & {dist['counting_errors']['count']} & "
                         f"{dist['counting_errors']['percentage']:.1f}\\% \\\\")
        
        # Voting expression errors
        latex_lines.append(f"Voting Expression Errors & {dist['voting_expression_errors']['count']} & "
                         f"{dist['voting_expression_errors']['percentage']:.1f}\\% \\\\")
        
        # Event detection errors
        latex_lines.append(f"Event Detection Errors & {dist['event_detection_errors']['count']} & "
                         f"{dist['event_detection_errors']['percentage']:.1f}\\% \\\\")
        
        # Multiple errors
        latex_lines.append(f"Multiple Simultaneous Errors & {dist['multiple_errors']['count']} & "
                         f"{dist['multiple_errors']['percentage']:.1f}\\% \\\\")
        
        latex_lines.append("\\midrule")
        latex_lines.append(f"\\textbf{{Total Failed Events}} & \\textbf{{{summary['summary']['failed_events']}}} & "
                         f"\\textbf{{100.0\\%}} \\\\")
        latex_lines.append("\\bottomrule")
        latex_lines.append("\\end{tabular}")
        latex_lines.append("\\end{table}")
        
        latex_file = output_file.parent / (output_file.stem + '_table.tex')
        latex_file.write_text('\n'.join(latex_lines))
        print(f"✓ Created LaTeX error table: {latex_file}")


def load_jsonl(file_path: Path) -> List[Dict]:
    """Load JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_json(file_path: Path) -> List[Dict]:
    """Load JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Handle both list and dict formats
    if isinstance(data, list):
        return data
    return data


def convert_bio_to_events(predictions: List[Dict]) -> List[Dict]:
    """Convert BIO predictions to structured events"""
    if not EVENT_CONSTRUCTOR_AVAILABLE:
        raise ImportError("EventConstructor not available. Cannot convert BIO to events.")

    print("Converting BIO predictions to structured events...")
    constructor = EventConstructor()
    event_predictions = []

    for pred in predictions:
        example_id = pred.get('id')
        tokens = pred.get('tokens', [])
        pred_labels = pred.get('pred_labels', [])

        # Construct event from predicted labels
        predicted_event = constructor.construct_event(
            tokens=tokens,
            labels=pred_labels,
            example_id=example_id
        )

        event_predictions.append(predicted_event)

    print(f"  Converted {len(event_predictions)} predictions to events")
    return event_predictions


def detect_prediction_format(predictions: List[Dict]) -> str:
    """Detect whether predictions are in BIO or event format"""
    if not predictions:
        return "unknown"

    first_pred = predictions[0]

    # Check for BIO format (has tokens and labels)
    if 'tokens' in first_pred and ('pred_labels' in first_pred or 'labels' in first_pred):
        return "bio"

    # Check for event format (has event and has_voting_event)
    if 'event' in first_pred and 'has_voting_event' in first_pred:
        return "event"

    return "unknown"


def main():
    parser = argparse.ArgumentParser(description='Analyze event extraction errors')
    parser.add_argument('--predictions', required=True, help='Path to predictions.jsonl file')
    parser.add_argument('--gold', required=True, help='Path to gold test.json file')
    parser.add_argument('--output', help='Path to save detailed JSON output (optional)')
    parser.add_argument('--examples', type=int, default=5, help='Number of examples to include per error type')

    args = parser.parse_args()

    # Load data
    print(f"Loading predictions from {args.predictions}...")
    predictions = load_jsonl(Path(args.predictions))
    print(f"  Loaded {len(predictions)} predictions")

    # Detect format and convert if necessary
    pred_format = detect_prediction_format(predictions)
    print(f"  Detected prediction format: {pred_format}")

    if pred_format == "bio":
        if not EVENT_CONSTRUCTOR_AVAILABLE:
            print("ERROR: Predictions are in BIO format but EventConstructor is not available.")
            print("Please ensure src/utils/event_constructor.py is accessible.")
            return 1
        predictions = convert_bio_to_events(predictions)
    elif pred_format == "unknown":
        print("WARNING: Could not detect prediction format. Assuming event format.")

    print(f"Loading gold data from {args.gold}...")
    gold_data = load_json(Path(args.gold))
    print(f"  Loaded {len(gold_data)} gold examples")
    print()

    # Analyze
    analyzer = EventErrorAnalyzer()
    summary = analyzer.analyze_predictions(predictions, gold_data)

    # Print summary
    analyzer.print_summary(summary)

    # Save detailed output if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Detailed analysis saved to: {output_path}")
        
        # Generate LaTeX table
        analyzer.generate_latex_error_table(summary, output_path)


if __name__ == '__main__':
    main()
