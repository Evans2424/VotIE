"""
Portuguese Voting Span Extraction Prompts for Gemini 2.0 Flash

Converts event-based examples to span-based extraction format.
Uses the same curated examples as the event extraction approach.
"""

from typing import List, Dict, Any
import langextract as lx
import logging
import warnings
import json
from pathlib import Path

# Suppress alignment warnings
warnings.filterwarnings("ignore", message=".*Prompt alignment: non-exact match.*")
logging.getLogger('absl').setLevel(logging.ERROR)


# Span extraction prompt (English, for consistency with event extraction)
PROMPT = """
Extract all voting-related entities from Portuguese municipal meeting minutes.

Entity types to identify:
- VOTER-FAVOR: Person or entity who voted in favor
- VOTER-AGAINST: Person or entity who voted against
- VOTER-ABSTENTION: Person or entity who abstained
- VOTER-ABSENT: Person or entity absent from the vote
- SUBJECT: Subject or matter being voted on (nouns/noun phrases only, WITHOUT action verbs like "approve", "ratify", "deliberate")
- VOTING: Voting action (deliberated, approved, rejected, etc.)
- COUNTING-MAJORITY: Vote counting by majority
- COUNTING-UNANIMITY: Vote counting by unanimity

IMPORTANT for SUBJECT:
- Extract only the object of the vote, DO NOT include verbs
- Correct: "minutes nº 28 of 20.12.2023", "budget amendments"
- Incorrect: "approve the minutes nº 28", "ratify the budget amendments"

Context: Extract all voting-related entities from municipal council deliberations,
including voters, subjects, and voting results as they appear in the text.
Extract only entities explicitly mentioned in the text.
"""


def load_event_examples() -> List[Dict[str, Any]]:
    """
    Load curated event examples from selected_event_examples.json

    Returns:
        List of event dictionaries
    """
    examples_file = Path(__file__).parent / "selected_event_examples.json"

    if not examples_file.exists():
        logging.warning(f"Examples file not found: {examples_file}")
        return []

    with open(examples_file, 'r', encoding='utf-8') as f:
        examples = json.load(f)

    return examples


def convert_event_to_span_extractions(event_data: Dict[str, Any]) -> List[lx.data.Extraction]:
    """
    Convert event-structured data to span-based extractions

    Args:
        event_data: Event dictionary with has_voting_event and event fields

    Returns:
        List of LangExtract Extraction objects
    """
    if not event_data.get('has_voting_event'):
        return []

    event = event_data.get('event', {})
    extractions = []

    # Extract SUBJECT
    subject = event.get('subject')
    if subject:
        extractions.append(
            lx.data.Extraction(
                extraction_class="SUBJECT",
                extraction_text=subject
            )
        )

    # Extract PARTICIPANTS with positions
    participants = event.get('participants', [])
    for participant in participants:
        text = participant.get('text')
        position = participant.get('position')

        if text and position:
            # Map position to entity type
            position_map = {
                'Favor': 'VOTER-FAVOR',
                'Against': 'VOTER-AGAINST',
                'Abstention': 'VOTER-ABSTENTION',
                'Absent': 'VOTER-ABSENT'
            }

            entity_type = position_map.get(position)
            if entity_type:
                extractions.append(
                    lx.data.Extraction(
                        extraction_class=entity_type,
                        extraction_text=text
                    )
                )

    # Extract COUNTING
    counting = event.get('counting')
    if counting:
        text = counting.get('text')
        count_type = counting.get('type')

        if text and count_type:
            # Map counting type to entity type
            type_map = {
                'Majority': 'COUNTING-MAJORITY',
                'Unanimity': 'COUNTING-UNANIMITY'
            }

            entity_type = type_map.get(count_type)
            if entity_type:
                extractions.append(
                    lx.data.Extraction(
                        extraction_class=entity_type,
                        extraction_text=text
                    )
                )

    # Extract VOTING EXPRESSION
    voting_expr = event.get('voting_expression')
    if voting_expr:
        extractions.append(
            lx.data.Extraction(
                extraction_class="VOTING",
                extraction_text=voting_expr
            )
        )

    return extractions


def get_examples() -> List[lx.data.ExampleData]:
    """
    Get Portuguese few-shot examples for voting span extraction

    Converts event-based examples to span extraction format.

    Returns:
        List of LangExtract ExampleData objects
    """
    event_examples = load_event_examples()

    span_examples = []

    for event_data in event_examples:
        text = event_data.get('text', '')
        extractions = convert_event_to_span_extractions(event_data)

        if text:  # Only add if text exists
            span_examples.append(
                lx.data.ExampleData(
                    text=text,
                    extractions=extractions
                )
            )

    logging.info(f"Loaded {len(span_examples)} span extraction examples from event examples")

    return span_examples


def get_voting_extraction_config() -> Dict[str, Any]:
    """Get extraction configuration for Gemini 2.0 Flash"""

    entity_types = [
        "VOTER-FAVOR", "VOTER-AGAINST", "VOTER-ABSTENTION", "VOTER-ABSENT",
        "SUBJECT", "VOTING", "COUNTING-MAJORITY", "COUNTING-UNANIMITY"
    ]

    return {
        "prompt": PROMPT,
        "examples": get_examples(),
        "entity_types": entity_types,
        "language": "pt"
    }
