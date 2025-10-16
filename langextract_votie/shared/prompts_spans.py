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
    Load curated event examples from short_examples.json
    
    CRITICAL: Uses SHORT examples (<100 chars) to avoid LangExtract validation hanging.
    LangExtract's difflib-based validation can hang on long texts (>1000 chars).

    Returns:
        List of event dictionaries
    """
    # Use SHORT examples to avoid validation hanging
    examples_file = Path(__file__).parent / "short_examples.json"

    if not examples_file.exists():
        logging.warning(f"Examples file not found: {examples_file}")
        # Fallback to long examples if short ones don't exist
        examples_file = Path(__file__).parent / "selected_event_examples.json"
        if examples_file.exists():
            logging.warning("Using long examples - may cause slow validation")

    if not examples_file.exists():
        logging.error(f"No examples file found!")
        return []

    with open(examples_file, 'r', encoding='utf-8') as f:
        examples = json.load(f)

    return examples


def convert_event_to_span_extractions(event_data: Dict[str, Any]) -> List[lx.data.Extraction]:
    """
    Convert event-structured data to span-based extractions
    
    CRITICAL: Only includes extractions whose text actually appears in the example text.
    This prevents LangExtract from hanging on invalid examples.

    Args:
        event_data: Event dictionary with has_voting_event and event fields

    Returns:
        List of LangExtract Extraction objects (validated against text)
    """
    if not event_data.get('has_voting_event'):
        return []

    event = event_data.get('event', {})
    text = event_data.get('text', '')
    extractions = []

    # Extract SUBJECT - VALIDATE it exists in text
    subject = event.get('subject')
    if subject and subject in text:
        extractions.append(
            lx.data.Extraction(
                extraction_class="SUBJECT",
                extraction_text=subject
            )
        )
    elif subject:
        logging.debug(f"Skipping subject '{subject[:50]}...' - not found in text")

    # Extract PARTICIPANTS with positions - VALIDATE they exist in text
    participants = event.get('participants', [])
    for participant in participants:
        participant_text = participant.get('text')
        position = participant.get('position')

        if participant_text and position and participant_text in text:
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
                        extraction_text=participant_text
                    )
                )
        elif participant_text:
            logging.debug(f"Skipping participant '{participant_text[:30]}...' - not found in text")

    # Extract COUNTING (counting is a list) - VALIDATE they exist in text
    counting_list = event.get('counting', [])
    for counting in counting_list:
        counting_text = counting.get('text')
        count_type = counting.get('type')

        if counting_text and count_type and counting_text in text:
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
                        extraction_text=counting_text
                    )
                )
        elif counting_text:
            logging.debug(f"Skipping counting '{counting_text[:30]}...' - not found in text")

    # Extract VOTING EXPRESSIONS (voting_expressions is a list) - VALIDATE they exist in text
    voting_expressions = event.get('voting_expressions', [])
    for voting_expr in voting_expressions:
        if voting_expr and voting_expr in text:
            extractions.append(
                lx.data.Extraction(
                    extraction_class="VOTING",
                    extraction_text=voting_expr
                )
            )
        elif voting_expr:
            logging.debug(f"Skipping voting expr '{voting_expr[:30]}...' - not found in text")

    return extractions


def get_examples() -> List[lx.data.ExampleData]:
    """
    Get Portuguese few-shot examples for voting span extraction

    Converts event-based examples to span extraction format.
    CRITICAL: Only includes examples with valid extractions (text found in example).

    Returns:
        List of LangExtract ExampleData objects
    """
    event_examples = load_event_examples()

    span_examples = []
    total_extractions = 0
    valid_extractions = 0

    for event_data in event_examples:
        text = event_data.get('text', '')
        extractions = convert_event_to_span_extractions(event_data)
        
        total_extractions += len(event_data.get('event', {}).get('participants', []))

        # Only add examples that have text AND valid extractions
        if text and extractions:
            span_examples.append(
                lx.data.ExampleData(
                    text=text,
                    extractions=extractions
                )
            )
            valid_extractions += len(extractions)
        elif text:
            logging.warning(f"Skipping example with no valid extractions (text len: {len(text)})")

    logging.info(f"Loaded {len(span_examples)} span extraction examples from {len(event_examples)} event examples")
    logging.info(f"Valid extractions: {valid_extractions} (filtered from potential mismatches)")

    return span_examples


def get_voting_extraction_config(max_examples: int = 3) -> Dict[str, Any]:
    """
    Get extraction configuration for Gemini 2.0 Flash
    
    Args:
        max_examples: Maximum number of examples to use (default: 3)
                     Using fewer examples can improve API response time
    """

    entity_types = [
        "VOTER-FAVOR", "VOTER-AGAINST", "VOTER-ABSTENTION", "VOTER-ABSENT",
        "SUBJECT", "VOTING", "COUNTING-MAJORITY", "COUNTING-UNANIMITY"
    ]
    
    # Limit examples to avoid API timeout
    all_examples = get_examples()
    limited_examples = all_examples[:max_examples]
    
    logging.info(f"Using {len(limited_examples)} examples (max: {max_examples}, available: {len(all_examples)})")

    return {
        "prompt": PROMPT,
        "examples": limited_examples,
        "entity_types": entity_types,
        "language": "pt"
    }
