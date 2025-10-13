"""
Select 10 few-shot examples from votie_events/train.jsonl

Criteria:
- One example per municipality (if possible)
- Cover all entity types (VOTING, SUBJECT, participants in all positions, both counting types)
- Prefer shorter, clearer examples
- Prefer examples with multiple entity types
"""
import json
from collections import defaultdict
from typing import List, Dict, Any, Set

def count_entity_types(event: Dict[str, Any]) -> Set[str]:
    """Count unique entity types in an event"""
    types = set()

    if not event.get("has_voting_event"):
        return types

    event_data = event.get("event", {})

    if event_data.get("subjects"):
        types.add("SUBJECT")

    if event_data.get("voting_expressions"):
        types.add("VOTING")

    for participant in event_data.get("participants", []):
        pos = participant.get("position", "")
        types.add(f"VOTER-{pos.upper()}")

    for counting in event_data.get("counting", []):
        c_type = counting.get("type", "")
        types.add(f"COUNTING-{c_type.upper()}")

    return types

def score_example(event: Dict[str, Any], covered_types: Set[str]) -> float:
    """Score an example based on utility for few-shot learning"""
    if not event.get("has_voting_event"):
        return 0.0

    event_data = event.get("event", {})
    example_types = count_entity_types(event)

    score = 0.0

    # Bonus for rare entity types
    rare_types = {"VOTER-AGAINST", "VOTER-ABSENT"}
    for t in rare_types:
        if t in example_types:
            score += 15.0

    # Bonus for uncovered types
    for t in example_types:
        if t not in covered_types:
            score += 10.0

    # Bonus for having multiple types
    score += len(example_types) * 2.0

    # Prefer shorter texts (more focused)
    text_len = len(event.get("text", ""))
    if 100 < text_len < 800:
        score += 5.0
    elif text_len < 100:
        score += 2.0

    # Bonus for having all participant roles
    participant_count = len(event_data.get("participants", []))
    if participant_count >= 2:
        score += 3.0

    return score

def select_best_examples(filepath: str, target_count: int = 10) -> List[Dict[str, Any]]:
    """Select best examples for few-shot learning"""

    # Load all examples
    examples_by_muni = defaultdict(list)

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                example = json.loads(line)
                muni = example.get("municipality", "Unknown")
                examples_by_muni[muni].append(example)

    print(f"Loaded examples from {len(examples_by_muni)} municipalities")
    for muni, examples in examples_by_muni.items():
        print(f"  {muni}: {len(examples)} examples")

    # Select best example per municipality
    selected = []
    covered_types = set()

    # Sort municipalities by example count (prioritize those with fewer to ensure coverage)
    sorted_munis = sorted(examples_by_muni.keys(), key=lambda m: len(examples_by_muni[m]))

    for muni in sorted_munis:
        examples = examples_by_muni[muni]

        # Score each example
        best_example = None
        best_score = -1.0

        for example in examples:
            score = score_example(example, covered_types)
            if score > best_score:
                best_score = score
                best_example = example

        if best_example:
            selected.append(best_example)
            covered_types.update(count_entity_types(best_example))
            print(f"\n✓ Selected from {muni}: {best_example['id']}")
            print(f"  Entity types: {sorted(count_entity_types(best_example))}")
            print(f"  Text length: {len(best_example['text'])} chars")
            print(f"  Score: {best_score:.1f}")

    # If we need more examples, select from remaining
    while len(selected) < target_count:
        best_example = None
        best_score = -1.0
        best_muni = None

        for muni, examples in examples_by_muni.items():
            for example in examples:
                # Skip already selected
                if any(e['id'] == example['id'] for e in selected):
                    continue

                score = score_example(example, covered_types)
                if score > best_score:
                    best_score = score
                    best_example = example
                    best_muni = muni

        if best_example:
            selected.append(best_example)
            covered_types.update(count_entity_types(best_example))
            print(f"\n✓ Additional from {best_muni}: {best_example['id']}")
            print(f"  Entity types: {sorted(count_entity_types(best_example))}")
            print(f"  Score: {best_score:.1f}")
        else:
            break

    print(f"\n{'='*80}")
    print(f"FINAL SELECTION: {len(selected)} examples")
    print(f"{'='*80}")
    print(f"Covered entity types: {sorted(covered_types)}")

    # Check coverage
    all_types = {
        "VOTING", "SUBJECT",
        "VOTER-FAVOR", "VOTER-AGAINST", "VOTER-ABSTENTION", "VOTER-ABSENT",
        "COUNTING-MAJORITY", "COUNTING-UNANIMITY"
    }
    missing = all_types - covered_types
    if missing:
        print(f"⚠️  Missing types: {sorted(missing)}")
    else:
        print(f"✅ All entity types covered!")

    return selected[:target_count]

if __name__ == "__main__":
    filepath = "/Users/josepedroevans/Desktop/VotIE/data/votie_events/train.jsonl"
    selected_examples = select_best_examples(filepath, target_count=10)

    # Save selected examples
    output_path = "/Users/josepedroevans/Desktop/VotIE/selected_event_examples.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(selected_examples, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved selected examples to: {output_path}")
