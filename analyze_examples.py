"""
Analyze train.jsonl to find best examples for few-shot learning
"""
import json
from collections import defaultdict
from typing import Dict, List, Set

def extract_entities_from_labels(labels: List[str]) -> Set[str]:
    """Extract unique entity types from BIO labels"""
    entities = set()
    for label in labels:
        if label != 'O' and not label.startswith('I-'):
            # Extract entity type from B- prefix
            entity_type = label.replace('B-', '')
            entities.add(entity_type)
    return entities

def analyze_train_data(filepath: str):
    """Analyze train.jsonl for municipalities and entity distribution"""

    municipalities = defaultdict(list)
    entity_coverage = defaultdict(int)

    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                data = json.loads(line)

                muni = data.get('municipality', 'Unknown')
                entities_in_example = extract_entities_from_labels(data['labels'])

                # Track entity coverage
                for entity in entities_in_example:
                    entity_coverage[entity] += 1

                # Store example with metadata
                municipalities[muni].append({
                    'id': data['id'],
                    'text': data['text'][:200] + '...' if len(data['text']) > 200 else data['text'],
                    'tokens': data['tokens'],
                    'labels': data['labels'],
                    'entities': entities_in_example,
                    'line_num': line_num
                })

    return municipalities, entity_coverage

def find_best_examples(municipalities: Dict, target_count: int = 10):
    """Find the best examples covering all entity types"""

    # All entity types we need to cover
    all_entities = {
        'VOTING', 'SUBJECT', 'VOTER-FAVOR', 'VOTER-AGAINST',
        'VOTER-ABSTENTION', 'VOTER-ABSENT', 'COUNTING-MAJORITY', 'COUNTING-UNANIMITY'
    }

    print("\n" + "="*80)
    print("MUNICIPALITY DISTRIBUTION")
    print("="*80)
    for muni, examples in sorted(municipalities.items()):
        print(f"\n{muni}: {len(examples)} examples")

    print("\n" + "="*80)
    print("FINDING BEST EXAMPLES (covering all entity types)")
    print("="*80)

    selected_examples = []
    covered_entities = set()

    # First pass: select one example per municipality that covers rare entities
    rare_entities = ['VOTER-AGAINST', 'VOTER-ABSENT']

    for muni in sorted(municipalities.keys()):
        examples = municipalities[muni]

        # Prioritize examples with rare entities
        best_example = None
        best_score = -1

        for example in examples:
            # Score based on:
            # 1. Has rare entities (+10 points each)
            # 2. Has uncovered entities (+5 points each)
            # 3. Total unique entities (+1 point each)
            # 4. Text length (prefer shorter, more focused examples)

            score = 0
            for entity in rare_entities:
                if entity in example['entities']:
                    score += 10

            for entity in example['entities']:
                if entity not in covered_entities:
                    score += 5

            score += len(example['entities'])

            # Prefer examples with reasonable length (not too long)
            text_len = len(' '.join(example['tokens']))
            if 100 < text_len < 800:
                score += 2

            if score > best_score:
                best_score = score
                best_example = example

        if best_example:
            selected_examples.append((muni, best_example))
            covered_entities.update(best_example['entities'])
            print(f"\n{muni}: {best_example['id']}")
            print(f"  Entities: {sorted(best_example['entities'])}")
            print(f"  Text length: {len(' '.join(best_example['tokens']))} chars")

    print("\n" + "="*80)
    print("COVERAGE SUMMARY")
    print("="*80)
    print(f"Selected {len(selected_examples)} examples")
    print(f"Covered entities: {sorted(covered_entities)}")
    print(f"Missing entities: {sorted(all_entities - covered_entities)}")

    # If we need more examples or missing entities, add more
    if len(selected_examples) < target_count or covered_entities != all_entities:
        missing = all_entities - covered_entities
        print(f"\nLooking for examples with missing entities: {sorted(missing)}")

        for entity_type in missing:
            for muni, examples in municipalities.items():
                for example in examples:
                    if entity_type in example['entities']:
                        # Check if we already have this example
                        if not any(ex['id'] == example['id'] for _, ex in selected_examples):
                            selected_examples.append((muni, example))
                            covered_entities.update(example['entities'])
                            print(f"\nAdding {muni}: {example['id']}")
                            print(f"  Entities: {sorted(example['entities'])}")
                            break
                if len(selected_examples) >= target_count:
                    break
            if len(selected_examples) >= target_count:
                break

    return selected_examples[:target_count]

if __name__ == '__main__':
    filepath = '/Users/josepedroevans/Desktop/VotIE/data/votie_bio/train.jsonl'

    print("Analyzing train.jsonl...")
    municipalities, entity_coverage = analyze_train_data(filepath)

    print("\n" + "="*80)
    print("ENTITY DISTRIBUTION ACROSS ALL EXAMPLES")
    print("="*80)
    for entity, count in sorted(entity_coverage.items()):
        print(f"{entity:25} : {count:5} examples")

    selected = find_best_examples(municipalities, target_count=10)

    print("\n" + "="*80)
    print("FINAL SELECTED EXAMPLES")
    print("="*80)
    for i, (muni, example) in enumerate(selected, 1):
        print(f"\n{i}. {muni} - {example['id']}")
        print(f"   Entities: {sorted(example['entities'])}")
        print(f"   Line number: {example['line_num']}")
