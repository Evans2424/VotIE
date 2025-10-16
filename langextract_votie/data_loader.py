"""
Unified JSONL Data Loader for LangExtract NER System

Simple, focused data loader for Portuguese and English municipal voting documents.
Only handles JSONL format with municipality-level grouping.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VotingDocument:
    """Document structure for JSONL voting data"""
    id: str
    text: str
    entities: List[Dict[str, Any]]
    metadata: Dict[str, Any]

    @property
    def municipality(self) -> str:
        """Get municipality from metadata"""
        return self.metadata.get('municipality', 'Unknown')

    def has_entities(self) -> bool:
        """Check if document has any entities"""
        return len(self.entities) > 0


class VotingDataLoader:
    """Unified loader for JSONL voting data"""

    def __init__(self, data_root: Path):
        """
        Initialize loader

        Args:
            data_root: Path to vote_ner data directory
        """
        self.data_root = Path(data_root)
        self.cached_data = {}

    def get_voting_data_path(self, language: str) -> Path:
        """Get voting data path for language"""
        lang_suffix = f"{language}_voting_no_faker"
        return self.data_root / "voting" / "jsonl_data" / "voting" / lang_suffix

    def load_municipality_data(self, language: str, municipality: str) -> List[VotingDocument]:
        """
        Load documents for a specific municipality

        Args:
            language: Language code ("pt" or "en")
            municipality: Municipality name

        Returns:
            List of VotingDocument objects
        """
        cache_key = f"{language}_{municipality}"
        if cache_key in self.cached_data:
            return self.cached_data[cache_key]

        voting_data_dir = self.get_voting_data_path(language)
        jsonl_file = voting_data_dir / f"{municipality}_voting_dataset.jsonl"

        if not jsonl_file.exists():
            logger.warning(f"Municipality data not found: {jsonl_file}")
            return []

        logger.info(f"Loading {municipality} data from {jsonl_file}")

        documents = []
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            data = json.loads(line)
                            doc = VotingDocument(
                                id=data['id'],
                                text=data['text'],
                                entities=data.get('entities', []),
                                metadata=data.get('metadata', {})
                            )
                            documents.append(doc)
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.warning(f"Error parsing line {line_num} in {jsonl_file}: {e}")

            self.cached_data[cache_key] = documents
            logger.info(f"Loaded {len(documents)} documents from {municipality}")
            return documents

        except Exception as e:
            logger.error(f"Error loading {jsonl_file}: {e}")
            return []

    def get_available_municipalities(self, language: str) -> List[str]:
        """
        Get list of available municipalities for a language

        Args:
            language: Language code ("pt" or "en")

        Returns:
            List of municipality names
        """
        voting_data_dir = self.get_voting_data_path(language)

        if not voting_data_dir.exists():
            logger.warning(f"Voting data directory not found: {voting_data_dir}")
            return []

        municipalities = []
        for jsonl_file in voting_data_dir.glob("*_voting_dataset.jsonl"):
            municipality_name = jsonl_file.stem.replace("_voting_dataset", "")
            municipalities.append(municipality_name)

        return sorted(municipalities)

    def load_all_municipalities(self, language: str) -> Dict[str, List[VotingDocument]]:
        """
        Load all municipalities for a language

        Args:
            language: Language code ("pt" or "en")

        Returns:
            Dictionary mapping municipality names to document lists
        """
        municipalities = self.get_available_municipalities(language)

        result = {}
        for municipality in municipalities:
            docs = self.load_municipality_data(language, municipality)
            if docs:
                result[municipality] = docs

        total_docs = sum(len(docs) for docs in result.values())
        logger.info(f"Loaded {total_docs} documents across {len(result)} municipalities for {language}")

        return result

    def get_statistics(self, language: str) -> Dict[str, Any]:
        """
        Get statistics about the data

        Args:
            language: Language code ("pt" or "en")

        Returns:
            Statistics dictionary
        """
        all_data = self.load_all_municipalities(language)

        total_docs = 0
        total_entities = 0
        entity_type_counts = {}
        docs_with_entities = 0

        for municipality, docs in all_data.items():
            total_docs += len(docs)
            for doc in docs:
                if doc.has_entities():
                    docs_with_entities += 1
                for entity in doc.entities:
                    total_entities += 1
                    entity_type = entity.get('label', 'UNKNOWN')
                    entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1

        return {
            'language': language,
            'total_documents': total_docs,
            'total_entities': total_entities,
            'documents_with_entities': docs_with_entities,
            'coverage_percentage': (docs_with_entities / total_docs * 100) if total_docs > 0 else 0,
            'municipalities': len(all_data),
            'entity_type_distribution': entity_type_counts,
            'avg_entities_per_doc': total_entities / total_docs if total_docs > 0 else 0
        }


def create_data_loader(data_root: Optional[Path] = None) -> VotingDataLoader:
    """
    Create a VotingDataLoader instance

    Args:
        data_root: Optional custom data root path

    Returns:
        VotingDataLoader instance
    """
    if data_root is None:
        # Default to project structure
        script_dir = Path(__file__).parent
        data_root = script_dir.parent / "data"

    return VotingDataLoader(data_root)


# Interactive selection functions
def select_municipalities_interactive(loader: VotingDataLoader, language: str) -> List[str]:
    """
    Interactive municipality selection

    Args:
        loader: VotingDataLoader instance
        language: Language code

    Returns:
        List of selected municipality names
    """
    municipalities = loader.get_available_municipalities(language)

    if not municipalities:
        print(f"❌ No municipalities available for language: {language}")
        return []

    print(f"\n🏛️ Available Municipalities for {language.upper()} ({len(municipalities)} total):")
    for i, municipality in enumerate(municipalities, 1):
        docs = loader.load_municipality_data(language, municipality)
        print(f"   {i:2d}. {municipality} ({len(docs)} documents)")

    while True:
        print(f"\n📋 Select municipalities (1-{len(municipalities)}):")
        print(f"   • Numbers: 1,3,5 or ranges: 1-3")
        print(f"   • Type 'all' for all municipalities")

        user_input = input("Municipality selection: ").strip()

        if not user_input:
            continue

        try:
            if user_input.lower() == 'all':
                return municipalities

            # Parse selection
            selected_indices = set()
            parts = user_input.replace(' ', '').split(',')

            for part in parts:
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    selected_indices.update(range(start, end + 1))
                else:
                    selected_indices.add(int(part))

            # Validate and collect
            selected_municipalities = []
            for i in selected_indices:
                if 1 <= i <= len(municipalities):
                    selected_municipalities.append(municipalities[i-1])
                else:
                    print(f"❌ Number {i} is out of range (1-{len(municipalities)})")
                    break
            else:
                if selected_municipalities:
                    print(f"✅ Selected: {', '.join(selected_municipalities)}")
                    return selected_municipalities
                else:
                    print("❌ No valid municipalities selected")

        except ValueError:
            print("❌ Invalid format. Use numbers, ranges (1-3), or 'all'")


def group_documents_by_base_id(docs: List[VotingDocument]) -> Dict[str, List[VotingDocument]]:
    """
    Group document segments by their base document ID

    Args:
        docs: List of VotingDocument objects

    Returns:
        Dictionary mapping base document IDs to lists of segments
    """
    grouped = {}

    for doc in docs:
        # Extract base document ID by removing the segment suffix
        # e.g., "Alandroal_cm_021_2024-09-11_1" -> "Alandroal_cm_021_2024-09-11"
        doc_id = doc.id
        if '_' in doc_id and doc_id.split('_')[-1].isdigit():
            base_id = '_'.join(doc_id.split('_')[:-1])
        else:
            base_id = doc_id

        if base_id not in grouped:
            grouped[base_id] = []
        grouped[base_id].append(doc)

    return grouped


def select_documents_interactive(loader: VotingDataLoader, language: str,
                               municipalities: List[str]) -> List[VotingDocument]:
    """
    Interactive document selection from municipalities (grouped by complete documents)

    Args:
        loader: VotingDataLoader instance
        language: Language code
        municipalities: Selected municipality names

    Returns:
        List of selected VotingDocument objects
    """
    all_selected_docs = []

    for municipality in municipalities:
        docs = loader.load_municipality_data(language, municipality)

        # Group documents by base ID (complete documents)
        grouped_docs = group_documents_by_base_id(docs)

        print(f"\n📄 {municipality} Complete Documents ({len(grouped_docs)} unique documents, {len(docs)} total segments):")

        # Show preview of complete documents
        document_list = list(grouped_docs.items())
        for i, (base_id, segments) in enumerate(document_list[:15], 1):
            total_entities = sum(len(seg.entities) for seg in segments)
            # Show preview from first segment
            first_segment = segments[0]
            preview = first_segment.text[:80] + "..." if len(first_segment.text) > 80 else first_segment.text
            print(f"   {i:2d}. {base_id} ({len(segments)} segments, {total_entities} entities)")
            print(f"       Preview: {preview}")

        if len(grouped_docs) > 15:
            print(f"   ... and {len(grouped_docs) - 15} more complete documents")

        while True:
            print(f"\n📋 Select complete documents from {municipality}:")
            print(f"   • Numbers: 1,5,10 or ranges: 1-20")
            print(f"   • Type 'all' for all {len(grouped_docs)} complete documents")
            print(f"   • Type 'with-entities' for documents with entities only")
            print(f"   • Type 'skip' to skip this municipality")

            doc_input = input(f"Document selection for {municipality}: ").strip()

            if not doc_input:
                continue

            if doc_input.lower() == 'skip':
                break
            elif doc_input.lower() == 'all':
                all_selected_docs.extend(docs)
                print(f"✅ Added all {len(docs)} segments from all {len(grouped_docs)} complete documents")
                break
            elif doc_input.lower() == 'with-entities':
                docs_with_entities = [doc for doc in docs if doc.has_entities()]
                all_selected_docs.extend(docs_with_entities)
                print(f"✅ Added {len(docs_with_entities)} segments with entities from {municipality}")
                break
            else:
                # Parse complete document selection
                try:
                    selected_indices = set()
                    parts = doc_input.replace(' ', '').split(',')

                    for part in parts:
                        if '-' in part:
                            start, end = map(int, part.split('-'))
                            selected_indices.update(range(start, end + 1))
                        else:
                            selected_indices.add(int(part))

                    # Validate and collect complete documents (all segments)
                    selected_docs = []
                    selected_doc_names = []
                    for i in selected_indices:
                        if 1 <= i <= len(document_list):
                            base_id, segments = document_list[i-1]
                            selected_docs.extend(segments)  # Add all segments for this document
                            selected_doc_names.append(base_id)
                        else:
                            print(f"❌ Number {i} is out of range (1-{len(document_list)})")
                            break
                    else:
                        if selected_docs:
                            all_selected_docs.extend(selected_docs)
                            print(f"✅ Added {len(selected_docs)} segments from {len(selected_doc_names)} complete documents:")
                            for doc_name in selected_doc_names:
                                print(f"     - {doc_name}")
                            break
                        else:
                            print("❌ No valid documents selected")

                except ValueError:
                    print("❌ Invalid format. Use numbers, ranges, 'all', 'with-entities', or 'skip'")

    return all_selected_docs