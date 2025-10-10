# Dynamic Few-Shot Example Selection Implementation Plan

## Overview

This plan implements **similarity-based dynamic example selection** for the langextract_ner module, replacing the current static 5-example approach with an intelligent system that selects the most relevant examples for each input text.

## Current Problem

The system currently uses the same 5 static examples for all predictions, which is suboptimal because different input texts may benefit from different example sets.

## Proposed Solution

**Dynamic Selection System** that:
- Uses semantic similarity to choose most relevant examples
- Maintains a pool of 20 diverse examples (vs. current 6)
- Selects top-5 most similar examples for each prediction
- Remains scientifically sound and easy to explain
- Maintains full compatibility with existing evaluation framework

## Technical Architecture

### Core Components

1. **DynamicExampleSelector**: Main similarity-based selection class
2. **ExamplePool**: Repository of 20 diverse Portuguese municipal examples
3. **SimilarityCalculator**: Semantic similarity using sentence transformers
4. **CacheManager**: Efficient embedding caching system
5. **EvaluationComparator**: A/B testing framework

### Similarity Methodology

```python
class DynamicExampleSelector:
    """
    Semantic similarity-based example selection for Portuguese NER.
    
    Process:
    1. Encode input text using multilingual sentence transformer
    2. Calculate cosine similarity with all 20 examples
    3. Select top-5 most similar examples
    4. Apply diversity filtering to avoid redundancy
    """
    
    def __init__(self, 
                 model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 top_k: int = 5,
                 diversity_threshold: float = 0.8):
        self.encoder = SentenceTransformer(model_name)
        self.top_k = top_k
        self.example_pool = self._load_20_examples()
        self.cached_embeddings = {}
    
    def select_examples(self, input_text: str) -> List[lx.data.ExampleData]:
        """Select 5 most relevant examples from pool of 20"""
        input_embedding = self.encoder.encode([input_text])[0]
        similarities = self._calculate_similarities(input_embedding)
        return self._select_diverse_examples(similarities)
```

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1)

**Files to Create/Modify:**
- `src/vote_ner/langextract_ner/shared/dynamic_selection.py` - Main selector class
- `src/vote_ner/langextract_ner/shared/expanded_prompts.py` - 20 example pool
- `src/vote_ner/langextract_ner/shared/models.py` - Integration with GeminiModel
- `src/vote_ner/langextract_ner/shared/citilink_data_adapter.py` - **NEW: Citilink dataset adapter**
- `src/vote_ner/langextract_ner/shared/processed_data_loader.py` - **MODIFIED: Extended for Citilink support**

**Tasks:**
1. Implement DynamicExampleSelector class
2. **Citilink Dataset Integration:**
   - Create CitilinkDataAdapter for handling Citilink dataset format
   - Support both Portuguese and English versions
   - Adapt segmented text format (`.txt` files) and metadata (`.json` files)
   - Ensure compatibility with existing ProcessedDataLoader system
3. Curate 20 diverse examples covering:
   - All voting patterns (unanimous, majority, rejection, abstention)
   - All entity types (VOTACAO, ASSUNTO, VOTANTE-*, CONTABILIZACAO-*)
   - Different municipalities and linguistic variations
   - **Both Portuguese and English examples from Citilink dataset**
   - Edge cases and boundary conditions
4. Integrate with GeminiModel class (backward compatible)

### Phase 2: Evaluation Framework (Week 2)

**Files to Create/Modify:**
- `src/vote_ner/langextract_ner/shared/evaluation_comparison.py` - A/B testing
- `src/vote_ner/langextract_ner/shared/evaluation.py` - Extended evaluator
- `src/vote_ner/langextract_ner/shared/multilingual_evaluation.py` - **NEW: Language-specific evaluation**

**Tasks:**
1. Implement A/B testing infrastructure
2. Compare static vs. dynamic approaches
3. Statistical significance testing
4. Integration with existing seqeval-based evaluation
5. **Multilingual Evaluation Support:**
   - Separate evaluation pipelines for Portuguese and English datasets
   - Language-specific performance metrics and comparisons
   - Cross-language similarity analysis for dynamic selection
   - Evaluation compatibility across both dataset formats (original processed_data and Citilink)

### Phase 3: Optimization & Testing (Week 3)

**Tasks:**
1. Performance optimization (embedding caching, batch processing)
2. Comprehensive testing and validation
3. Documentation and examples

## Portuguese Language Optimization

### Model Selection
- **Primary**: `paraphrase-multilingual-MiniLM-L12-v2` (384 dims, efficient)
- **Alternative**: `distiluse-base-multilingual-cased-v1` (512 dims, higher quality)
- Both proven effective for Portuguese semantic similarity

### Example Pool Strategy (20 Examples)
- **Voting Patterns**: 6 examples (unanimous, majority, rejection, abstention)
- **Entity Complexity**: 4 examples (simple vs. complex ASSUNTO entities)
- **Linguistic Variations**: 4 examples (different Portuguese formulations)
- **Multilingual Examples**: 4 examples (English versions from Citilink dataset)
- **Edge Cases**: 2 examples (boundary conditions)

## Integration Approach

### API Compatibility

```python
# Current API (still works)
model = GeminiModel()
result = model.extract(text, prompt, static_examples)

# New dynamic API (recommended)
model = GeminiModel(use_dynamic_selection=True)
result = model.extract(text, prompt)  # examples selected automatically

# Advanced configuration
selector = DynamicExampleSelector(top_k=5, diversity_threshold=0.8)
model = GeminiModel(example_selector=selector)
result = model.extract(text, prompt)
```

### Configuration

Add to `config.py`:
```python
DYNAMIC_SELECTION_CONFIG = {
    'enabled': True,
    'model_name': 'paraphrase-multilingual-MiniLM-L12-v2',
    'top_k_examples': 5,
    'diversity_threshold': 0.8,
    'cache_embeddings': True,
    'language_support': ['pt', 'en'],
    'dataset_formats': ['processed_data', 'citilink']
}

# Citilink Dataset Configuration
CITILINK_CONFIG = {
    'dataset_path': 'data/citilink_dataset',
    'ground_truth_path': 'data/citilink_dataset/ground_truth',
    'meetings_path': 'data/citilink_dataset/meetings',
    'supported_languages': ['pt', 'en'],
    'segmented_format': True
}
```

## Evaluation Methodology

### Comparison Framework
```python
class DynamicVsStaticEvaluator:
    def run_comparison(self, test_documents: List[str]) -> Dict[str, Any]:
        """
        A/B test comparing static vs dynamic selection
        
        Returns:
        - Performance metrics (F1, precision, recall)
        - Statistical significance tests  
        - Example diversity analysis
        - Computational efficiency comparison
        """
```

### Success Metrics
- **Performance**: ≥2% F1 improvement over static approach
- **Efficiency**: <100ms additional latency per query
- **Reliability**: >99% successful selection rate
- **Compatibility**: 100% backward compatibility

## Performance Expectations

### Computational Impact
- **Initial Setup**: ~2-3 seconds (model loading + caching 20 examples)
- **Per-Query Overhead**: ~50-100ms (similarity calculation + selection)
- **Memory Usage**: ~200MB additional (model + cached embeddings)
- **Expected Improvement**: 2-5% F1 score improvement

### Efficiency Optimizations
1. **Embedding Caching**: Pre-compute embeddings for all 20 examples
2. **Batch Processing**: Optimize similarity calculations
3. **Lazy Loading**: Load model only when needed
4. **Memory Management**: Efficient embedding storage

## Timeline

**Week 1: Core Development**
- Implement DynamicExampleSelector
- Create CitilinkDataAdapter for dataset integration
- Create 20-example pool (including English examples)
- Basic GeminiModel integration with language support

**Week 2: Evaluation Framework**
- A/B testing infrastructure
- Multilingual evaluation system
- Integration with existing evaluation metrics
- Statistical comparison tools for both languages

**Week 3: Optimization & Testing**
- Performance optimization and caching
- Comprehensive testing and validation (Portuguese + English)
- Cross-dataset compatibility testing
- Documentation and examples

## Files Structure

```
src/vote_ner/langextract_ner/
├── shared/
│   ├── dynamic_selection.py          # NEW: Main selector class
│   ├── expanded_prompts.py           # NEW: 20 example pool (PT + EN)
│   ├── citilink_data_adapter.py      # NEW: Citilink dataset adapter
│   ├── multilingual_evaluation.py    # NEW: Language-specific evaluation
│   ├── evaluation_comparison.py      # NEW: A/B testing
│   ├── models.py                     # MODIFIED: Integration + language support
│   ├── evaluation.py                 # MODIFIED: Extended evaluator
│   ├── processed_data_loader.py      # MODIFIED: Citilink support
│   └── config.py                     # MODIFIED: Dynamic + Citilink config
├── tests/
│   ├── test_dynamic_selection.py     # NEW: Unit tests
│   ├── test_citilink_adapter.py      # NEW: Citilink adapter tests
│   ├── test_multilingual_eval.py     # NEW: Multilingual evaluation tests
│   └── test_evaluation_comparison.py # NEW: Evaluation tests
└── DYNAMIC_SELECTION_PLAN.md         # This file
```

## Next Steps

1. **Review and approve** this implementation plan
2. **Begin Phase 1** with core infrastructure development
3. **Curate 20 examples** from existing processed data
4. **Implement and test** dynamic selection system
5. **Evaluate performance** against static baseline

## Citilink Dataset Integration

### Dataset Format Support

The system will support the **Citilink dataset format** located at `data/citilink_dataset/` with the following structure:

**Current Citilink Structure:**
```
data/citilink_dataset/
├── ground_truth/                    # Segmented text with metadata
│   ├── {Municipality}_{doc}_segmented.txt      # BIO-tagged text
│   └── {Municipality}_{doc}_segmented_metadata.json  # Segment metadata
├── meetings/                        # Raw meeting transcripts
│   └── {Municipality}_{doc}.txt
└── splits_municipality/             # Train/test splits
    ├── train/, val/, test/
    └── split_info.json
```

### Language Support Architecture

**Dual Language Processing:**
1. **Portuguese Version**: Native language processing (existing functionality)
2. **English Version**: Translated/parallel corpus processing (new functionality)

**CitilinkDataAdapter Features:**
```python
class CitilinkDataAdapter:
    """
    Adapter for Citilink dataset format with bilingual support.

    Features:
    - Load segmented text files (.txt) with BIO tags
    - Parse metadata files (.json) with segment information
    - Support both Portuguese and English versions
    - Convert to ProcessedDocument format for compatibility
    - Maintain evaluation framework compatibility
    """

    def __init__(self, dataset_path: Path, language: str = 'pt'):
        self.dataset_path = dataset_path
        self.language = language  # 'pt' or 'en'
        self.ground_truth_path = dataset_path / 'ground_truth'

    def load_document(self, doc_id: str) -> ProcessedDocument:
        """Load Citilink document and convert to ProcessedDocument format"""

    def load_all_documents(self) -> Dict[str, ProcessedDocument]:
        """Load all Citilink documents for specified language"""

    def get_language_specific_examples(self) -> List[lx.data.ExampleData]:
        """Extract examples from Citilink data for dynamic selection pool"""
```

### Integration Points

**1. ProcessedDataLoader Extension:**
- Add Citilink format detection and loading
- Maintain backward compatibility with existing processed_data format
- Support language-specific loading (`language='pt'` or `language='en'`)

**2. Dynamic Example Selection:**
- Include Citilink examples in the 20-example pool
- Language-aware similarity calculation
- Cross-language example selection capabilities

**3. Evaluation Framework:**
- Separate evaluation pipelines for Portuguese and English
- Cross-dataset evaluation (processed_data vs. Citilink)
- Language-specific performance metrics

### Multilingual Dynamic Selection

**Language-Aware Selection Strategy:**
```python
class MultilingualDynamicSelector(DynamicExampleSelector):
    """
    Extended selector with multilingual support for Citilink datasets.
    """

    def __init__(self, languages: List[str] = ['pt', 'en']):
        self.languages = languages
        self.language_pools = {}  # Separate example pools per language
        self.cross_language_similarity = True  # Enable cross-language selection

    def select_examples(self, input_text: str, target_language: str = 'pt') -> List[lx.data.ExampleData]:
        """
        Select examples with language preference:
        1. Prioritize examples in target language
        2. Include cross-language examples if beneficial
        3. Apply diversity filtering across languages
        """
```

### Separate Language Experiences

**Independent Processing Pipelines:**

1. **Portuguese Pipeline:**
   - Uses existing processed_data + Citilink Portuguese data
   - Portuguese-optimized sentence transformer models
   - Portuguese-specific evaluation metrics

2. **English Pipeline:**
   - Uses Citilink English data exclusively
   - English-optimized sentence transformer models
   - English-specific evaluation metrics

3. **Cross-Language Analysis:**
   - Compare Portuguese vs. English performance
   - Analyze cross-language similarity patterns
   - Evaluate translation quality impact on NER performance

**Configuration Support:**
```python
# Language-specific configurations
LANGUAGE_CONFIGS = {
    'pt': {
        'model_name': 'paraphrase-multilingual-MiniLM-L12-v2',
        'dataset_sources': ['processed_data', 'citilink'],
        'example_pool_size': 12  # 12 Portuguese examples
    },
    'en': {
        'model_name': 'paraphrase-MiniLM-L6-v2',  # English-optimized
        'dataset_sources': ['citilink'],
        'example_pool_size': 8   # 8 English examples
    }
}
```

This approach maintains scientific rigor while being computationally efficient and easy to understand, providing a solid foundation for improved few-shot learning in both Portuguese and English municipal voting NER with comprehensive Citilink dataset support.
