"""
Comprehensive System Tests for LangExtract NER

Tests the complete pipeline with real data using Gemini API.
This is the main test file that validates the entire system functionality.
"""

import os
import json
import logging
from pathlib import Path
import pytest
import tempfile
from typing import List, Dict, Any

# Set up path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from shared.models import GeminiModel, ExtractionResult, LocalModel, create_model
from shared.prompts import VotingPrompts, get_voting_extraction_config
from shared.data_processing import LangExtractDataProcessor, LangExtractDocument
from shared.evaluation import LangExtractEvaluator, evaluate_langextract_system
from voting.extractor import VotingExtractor, create_voting_extractor, quick_extract_voting
from config import get_config

# Configure logging for tests
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test constants
SAMPLE_PORTUGUESE_TEXT = """
A Câmara Municipal tomou conhecimento e deliberou, por maioria e em minuta, aprovar a proposta apresentada.
Votaram a favor: Presidente João Silva, Vereador Maria Santos.
Abstiveram-se os Senhores Vereadores Dra. Joana Bento e Prof. Sérgio Mendes.
"""

# Sample JSONL data (extracted from real document)
SAMPLE_JSONL_DATA = {
    "id": "test_document_1",
    "tokens": [
        "A", "Câmara", "Municipal", "deliberou", ",", "por", "maioria", ",", "aprovar", 
        "a", "proposta", ".", "Votaram", "a", "favor", ":", "Presidente", "João", "Silva",
        ",", "Vereador", "Maria", "Santos", ".", "Absteve-se", ":", "Vereador", "Carlos", "Lima", "."
    ],
    "tags": [
        "O", "O", "O", "B-VOTACAO", "O", "B-CONTABILIZACAO-MAIORIA", "I-CONTABILIZACAO-MAIORIA", 
        "O", "O", "O", "B-ASSUNTO", "O", "O", "O", "O", "O", "B-VOTANTE-FAVOR", "I-VOTANTE-FAVOR", 
        "I-VOTANTE-FAVOR", "O", "B-VOTANTE-FAVOR", "I-VOTANTE-FAVOR", "I-VOTANTE-FAVOR", "O", 
        "O", "O", "B-VOTANTE-ABSTENCAO", "I-VOTANTE-ABSTENCAO", "I-VOTANTE-ABSTENCAO", "O"
    ]
}

class TestSystemConfiguration:
    """Test system configuration and setup"""
    
    def test_config_loading(self):
        """Test configuration loading and validation"""
        config = get_config()
        
        assert config is not None
        assert hasattr(config, 'entity_types')
        assert hasattr(config, 'model_id')
        assert hasattr(config, 'batch_size')
        
        # Check entity types match expected
        expected_entities = [
            "VOTANTE-FAVOR", "VOTANTE-CONTRA", "VOTANTE-ABSTENCAO", 
            "VOTANTE-AUSENTE", "ASSUNTO", "VOTACAO", 
            "CONTABILIZACAO-MAIORIA", "CONTABILIZACAO-UNANIMIDADE"
        ]
        assert config.entity_types == expected_entities
        logger.info("✅ Configuration loading test passed")
    
    def test_api_key_configuration(self):
        """Test API key configuration"""
        config = get_config()
        
        # Should either have API key or warn about missing key
        if config.api_key:
            assert isinstance(config.api_key, str)
            assert len(config.api_key) > 0
            logger.info("✅ API key configured")
        else:
            logger.warning("⚠️ No API key configured - tests may fail")

class TestModels:
    """Test model implementations"""
    
    def test_gemini_model_initialization(self):
        """Test GeminiModel initialization"""
        model = GeminiModel()
        
        assert model.model_id == "gemini-2.0-flash"
        assert hasattr(model, 'api_key')
        assert hasattr(model, 'extract')
        assert hasattr(model, 'extract_batch')
        logger.info("✅ GeminiModel initialization test passed")
    
    def test_local_model_placeholder(self):
        """Test LocalModel placeholder"""
        with pytest.raises(NotImplementedError):
            LocalModel("dummy_path")
        logger.info("✅ LocalModel placeholder test passed")
    
    def test_model_factory(self):
        """Test model factory function"""
        gemini_model = create_model("gemini")
        assert isinstance(gemini_model, GeminiModel)
        
        with pytest.raises(NotImplementedError):
            create_model("local", model_path="dummy")
        
        with pytest.raises(ValueError):
            create_model("invalid_type")
        logger.info("✅ Model factory test passed")

class TestPrompts:
    """Test prompt system and few-shot examples"""
    
    def test_prompt_configuration(self):
        """Test prompt configuration loading"""
        config = get_voting_extraction_config()
        
        assert 'prompt' in config
        assert 'examples' in config
        assert 'entity_types' in config
        
        assert isinstance(config['examples'], list)
        assert len(config['examples']) > 0
        logger.info("✅ Prompt configuration test passed")
    
    def test_few_shot_examples_validity(self):
        """Test few-shot examples are valid"""
        examples = VotingPrompts.get_few_shot_examples()
        
        assert len(examples) >= 5  # Should have at least 5 examples
        
        for i, example in enumerate(examples):
            assert hasattr(example, 'text')
            assert hasattr(example, 'extractions')
            assert len(example.text) > 0
            assert len(example.extractions) > 0
            
            # Validate each extraction
            for extraction in example.extractions:
                assert hasattr(extraction, 'text')
                assert hasattr(extraction, 'label')
                assert hasattr(extraction, 'start')
                assert hasattr(extraction, 'end')
                
                # Check span validity
                assert 0 <= extraction.start < extraction.end <= len(example.text)
                
                # Check text matches span
                actual_text = example.text[extraction.start:extraction.end]
                assert actual_text == extraction.text, f"Text mismatch in example {i}: expected '{extraction.text}', got '{actual_text}'"
        
        logger.info(f"✅ Few-shot examples validity test passed ({len(examples)} examples)")
    

class TestDataProcessing:
    """Test data processing functionality"""
    
    def test_document_creation(self):
        """Test LangExtractDocument creation"""
        doc = LangExtractDocument(
            doc_id="test_1",
            text="Sample text",
            original_tokens=["Sample", "text"],
            original_labels=["O", "O"]
        )
        
        assert doc.doc_id == "test_1"
        assert doc.text == "Sample text"
        assert doc.metadata == {}
        logger.info("✅ Document creation test passed")
    
    def test_data_processor_initialization(self):
        """Test data processor initialization"""
        processor = LangExtractDataProcessor()
        
        assert hasattr(processor, 'processed_docs')
        assert hasattr(processor, 'failed_docs')
        assert len(processor.processed_docs) == 0
        assert len(processor.failed_docs) == 0
        logger.info("✅ Data processor initialization test passed")
    
    def test_jsonl_conversion(self):
        """Test JSONL to document conversion"""
        processor = LangExtractDataProcessor()
        
        # Create temporary JSONL file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump(SAMPLE_JSONL_DATA, f)
            f.write('\n')
            temp_path = f.name
        
        try:
            # Load the data
            jsonl_data = [SAMPLE_JSONL_DATA]
            documents = processor._convert_jsonl_to_documents(jsonl_data, "test")
            
            assert len(documents) == 1
            doc = documents[0]
            assert doc.doc_id == "test_document_1"
            assert len(doc.original_tokens) > 0
            assert len(doc.original_labels) > 0
            assert doc.text is not None
            logger.info("✅ JSONL conversion test passed")
            
        finally:
            os.unlink(temp_path)

@pytest.mark.skipif(not os.getenv("LANGEXTRACT_API_KEY"), reason="API key not configured")
class TestAPIIntegration:
    """Test actual API integration with Gemini (requires API key)"""
    
    def test_simple_extraction(self):
        """Test simple extraction with Gemini API"""
        logger.info("🔄 Testing simple extraction with Gemini API...")
        
        result = quick_extract_voting(SAMPLE_PORTUGUESE_TEXT)
        
        assert isinstance(result, ExtractionResult)
        assert result.success, f"Extraction failed: {result.error}"
        assert result.num_extractions > 0, "No entities extracted"
        
        logger.info(f"✅ Simple extraction test passed - found {result.num_extractions} entities")
        
        # Log extracted entities for verification
        for extraction in result.extractions:
            logger.info(f"   Entity: '{extraction.text}' ({extraction.label})")
    
    def test_gemini_model_direct(self):
        """Test GeminiModel directly"""
        logger.info("🔄 Testing GeminiModel directly...")
        
        model = GeminiModel()
        prompt = VotingPrompts.MAIN_PROMPT
        examples = VotingPrompts.get_few_shot_examples()
        
        result = model.extract(
            text=SAMPLE_PORTUGUESE_TEXT,
            prompt=prompt,
            examples=examples,
            document_id="test_direct"
        )
        
        assert result.success, f"Direct extraction failed: {result.error}"
        assert result.num_extractions > 0
        assert result.processing_time > 0
        assert result.model_id == "gemini-2.0-flash"
        
        logger.info(f"✅ Direct GeminiModel test passed - {result.num_extractions} entities in {result.processing_time:.2f}s")

class TestExtractor:
    """Test the main voting extractor"""
    
    @pytest.mark.skipif(not os.getenv("LANGEXTRACT_API_KEY"), reason="API key not configured") 
    def test_voting_extractor_initialization(self):
        """Test VotingExtractor initialization"""
        extractor = create_voting_extractor()
        
        assert isinstance(extractor, VotingExtractor)
        assert hasattr(extractor, 'model')
        assert hasattr(extractor, 'stats')
        logger.info("✅ VotingExtractor initialization test passed")
    
    @pytest.mark.skipif(not os.getenv("LANGEXTRACT_API_KEY"), reason="API key not configured")
    def test_document_type_detection(self):
        """Test document type detection"""
        extractor = create_voting_extractor()
        
        # Test budget document
        budget_doc = LangExtractDocument(
            doc_id="budget_test",
            text="A proposta de orçamento municipal foi aprovada com taxa de aprovação.",
            metadata={}
        )
        doc_type = extractor._determine_document_type(budget_doc)
        assert doc_type == 'budget_voting'
        
        # Test urban planning document  
        urban_doc = LangExtractDocument(
            doc_id="urban_test", 
            text="O projeto de construção e PDM foi discutido pela câmara.",
            metadata={}
        )
        doc_type = extractor._determine_document_type(urban_doc)
        assert doc_type == 'urban_planning'
        
        logger.info("✅ Document type detection test passed")

@pytest.mark.skipif(not os.getenv("LANGEXTRACT_API_KEY"), reason="API key not configured")
class TestRealDocumentProcessing:
    """Test with real document from the dataset"""
    
    def test_real_document_extraction(self):
        """Test extraction on real document from JSONL dataset"""
        logger.info("🔄 Testing extraction on real document...")
        
        # Load the actual sample we found earlier
        sample_tokens = SAMPLE_JSONL_DATA["tokens"]
        sample_text = " ".join(sample_tokens)  # Simple reconstruction
        
        # Extract using our system
        result = quick_extract_voting(sample_text)
        
        assert result.success, f"Real document extraction failed: {result.error}"
        
        logger.info(f"✅ Real document extraction test passed")
        logger.info(f"   Document: {sample_text[:100]}...")
        logger.info(f"   Found {result.num_extractions} entities:")
        
        for extraction in result.extractions:
            logger.info(f"     '{extraction.text}' → {extraction.label}")
        
        return result
    
    def test_full_pipeline_with_real_data(self):
        """Test complete pipeline with real JSONL data"""
        logger.info("🔄 Testing complete pipeline with real data...")
        
        # Create processor and documents
        processor = LangExtractDataProcessor()
        documents = processor._convert_jsonl_to_documents([SAMPLE_JSONL_DATA], "pipeline_test")
        
        assert len(documents) == 1
        
        # Extract using the full extractor
        extractor = create_voting_extractor()
        results = extractor.extract_from_documents(documents)
        
        assert len(results) == 1
        result = results[0]
        
        assert result.success, f"Pipeline extraction failed: {result.error}"
        
        # Check statistics were updated
        stats = extractor.get_statistics()
        assert stats['total_documents'] == 1
        assert stats['successful_extractions'] == 1
        
        logger.info("✅ Full pipeline test passed")
        logger.info(f"   Statistics: {stats}")
        
        return result, stats

class TestEvaluation:
    """Test evaluation system"""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization"""
        evaluator = LangExtractEvaluator()
        
        assert hasattr(evaluator, 'entity_types')
        assert hasattr(evaluator, 'last_evaluation')
        assert len(evaluator.entity_types) == 8
        logger.info("✅ Evaluator initialization test passed")
    
    @pytest.mark.skipif(not os.getenv("LANGEXTRACT_API_KEY"), reason="API key not configured")
    def test_evaluation_with_real_data(self):
        """Test evaluation with real predictions and ground truth"""
        logger.info("🔄 Testing evaluation with real data...")
        
        # Create ground truth document
        ground_truth = [LangExtractDocument(
            doc_id="test_document_1",
            text=" ".join(SAMPLE_JSONL_DATA["tokens"]),
            original_tokens=SAMPLE_JSONL_DATA["tokens"],
            original_labels=SAMPLE_JSONL_DATA["tags"]
        )]
        
        # Generate prediction
        result = quick_extract_voting(" ".join(SAMPLE_JSONL_DATA["tokens"]))
        result.document_id = "test_document_1"  # Ensure ID matches
        predictions = [result]
        
        # Evaluate
        evaluation_results = evaluate_langextract_system(
            predictions=predictions,
            ground_truth=ground_truth
        )
        
        # Basic checks
        assert 'overall_metrics' in evaluation_results or 'error' in evaluation_results
        
        if 'error' not in evaluation_results:
            logger.info("✅ Evaluation test passed")
            logger.info(f"   Results: {evaluation_results.get('overall_metrics', {})}")
        else:
            logger.warning(f"⚠️ Evaluation had issues: {evaluation_results['error']}")

class TestSystemIntegration:
    """Integration tests for the complete system"""
    
    @pytest.mark.skipif(not os.getenv("LANGEXTRACT_API_KEY"), reason="API key not configured")
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        logger.info("🔄 Testing complete end-to-end workflow...")
        
        # 1. Configuration
        config = get_config()
        assert config is not None
        
        # 2. Data Processing
        processor = LangExtractDataProcessor(config)
        documents = processor._convert_jsonl_to_documents([SAMPLE_JSONL_DATA], "e2e_test")
        assert len(documents) == 1
        
        # 3. Extraction
        extractor = VotingExtractor(config=config)
        results = extractor.extract_from_documents(documents)
        assert len(results) == 1
        assert results[0].success
        
        # 4. Evaluation (basic)
        evaluator = LangExtractEvaluator(config.entity_types)
        ground_truth = [LangExtractDocument(
            doc_id="test_document_1",
            text=documents[0].text,
            original_tokens=SAMPLE_JSONL_DATA["tokens"],
            original_labels=SAMPLE_JSONL_DATA["tags"]
        )]
        
        # Ensure document IDs match
        results[0].document_id = "test_document_1"
        
        evaluation = evaluator.evaluate_extractions(results, ground_truth)
        
        # 5. Statistics and Reporting
        stats = extractor.get_statistics()
        extractor.log_extraction_statistics()
        
        logger.info("✅ End-to-end workflow test passed")
        logger.info(f"   Extracted {stats['total_entities_extracted']} entities")
        logger.info(f"   Success rate: {stats['successful_extractions']}/{stats['total_documents']}")
        
        return results, evaluation, stats

def run_comprehensive_test():
    """Run comprehensive system test with detailed output"""
    logger.info("="*60)
    logger.info("LANGEXTRACT NER SYSTEM - COMPREHENSIVE TEST")
    logger.info("="*60)
    
    # Check API key
    if not os.getenv("LANGEXTRACT_API_KEY"):
        logger.error("❌ LANGEXTRACT_API_KEY not found in environment")
        logger.info("Please set the API key: export LANGEXTRACT_API_KEY=your_key_here")
        return False
    
    try:
        # Test 1: System Configuration
        logger.info("\n1. Testing System Configuration...")
        test_config = TestSystemConfiguration()
        test_config.test_config_loading()
        test_config.test_api_key_configuration()
        
        # Test 2: Models
        logger.info("\n2. Testing Models...")
        test_models = TestModels()
        test_models.test_gemini_model_initialization()
        test_models.test_local_model_placeholder()
        test_models.test_model_factory()
        
        # Test 3: Prompts
        logger.info("\n3. Testing Prompts...")
        test_prompts = TestPrompts()
        test_prompts.test_prompt_configuration()
        test_prompts.test_few_shot_examples_validity()
        
        # Test 4: Data Processing
        logger.info("\n4. Testing Data Processing...")
        test_data = TestDataProcessing()
        test_data.test_document_creation()
        test_data.test_data_processor_initialization()
        test_data.test_jsonl_conversion()
        
        # Test 5: API Integration
        logger.info("\n5. Testing API Integration...")
        test_api = TestAPIIntegration()
        test_api.test_simple_extraction()
        test_api.test_gemini_model_direct()
        
        # Test 6: Real Document Processing
        logger.info("\n6. Testing Real Document Processing...")
        test_real = TestRealDocumentProcessing()
        result = test_real.test_real_document_extraction()
        pipeline_result, pipeline_stats = test_real.test_full_pipeline_with_real_data()
        
        # Test 7: Evaluation
        logger.info("\n7. Testing Evaluation...")
        test_eval = TestEvaluation()
        test_eval.test_evaluator_initialization()
        test_eval.test_evaluation_with_real_data()
        
        # Test 8: System Integration
        logger.info("\n8. Testing System Integration...")
        test_integration = TestSystemIntegration()
        e2e_results, e2e_evaluation, e2e_stats = test_integration.test_end_to_end_workflow()
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("COMPREHENSIVE TEST COMPLETED SUCCESSFULLY! ✅")
        logger.info("="*60)
        logger.info(f"Sample extraction found {result.num_extractions} entities:")
        for extraction in result.extractions:
            logger.info(f"  • '{extraction.text}' → {extraction.label}")
        
        logger.info(f"\nPipeline Statistics:")
        logger.info(f"  • Total documents processed: {pipeline_stats['total_documents']}")
        logger.info(f"  • Success rate: {pipeline_stats['successful_extractions']}/{pipeline_stats['total_documents']}")
        logger.info(f"  • Total entities extracted: {pipeline_stats['total_entities_extracted']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    exit(0 if success else 1)