"""Tests for fact extraction functionality."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from datetime import datetime

from morag_graph.models.fact import Fact, FactRelation, FactType, FactRelationType
from morag_graph.extraction.fact_extractor import FactExtractor
from morag_graph.extraction.fact_validator import FactValidator
from morag_graph.extraction.fact_graph_builder import FactGraphBuilder
from morag_graph.services.fact_extraction_service import FactExtractionService


class TestFactModel:
    """Test the Fact model."""
    
    def test_fact_creation(self):
        """Test basic fact creation."""
        fact = Fact(
            subject="Machine Learning",
            object="image classification accuracy",
            approach="convolutional neural networks",
            solution="95% accuracy on ImageNet dataset",
            source_chunk_id="chunk_123",
            source_document_id="doc_456",
            extraction_confidence=0.9,
            fact_type=FactType.RESEARCH
        )
        
        assert fact.subject == "Machine Learning"
        assert fact.object == "image classification accuracy"
        assert fact.approach == "convolutional neural networks"
        assert fact.solution == "95% accuracy on ImageNet dataset"
        assert fact.extraction_confidence == 0.9
        assert fact.fact_type == FactType.RESEARCH
        assert fact.id.startswith("fact_")
    
    def test_fact_completeness(self):
        """Test fact completeness checking."""
        # Complete fact
        complete_fact = Fact(
            subject="Python",
            object="web development",
            approach="Django framework",
            solution="rapid application development",
            source_chunk_id="chunk_1",
            source_document_id="doc_1",
            extraction_confidence=0.8,
            fact_type=FactType.PROCESS
        )
        assert complete_fact.is_complete()
        
        # Incomplete fact (no approach or solution)
        incomplete_fact = Fact(
            subject="Python",
            object="web development",
            source_chunk_id="chunk_1",
            source_document_id="doc_1",
            extraction_confidence=0.8,
            fact_type=FactType.PROCESS
        )
        assert not incomplete_fact.is_complete()
    
    def test_fact_neo4j_properties(self):
        """Test Neo4j properties conversion."""
        fact = Fact(
            subject="Test Subject",
            object="Test Object",
            approach="Test Approach",
            source_chunk_id="chunk_1",
            source_document_id="doc_1",
            extraction_confidence=0.85,
            fact_type=FactType.DEFINITION,
            keywords=["test", "example"]
        )
        
        props = fact.get_neo4j_properties()
        
        assert props["subject"] == "Test Subject"
        assert props["object"] == "Test Object"
        assert props["approach"] == "Test Approach"
        assert props["confidence"] == 0.85
        assert props["fact_type"] == FactType.DEFINITION
        assert props["keywords"] == "test,example"


class TestFactValidator:
    """Test the FactValidator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = FactValidator(min_confidence=0.3)
    
    def test_valid_fact(self):
        """Test validation of a valid fact."""
        fact = Fact(
            subject="Machine Learning algorithms",
            object="pattern recognition in large datasets",
            approach="supervised learning with neural networks",
            solution="improved accuracy and reduced processing time",
            source_chunk_id="chunk_1",
            source_document_id="doc_1",
            extraction_confidence=0.9,
            fact_type=FactType.RESEARCH
        )
        
        is_valid, issues = self.validator.validate_fact(fact)
        assert is_valid
        assert len(issues) == 0
    
    def test_invalid_fact_generic_subject(self):
        """Test validation fails for generic subject."""
        fact = Fact(
            subject="it",
            object="pattern recognition",
            approach="machine learning",
            source_chunk_id="chunk_1",
            source_document_id="doc_1",
            extraction_confidence=0.8,
            fact_type=FactType.RESEARCH
        )
        
        is_valid, issues = self.validator.validate_fact(fact)
        assert not is_valid
        assert any("Generic subject" in issue for issue in issues)
    
    def test_invalid_fact_low_confidence(self):
        """Test validation fails for low confidence."""
        fact = Fact(
            subject="Machine Learning",
            object="pattern recognition",
            approach="neural networks",
            source_chunk_id="chunk_1",
            source_document_id="doc_1",
            extraction_confidence=0.5,  # Below threshold
            fact_type=FactType.RESEARCH
        )
        
        is_valid, issues = self.validator.validate_fact(fact)
        assert not is_valid
        assert any("Confidence" in issue and "below threshold" in issue for issue in issues)
    
    def test_invalid_fact_missing_actionable_content(self):
        """Test validation fails for missing actionable content."""
        fact = Fact(
            subject="Something",
            object="something else",
            source_chunk_id="chunk_1",
            source_document_id="doc_1",
            extraction_confidence=0.8,
            fact_type=FactType.RESEARCH
        )
        
        is_valid, issues = self.validator.validate_fact(fact)
        assert not is_valid
        assert any("Missing both approach and solution" in issue for issue in issues)
    
    def test_batch_validation(self):
        """Test batch validation of multiple facts."""
        facts = [
            Fact(
                subject="Valid fact",
                object="valid content",
                approach="valid approach",
                source_chunk_id="chunk_1",
                source_document_id="doc_1",
                extraction_confidence=0.9,
                fact_type=FactType.RESEARCH
            ),
            Fact(
                subject="it",  # Invalid - generic subject
                object="invalid content",
                approach="some approach",
                source_chunk_id="chunk_2",
                source_document_id="doc_1",
                extraction_confidence=0.8,
                fact_type=FactType.RESEARCH
            )
        ]
        
        result = self.validator.validate_facts_batch(facts)
        
        assert result['total_facts'] == 2
        assert result['valid_facts'] == 1
        assert result['invalid_facts'] == 1
        assert result['validation_rate'] == 0.5


class TestFactExtractor:
    """Test the FactExtractor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock the LLM client
        self.mock_llm_client = Mock()
        self.extractor = FactExtractor(
            model_id="test-model",
            api_key="test-key",
            min_confidence=0.7
        )
        self.extractor.llm_client = self.mock_llm_client
    
    def test_preprocess_chunk(self):
        """Test text preprocessing."""
        text = """
        # Header
        
        This is **bold** text with `code` and [link](url).
        
        Multiple    spaces    should    be    normalized.
        """
        
        processed = self.extractor._preprocess_chunk(text)
        
        assert "# Header" not in processed
        assert "**bold**" not in processed
        assert "`code`" not in processed
        assert "[link](url)" not in processed
        assert "  " not in processed  # Multiple spaces normalized
        assert "bold" in processed
        assert "code" in processed
        assert "link" in processed
    
    def test_parse_llm_response(self):
        """Test parsing LLM response."""
        response = """
        Here are the extracted facts:
        
        [
          {
            "subject": "Machine Learning",
            "object": "data analysis",
            "approach": "neural networks",
            "confidence": 0.9,
            "fact_type": "research"
          }
        ]
        """
        
        candidates = self.extractor._parse_llm_response(response)
        
        assert len(candidates) == 1
        assert candidates[0]["subject"] == "Machine Learning"
        assert candidates[0]["object"] == "data analysis"
        assert candidates[0]["approach"] == "neural networks"
    
    def test_generate_keywords(self):
        """Test keyword generation."""
        fact = Fact(
            subject="Machine Learning algorithms",
            object="image classification using convolutional neural networks",
            approach="deep learning with backpropagation",
            solution="improved accuracy on computer vision tasks",
            source_chunk_id="chunk_1",
            source_document_id="doc_1",
            extraction_confidence=0.9,
            fact_type=FactType.RESEARCH
        )
        
        keywords = self.extractor._generate_fact_keywords(fact)
        
        assert "machine" in keywords
        assert "learning" in keywords
        assert "algorithms" in keywords
        assert "image" in keywords
        assert "classification" in keywords
        assert "neural" in keywords
        assert "networks" in keywords
        # Stop words should be filtered out
        assert "the" not in keywords
        assert "and" not in keywords


class TestFactRelation:
    """Test the FactRelation model."""
    
    def test_fact_relation_creation(self):
        """Test basic fact relation creation."""
        relation = FactRelation(
            source_fact_id="fact_1",
            target_fact_id="fact_2",
            relation_type=FactRelationType.SUPPORTS,
            confidence=0.8,
            context="Fact 1 provides evidence for Fact 2"
        )
        
        assert relation.source_fact_id == "fact_1"
        assert relation.target_fact_id == "fact_2"
        assert relation.relation_type == FactRelationType.SUPPORTS
        assert relation.confidence == 0.8
        assert relation.id.startswith("fact_rel_")
    
    def test_fact_relation_neo4j_properties(self):
        """Test Neo4j properties conversion."""
        relation = FactRelation(
            source_fact_id="fact_1",
            target_fact_id="fact_2",
            relation_type=FactRelationType.ELABORATES,
            confidence=0.75,
            context="Additional details"
        )
        
        props = relation.get_neo4j_properties()
        
        assert props["relation_type"] == FactRelationType.ELABORATES
        assert props["confidence"] == 0.75
        assert props["context"] == "Additional details"


@pytest.mark.asyncio
class TestFactExtractionIntegration:
    """Integration tests for fact extraction."""
    
    async def test_end_to_end_extraction(self):
        """Test end-to-end fact extraction process."""
        # Mock components
        mock_storage = Mock()
        mock_storage.driver = Mock()
        
        service = FactExtractionService(
            neo4j_storage=mock_storage,
            enable_relationships=False  # Disable for simpler test
        )
        
        # Mock the fact extractor
        service.fact_extractor.extract_facts = AsyncMock(return_value=[
            Fact(
                subject="Test Subject",
                object="test object",
                approach="test approach",
                source_chunk_id="chunk_1",
                source_document_id="doc_1",
                extraction_confidence=0.9,
                fact_type=FactType.RESEARCH
            )
        ])
        
        # Mock fact operations
        service.fact_operations.store_facts = AsyncMock(return_value=["fact_1"])
        
        # Mock document chunks
        from morag_graph.models.document_chunk import DocumentChunk
        chunks = [
            DocumentChunk(
                id="chunk_1",
                document_id="doc_1",
                content="Test content for fact extraction.",
                index=0
            )
        ]
        
        result = await service.extract_facts_from_chunks(chunks, domain="test")
        
        assert result['statistics']['chunks_processed'] == 1
        assert result['statistics']['facts_extracted'] == 1
        assert len(result['facts']) == 1
        assert result['facts'][0].subject == "Test Subject"


if __name__ == "__main__":
    pytest.main([__file__])
