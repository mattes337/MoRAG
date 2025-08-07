"""Tests for hybrid fact extraction approach."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from morag_graph.models.fact import Fact, StructuredMetadata
from morag_graph.extraction.fact_extractor import FactExtractor
from morag_graph.extraction.fact_entity_converter import FactEntityConverter


class TestStructuredMetadata:
    """Test the StructuredMetadata model."""
    
    def test_structured_metadata_creation(self):
        """Test basic structured metadata creation."""
        metadata = StructuredMetadata(
            primary_entities=["entity1", "entity2"],
            relationships=["relates_to", "affects"],
            domain_concepts=["concept1", "concept2"]
        )
        
        assert metadata.primary_entities == ["entity1", "entity2"]
        assert metadata.relationships == ["relates_to", "affects"]
        assert metadata.domain_concepts == ["concept1", "concept2"]
        assert metadata.subject is None
        assert metadata.object is None
    
    def test_structured_metadata_with_legacy_fields(self):
        """Test structured metadata with legacy fields."""
        metadata = StructuredMetadata(
            primary_entities=["entity1"],
            relationships=["relates_to"],
            domain_concepts=["concept1"],
            subject="legacy_subject",
            object="legacy_object",
            approach="legacy_approach",
            solution="legacy_solution"
        )
        
        assert metadata.subject == "legacy_subject"
        assert metadata.object == "legacy_object"
        assert metadata.approach == "legacy_approach"
        assert metadata.solution == "legacy_solution"


class TestHybridFact:
    """Test the hybrid Fact model."""
    
    def test_hybrid_fact_creation(self):
        """Test basic hybrid fact creation."""
        metadata = StructuredMetadata(
            primary_entities=["Ashwagandha", "stress", "anxiety"],
            relationships=["treats", "reduces"],
            domain_concepts=["herbal medicine", "adaptogen", "dosage"]
        )
        
        fact = Fact(
            fact_text="Ashwagandha extract containing 5% withanolides should be taken at 300-600mg twice daily with meals for 8-12 weeks to effectively manage chronic stress and anxiety.",
            structured_metadata=metadata,
            source_chunk_id="chunk_123",
            source_document_id="doc_456",
            extraction_confidence=0.9,
            fact_type="methodological",
            keywords=["ashwagandha", "stress", "anxiety", "dosage"]
        )
        
        assert fact.fact_text.startswith("Ashwagandha extract")
        assert fact.structured_metadata.primary_entities == ["Ashwagandha", "stress", "anxiety"]
        assert fact.extraction_confidence == 0.9
        assert fact.fact_type == "methodological"
        assert fact.id.startswith("fact_")
    
    def test_fact_display_text(self):
        """Test fact display text returns fact_text."""
        fact = Fact(
            fact_text="This is a complete fact statement.",
            source_chunk_id="chunk_123",
            source_document_id="doc_456",
            extraction_confidence=0.8,
            fact_type="definition"
        )
        
        assert fact.get_display_text() == "This is a complete fact statement."
    
    def test_fact_is_complete(self):
        """Test fact completeness check."""
        # Complete fact with entities
        metadata = StructuredMetadata(primary_entities=["entity1"])
        fact = Fact(
            fact_text="Complete fact statement.",
            structured_metadata=metadata,
            source_chunk_id="chunk_123",
            source_document_id="doc_456",
            extraction_confidence=0.8,
            fact_type="definition"
        )
        assert fact.is_complete()
        
        # Incomplete fact without entities or text
        empty_fact = Fact(
            fact_text="",
            source_chunk_id="chunk_123",
            source_document_id="doc_456",
            extraction_confidence=0.8,
            fact_type="definition"
        )
        assert not empty_fact.is_complete()
    
    def test_fact_search_text(self):
        """Test fact search text generation."""
        metadata = StructuredMetadata(
            primary_entities=["entity1", "entity2"],
            relationships=["relates_to"],
            domain_concepts=["concept1"]
        )
        
        fact = Fact(
            fact_text="This is a fact about entities.",
            structured_metadata=metadata,
            source_chunk_id="chunk_123",
            source_document_id="doc_456",
            extraction_confidence=0.8,
            fact_type="definition",
            keywords=["keyword1", "keyword2"]
        )
        
        search_text = fact.get_search_text()
        assert "This is a fact about entities." in search_text
        assert "entity1" in search_text
        assert "entity2" in search_text
        assert "relates_to" in search_text
        assert "concept1" in search_text
        assert "keyword1" in search_text
        assert "keyword2" in search_text
    
    def test_fact_to_dict_and_from_dict(self):
        """Test fact serialization and deserialization."""
        metadata = StructuredMetadata(
            primary_entities=["entity1"],
            relationships=["relates_to"],
            domain_concepts=["concept1"]
        )
        
        original_fact = Fact(
            fact_text="Original fact text.",
            structured_metadata=metadata,
            source_chunk_id="chunk_123",
            source_document_id="doc_456",
            extraction_confidence=0.8,
            fact_type="definition"
        )
        
        # Convert to dict
        fact_dict = original_fact.to_dict()
        assert fact_dict["fact_text"] == "Original fact text."
        assert "structured_metadata" in fact_dict
        assert fact_dict["structured_metadata"]["primary_entities"] == ["entity1"]
        
        # Convert back to fact
        restored_fact = Fact.from_dict(fact_dict)
        assert restored_fact.fact_text == original_fact.fact_text
        assert restored_fact.structured_metadata.primary_entities == original_fact.structured_metadata.primary_entities
        assert restored_fact.extraction_confidence == original_fact.extraction_confidence


class TestHybridFactExtraction:
    """Test hybrid fact extraction process."""
    
    @pytest.fixture
    def mock_llm_client(self):
        """Mock LLM client for testing."""
        client = AsyncMock()
        client.generate = AsyncMock()
        return client
    
    @pytest.fixture
    def fact_extractor(self, mock_llm_client):
        """Create fact extractor with mocked LLM."""
        return FactExtractor(
            llm_client=mock_llm_client,
            max_facts_per_chunk=5
        )
    
    @pytest.mark.asyncio
    async def test_hybrid_fact_extraction_response_parsing(self, fact_extractor):
        """Test parsing of hybrid fact extraction response."""
        # Mock LLM response with hybrid format
        mock_response = '''[
          {
            "fact_text": "PostgreSQL query performance can be optimized by creating B-tree indexes on frequently queried columns using CREATE INDEX syntax, with composite queries requiring multi-column indexes where the most selective column is placed first.",
            "structured_metadata": {
              "primary_entities": ["PostgreSQL", "B-tree index", "query performance"],
              "relationships": ["optimizes", "improves", "requires"],
              "domain_concepts": ["CREATE INDEX", "composite queries", "column selectivity"]
            },
            "fact_type": "methodological",
            "confidence": 0.95,
            "keywords": ["PostgreSQL", "B-tree index", "query optimization"]
          }
        ]'''
        
        fact_extractor.llm_client.generate.return_value = mock_response
        
        # Extract facts
        facts = await fact_extractor.extract_facts(
            chunk_text="Sample text about PostgreSQL optimization",
            chunk_id="chunk_123",
            document_id="doc_456",
            context={"domain": "technical", "language": "en"}
        )
        
        assert len(facts) == 1
        fact = facts[0]
        assert fact.fact_text.startswith("PostgreSQL query performance")
        assert "PostgreSQL" in fact.structured_metadata.primary_entities
        assert "optimizes" in fact.structured_metadata.relationships
        assert "CREATE INDEX" in fact.structured_metadata.domain_concepts
        assert fact.fact_type == "methodological"
        assert fact.extraction_confidence == 0.95
    
    @pytest.mark.asyncio
    async def test_legacy_format_fallback(self, fact_extractor):
        """Test fallback to legacy format when fact_text is missing."""
        # Mock LLM response with legacy format
        mock_response = '''[
          {
            "subject": "Ashwagandha extract",
            "object": "chronic stress and anxiety",
            "approach": "300-600mg twice daily with meals",
            "solution": "reduction of stress symptoms",
            "fact_type": "methodological",
            "confidence": 0.9,
            "keywords": ["ashwagandha", "stress", "anxiety"]
          }
        ]'''
        
        fact_extractor.llm_client.generate.return_value = mock_response
        
        # Extract facts
        facts = await fact_extractor.extract_facts(
            chunk_text="Sample text about Ashwagandha",
            chunk_id="chunk_123",
            document_id="doc_456",
            context={"domain": "medical", "language": "en"}
        )
        
        assert len(facts) == 1
        fact = facts[0]
        # Should construct fact_text from legacy fields
        assert "Ashwagandha extract" in fact.fact_text
        assert "chronic stress and anxiety" in fact.fact_text
        # Legacy fields should be preserved in metadata
        assert fact.structured_metadata.subject == "Ashwagandha extract"
        assert fact.structured_metadata.object == "chronic stress and anxiety"


class TestHybridFactEntityConversion:
    """Test entity conversion from hybrid facts."""
    
    @pytest.fixture
    def fact_entity_converter(self):
        """Create fact entity converter."""
        return FactEntityConverter()
    
    def test_hybrid_fact_to_entities(self, fact_entity_converter):
        """Test conversion of hybrid fact to entities and relationships."""
        metadata = StructuredMetadata(
            primary_entities=["PostgreSQL", "B-tree index", "query performance"],
            relationships=["optimizes", "improves"],
            domain_concepts=["database optimization", "indexing strategy"]
        )
        
        fact = Fact(
            fact_text="PostgreSQL B-tree indexes optimize query performance through strategic column selection.",
            structured_metadata=metadata,
            source_chunk_id="chunk_123",
            source_document_id="doc_456",
            extraction_confidence=0.9,
            fact_type="methodological",
            keywords=["PostgreSQL", "optimization"]
        )
        
        entities, relationships = fact_entity_converter.convert_facts_to_entities([fact])
        
        # Should create entities from primary_entities, domain_concepts, and keywords
        entity_names = [e.name for e in entities]
        assert "PostgreSQL" in entity_names
        assert "B-tree index" in entity_names
        assert "query performance" in entity_names
        assert "database optimization" in entity_names
        assert "indexing strategy" in entity_names
        
        # Should create relationships
        assert len(relationships) > 0
        relation_types = [r.type for r in relationships]
        assert "MENTIONED_IN" in relation_types
        assert "RELATES_TO" in relation_types


if __name__ == "__main__":
    pytest.main([__file__])
