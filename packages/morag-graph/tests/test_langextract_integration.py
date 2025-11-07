"""Integration tests for LangExtract-based extraction system."""

import pytest
import asyncio
from unittest.mock import Mock, patch

from morag_graph.extraction import EntityExtractor, RelationExtractor
from morag_graph.models import Entity, Relation


class TestLangExtractIntegration:
    """Test LangExtract integration with MoRAG."""

    @pytest.mark.asyncio
    async def test_entity_extractor_initialization(self):
        """Test that EntityExtractor initializes correctly."""
        extractor = EntityExtractor(domain="medical")

        assert extractor.domain == "medical"
        assert hasattr(extractor, 'extract')
        assert hasattr(extractor, 'get_system_prompt')

    @pytest.mark.asyncio
    async def test_relation_extractor_initialization(self):
        """Test that RelationExtractor initializes correctly."""
        extractor = RelationExtractor(domain="technical")

        assert extractor.domain == "technical"
        assert hasattr(extractor, 'extract')
        assert hasattr(extractor, 'get_system_prompt')

    @pytest.mark.asyncio
    async def test_entity_extraction_without_api_key(self):
        """Test entity extraction without API key returns empty list."""
        extractor = EntityExtractor()

        # Should return empty list when no API key is available
        entities = await extractor.extract("Test text")
        assert entities == []

    @pytest.mark.asyncio
    async def test_relation_extraction_without_api_key(self):
        """Test relation extraction without API key returns empty list."""
        extractor = RelationExtractor()

        # Create mock entities
        entities = [
            Entity(name="John", type="PERSON", source_doc_id="test"),
            Entity(name="Google", type="ORGANIZATION", source_doc_id="test")
        ]

        # Should return empty list when no API key is available
        relations = await extractor.extract("Test text", entities=entities)
        assert relations == []

    @pytest.mark.asyncio
    async def test_extractors_with_custom_types(self):
        """Test extractors with custom entity and relation types."""
        entity_types = {"medication": "Pharmaceutical drugs", "symptom": "Medical symptoms"}
        relation_types = {"treats": "Treatment relationship", "causes": "Causation relationship"}

        entity_extractor = EntityExtractor(
            domain="medical",
            entity_types=entity_types
        )

        relation_extractor = RelationExtractor(
            domain="medical",
            relation_types=relation_types
        )

        assert entity_extractor.entity_types == entity_types
        assert relation_extractor.relation_types == relation_types

    @pytest.mark.asyncio
    async def test_async_context_managers(self):
        """Test that extractors work as async context managers."""
        async with EntityExtractor() as entity_extractor:
            assert entity_extractor is not None

        async with RelationExtractor() as relation_extractor:
            assert relation_extractor is not None

    @pytest.mark.asyncio
    async def test_domain_specific_extractors(self):
        """Test domain-specific extractor initialization."""
        domains = ["general", "medical", "technical", "legal", "business", "scientific"]

        for domain in domains:
            entity_extractor = EntityExtractor(domain=domain)
            relation_extractor = RelationExtractor(domain=domain)

            assert entity_extractor.domain == domain
            assert relation_extractor.domain == domain

            # Check that domain-specific types are loaded
            if domain != "general":
                assert len(entity_extractor.entity_types) > 0
                assert len(relation_extractor.relation_types) > 0



    @pytest.mark.asyncio
    async def test_system_prompts(self):
        """Test that system prompts are generated correctly."""
        entity_extractor = EntityExtractor(domain="medical")
        relation_extractor = RelationExtractor(domain="medical")

        entity_prompt = entity_extractor.get_system_prompt()
        relation_prompt = relation_extractor.get_system_prompt()

        assert isinstance(entity_prompt, str)
        assert len(entity_prompt) > 0
        assert isinstance(relation_prompt, str)
        assert len(relation_prompt) > 0

        # Check that medical domain is reflected in prompts
        assert "medical" in entity_prompt.lower() or "health" in entity_prompt.lower()
        assert "medical" in relation_prompt.lower() or "health" in relation_prompt.lower()


class TestLangExtractVisualization:
    """Test LangExtract visualization integration."""

    def test_visualizer_import(self):
        """Test that visualizer can be imported."""
        try:
            from morag_graph.visualization import LangExtractVisualizer
            assert LangExtractVisualizer is not None
        except ImportError:
            # Skip if LangExtract is not available
            pytest.skip("LangExtract not available")

    def test_visualizer_initialization_without_api_key(self):
        """Test visualizer initialization without API key."""
        try:
            from morag_graph.visualization import LangExtractVisualizer

            with pytest.raises(ValueError, match="No API key found"):
                LangExtractVisualizer()
        except ImportError:
            # Skip if LangExtract is not available
            pytest.skip("LangExtract not available")


class TestGraphBuilderIntegration:
    """Test graph builder integration with LangExtract."""

    def test_graph_builder_import(self):
        """Test that graph builders can be imported."""
        from morag_graph.builders import GraphBuilder, EnhancedGraphBuilder

        assert GraphBuilder is not None
        assert EnhancedGraphBuilder is not None

    def test_graph_builder_initialization(self):
        """Test graph builder initialization."""
        from morag_graph.builders import GraphBuilder
        from morag_graph.storage.base import BaseStorage

        # Create a minimal mock storage
        storage = Mock(spec=BaseStorage)

        # Test basic initialization
        builder = GraphBuilder(storage=storage)
        assert builder.storage == storage
        assert hasattr(builder, 'entity_extractor')
        assert hasattr(builder, 'relation_extractor')

        # Test with domain
        builder_with_domain = GraphBuilder(storage=storage, domain="medical")
        assert builder_with_domain.domain == "medical"
        assert builder_with_domain.entity_extractor.domain == "medical"
        assert builder_with_domain.relation_extractor.domain == "medical"

    def test_enhanced_graph_builder_initialization(self):
        """Test enhanced graph builder initialization."""
        from morag_graph.builders import EnhancedGraphBuilder
        from morag_graph.storage.base import BaseStorage

        # Create a minimal mock storage
        storage = Mock(spec=BaseStorage)

        # Test basic initialization
        builder = EnhancedGraphBuilder(storage=storage)
        assert builder.storage == storage
        assert hasattr(builder, 'entity_extractor')
        assert hasattr(builder, 'relation_extractor')

        # Test with custom parameters
        builder_custom = EnhancedGraphBuilder(
            storage=storage,
            domain="technical",
            min_confidence=0.8,
            chunk_size=1500,
            max_workers=5
        )
        assert builder_custom.domain == "technical"
        assert builder_custom.min_confidence == 0.8
        assert builder_custom.chunk_size == 1500
        assert builder_custom.max_workers == 5


@pytest.mark.asyncio
async def test_end_to_end_extraction_flow():
    """Test end-to-end extraction flow without API calls."""
    # Initialize extractors
    entity_extractor = EntityExtractor(domain="general")
    relation_extractor = RelationExtractor(domain="general")

    # Test text
    text = "Dr. Smith works at Google in Mountain View."

    # Extract entities (will return empty without API key)
    entities = await entity_extractor.extract(text, source_doc_id="test_doc")
    assert isinstance(entities, list)

    # Extract relations (will return empty without API key)
    relations = await relation_extractor.extract(text, entities=entities, source_doc_id="test_doc")
    assert isinstance(relations, list)

    # Test that the flow completes without errors
    assert True  # If we get here, the flow worked


if __name__ == "__main__":
    pytest.main([__file__])
