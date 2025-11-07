"""Tests for PydanticAI-based entity and relation extraction."""

import asyncio
import os
import sys
from typing import List

import pytest

# Add the packages directory to the Python path
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "packages", "morag-core", "src")
)
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "packages", "morag-graph", "src")
)

from morag_core.ai import AgentConfig, ConfidenceLevel
from morag_core.ai import Entity as AIEntity
from morag_core.ai import EntityExtractionResult, MoRAGBaseAgent
from morag_core.ai import Relation as AIRelation
from morag_core.ai import RelationExtractionResult
from morag_graph.ai import EntityExtractionAgent, RelationExtractionAgent
from morag_graph.extraction import EntityExtractor, RelationExtractor
from morag_graph.models import Entity as GraphEntity
from morag_graph.models import Relation as GraphRelation


class TestEntityExtractionAgent:
    """Test the PydanticAI entity extraction agent."""

    def test_agent_creation(self):
        """Test creating entity extraction agent."""
        agent = EntityExtractionAgent(min_confidence=0.7)

        assert agent.min_confidence == 0.7
        assert agent.get_result_type() == EntityExtractionResult
        assert "entity extraction agent" in agent.get_system_prompt().lower()

    def test_agent_system_prompt(self):
        """Test the system prompt contains required elements."""
        agent = EntityExtractionAgent()
        prompt = agent.get_system_prompt()

        # Check for key elements
        assert "PERSON" in prompt
        assert "ORGANIZATION" in prompt
        assert "LOCATION" in prompt
        assert "confidence" in prompt.lower()
        assert "0.0 to 1.0" in prompt

    def test_text_chunking(self):
        """Test text chunking functionality."""
        agent = EntityExtractionAgent()

        # Test small text (no chunking needed)
        small_text = "Apple Inc. is a technology company."
        chunks = agent._split_text_into_chunks(small_text, 1000)
        assert len(chunks) == 1
        assert chunks[0] == small_text

        # Test large text (chunking needed)
        large_text = "Apple Inc. " * 500  # Create a large text
        chunks = agent._split_text_into_chunks(large_text, 100)
        assert len(chunks) > 1

        # Check that chunks respect word boundaries
        for chunk in chunks:
            assert len(chunk) <= 100 or not any(c in chunk for c in " \n\t")

    def test_entity_conversion(self):
        """Test conversion from AI entity to graph entity."""
        agent = EntityExtractionAgent()

        ai_entity = AIEntity(
            name="Apple Inc.",
            type="ORGANIZATION",
            confidence=0.95,
            context="Technology company",
            metadata={"industry": "technology"},
        )

        graph_entity = agent._convert_to_graph_entity(ai_entity, "doc123")

        assert graph_entity.name == "Apple Inc."
        assert graph_entity.type == "ORGANIZATION"
        assert graph_entity.confidence == 0.95
        assert graph_entity.source_doc_id == "doc123"
        assert graph_entity.attributes["context"] == "Technology company"
        assert graph_entity.attributes["industry"] == "technology"

    def test_entity_deduplication(self):
        """Test entity deduplication."""
        agent = EntityExtractionAgent()

        # Create duplicate entities with different confidence scores
        entity1 = GraphEntity(name="Apple Inc.", type="ORGANIZATION", confidence=0.8)
        entity2 = GraphEntity(
            name="apple inc.", type="ORGANIZATION", confidence=0.9  # Different case
        )
        entity3 = GraphEntity(name="Microsoft", type="ORGANIZATION", confidence=0.85)

        entities = [entity1, entity2, entity3]
        deduplicated = agent._deduplicate_entities(entities)

        # Should have 2 entities (Apple and Microsoft)
        assert len(deduplicated) == 2

        # Should keep the higher confidence Apple entity
        apple_entity = next(
            e for e in deduplicated if e.name.lower().startswith("apple")
        )
        assert apple_entity.confidence == 0.9


class TestRelationExtractionAgent:
    """Test the PydanticAI relation extraction agent."""

    def test_agent_creation(self):
        """Test creating relation extraction agent."""
        agent = RelationExtractionAgent(min_confidence=0.7)

        assert agent.min_confidence == 0.7
        assert agent.get_result_type() == RelationExtractionResult
        assert "relation extraction agent" in agent.get_system_prompt().lower()

    def test_agent_system_prompt(self):
        """Test the system prompt contains required elements."""
        agent = RelationExtractionAgent()
        prompt = agent.get_system_prompt()

        # Check for key relation types
        assert "WORKS_FOR" in prompt
        assert "LOCATED_IN" in prompt
        assert "PART_OF" in prompt
        assert "confidence" in prompt.lower()
        assert "0.0 to 1.0" in prompt

    def test_relation_conversion(self):
        """Test conversion from AI relation to graph relation."""
        agent = RelationExtractionAgent()

        # Create test entities with proper IDs
        entities = [
            GraphEntity(name="John Doe", type="PERSON"),
            GraphEntity(name="Apple Inc.", type="ORGANIZATION"),
        ]
        # Set IDs manually after creation to avoid validation
        entities[0].id = "ent1"
        entities[1].id = "ent2"

        ai_relation = AIRelation(
            source_entity="John Doe",
            target_entity="Apple Inc.",
            relation_type="WORKS_FOR",
            confidence=0.9,
            context="John works at Apple",
        )

        graph_relation = agent._convert_to_graph_relation(
            ai_relation, entities, "doc123"
        )

        assert graph_relation is not None
        assert graph_relation.source_entity_id == "ent1"
        assert graph_relation.target_entity_id == "ent2"
        assert graph_relation.confidence == 0.9
        assert graph_relation.source_doc_id == "doc123"
        assert graph_relation.attributes["context"] == "John works at Apple"

    def test_relation_conversion_missing_entity(self):
        """Test relation conversion when entity is missing."""
        agent = RelationExtractionAgent()

        # Create test entities (missing one entity)
        entities = [GraphEntity(name="John Doe", type="PERSON")]
        # Set ID manually after creation to avoid validation
        entities[0].id = "ent1"

        ai_relation = AIRelation(
            source_entity="John Doe",
            target_entity="Unknown Company",  # This entity doesn't exist
            relation_type="WORKS_FOR",
            confidence=0.9,
        )

        graph_relation = agent._convert_to_graph_relation(
            ai_relation, entities, "doc123"
        )

        # Should return None when entity can't be resolved
        assert graph_relation is None

    def test_relation_deduplication(self):
        """Test relation deduplication."""
        agent = RelationExtractionAgent()

        # Create duplicate relations with different confidence scores
        relation1 = GraphRelation(
            source_entity_id="ent1",
            target_entity_id="ent2",
            type="WORKS_FOR",
            confidence=0.8,
        )
        relation2 = GraphRelation(
            source_entity_id="ent1",
            target_entity_id="ent2",
            type="WORKS_FOR",
            confidence=0.9,
        )
        relation3 = GraphRelation(
            source_entity_id="ent2",
            target_entity_id="ent3",
            type="LOCATED_IN",
            confidence=0.85,
        )

        relations = [relation1, relation2, relation3]
        deduplicated = agent._deduplicate_relations(relations)

        # Should have 2 relations (one WORKS_FOR and one LOCATED_IN)
        assert len(deduplicated) == 2

        # Should keep the higher confidence WORKS_FOR relation
        works_for_relation = next(r for r in deduplicated if r.type == "WORKS_FOR")
        assert works_for_relation.confidence == 0.9


class TestEntityExtractor:
    """Test the new PydanticAI-based EntityExtractor."""

    def test_extractor_creation(self):
        """Test creating entity extractor."""
        extractor = EntityExtractor(min_confidence=0.7, chunk_size=5000)

        assert extractor.min_confidence == 0.7
        assert extractor.chunk_size == 5000
        assert isinstance(extractor.agent, EntityExtractionAgent)

    def test_extractor_backward_compatibility(self):
        """Test backward compatibility with old interface."""
        extractor = EntityExtractor()

        # Test that old parameter names still work
        sample_text = "Apple Inc. is a technology company founded by Steve Jobs."

        # This would normally call the API, but we're just testing the interface
        # In a real test with API key, you would await the actual extraction
        assert hasattr(extractor, "extract")
        assert hasattr(extractor, "extract_entities")
        assert hasattr(extractor, "extract_with_context")


class TestRelationExtractor:
    """Test the new PydanticAI-based RelationExtractor."""

    def test_extractor_creation(self):
        """Test creating relation extractor."""
        extractor = RelationExtractor(min_confidence=0.7, chunk_size=4000)

        assert extractor.min_confidence == 0.7
        assert extractor.chunk_size == 4000
        assert isinstance(extractor.agent, RelationExtractionAgent)

    def test_extractor_backward_compatibility(self):
        """Test backward compatibility with old interface."""
        extractor = RelationExtractor()

        # Test that old method names still work
        assert hasattr(extractor, "extract")
        assert hasattr(extractor, "extract_relations")
        assert hasattr(extractor, "extract_with_entities")
        assert hasattr(extractor, "extract_from_entity_pairs")

    def test_entity_pair_filtering(self):
        """Test entity pair filtering logic."""
        extractor = RelationExtractor()

        # Create test entities
        entity1 = GraphEntity(name="John", type="PERSON")
        entity2 = GraphEntity(name="Apple", type="ORGANIZATION")
        entity3 = GraphEntity(name="Microsoft", type="ORGANIZATION")
        # Set IDs manually after creation to avoid validation
        entity1.id = "ent1"
        entity2.id = "ent2"
        entity3.id = "ent3"

        entity_pairs = [(entity1, entity2)]

        # Convert pairs to flat list
        entities = []
        for source_entity, target_entity in entity_pairs:
            if source_entity not in entities:
                entities.append(source_entity)
            if target_entity not in entities:
                entities.append(target_entity)

        assert len(entities) == 2
        assert entity1 in entities
        assert entity2 in entities
        assert entity3 not in entities


class TestIntegration:
    """Integration tests for the complete extraction pipeline."""

    def test_extraction_pipeline_interface(self):
        """Test that the extraction pipeline interface is maintained."""
        entity_extractor = EntityExtractor()
        relation_extractor = RelationExtractor()

        # Test that the interface matches what the rest of the system expects
        sample_text = "Apple Inc. was founded by Steve Jobs in Cupertino."

        # These would normally be async calls with actual API interaction
        # Here we're just testing that the interface is correct
        assert callable(entity_extractor.extract)
        assert callable(relation_extractor.extract)

        # Test parameter compatibility
        import inspect

        # Check entity extractor signature
        entity_sig = inspect.signature(entity_extractor.extract)
        assert "text" in entity_sig.parameters
        assert "source_doc_id" in entity_sig.parameters

        # Check relation extractor signature
        relation_sig = inspect.signature(relation_extractor.extract)
        assert "text" in relation_sig.parameters
        assert "entities" in relation_sig.parameters
        assert "source_doc_id" in relation_sig.parameters


if __name__ == "__main__":
    pytest.main([__file__])
