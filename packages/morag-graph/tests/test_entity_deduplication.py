"""Tests for entity deduplication improvements."""

import pytest
from unittest.mock import Mock, AsyncMock
from typing import List

from morag_graph.models import Entity as GraphEntity
from morag_graph.ai.entity_agent import EntityExtractionAgent
from morag_graph.utils.id_generation import UnifiedIDGenerator
from morag_core.ai.models import Entity as AIEntity
from morag_core.ai import AgentConfig





class TestEntityIDGeneration:
    """Test entity ID generation for deduplication."""
    
    def test_id_based_on_name_only(self):
        """Test that entity IDs are based only on name, not type."""
        # Same name, different types should generate same ID
        id1 = UnifiedIDGenerator.generate_entity_id("Zirbeldrüse", "GLAND", "doc_test_123")
        id2 = UnifiedIDGenerator.generate_entity_id("Zirbeldrüse", "ORGAN", "doc_test_123")
        id3 = UnifiedIDGenerator.generate_entity_id("Zirbeldrüse", "ANATOMICAL_STRUCTURE", "doc_test_456")
        
        assert id1 == id2 == id3
    
    def test_different_names_different_ids(self):
        """Test that different names generate different IDs."""
        id1 = UnifiedIDGenerator.generate_entity_id("Zirbeldrüse", "GLAND", "doc_test_123")
        id2 = UnifiedIDGenerator.generate_entity_id("Thymus", "GLAND", "doc_test_123")
        
        assert id1 != id2
    
    def test_name_normalization(self):
        """Test that names are normalized for consistent IDs."""
        # Different spacing and casing should generate same ID
        id1 = UnifiedIDGenerator.generate_entity_id("Zirbeldrüse", "GLAND", "doc_test_123")
        id2 = UnifiedIDGenerator.generate_entity_id("zirbeldrüse", "ORGAN", "doc_test_123")
        id3 = UnifiedIDGenerator.generate_entity_id("ZIRBELDRÜSE", "TISSUE", "doc_test_123")
        
        assert id1 == id2 == id3


class TestEntityDeduplication:
    """Test entity deduplication logic."""
    
    def create_mock_agent(self):
        """Create a mock entity extraction agent."""
        config = AgentConfig(model="mock")
        agent = EntityExtractionAgent(
            config=config,
            dynamic_types=True,
            min_confidence=0.5
        )
        # Mock the run method to avoid actual LLM calls
        agent.run = AsyncMock()
        return agent
    
    def test_deduplicate_same_name_different_types(self):
        """Test that entities with same name but different types are merged."""
        agent = self.create_mock_agent()
        
        # Create entities with same name but different types
        entities = [
            GraphEntity(
                name="Zirbeldrüse",
                type="GLAND",
                confidence=0.8,
                source_doc_id="doc_test_123"
            ),
            GraphEntity(
                name="Zirbeldrüse",
                type="ORGAN",
                confidence=0.9,
                source_doc_id="doc_test_123"
            ),
            GraphEntity(
                name="Zirbeldrüse",
                type="ANATOMICAL_STRUCTURE",
                confidence=0.7,
                source_doc_id="doc_test_123"
            )
        ]
        
        deduplicated = agent._deduplicate_entities(entities)
        
        # Should have only one entity
        assert len(deduplicated) == 1
        
        # Should keep the highest confidence
        assert deduplicated[0].confidence == 0.9

        # Should use the type from the highest confidence entity
        assert deduplicated[0].type == "ORGAN"  # This was the type of the 0.9 confidence entity

        # Should keep the name
        assert deduplicated[0].name == "Zirbeldrüse"
    
    def test_deduplicate_different_names(self):
        """Test that entities with different names are not merged."""
        agent = self.create_mock_agent()
        
        entities = [
            GraphEntity(
                name="Zirbeldrüse",
                type="GLAND",
                confidence=0.8,
                source_doc_id="doc_test_123"
            ),
            GraphEntity(
                name="Thymus",
                type="GLAND",
                confidence=0.9,
                source_doc_id="doc_test_123"
            )
        ]
        
        deduplicated = agent._deduplicate_entities(entities)
        
        # Should have two entities
        assert len(deduplicated) == 2
        
        # Names should be preserved
        names = {entity.name for entity in deduplicated}
        assert names == {"Zirbeldrüse", "Thymus"}
    
    def test_merge_attributes(self):
        """Test that attributes are merged when deduplicating."""
        agent = self.create_mock_agent()
        
        entities = [
            GraphEntity(
                name="Zirbeldrüse",
                type="GLAND",
                confidence=0.8,
                source_doc_id="doc_test_123",
                attributes={"context": "endocrine system", "size": "small"}
            ),
            GraphEntity(
                name="Zirbeldrüse",
                type="ORGAN",
                confidence=0.9,
                source_doc_id="doc_test_123",
                attributes={"function": "melatonin production", "location": "brain"}
            )
        ]
        
        deduplicated = agent._deduplicate_entities(entities)
        
        # Should have one entity with merged attributes
        assert len(deduplicated) == 1
        
        merged_attrs = deduplicated[0].attributes
        assert "context" in merged_attrs
        assert "size" in merged_attrs
        assert "function" in merged_attrs
        assert "location" in merged_attrs
    
    def test_case_insensitive_deduplication(self):
        """Test that deduplication is case insensitive."""
        agent = self.create_mock_agent()
        
        entities = [
            GraphEntity(
                name="Zirbeldrüse",
                type="GLAND",
                confidence=0.8,
                source_doc_id="doc_test_123"
            ),
            GraphEntity(
                name="zirbeldrüse",
                type="ORGAN",
                confidence=0.9,
                source_doc_id="doc_test_123"
            ),
            GraphEntity(
                name="ZIRBELDRÜSE",
                type="TISSUE",
                confidence=0.7,
                source_doc_id="doc_test_123"
            )
        ]
        
        deduplicated = agent._deduplicate_entities(entities)
        
        # Should have only one entity
        assert len(deduplicated) == 1
        
        # Should keep the highest confidence entity's exact name
        assert deduplicated[0].name == "zirbeldrüse"
        assert deduplicated[0].confidence == 0.9


class TestIntegrationScenario:
    """Test the complete integration scenario."""
    
    def test_zirbeldruse_scenario(self):
        """Test the specific Zirbeldrüse scenario mentioned in the issue."""
        # Simulate the scenario where Zirbeldrüse is extracted with different types
        
        # 1. Test ID generation - same name should generate same ID regardless of type
        id1 = UnifiedIDGenerator.generate_entity_id("Zirbeldrüse", "GLAND", "doc_medical_123")
        id2 = UnifiedIDGenerator.generate_entity_id("Zirbeldrüse", "PINEAL_GLAND", "doc_medical_123")
        id3 = UnifiedIDGenerator.generate_entity_id("Zirbeldrüse", "ENDOCRINE_ORGAN", "doc_medical_123")
        id4 = UnifiedIDGenerator.generate_entity_id("Zirbeldrüse", "BRAIN_STRUCTURE", "doc_medical_123")
        id5 = UnifiedIDGenerator.generate_entity_id("Zirbeldrüse", "ANATOMICAL_STRUCTURE", "doc_medical_123")
        
        assert id1 == id2 == id3 == id4 == id5

        # 2. Test deduplication - multiple entities should be merged into one
        config = AgentConfig(model="mock")
        agent = EntityExtractionAgent(config=config, dynamic_types=True, min_confidence=0.5)
        agent.run = AsyncMock()
        
        entities = [
            GraphEntity(name="Zirbeldrüse", type="GLAND", confidence=0.8, source_doc_id="doc_medical_123"),
            GraphEntity(name="Zirbeldrüse", type="PINEAL_GLAND", confidence=0.9, source_doc_id="doc_medical_123"),
            GraphEntity(name="Zirbeldrüse", type="ENDOCRINE_ORGAN", confidence=0.7, source_doc_id="doc_medical_123"),
            GraphEntity(name="Zirbeldrüse", type="BRAIN_STRUCTURE", confidence=0.6, source_doc_id="doc_medical_123"),
            GraphEntity(name="Zirbeldrüse", type="ANATOMICAL_STRUCTURE", confidence=0.85, source_doc_id="doc_medical_123"),
        ]
        
        deduplicated = agent._deduplicate_entities(entities)
        
        # Should result in exactly one entity
        assert len(deduplicated) == 1
        
        final_entity = deduplicated[0]
        assert final_entity.name == "Zirbeldrüse"
        assert final_entity.type == "PINEAL_GLAND"  # Type from highest confidence entity (0.9)
        assert final_entity.confidence == 0.9  # Highest confidence should be kept
