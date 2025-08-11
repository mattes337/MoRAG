"""Test enhanced entity normalization and relationship creation."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from morag_graph.models.fact import Fact
from morag_graph.models.entity import Entity
from morag_graph.services.enhanced_fact_processing_service import EnhancedFactProcessingService
from morag_graph.extraction.entity_normalizer import LLMEntityNormalizer, EntityVariation


@pytest.fixture
def mock_neo4j_storage():
    """Mock Neo4j storage."""
    storage = AsyncMock()
    storage.store_entity = AsyncMock()
    storage._connection_ops = AsyncMock()
    storage._connection_ops._execute_query = AsyncMock(return_value=[{"r": "relation"}])
    return storage


@pytest.fixture
def mock_entity_normalizer():
    """Mock entity normalizer."""
    normalizer = AsyncMock()
    
    # Define normalization mappings
    normalization_map = {
        "Silizium Pur": "Silizium",
        "natural vitamins": "Vitamin",
        "heavy metals": "Metal",
        "Engelwurz (Wurzel)": "Engelwurz",
        "männliche Hormone": "Hormon",
        "weibliche Geschlechtshormone": "Geschlechtshormon",
        "ADHD": "ADHD",
        "Stress": "Stress"
    }
    
    async def normalize_side_effect(entity_name):
        normalized = normalization_map.get(entity_name, entity_name)
        return EntityVariation(
            original=entity_name,
            normalized=normalized,
            confidence=0.9,
            rule_applied="test_normalization"
        )
    
    normalizer.normalize_entity.side_effect = normalize_side_effect
    return normalizer


@pytest.fixture
def enhanced_service(mock_neo4j_storage, mock_entity_normalizer):
    """Enhanced fact processing service with mocked dependencies."""
    return EnhancedFactProcessingService(mock_neo4j_storage, mock_entity_normalizer)


@pytest.mark.asyncio
async def test_entity_normalization_and_uniqueness(enhanced_service, mock_neo4j_storage):
    """Test that entities are normalized and unique by name."""
    facts = [
        Fact(
            id="fact1",
            subject="Silizium Pur",
            object="ADHD",
            source_chunk_id="chunk1",
            fact_type="treatment",
            extraction_confidence=0.9,
            source_document_id="doc1",
            domain="health",
            language="en"
        ),
        Fact(
            id="fact2",
            subject="natural vitamins",
            object="Stress",
            source_chunk_id="chunk2",
            fact_type="treatment",
            extraction_confidence=0.8,
            source_document_id="doc2",
            domain="health",
            language="en"
        ),
        Fact(
            id="fact3",
            subject="Silizium Pur",  # Same as fact1 subject - should be normalized to same entity
            object="heavy metals",
            source_chunk_id="chunk3",
            fact_type="treatment",
            extraction_confidence=0.85,
            source_document_id="doc3",
            domain="health",
            language="en"
        )
    ]
    
    # Process facts
    result = await enhanced_service.process_facts_with_entities(
        facts=facts,
        create_keyword_entities=False,
        create_mandatory_relations=True
    )
    
    # Verify entities were created
    assert mock_neo4j_storage.store_entity.called
    
    # Get all stored entities
    stored_entities = []
    for call in mock_neo4j_storage.store_entity.call_args_list:
        entity = call[0][0]  # First argument of the call
        stored_entities.append(entity)
    
    # Check that entities use generic ENTITY label
    for entity in stored_entities:
        assert entity.type == "ENTITY"
        assert entity.get_neo4j_label() == "ENTITY"
    
    # Check that entity names are normalized
    entity_names = [entity.name for entity in stored_entities]
    assert "Silizium" in entity_names  # Normalized from "Silizium Pur"
    assert "Vitamin" in entity_names   # Normalized from "natural vitamins"
    assert "Metal" in entity_names     # Normalized from "heavy metals"
    assert "ADHD" in entity_names      # Already normalized
    assert "Stress" in entity_names    # Already normalized
    
    # Check that duplicate entities are not created (Silizium appears in fact1 and fact3)
    silizium_entities = [e for e in stored_entities if e.name == "Silizium"]
    assert len(silizium_entities) == 1  # Should only be one Silizium entity


@pytest.mark.asyncio
async def test_semantic_relationship_types(enhanced_service, mock_neo4j_storage):
    """Test that semantic relationship types are determined correctly."""
    facts = [
        Fact(
            id="fact1",
            subject="Engelwurz",
            object="ADHD",
            approach="treats symptoms effectively",
            source_chunk_id="chunk1",
            fact_type="treatment",
            extraction_confidence=0.9,
            source_document_id="doc1",
            domain="health",
            language="en"
        ),
        Fact(
            id="fact2",
            subject="Stress",
            object="Hormon",
            approach="causes hormonal imbalances",
            source_chunk_id="chunk2",
            fact_type="causation",
            extraction_confidence=0.8,
            source_document_id="doc2",
            domain="health",
            language="en"
        ),
        Fact(
            id="fact3",
            subject="Vitamin",
            object="Stress",
            approach="reduces stress levels",
            source_chunk_id="chunk3",
            fact_type="treatment",
            extraction_confidence=0.85,
            source_document_id="doc3",
            domain="health",
            language="en"
        )
    ]
    
    # Process facts with relationships
    result = await enhanced_service.process_facts_with_entities(
        facts=facts,
        create_keyword_entities=False,
        create_mandatory_relations=True
    )
    
    # Verify relationships were created with semantic types
    assert mock_neo4j_storage._connection_ops._execute_query.called
    
    # Check the relationship creation calls
    relation_calls = mock_neo4j_storage._connection_ops._execute_query.call_args_list
    
    # Extract relation types from the calls
    relation_types = []
    for call in relation_calls:
        args, kwargs = call
        if len(args) > 1 and isinstance(args[1], dict) and 'relation_type' in args[1]:
            relation_types.append(args[1]['relation_type'])
    
    # Verify semantic relationship types are used
    assert 'TREATS' in relation_types  # Engelwurz treats ADHD
    assert 'CAUSES' in relation_types  # Stress causes hormonal imbalances
    assert 'REDUCES' in relation_types # Vitamin reduces stress


@pytest.mark.asyncio
async def test_keyword_entity_normalization(enhanced_service, mock_neo4j_storage):
    """Test that keyword entities are normalized correctly."""
    facts = [
        Fact(
            id="fact1",
            subject="Engelwurz",
            object="ADHD",
            source_chunk_id="chunk1",
            fact_type="treatment",
            keywords=["natural vitamins", "heavy metals", "Engelwurz (Wurzel)"],
            extraction_confidence=0.9,
            source_document_id="doc1",
            domain="health",
            language="en"
        )
    ]
    
    # Process facts with keyword entities
    result = await enhanced_service.process_facts_with_entities(
        facts=facts,
        create_keyword_entities=True,
        create_mandatory_relations=True
    )
    
    # Get all stored entities
    stored_entities = []
    for call in mock_neo4j_storage.store_entity.call_args_list:
        entity = call[0][0]
        stored_entities.append(entity)
    
    # Check that keyword entities are normalized
    entity_names = [entity.name for entity in stored_entities]
    assert "Vitamin" in entity_names   # Normalized from "natural vitamins"
    assert "Metal" in entity_names     # Normalized from "heavy metals"
    assert "Engelwurz" in entity_names # Normalized from "Engelwurz (Wurzel)"
    
    # Check that all entities use generic ENTITY label
    for entity in stored_entities:
        assert entity.type == "ENTITY"
        assert entity.get_neo4j_label() == "ENTITY"


def test_entity_model_generic_label():
    """Test that Entity model always returns generic ENTITY label."""
    # Test with different entity types
    entity1 = Entity(name="Test Entity", type="SUBJECT", confidence=0.9)
    entity2 = Entity(name="Another Entity", type="OBJECT", confidence=0.8)
    entity3 = Entity(name="Keyword Entity", type="KEYWORD", confidence=0.7)
    
    # All should return generic ENTITY label
    assert entity1.get_neo4j_label() == "ENTITY"
    assert entity2.get_neo4j_label() == "ENTITY"
    assert entity3.get_neo4j_label() == "ENTITY"
    
    # But preserve original type in the type field
    assert entity1.type == "SUBJECT"
    assert entity2.type == "OBJECT"
    assert entity3.type == "KEYWORD"


if __name__ == "__main__":
    pytest.main([__file__])
