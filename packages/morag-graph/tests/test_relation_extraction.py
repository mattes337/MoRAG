"""Tests for relation extraction functionality."""

import asyncio
import json
import os
from typing import List, Dict, Any

import pytest
from dotenv import load_dotenv

from morag_graph.extraction import EntityExtractor, RelationExtractor
from morag_graph.models import Entity, Relation, Graph
from morag_graph.models.types import EntityType, RelationType

# Load environment variables from .env file
load_dotenv()

# Sample texts for testing relation extraction
SAMPLE_TEXTS = [
    """Apple Inc. is an American multinational technology company headquartered in Cupertino, California. 
    Tim Cook is the CEO of Apple. The company was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.""",
    
    """The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. 
    It is named after the engineer Gustave Eiffel, whose company designed and built the tower from 1887 to 1889.""",
    
    """Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability. 
    Guido van Rossum created Python and first released it in 1991."""
]


@pytest.fixture
def gemini_api_key() -> str:
    """Get Gemini API key from environment variables."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY environment variable not set")
    return api_key


@pytest.fixture
def entity_extractor(gemini_api_key: str) -> EntityExtractor:
    """Create an EntityExtractor instance for testing."""
    return EntityExtractor(
        llm_config={
            "provider": "gemini",
            "api_key": gemini_api_key,
            "model": "gemini-1.5-flash",  # Use Gemini Flash for testing
            "temperature": 0.0,  # Set to 0 for deterministic results
            "max_tokens": 1000
        }
    )


@pytest.fixture
def relation_extractor(gemini_api_key: str) -> RelationExtractor:
    """Create a RelationExtractor instance for testing."""
    return RelationExtractor(
        llm_config={
            "provider": "gemini",
            "api_key": gemini_api_key,
            "model": "gemini-1.5-flash",  # Use Gemini Flash for testing
            "temperature": 0.0,  # Set to 0 for deterministic results
            "max_tokens": 1000
        }
    )


@pytest.fixture
async def sample_entities(entity_extractor: EntityExtractor) -> List[List[Entity]]:
    """Extract entities from sample texts for relation extraction tests."""
    all_entities = []
    for text in SAMPLE_TEXTS:
        entities = await entity_extractor.extract(text)
        all_entities.append(entities)
    return all_entities


@pytest.mark.asyncio
async def test_relation_extraction_basic(relation_extractor: RelationExtractor, sample_entities: List[List[Entity]]):
    """Test basic relation extraction functionality."""
    # Extract relations from the first sample text with its entities
    text = SAMPLE_TEXTS[0]
    entities = sample_entities[0]
    
    relations = await relation_extractor.extract(text, entities)
    
    # Verify that relations were extracted
    assert len(relations) > 0, "No relations were extracted"
    
    # Check that relations connect existing entities
    entity_ids = {entity.id for entity in entities}
    for relation in relations:
        assert relation.source_entity_id in entity_ids, f"Relation source {relation.source_entity_id} not in entities"
        assert relation.target_entity_id in entity_ids, f"Relation target {relation.target_entity_id} not in entities"
    
    # Check that specific relation types are present
    relation_types = {relation.type for relation in relations}
    expected_types = {RelationType.WORKS_FOR, RelationType.FOUNDED, RelationType.LOCATED_IN}
    for expected_type in expected_types:
        assert expected_type in relation_types, f"Expected relation type {expected_type} not found"
    
    # Check that relations have required attributes
    for relation in relations:
        assert relation.id, "Relation is missing ID"
        assert relation.source_entity_id, "Relation is missing source entity ID"
        assert relation.target_entity_id, "Relation is missing target entity ID"
        assert relation.type, "Relation is missing type"
        assert relation.confidence is not None, "Relation is missing confidence score"


@pytest.mark.asyncio
async def test_relation_extraction_with_context(relation_extractor: RelationExtractor, sample_entities: List[List[Entity]]):
    """Test relation extraction with additional context."""
    # Add context about company relationships
    context = "Focus on employment and founding relationships between people and companies."
    
    # Extract relations from the first sample text with context
    text = SAMPLE_TEXTS[0]
    entities = sample_entities[0]
    
    relations = await relation_extractor.extract(text, entities, context=context)
    
    # Verify that relations were extracted
    assert len(relations) > 0, "No relations were extracted"
    
    # Check that employment and founding relations are present
    employment_relations = [r for r in relations if r.type == RelationType.WORKS_FOR]
    founding_relations = [r for r in relations if r.type == RelationType.FOUNDED]
    
    assert len(employment_relations) > 0, "No employment relations found"
    assert len(founding_relations) > 0, "No founding relations found"
    
    # Check for specific relationships
    # Find Tim Cook and Apple entities
    tim_cook = next((e for e in entities if "tim cook" in e.name.lower()), None)
    apple = next((e for e in entities if "apple" in e.name.lower() and e.type == EntityType.ORGANIZATION), None)
    
    if tim_cook and apple:
        # Check if there's a WORKS_FOR relation between Tim Cook and Apple
        tim_cook_works_for_apple = any(
            r.source_entity_id == tim_cook.id and r.target_entity_id == apple.id and r.type == RelationType.WORKS_FOR
            for r in relations
        )
        assert tim_cook_works_for_apple, "Expected WORKS_FOR relation between Tim Cook and Apple not found"


@pytest.mark.asyncio
async def test_relation_extraction_multiple_texts(relation_extractor: RelationExtractor, sample_entities: List[List[Entity]]):
    """Test relation extraction from multiple texts."""
    all_relations = []
    
    # Extract relations from all sample texts
    for i, (text, entities) in enumerate(zip(SAMPLE_TEXTS, sample_entities)):
        relations = await relation_extractor.extract(text, entities)
        all_relations.append(relations)
        
        # Verify that relations were extracted
        assert len(relations) > 0, f"No relations were extracted from text {i}"
        
        # Check that relations connect existing entities
        entity_ids = {entity.id for entity in entities}
        for relation in relations:
            assert relation.source_entity_id in entity_ids, f"Relation source not in entities for text {i}"
            assert relation.target_entity_id in entity_ids, f"Relation target not in entities for text {i}"
    
    # Check that we have different relations for different texts
    assert len(set(r.id for relations in all_relations for r in relations)) >= sum(len(relations) for relations in all_relations) * 0.8, \
        "Too many duplicate relations across different texts"


@pytest.mark.asyncio
async def test_relation_extraction_specific_pairs(relation_extractor: RelationExtractor, sample_entities: List[List[Entity]]):
    """Test relation extraction for specific entity pairs."""
    # Get entities from the first sample text
    text = SAMPLE_TEXTS[0]
    entities = sample_entities[0]
    
    # Find specific entities
    apple = next((e for e in entities if "apple" in e.name.lower() and e.type == EntityType.ORGANIZATION), None)
    steve_jobs = next((e for e in entities if "steve jobs" in e.name.lower()), None)
    tim_cook = next((e for e in entities if "tim cook" in e.name.lower()), None)
    
    if not all([apple, steve_jobs, tim_cook]):
        pytest.skip("Required entities not found in extracted entities")
    
    # Create specific entity pairs
    entity_pairs = [
        (steve_jobs.id, apple.id),
        (tim_cook.id, apple.id)
    ]
    
    # Extract relations for specific entity pairs
    relations = await relation_extractor.extract_for_entity_pairs(text, entities, entity_pairs)
    
    # Verify that relations were extracted
    assert len(relations) > 0, "No relations were extracted for specific entity pairs"
    
    # Check that only relations between specified pairs were extracted (in either direction)
    for relation in relations:
        pair_forward = (relation.source_entity_id, relation.target_entity_id)
        pair_reverse = (relation.target_entity_id, relation.source_entity_id)
        assert pair_forward in entity_pairs or pair_reverse in entity_pairs, \
            f"Relation between {relation.source_entity_id} and {relation.target_entity_id} not in specified pairs"
    
    # Check for specific relationships (bidirectional)
    steve_jobs_founded_apple = any(
        ((r.source_entity_id == steve_jobs.id and r.target_entity_id == apple.id) or
         (r.source_entity_id == apple.id and r.target_entity_id == steve_jobs.id)) and r.type == RelationType.FOUNDED
        for r in relations
    )
    
    tim_cook_works_for_apple = any(
        ((r.source_entity_id == tim_cook.id and r.target_entity_id == apple.id) or
         (r.source_entity_id == apple.id and r.target_entity_id == tim_cook.id)) and r.type == RelationType.WORKS_FOR
        for r in relations
    )
    
    assert steve_jobs_founded_apple, "Expected FOUNDED relation between Steve Jobs and Apple not found"
    assert tim_cook_works_for_apple, "Expected WORKS_FOR relation between Tim Cook and Apple not found"


@pytest.mark.asyncio
async def test_relation_extraction_with_document_id(relation_extractor: RelationExtractor, sample_entities: List[List[Entity]]):
    """Test relation extraction with document ID."""
    doc_id = "test-document-123"
    
    # Extract relations with document ID
    text = SAMPLE_TEXTS[0]
    entities = sample_entities[0]
    
    relations = await relation_extractor.extract(text, entities, doc_id=doc_id)
    
    # Verify that relations were extracted
    assert len(relations) > 0, "No relations were extracted"
    
    # Check that all relations have the correct document ID
    for relation in relations:
        # Document-specific attributes removed for document-agnostic extraction
        # assert relation.source_doc_id == doc_id, f"Relation {relation.id} has incorrect document ID"
        pass  # Placeholder since document-specific checks are removed


@pytest.mark.asyncio
async def test_relation_serialization(relation_extractor: RelationExtractor, sample_entities: List[List[Entity]]):
    """Test relation serialization to/from JSON."""
    # Extract relations
    text = SAMPLE_TEXTS[0]
    entities = sample_entities[0]
    
    relations = await relation_extractor.extract(text, entities)
    assert len(relations) > 0, "No relations were extracted"
    
    # Serialize to JSON
    relations_json = [relation.model_dump_json() for relation in relations]
    
    # Deserialize from JSON
    deserialized_relations = [Relation.model_validate_json(relation_json) for relation_json in relations_json]
    
    # Check that deserialized relations match original relations
    for original, deserialized in zip(relations, deserialized_relations):
        assert original.id == deserialized.id
        assert original.source_entity_id == deserialized.source_entity_id
        assert original.target_entity_id == deserialized.target_entity_id
        assert original.type == deserialized.type
        assert original.confidence == deserialized.confidence
        # source_text field removed in new document structure
        # Document-specific attributes removed for document-agnostic extraction
    # assert original.source_doc_id == deserialized.source_doc_id


@pytest.mark.asyncio
async def test_relation_extraction_batch(relation_extractor: RelationExtractor, sample_entities: List[List[Entity]]):
    """Test batch relation extraction."""
    # Extract relations from all texts in a batch
    extraction_tasks = [
        relation_extractor.extract(text, entities)
        for text, entities in zip(SAMPLE_TEXTS, sample_entities)
    ]
    
    all_relations = await asyncio.gather(*extraction_tasks)
    
    # Verify that relations were extracted for each text
    for i, relations in enumerate(all_relations):
        assert len(relations) > 0, f"No relations were extracted from text {i}"
        
        # Check that relations connect existing entities
        entity_ids = {entity.id for entity in sample_entities[i]}
        for relation in relations:
            assert relation.source_entity_id in entity_ids, f"Relation source not in entities for text {i}"
            assert relation.target_entity_id in entity_ids, f"Relation target not in entities for text {i}"


def test_relation_neo4j_conversion():
    """Test relation conversion to/from Neo4J format."""
    # Create a test relation
    relation = Relation(
        id="test-relation-1",
        source_entity_id="entity-1",
        target_entity_id="entity-2",
        type=RelationType.WORKS_FOR,
        attributes={"role": "CEO", "since": 2011},
        # source_doc_id removed for document-agnostic extraction
        confidence=0.95,
        weight=1.0
    )
    
    # Convert to Neo4J relationship properties
    rel_props = relation.to_neo4j_relationship()
    
    # Check that all properties are present
    assert rel_props["id"] == relation.id
    assert json.loads(rel_props["attributes"]) == relation.attributes
    # Document-specific attributes removed for document-agnostic extraction
    # assert rel_props["source_doc_id"] == relation.source_doc_id
    assert rel_props["confidence"] == relation.confidence
    assert rel_props["weight"] == relation.weight
    
    # Convert back from Neo4J relationship properties
    reconstructed = Relation.from_neo4j_relationship(
        rel_props, 
        relation.source_entity_id, 
        relation.target_entity_id
    )
    
    # Check that reconstructed relation matches original
    assert reconstructed.id == relation.id
    assert reconstructed.source_entity_id == relation.source_entity_id
    assert reconstructed.target_entity_id == relation.target_entity_id
    assert reconstructed.type == relation.type
    assert reconstructed.attributes == relation.attributes
    # source_text field removed in new document structure
    # Document-specific attributes removed for document-agnostic extraction
    # assert reconstructed.source_doc_id == relation.source_doc_id
    assert reconstructed.confidence == relation.confidence
    assert reconstructed.weight == relation.weight


@pytest.mark.asyncio
async def test_graph_construction(entity_extractor: EntityExtractor, relation_extractor: RelationExtractor):
    """Test constructing a graph from extracted entities and relations."""
    # Extract entities and relations from the first sample text
    text = SAMPLE_TEXTS[0]
    
    # Extract entities
    entities = await entity_extractor.extract(text)
    assert len(entities) > 0, "No entities were extracted"
    
    # Extract relations
    relations = await relation_extractor.extract(text, entities)
    assert len(relations) > 0, "No relations were extracted"
    
    # Create a graph
    graph = Graph()
    
    # Add entities and relations to the graph
    for entity in entities:
        graph.add_entity(entity)
    
    for relation in relations:
        graph.add_relation(relation)
    
    # Verify graph construction
    assert len(graph.entities) == len(entities), "Not all entities were added to the graph"
    assert len(graph.relations) == len(relations), "Not all relations were added to the graph"
    
    # Test graph queries
    # Find Apple entity
    apple = next((e for e in entities if "apple" in e.name.lower() and e.type == EntityType.ORGANIZATION), None)
    if apple:
        # Get relations involving Apple
        apple_relations = graph.get_entity_relations(apple.id)
        assert len(apple_relations) > 0, "No relations found for Apple entity"
        
        # Get neighbors of Apple
        neighbors = graph.get_neighbors(apple.id)
        assert len(neighbors) > 0, "No neighbors found for Apple entity"


if __name__ == "__main__":
    # This allows running the tests directly with asyncio
    asyncio.run(pytest.main([__file__]))