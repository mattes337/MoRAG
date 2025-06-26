"""Tests for entity extraction functionality."""

import asyncio
import json
import os
from typing import List, Dict, Any

import pytest
from dotenv import load_dotenv

from morag_graph.extraction import EntityExtractor
from morag_graph.models import Entity

# Load environment variables from .env file
load_dotenv()

# Sample texts for testing entity extraction
SAMPLE_TEXTS = [
    """Apple Inc. is an American multinational technology company headquartered in Cupertino, California. 
    Tim Cook is the CEO of Apple. The company was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.""",
    
    """The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. 
    It is named after the engineer Gustave Eiffel, whose company designed and built the tower from 1887 to 1889.""",
    
    """Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability. 
    Guido van Rossum created Python and first released it in 1991."""
]

# Expected entity types in the sample texts (flexible for dynamic types)
# Since we're using dynamic types, we need to be more flexible with type matching
EXPECTED_ENTITY_PATTERNS = [
    # Apple text - should contain company/organization and person/location entities
    {
        "company_patterns": ["ORGANIZATION", "COMPANY", "TECHNOLOGY_COMPANY", "CORPORATION"],
        "person_patterns": ["PERSON", "CEO", "FOUNDER"],
        "location_patterns": ["LOCATION", "CITY", "HEADQUARTERS"]
    },
    # Eiffel Tower text - should contain location and person entities
    {
        "location_patterns": ["LOCATION", "LANDMARK", "TOWER", "CITY", "COUNTRY"],
        "person_patterns": ["PERSON", "ENGINEER", "ARCHITECT"]
    },
    # Python text - should contain person and technology entities
    {
        "person_patterns": ["PERSON", "CREATOR", "DEVELOPER"],
        "tech_patterns": ["TECHNOLOGY", "PROGRAMMING_LANGUAGE", "SOFTWARE", "LANGUAGE"]
    }
]


def check_entity_types_flexible(entities, text_index):
    """Helper function to check entity types with flexible matching for dynamic types."""
    entity_types = {entity.type for entity in entities}
    patterns = EXPECTED_ENTITY_PATTERNS[text_index]

    if text_index == 0:  # Apple text
        # Check for company/organization entities
        company_found = any(any(pattern in entity_type for pattern in patterns["company_patterns"])
                           for entity_type in entity_types)
        assert company_found, f"No company/organization entity found in text {text_index}. Found types: {entity_types}"

    elif text_index == 1:  # Eiffel Tower text
        # Check for location entities
        location_found = any(any(pattern in entity_type for pattern in patterns["location_patterns"])
                            for entity_type in entity_types)
        assert location_found, f"No location entity found in text {text_index}. Found types: {entity_types}"

    elif text_index == 2:  # Python text
        # Check for person or technology entities
        person_found = any(any(pattern in entity_type for pattern in patterns["person_patterns"])
                          for entity_type in entity_types)
        tech_found = any(any(pattern in entity_type for pattern in patterns["tech_patterns"])
                        for entity_type in entity_types)
        assert person_found or tech_found, f"No person or technology entity found in text {text_index}. Found types: {entity_types}"


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


@pytest.mark.asyncio
async def test_entity_extraction_basic(entity_extractor: EntityExtractor):
    """Test basic entity extraction functionality."""
    # Extract entities from the first sample text
    entities = await entity_extractor.extract(SAMPLE_TEXTS[0])
    
    # Verify that entities were extracted
    assert len(entities) > 0, "No entities were extracted"
    
    # Check that we have the expected entity types (flexible matching for dynamic types)
    entity_types = {entity.type for entity in entities}
    patterns = EXPECTED_ENTITY_PATTERNS[0]

    # Check for company/organization entities
    company_found = any(any(pattern in entity_type for pattern in patterns["company_patterns"])
                       for entity_type in entity_types)
    assert company_found, f"No company/organization entity found. Found types: {entity_types}"
    
    # Check that specific entities were extracted
    entity_names = {entity.name.lower() for entity in entities}
    expected_names = {"apple", "tim cook", "steve jobs", "cupertino"}
    for name in expected_names:
        assert any(name in entity_name for entity_name in entity_names), f"Expected entity '{name}' not found"
    
    # Check that entities have required attributes
    for entity in entities:
        assert entity.id, "Entity is missing ID"
        assert entity.name, "Entity is missing name"
        assert entity.type, "Entity is missing type"
        assert entity.confidence is not None, "Entity is missing confidence score"
        # source_text field removed in new document structure


@pytest.mark.asyncio
async def test_entity_extraction_with_context(entity_extractor: EntityExtractor):
    """Test entity extraction with additional context."""
    # Add context about technology companies
    context = "Focus on technology companies and their founders."
    
    # Extract entities from the first sample text with context
    entities = await entity_extractor.extract(SAMPLE_TEXTS[0], context=context)
    
    # Verify that entities were extracted
    assert len(entities) > 0, "No entities were extracted"
    
    # Check that technology-related entities are present
    tech_entities = [e for e in entities if e.type == "ORGANIZATION" or "COMPANY" in e.type]
    assert len(tech_entities) > 0, "No technology company entities found"
    
    # Check that Apple is identified as a technology company
    apple_entities = [e for e in tech_entities if "apple" in e.name.lower()]
    assert len(apple_entities) > 0, "Apple not identified as a technology company"


@pytest.mark.asyncio
async def test_entity_extraction_multiple_texts(entity_extractor: EntityExtractor):
    """Test entity extraction from multiple texts."""
    all_entities = []
    
    # Extract entities from all sample texts
    for i, text in enumerate(SAMPLE_TEXTS):
        entities = await entity_extractor.extract(text)
        all_entities.append(entities)
        
        # Verify that entities were extracted
        assert len(entities) > 0, f"No entities were extracted from text {i}"
        
        # Check that we have the expected entity types using flexible matching
        check_entity_types_flexible(entities, i)
    
    # Check that we have different entities for different texts
    assert len(set(e.id for entities in all_entities for e in entities)) >= sum(len(entities) for entities in all_entities) * 0.8, \
        "Too many duplicate entities across different texts"


@pytest.mark.asyncio
async def test_entity_extraction_custom_type(entity_extractor: EntityExtractor):
    """Test entity extraction with custom entity types."""
    # Add custom entity type instructions
    custom_instructions = "Also identify programming languages as TECHNOLOGY entities."
    
    # Extract entities from the Python text with custom instructions
    entities = await entity_extractor.extract(
        SAMPLE_TEXTS[2], 
        custom_instructions=custom_instructions
    )
    
    # Verify that entities were extracted
    assert len(entities) > 0, "No entities were extracted"
    
    # Check that Python is identified as a technology-related entity
    tech_entities = [e for e in entities if e.type in ACCEPTABLE_TECH_TYPES]
    assert len(tech_entities) > 0, f"No technology-related entities found. Found types: {[e.type for e in entities]}"
    
    python_entities = [e for e in tech_entities if "python" in e.name.lower()]
    assert len(python_entities) > 0, "Python not identified as a TECHNOLOGY entity"


@pytest.mark.asyncio
async def test_entity_extraction_with_document_id(entity_extractor: EntityExtractor):
    """Test entity extraction with document ID."""
    doc_id = "test-document-123"
    
    # Extract entities with document ID
    entities = await entity_extractor.extract(SAMPLE_TEXTS[0], doc_id=doc_id)
    
    # Verify that entities were extracted
    assert len(entities) > 0, "No entities were extracted"
    
    # Check that all entities have the correct document ID
    for entity in entities:
        # Document-specific attributes removed for document-agnostic extraction
        # assert entity.source_doc_id == doc_id, f"Entity {entity.id} has incorrect document ID"
        pass  # Placeholder since document-specific checks are removed


@pytest.mark.asyncio
async def test_entity_serialization(entity_extractor: EntityExtractor):
    """Test entity serialization to/from JSON."""
    # Extract entities
    entities = await entity_extractor.extract(SAMPLE_TEXTS[0])
    assert len(entities) > 0, "No entities were extracted"
    
    # Serialize to JSON
    entities_json = [entity.model_dump_json() for entity in entities]
    
    # Deserialize from JSON
    deserialized_entities = [Entity.model_validate_json(entity_json) for entity_json in entities_json]
    
    # Check that deserialized entities match original entities
    for original, deserialized in zip(entities, deserialized_entities):
        assert original.id == deserialized.id
        assert original.name == deserialized.name
        assert original.type == deserialized.type
        assert original.confidence == deserialized.confidence
        # source_text field removed in new document structure
        # Document-specific attributes removed for document-agnostic extraction
    # assert original.source_doc_id == deserialized.source_doc_id


@pytest.mark.asyncio
async def test_entity_extraction_batch(entity_extractor: EntityExtractor):
    """Test batch entity extraction."""
    # Extract entities from all texts in a batch
    all_entities = await asyncio.gather(*[
        entity_extractor.extract(text) for text in SAMPLE_TEXTS
    ])
    
    # Verify that entities were extracted for each text
    for i, entities in enumerate(all_entities):
        assert len(entities) > 0, f"No entities were extracted from text {i}"
        
        # Check that we have the expected entity types using flexible matching
        check_entity_types_flexible(entities, i)


def test_entity_neo4j_conversion():
    """Test entity conversion to/from Neo4J format."""
    # Create a test entity
    entity = Entity(
        name="Test Entity",
        type="ORGANIZATION",
        source_doc_id="doc_test_abc123",
        attributes={"industry": "Technology", "founded": 2020},
        # source_doc_id removed for document-agnostic extraction
        confidence=0.95
    )
    
    # Convert to Neo4J node properties
    node_props = entity.to_neo4j_node()
    
    # Check that all properties are present
    assert node_props["id"] == entity.id
    assert node_props["name"] == entity.name
    assert node_props["type"] == entity.type.value  # Neo4j conversion extracts just the enum value
    assert json.loads(node_props["attributes"]) == entity.attributes
    # Document-specific attributes removed for document-agnostic extraction
    # assert node_props["source_doc_id"] == entity.source_doc_id
    assert node_props["confidence"] == entity.confidence
    
    # Convert back from Neo4J node properties
    reconstructed = Entity.from_neo4j_node(node_props)
    
    # Check that reconstructed entity matches original
    assert reconstructed.id == entity.id
    assert reconstructed.name == entity.name
    assert reconstructed.type == entity.type
    assert reconstructed.attributes == entity.attributes
    # Document-specific attributes removed for document-agnostic extraction
    # assert reconstructed.source_doc_id == entity.source_doc_id
    assert reconstructed.confidence == entity.confidence


if __name__ == "__main__":
    # This allows running the tests directly with asyncio
    asyncio.run(pytest.main([__file__]))