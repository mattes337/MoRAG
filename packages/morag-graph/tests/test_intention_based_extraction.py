"""Test intention-based entity and relation extraction."""

import pytest
import asyncio
import os
from unittest.mock import AsyncMock, patch

from morag_graph.extraction.entity_extractor import EntityExtractor
from morag_graph.extraction.relation_extractor import RelationExtractor
# Import LLMConfig from morag-reasoning package
try:
    from morag_reasoning.llm import LLMConfig
except ImportError:
    # Fallback LLMConfig for compatibility
    from pydantic import BaseModel
    class LLMConfig(BaseModel):
        provider: str = "gemini"
        model: str = "gemini-1.5-flash"
        api_key: str = None
        temperature: float = 0.1
        max_tokens: int = 2000
from morag_graph.models import Entity, Relation


@pytest.fixture
def llm_config():
    """Create test LLM configuration."""
    return LLMConfig(
        provider="gemini",
        model="gemini-1.5-flash",
        api_key="test-key",
        temperature=0.1
    )


@pytest.fixture
def entity_extractor(llm_config):
    """Create entity extractor for testing."""
    return EntityExtractor(config=llm_config)


@pytest.fixture
def relation_extractor(llm_config):
    """Create relation extractor for testing."""
    return RelationExtractor(config=llm_config)


class TestIntentionBasedExtraction:
    """Test intention-based extraction functionality."""
    
    @pytest.mark.asyncio
    async def test_entity_extraction_with_intention(self, entity_extractor):
        """Test entity extraction with intention context."""
        # Mock the LLM response
        mock_response = '''[
            {
                "name": "pineal gland",
                "type": "ANATOMICAL",
                "context": "gland mentioned for spiritual healing",
                "confidence": 0.9
            },
            {
                "name": "meditation",
                "type": "PRACTICE",
                "context": "spiritual practice for healing",
                "confidence": 0.85
            }
        ]'''
        
        with patch.object(entity_extractor, 'call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            
            text = "The pineal gland can be healed through meditation and spiritual practices."
            intention = "Heal the pineal gland for spiritual enlightenment"
            
            entities = await entity_extractor.extract(text, intention=intention)
            
            assert len(entities) == 2
            assert entities[0].name == "pineal gland"
            assert entities[0].type == "ANATOMICAL"  # Should be abstract type
            assert entities[1].name == "meditation"
            assert entities[1].type == "PRACTICE"
            
            # Verify intention was passed in the prompt
            call_args = mock_llm.call_args[0][0]  # Get the messages
            user_message = call_args[1]["content"]
            assert "Document intention: Heal the pineal gland for spiritual enlightenment" in user_message
    
    @pytest.mark.asyncio
    async def test_relation_extraction_with_intention(self, relation_extractor):
        """Test relation extraction with intention context."""
        # Create test entities
        entities = [
            Entity(name="meditation", type="PRACTICE", confidence=0.9),
            Entity(name="pineal gland", type="ANATOMICAL", confidence=0.9)
        ]
        
        # Mock the LLM response
        mock_response = '''[
            {
                "source_entity": "meditation",
                "target_entity": "pineal gland",
                "relation_type": "AFFECTS",
                "context": "meditation heals the pineal gland",
                "confidence": 0.9
            }
        ]'''
        
        with patch.object(relation_extractor, 'call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            
            text = "Meditation can heal and activate the pineal gland for spiritual awakening."
            intention = "Heal the pineal gland for spiritual enlightenment"
            
            relations = await relation_extractor.extract(text, entities=entities, intention=intention)
            
            assert len(relations) == 1
            assert relations[0].attributes["source_entity_name"] == "meditation"
            assert relations[0].attributes["target_entity_name"] == "pineal gland"
            assert relations[0].type == "AFFECTS"  # Should be abstract type
            
            # Verify intention was passed in the prompt
            call_args = mock_llm.call_args[0][0]  # Get the messages
            user_message = call_args[1]["content"]
            assert "Document intention: Heal the pineal gland for spiritual enlightenment" in user_message
    
    @pytest.mark.asyncio
    async def test_organizational_intention_extraction(self, entity_extractor, relation_extractor):
        """Test extraction with organizational document intention."""
        # Test entity extraction
        entity_mock_response = '''[
            {
                "name": "John Smith",
                "type": "PERSON",
                "context": "CEO of the company",
                "confidence": 0.9
            },
            {
                "name": "TechCorp",
                "type": "ORGANIZATION",
                "context": "the company being described",
                "confidence": 0.95
            }
        ]'''
        
        # Test relation extraction
        relation_mock_response = '''[
            {
                "source_entity": "John Smith",
                "target_entity": "TechCorp",
                "relation_type": "IS_MEMBER",
                "context": "John Smith is the CEO of TechCorp",
                "confidence": 0.9
            }
        ]'''
        
        text = "John Smith is the CEO of TechCorp and leads the engineering division."
        intention = "Document explaining the structure of the organization/company"
        
        # Test entity extraction
        with patch.object(entity_extractor, 'call_llm', new_callable=AsyncMock) as mock_entity_llm:
            mock_entity_llm.return_value = entity_mock_response
            
            entities = await entity_extractor.extract(text, intention=intention)
            
            assert len(entities) == 2
            assert any(e.name == "John Smith" and e.type == "PERSON" for e in entities)
            assert any(e.name == "TechCorp" and e.type == "ORGANIZATION" for e in entities)
        
        # Test relation extraction
        with patch.object(relation_extractor, 'call_llm', new_callable=AsyncMock) as mock_relation_llm:
            mock_relation_llm.return_value = relation_mock_response
            
            relations = await relation_extractor.extract(text, entities=entities, intention=intention)
            
            assert len(relations) == 1
            assert relations[0].type == "IS_MEMBER"  # Should use abstract relation type
    
    def test_entity_prompt_includes_abstraction_guidance(self, entity_extractor):
        """Test that entity extraction prompts include abstraction guidance."""
        system_prompt = entity_extractor.get_system_prompt()

        # Check for abstraction guidance
        assert "BROAD, REUSABLE entity types" in system_prompt
        assert "SINGULAR" in system_prompt
        assert "BODY_PART" in system_prompt  # Example of broad typing
        assert "TECHNOLOGY" in system_prompt  # Example of broad typing
        assert "UNCONJUGATED" in system_prompt  # Entity name normalization
    
    def test_relation_prompt_includes_abstraction_guidance(self, relation_extractor):
        """Test that relation extraction prompts include abstraction guidance."""
        system_prompt = relation_extractor.get_system_prompt()

        # Check for abstraction guidance
        assert "SIMPLE, descriptive relation type" in system_prompt
        assert "NORMALIZED name" in system_prompt
        assert "SINGULAR, UNCONJUGATED form" in system_prompt
        assert "GENERAL types over overly specific ones" in system_prompt
    
    @pytest.mark.asyncio
    async def test_extract_entities_alias_method(self, entity_extractor):
        """Test the extract_entities alias method."""
        mock_response = '''[
            {
                "name": "test entity",
                "type": "CONCEPT",
                "context": "test context",
                "confidence": 0.9
            }
        ]'''
        
        with patch.object(entity_extractor, 'call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            
            entities = await entity_extractor.extract_entities(
                "Test text", 
                source_doc_id="test-doc",
                intention="Test intention"
            )
            
            assert len(entities) == 1
            assert entities[0].name == "test entity"
    
    @pytest.mark.asyncio
    async def test_extract_relations_alias_method(self, relation_extractor):
        """Test the extract_relations alias method."""
        entities = [Entity(name="entity1", type="CONCEPT", confidence=0.9)]
        
        mock_response = '''[
            {
                "source_entity": "entity1",
                "target_entity": "entity2",
                "relation_type": "RELATED_TO",
                "context": "test relation",
                "confidence": 0.9
            }
        ]'''
        
        with patch.object(relation_extractor, 'call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            
            relations = await relation_extractor.extract_relations(
                "Test text",
                entities=entities,
                source_doc_id="test-doc",
                intention="Test intention"
            )
            
            assert len(relations) == 1
            assert relations[0].attributes["source_entity_name"] == "entity1"


if __name__ == "__main__":
    pytest.main([__file__])
