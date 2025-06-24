"""Async tests for dynamic entity and relation types functionality.

These tests cover the async extraction methods with dynamic types,
ensuring proper integration with LLM calls and response parsing.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Optional
import json

from morag_graph.extraction import EntityExtractor, RelationExtractor
from morag_graph.extraction.base import LLMConfig
from morag_graph.models import Entity, Relation, EntityType, RelationType


class TestEntityExtractorAsyncDynamicTypes:
    """Test async entity extraction with dynamic types."""
    
    @pytest.fixture
    def mock_llm_response_entities(self):
        """Mock LLM response for entity extraction."""
        return [
            {
                "name": "John Doe",
                "type": "PERSON",
                "confidence": 0.95,
                "source_text": "John Doe is a researcher"
            },
            {
                "name": "COVID-19",
                "type": "DISEASE",
                "confidence": 0.98,
                "source_text": "COVID-19 pandemic"
            }
        ]
    
    @pytest.fixture
    def mock_llm_response_custom_types(self):
        """Mock LLM response with custom entity types."""
        return [
            {
                "name": "Diabetes",
                "type": "MEDICAL_CONDITION",
                "confidence": 0.92,
                "source_text": "Type 2 diabetes"
            },
            {
                "name": "Insulin",
                "type": "MEDICATION",
                "confidence": 0.89,
                "source_text": "insulin therapy"
            }
        ]
    
    @pytest.mark.asyncio
    async def test_extract_with_default_types(self, mock_llm_response_entities):
        """Test async extraction with default entity types."""
        config = LLMConfig(provider="mock", model="test")
        extractor = EntityExtractor(config)
        
        with patch.object(extractor, 'call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = json.dumps(mock_llm_response_entities)
            
            text = "John Doe is researching COVID-19."
            entities = await extractor.extract(text)
            
            # Verify extraction worked
            assert len(entities) == 2
            assert entities[0].name == "John Doe"
            assert entities[0].type == EntityType.PERSON
            assert entities[1].name == "COVID-19"
            assert entities[1].type == "DISEASE"  # Custom type from response
            
            # Verify LLM was called with default types in prompt
            mock_llm.assert_called_once()
            call_args = mock_llm.call_args[0]
            messages = call_args[0]
            system_prompt = messages[0]['content']  # Extract the actual system prompt content
            
            # Should use dynamic prompt without predefined types
            assert "semantic meaning" in system_prompt or "not limit yourself" in system_prompt
    
    @pytest.mark.asyncio
    async def test_extract_with_custom_types(self, mock_llm_response_custom_types):
        """Test async extraction with custom entity types."""
        config = LLMConfig(provider="mock", model="test")
        custom_types = {
            "MEDICAL_CONDITION": "Disease or health condition",
            "MEDICATION": "Pharmaceutical drug or treatment"
        }
        extractor = EntityExtractor(config, entity_types=custom_types)
        
        with patch.object(extractor, 'call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = json.dumps(mock_llm_response_custom_types)
            
            text = "Type 2 diabetes is treated with insulin therapy."
            entities = await extractor.extract(text)
            
            # Verify extraction worked with custom types
            assert len(entities) == 2
            assert entities[0].name == "Diabetes"
            assert entities[0].type == "MEDICAL_CONDITION"
            assert entities[1].name == "Insulin"
            assert entities[1].type == "MEDICATION"
            
            # Verify LLM was called with custom types in prompt
            mock_llm.assert_called_once()
            call_args = mock_llm.call_args[0]
            messages = call_args[0]
            system_prompt = messages[0]['content']  # Extract the actual system prompt content
            
            # Should contain custom types, not defaults
            assert "- MEDICAL_CONDITION: Disease or health condition" in system_prompt
            assert "- MEDICATION: Pharmaceutical drug" in system_prompt
            assert "- PERSON: Names of people" not in system_prompt
    
    @pytest.mark.asyncio
    async def test_extract_with_empty_types(self):
        """Test async extraction with empty entity types."""
        config = LLMConfig(provider="mock", model="test")
        extractor = EntityExtractor(config, entity_types={})
        
        mock_response = []
        
        with patch.object(extractor, 'call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = json.dumps(mock_response)
            
            text = "Some text to extract from."
            entities = await extractor.extract(text)
            
            # Should handle empty types gracefully
            assert entities == []
            
            # Verify LLM was called with empty types
            mock_llm.assert_called_once()
            call_args = mock_llm.call_args[0]
            messages = call_args[0]
            system_prompt = messages[0]['content']  # Extract the actual system prompt content
            
            # Should not contain any default types
            assert "- PERSON:" not in system_prompt
            assert "- ORGANIZATION:" not in system_prompt
    
    @pytest.mark.asyncio
    async def test_extract_with_context_and_custom_types(self):
        """Test async extraction with both context and custom types."""
        config = LLMConfig(provider="mock", model="test")
        custom_types = {"SYMPTOM": "Observable sign of illness"}
        extractor = EntityExtractor(config, entity_types=custom_types)
        
        mock_response = [
            {
                "name": "fever",
                "type": "SYMPTOM",
                "confidence": 0.91,
                "source_text": "patient has fever"
            }
        ]
        
        with patch.object(extractor, 'call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = json.dumps(mock_response)
            
            text = "Patient presents with fever and fatigue."
            context = "Focus on medical symptoms."
            entities = await extractor.extract(text, context=context)
            
            # Verify extraction worked
            assert len(entities) == 1
            assert entities[0].name == "fever"
            assert entities[0].type == "SYMPTOM"
            
            # Verify both custom types and context were used
            mock_llm.assert_called_once()
            call_args = mock_llm.call_args[0]
            messages = call_args[0]
            system_prompt = messages[0]['content']  # Extract the actual system prompt content
            user_prompt = messages[1]['content']  # Extract the user prompt content
            
            assert "- SYMPTOM: Observable sign of illness" in system_prompt
            assert "Additional context: Focus on medical symptoms" in user_prompt


class TestRelationExtractorAsyncDynamicTypes:
    """Test async relation extraction with dynamic types."""
    
    @pytest.fixture
    def sample_entities(self):
        """Sample entities for relation extraction tests."""
        entities = [
            Entity(
                name="John Doe",
                type=EntityType.PERSON,
                source_doc_id="doc_test_abc123",
                confidence=0.95
            ),
            Entity(
                name="Acme Corp",
                type=EntityType.ORGANIZATION,
                source_doc_id="doc_test_abc123",
                confidence=0.92
            )
        ]
        entities[0].attributes["source_text"] = "John Doe"
        entities[1].attributes["source_text"] = "Acme Corp"
        return entities
    
    @pytest.fixture
    def mock_llm_response_relations(self):
        """Mock LLM response for relation extraction."""
        return [
            {
                "source_entity": "John Doe",
                "target_entity": "Acme Corp",
                "relation_type": "WORKS_FOR",
                "confidence": 0.88,
                "context": "John Doe works for Acme Corp"
            }
        ]
    
    @pytest.fixture
    def mock_llm_response_custom_relations(self):
        """Mock LLM response with custom relation types."""
        return [
            {
                "source_entity": "Dr. Smith",
                "target_entity": "patient",
                "relation_type": "TREATS_PATIENT",
                "confidence": 0.94,
                "context": "Dr. Smith treats patient"
            }
        ]
    
    @pytest.mark.asyncio
    async def test_extract_with_default_types(self, sample_entities, mock_llm_response_relations):
        """Test async relation extraction with default types."""
        config = LLMConfig(provider="mock", model="test")
        extractor = RelationExtractor(config)
        
        with patch.object(extractor, 'call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = json.dumps(mock_llm_response_relations)
            
            text = "John Doe works for Acme Corp."
            relations = await extractor.extract(text, sample_entities)
            
            # Verify extraction worked
            assert len(relations) == 1
            assert relations[0].attributes["source_entity_name"] == "John Doe"
            assert relations[0].attributes["target_entity_name"] == "Acme Corp"
            assert relations[0].type == RelationType.WORKS_FOR
            
            # Verify LLM was called with default types
            mock_llm.assert_called_once()
            call_args = mock_llm.call_args[0]
            messages = call_args[0]
            system_prompt = messages[0]['content']  # Extract the actual system prompt content
            
            # Should use dynamic prompt without predefined types
            assert "semantic meaning" in system_prompt or "not limit yourself" in system_prompt
    
    @pytest.mark.asyncio
    async def test_extract_with_custom_types(self, sample_entities, mock_llm_response_custom_relations):
        """Test async relation extraction with custom types."""
        config = LLMConfig(provider="mock", model="test")
        custom_types = {
            "TREATS_PATIENT": "Medical professional treats patient",
            "PRESCRIBES": "Doctor prescribes medication"
        }
        extractor = RelationExtractor(config, relation_types=custom_types)
        
        with patch.object(extractor, 'call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = json.dumps(mock_llm_response_custom_relations)
            
            # Create entities that match the mock response
            entities = [
                Entity(
                    name="Dr. Smith",
                    type=EntityType.PERSON,
                    source_doc_id="doc_test_abc123",
                    confidence=0.95
                ),
                Entity(
                    name="patient",
                    type=EntityType.PERSON,
                    source_doc_id="doc_test_abc123",
                    confidence=0.92
                )
            ]
            entities[0].attributes["source_text"] = "Dr. Smith"
            entities[1].attributes["source_text"] = "patient"
            
            text = "Dr. Smith treats the patient."
            relations = await extractor.extract(text, entities)
            
            # Verify extraction worked with custom types
            assert len(relations) == 1
            assert relations[0].attributes["source_entity_name"] == "Dr. Smith"
            assert relations[0].attributes["target_entity_name"] == "patient"
            assert relations[0].type == "TREATS_PATIENT"  # Custom types are kept as strings
            
            # Verify LLM was called with custom types
            mock_llm.assert_called_once()
            call_args = mock_llm.call_args[0]
            messages = call_args[0]
            system_prompt = messages[0]['content']  # Extract the actual system prompt content
            
            assert "- TREATS_PATIENT: Medical professional treats patient" in system_prompt
            assert "- PRESCRIBES: Doctor prescribes medication" in system_prompt
            assert "- WORKS_FOR: Person works for" not in system_prompt
    
    @pytest.mark.asyncio
    async def test_extract_with_empty_types(self, sample_entities):
        """Test async relation extraction with empty types."""
        config = LLMConfig(provider="mock", model="test")
        extractor = RelationExtractor(config, relation_types={})
        
        mock_response = {"relations": []}
        
        with patch.object(extractor, 'call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = json.dumps(mock_response)
            
            text = "Some text with entities."
            relations = await extractor.extract(text, sample_entities)
            
            # Should handle empty types gracefully
            assert relations == []
            
            # Verify LLM was called with empty types
            mock_llm.assert_called_once()
            call_args = mock_llm.call_args[0]
            messages = call_args[0]
            system_prompt = messages[0]['content']  # Extract the actual system prompt content
            
            # Should not contain default types
            assert "- WORKS_FOR:" not in system_prompt
            assert "- LOCATED_IN:" not in system_prompt
    
    @pytest.mark.asyncio
    async def test_extract_with_context_and_custom_types(self, sample_entities):
        """Test async relation extraction with context and custom types."""
        config = LLMConfig(provider="mock", model="test")
        custom_types = {"COLLABORATES_WITH": "Professional collaboration"}
        extractor = RelationExtractor(config, relation_types=custom_types)
        
        mock_response = [
            {
                "source_entity": "John",
                "target_entity": "Acme Corp",
                "relation_type": "COLLABORATES_WITH",
                "confidence": 0.87,
                "context": "collaboration between entities"
            }
        ]
        
        with patch.object(extractor, 'call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = json.dumps(mock_response)
            
            # Create entities that match the mock response
            entities = [
                Entity(
                    name="John",
                    type=EntityType.PERSON,
                    source_doc_id="doc_test_abc123",
                    confidence=0.95
                ),
                Entity(
                    name="Acme Corp",
                    type=EntityType.ORGANIZATION,
                    source_doc_id="doc_test_abc123",
                    confidence=0.92
                )
            ]
            entities[0].attributes["source_text"] = "John"
            entities[1].attributes["source_text"] = "Acme Corp"
            
            text = "John collaborates with Acme Corp on projects."
            context = "Focus on professional relationships."
            relations = await extractor.extract(text, entities, context=context)
            
            # Verify extraction worked
            assert len(relations) == 1
            assert relations[0].type == "COLLABORATES_WITH"  # Custom types are kept as strings
            assert relations[0].attributes["source_entity_name"] == "John"
            assert relations[0].attributes["target_entity_name"] == "Acme Corp"
            
            # Verify both custom types and context were used
            mock_llm.assert_called_once()
            call_args = mock_llm.call_args[0]
            messages = call_args[0]
            system_prompt = messages[0]['content']  # Extract the actual system prompt content
            user_prompt = messages[1]['content']  # Extract the user prompt content
            
            assert "- COLLABORATES_WITH: Professional collaboration" in system_prompt
            assert "Additional context: Focus on professional relationships." in user_prompt


class TestDynamicTypesErrorHandling:
    """Test error handling in async extraction with dynamic types."""
    
    @pytest.mark.asyncio
    async def test_invalid_json_response_with_custom_types(self):
        """Test handling of invalid JSON response with custom types."""
        config = LLMConfig(provider="mock", model="test")
        custom_types = {"CUSTOM_TYPE": "Custom entity type"}
        extractor = EntityExtractor(config, entity_types=custom_types)
        
        with patch.object(extractor, 'call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Invalid JSON response"
            
            text = "Some text to extract from."
            
            # Should handle invalid JSON gracefully and return empty list
            entities = await extractor.extract(text)
            assert entities == []
    
    @pytest.mark.asyncio
    async def test_missing_entities_field_with_custom_types(self):
        """Test handling of response missing entities field."""
        config = LLMConfig(provider="mock", model="test")
        custom_types = {"CUSTOM_TYPE": "Custom entity type"}
        extractor = EntityExtractor(config, entity_types=custom_types)
        
        with patch.object(extractor, 'call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = json.dumps({"wrong_field": []})
            
            text = "Some text to extract from."
            entities = await extractor.extract(text)
            
            # Should return empty list when entities field is missing
            assert entities == []
    
    @pytest.mark.asyncio
    async def test_malformed_entity_data_with_custom_types(self):
        """Test handling of malformed entity data."""
        config = LLMConfig(provider="mock", model="test")
        custom_types = {"CUSTOM_TYPE": "Custom entity type"}
        extractor = EntityExtractor(config, entity_types=custom_types)
        
        malformed_response = [
            {"name": "Valid Entity", "type": "CUSTOM_TYPE", "confidence": 0.9},
            {"missing_name": "Invalid", "type": "CUSTOM_TYPE"},  # Missing name
            {"name": "Another Valid", "type": "CUSTOM_TYPE", "confidence": "invalid"}  # Invalid confidence
        ]
        
        with patch.object(extractor, 'call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = json.dumps(malformed_response)
            
            text = "Some text to extract from."
            entities = await extractor.extract(text)
            
            # Should extract valid entities and skip malformed ones
            valid_entities = [e for e in entities if e.name in ["Valid Entity", "Another Valid"]]
            assert len(valid_entities) >= 1  # At least one valid entity should be extracted
    
    @pytest.mark.asyncio
    async def test_llm_call_failure_with_custom_types(self):
        """Test handling of LLM call failure."""
        config = LLMConfig(provider="mock", model="test")
        custom_types = {"CUSTOM_TYPE": "Custom entity type"}
        extractor = EntityExtractor(config, entity_types=custom_types)
        
        with patch.object(extractor, 'call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = Exception("LLM call failed")
            
            text = "Some text to extract from."
            
            # Should handle LLM call failure gracefully and return empty list
            entities = await extractor.extract(text)
            assert entities == []


class TestDynamicTypesPerformance:
    """Test performance aspects of dynamic types."""
    
    def test_system_prompt_caching_with_custom_types(self):
        """Test that system prompts are efficiently generated."""
        config = LLMConfig(provider="mock", model="test")
        custom_types = {f"TYPE_{i}": f"Description {i}" for i in range(50)}
        extractor = EntityExtractor(config, entity_types=custom_types)
        
        # Generate system prompt multiple times
        prompt1 = extractor.get_system_prompt()
        prompt2 = extractor.get_system_prompt()
        prompt3 = extractor.get_system_prompt()
        
        # Should generate consistent prompts
        assert prompt1 == prompt2 == prompt3
        
        # Should contain all custom types
        for i in range(50):
            assert f"TYPE_{i}: Description {i}" in prompt1
    
    def test_large_custom_types_handling(self):
        """Test handling of large numbers of custom types."""
        config = LLMConfig(provider="mock", model="test")
        
        # Create 200 custom entity types
        large_entity_types = {
            f"ENTITY_TYPE_{i:03d}": f"Description for entity type {i}"
            for i in range(200)
        }
        
        # Create 200 custom relation types
        large_relation_types = {
            f"RELATION_TYPE_{i:03d}": f"Description for relation type {i}"
            for i in range(200)
        }
        
        # Should handle large type sets without errors
        entity_extractor = EntityExtractor(config, entity_types=large_entity_types)
        relation_extractor = RelationExtractor(config, relation_types=large_relation_types)
        
        assert len(entity_extractor.entity_types) == 200
        assert len(relation_extractor.relation_types) == 200
        
        # System prompts should be generated successfully
        entity_prompt = entity_extractor.get_system_prompt()
        relation_prompt = relation_extractor.get_system_prompt()
        
        assert entity_prompt is not None
        assert relation_prompt is not None
        assert len(entity_prompt) > 1000  # Should be substantial
        assert len(relation_prompt) > 1000