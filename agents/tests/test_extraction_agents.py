"""Tests for extraction agents."""

import pytest
import asyncio
import os
from unittest.mock import Mock, AsyncMock, patch

# Set up test environment
os.environ["GEMINI_API_KEY"] = "test-key"

from agents.extraction.fact_extraction import FactExtractionAgent
from agents.extraction.entity_extraction import EntityExtractionAgent
from agents.extraction.relation_extraction import RelationExtractionAgent
from agents.extraction.keyword_extraction import KeywordExtractionAgent
from agents.extraction.models import (
    FactExtractionResult, ExtractedFact, ConfidenceLevel,
    EntityExtractionResult, ExtractedEntity, EntityTypeExamples,
    RelationExtractionResult, ExtractedRelation, RelationTypeExamples,
    KeywordExtractionResult, ExtractedKeyword
)
from agents.base.config import AgentConfig


class TestFactExtractionAgent:
    """Test fact extraction agent."""

    @pytest.fixture
    def fact_agent(self):
        """Create a fact extraction agent for testing."""
        config = AgentConfig(name="fact_extraction")
        return FactExtractionAgent(config)

    def test_agent_initialization(self, fact_agent):
        """Test agent initialization."""
        assert fact_agent.config.name == "fact_extraction"
        assert fact_agent.config.model.provider == "gemini"

    def test_config_update(self, fact_agent):
        """Test updating agent configuration."""
        fact_agent.update_config(agent_config={"max_facts": 15, "domain": "medical"})
        assert fact_agent.config.get_agent_config("max_facts") == 15
        assert fact_agent.config.get_agent_config("domain") == "medical"

    @pytest.mark.asyncio
    async def test_fact_extraction_empty_text(self, fact_agent):
        """Test fact extraction with empty text."""
        result = await fact_agent.extract_facts("")
        assert isinstance(result, FactExtractionResult)
        assert len(result.facts) == 0
        assert result.total_facts == 0
        assert "error" in result.metadata

    @pytest.mark.asyncio
    async def test_fact_extraction_medical_text(self, fact_agent):
        """Test fact extraction with medical text."""
        text = """
        Dr. Smith prescribed aspirin to treat the patient's headache.
        The medication was effective in reducing pain within 30 minutes.
        Vitamin D deficiency increases risk of respiratory infections.
        """

        with patch.object(fact_agent, '_call_model') as mock_llm:
            mock_llm.return_value = """{
                "facts": [
                    {
                        "subject": "Dr. Smith",
                        "object": "patient's headache",
                        "approach": "aspirin prescription",
                        "solution": "pain reduction within 30 minutes",
                        "condition": "headache symptoms",
                        "remarks": "effective treatment",
                        "fact_type": "MEDICAL_TREATMENT",
                        "confidence": 0.9,
                        "keywords": ["aspirin", "headache", "treatment"],
                        "source_text": "Dr. Smith prescribed aspirin"
                    }
                ],
                "total_facts": 1,
                "confidence": "high",
                "domain": "medical",
                "metadata": {}
            }"""

            result = await fact_agent.extract_facts(text, domain="medical")

            assert isinstance(result, FactExtractionResult)
            assert len(result.facts) == 1
            assert result.facts[0].subject == "Dr. Smith"
            assert result.facts[0].fact_type == "MEDICAL_TREATMENT"
            assert result.domain == "medical"


class TestEntityExtractionAgent:
    """Test entity extraction agent."""

    @pytest.fixture
    def entity_agent(self):
        """Create an entity extraction agent for testing."""
        config = AgentConfig(name="entity_extraction")
        return EntityExtractionAgent(config)

    def test_agent_initialization(self, entity_agent):
        """Test agent initialization."""
        assert entity_agent.config.name == "entity_extraction"

    @pytest.mark.asyncio
    async def test_entity_extraction(self, entity_agent):
        """Test entity extraction."""
        text = "Dr. John Smith works at Mayo Clinic in Rochester, Minnesota."

        with patch.object(entity_agent, '_call_model') as mock_llm:
            mock_llm.return_value = """{
                "entities": [
                    {
                        "name": "Dr. John Smith",
                        "canonical_name": "Dr. John Smith",
                        "entity_type": "PERSON",
                        "confidence": 0.95,
                        "start_offset": 0,
                        "end_offset": 13,
                        "attributes": {"title": "Doctor"},
                        "context": "Dr. John Smith works"
                    },
                    {
                        "name": "Mayo Clinic",
                        "canonical_name": "Mayo Clinic",
                        "entity_type": "ORGANIZATION",
                        "confidence": 0.9,
                        "start_offset": 23,
                        "end_offset": 34,
                        "attributes": {"type": "hospital"},
                        "context": "works at Mayo Clinic"
                    }
                ],
                "total_entities": 2,
                "confidence": "high",
                "metadata": {}
            }"""

            result = await entity_agent.extract_entities(text)

            assert isinstance(result, EntityExtractionResult)
            assert len(result.entities) == 2
            assert result.entities[0].name == "Dr. John Smith"
            assert result.entities[0].entity_type == "PERSON"
            assert result.entities[0].name == "Dr. John Smith"
            assert result.entities[1].entity_type == "ORGANIZATION"
            assert result.entities[1].name == "Mayo Clinic"


class TestRelationExtractionAgent:
    """Test relation extraction agent."""

    @pytest.fixture
    def relation_agent(self):
        """Create a relation extraction agent for testing."""
        config = AgentConfig(name="relation_extraction")
        return RelationExtractionAgent(config)

    @pytest.mark.asyncio
    async def test_relation_extraction(self, relation_agent):
        """Test relation extraction."""
        text = "Dr. Smith works at Mayo Clinic."
        entities = [
            ExtractedEntity(
                name="Dr. Smith",
                canonical_name="Dr. Smith",
                entity_type="PERSON",
                confidence=0.9,
                start_offset=0,
                end_offset=9
            ),
            ExtractedEntity(
                name="Mayo Clinic",
                canonical_name="Mayo Clinic",
                entity_type="ORGANIZATION",
                confidence=0.9,
                start_offset=19,
                end_offset=30
            )
        ]

        with patch.object(relation_agent, '_call_model') as mock_llm:
            mock_llm.return_value = """{
                "relations": [
                    {
                        "source_entity": "Dr. Smith",
                        "target_entity": "Mayo Clinic",
                        "relation_type": "WORKS_FOR",
                        "description": "Employment relationship between Dr. Smith and Mayo Clinic",
                        "confidence": 0.9,
                        "context": "Dr. Smith works at Mayo Clinic"
                    }
                ],
                "total_relations": 1,
                "confidence": "high",
                "metadata": {}
            }"""

            result = await relation_agent.extract_relations(text, entities)

            assert isinstance(result, RelationExtractionResult)
            assert len(result.relations) == 1
            assert result.relations[0].source_entity == "Dr. Smith"
            assert result.relations[0].relation_type == "WORKS_FOR"
            assert result.relations[0].source_entity == "Dr. Smith"
            assert result.relations[0].target_entity == "Mayo Clinic"


class TestKeywordExtractionAgent:
    """Test keyword extraction agent."""

    @pytest.fixture
    def keyword_agent(self):
        """Create a keyword extraction agent for testing."""
        config = AgentConfig(name="keyword_extraction")
        return KeywordExtractionAgent(config)

    @pytest.mark.asyncio
    async def test_keyword_extraction(self, keyword_agent):
        """Test keyword extraction."""
        text = """
        Machine learning algorithms are used for image classification.
        Deep neural networks achieve high accuracy on computer vision tasks.
        """

        with patch.object(keyword_agent, '_call_model') as mock_llm:
            mock_llm.return_value = """{
                "keywords": [
                    {
                        "keyword": "machine learning",
                        "importance": 0.9,
                        "category": "technology",
                        "frequency": 1,
                        "context": "Machine learning algorithms are used"
                    },
                    {
                        "keyword": "image classification",
                        "importance": 0.85,
                        "category": "task",
                        "frequency": 1,
                        "context": "used for image classification"
                    },
                    {
                        "keyword": "neural networks",
                        "importance": 0.8,
                        "category": "technology",
                        "frequency": 1,
                        "context": "Deep neural networks achieve"
                    }
                ],
                "total_keywords": 3,
                "confidence": "high",
                "metadata": {}
            }"""

            result = await keyword_agent.extract_keywords(text)

            assert isinstance(result, KeywordExtractionResult)
            assert len(result.keywords) >= 3
            assert any(kw.keyword == "machine learning" for kw in result.keywords)
            assert all(kw.importance > 0.7 for kw in result.keywords)


class TestExtractionAgentsIntegration:
    """Test integration between extraction agents."""

    @pytest.mark.asyncio
    async def test_extraction_pipeline(self):
        """Test complete extraction pipeline."""
        text = "Dr. Smith prescribed aspirin for headache treatment at Mayo Clinic."

        # Initialize agents
        fact_config = AgentConfig(name="fact_extraction")
        entity_config = AgentConfig(name="entity_extraction")
        relation_config = AgentConfig(name="relation_extraction")

        fact_agent = FactExtractionAgent(fact_config)
        entity_agent = EntityExtractionAgent(entity_config)
        relation_agent = RelationExtractionAgent(relation_config)

        # Mock LLM responses
        with patch.object(entity_agent, '_call_model') as mock_entity_llm, \
             patch.object(fact_agent, '_call_model') as mock_fact_llm, \
             patch.object(relation_agent, '_call_model') as mock_relation_llm:

            mock_entity_llm.return_value = """{
                "entities": [
                    {
                        "name": "Dr. Smith",
                        "canonical_name": "Dr. Smith",
                        "entity_type": "PERSON",
                        "confidence": 0.9,
                        "start_offset": 0,
                        "end_offset": 9,
                        "attributes": {},
                        "context": "Dr. Smith prescribed"
                    }
                ],
                "total_entities": 1,
                "confidence": "high",
                "metadata": {}
            }"""

            mock_fact_llm.return_value = """{
                "facts": [
                    {
                        "subject": "Dr. Smith",
                        "object": "headache treatment",
                        "approach": "aspirin prescription",
                        "solution": "pain relief",
                        "condition": "",
                        "remarks": "",
                        "fact_type": "MEDICAL_TREATMENT",
                        "confidence": 0.9,
                        "keywords": ["aspirin", "headache"],
                        "source_text": "Dr. Smith prescribed aspirin"
                    }
                ],
                "total_facts": 1,
                "confidence": "high",
                "metadata": {}
            }"""

            mock_relation_llm.return_value = """{
                "relations": [],
                "total_relations": 0,
                "confidence": "high",
                "metadata": {}
            }"""

            # Extract entities
            entity_result = await entity_agent.extract_entities(text)
            assert len(entity_result.entities) == 1

            # Extract facts
            fact_result = await fact_agent.extract_facts(text, domain="medical")
            assert len(fact_result.facts) == 1

            # Extract relations
            relation_result = await relation_agent.extract_relations(text, entity_result.entities)
            assert isinstance(relation_result, RelationExtractionResult)

            print("✅ Extraction pipeline test completed successfully")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
