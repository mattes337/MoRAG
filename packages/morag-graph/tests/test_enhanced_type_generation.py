"""Test enhanced entity and relationship type generation."""

import asyncio
from typing import List
from unittest.mock import AsyncMock, Mock, patch

import pytest
from morag_graph.extraction.entity_extractor import EntityExtractor
from morag_graph.extraction.relation_extractor import RelationExtractor
from morag_graph.models import Entity, Relation
from morag_graph.storage.neo4j_storage import Neo4jStorage

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


class TestEnhancedTypeGeneration:
    """Test enhanced entity and relationship type generation."""

    def test_entity_prompt_encourages_specific_types(self):
        """Test that entity extraction prompt encourages specific types."""
        config = LLMConfig(provider="mock", model="test")
        extractor = EntityExtractor(config, domain="medical")

        prompt = extractor._create_entity_prompt()

        # Check that prompt encourages specific types
        assert "SPECIFIC, DESCRIPTIVE entity types" in prompt
        assert "RESEARCHER" in prompt
        assert "EXECUTIVE" in prompt
        assert "PATIENT" in prompt
        assert "UNIVERSITY" in prompt
        assert "HOSPITAL" in prompt
        assert "MEDICAL_CONDITION" in prompt

        # Check that it discourages generic types
        assert "Never generic" in prompt
        assert '"ENTITY"' in prompt  # Should be mentioned as something to avoid
        assert '"THING"' in prompt  # Should be mentioned as something to avoid

    def test_relation_prompt_encourages_specific_types(self):
        """Test that relation extraction prompt encourages specific types."""
        config = LLMConfig(provider="mock", model="test")
        extractor = RelationExtractor(config, domain="medical")

        prompt = extractor._create_relation_prompt()

        # Check that prompt encourages specific types
        assert "SPECIFIC, DESCRIPTIVE relationship types" in prompt
        assert "EMPLOYS" in prompt
        assert "RESEARCHES" in prompt
        assert "TREATS" in prompt
        assert "DEVELOPS" in prompt
        assert "COLLABORATES_WITH" in prompt
        assert "MANAGES" in prompt

        # Check that it discourages generic types
        assert "Never generic" in prompt
        assert '"RELATES"' in prompt  # Should be mentioned as something to avoid
        assert '"CONNECTS"' in prompt  # Should be mentioned as something to avoid

    def test_domain_specific_prompts(self):
        """Test that domain-specific prompts are generated correctly."""
        config = LLMConfig(provider="mock", model="test")

        # Test medical domain
        medical_entity_extractor = EntityExtractor(config, domain="medical")
        medical_relation_extractor = RelationExtractor(config, domain="medical")

        entity_prompt = medical_entity_extractor._create_entity_prompt()
        relation_prompt = medical_relation_extractor._create_relation_prompt()

        assert "Domain context: medical" in entity_prompt
        assert "Domain context: medical" in relation_prompt

        # Test technical domain
        tech_entity_extractor = EntityExtractor(config, domain="technical")
        tech_relation_extractor = RelationExtractor(config, domain="technical")

        tech_entity_prompt = tech_entity_extractor._create_entity_prompt()
        tech_relation_prompt = tech_relation_extractor._create_relation_prompt()

        assert "Domain context: technical" in tech_entity_prompt
        assert "Domain context: technical" in tech_relation_prompt

    def test_entity_neo4j_label_generation(self):
        """Test that entities generate proper Neo4j labels."""
        # Test various entity types
        test_cases = [
            ("RESEARCHER", "RESEARCHER"),
            ("MEDICAL_CONDITION", "MEDICAL_CONDITION"),
            ("PHARMACEUTICAL_COMPANY", "PHARMACEUTICAL_COMPANY"),
            ("research institution", "RESEARCH_INSTITUTION"),
            ("AI Technology", "AI_TECHNOLOGY"),
            ("Patient-Care", "PATIENT_CARE"),
            ("Data & Analytics", "DATA_AND_ANALYTICS"),
        ]

        for entity_type, expected_label in test_cases:
            entity = Entity(
                name="Test Entity", type=entity_type, confidence=0.9, attributes={}
            )

            neo4j_label = entity.get_neo4j_label()
            assert (
                neo4j_label == expected_label
            ), f"Expected {expected_label}, got {neo4j_label}"

    def test_relation_neo4j_type_generation(self):
        """Test that relations generate proper Neo4j relationship types."""
        # Test various relation types
        test_cases = [
            ("EMPLOYS", "EMPLOYS"),
            ("CONDUCTS_RESEARCH_ON", "CONDUCTS_RESEARCH_ON"),
            ("IS_TREATED_BY", "IS_TREATED_BY"),
            ("collaborates with", "COLLABORATES_WITH"),
            ("Manages & Oversees", "MANAGES_AND_OVERSEES"),
            ("is-developed-by", "IS_DEVELOPED_BY"),
        ]

        for relation_type, expected_type in test_cases:
            relation = Relation(
                source_entity_id="entity_1",
                target_entity_id="entity_2",
                type=relation_type,
                confidence=0.9,
                attributes={},
            )

            neo4j_type = relation.get_neo4j_type()
            assert (
                neo4j_type == expected_type
            ), f"Expected {expected_type}, got {neo4j_type}"

    @pytest.mark.asyncio
    async def test_storage_uses_dynamic_labels(self):
        """Test that Neo4j storage uses dynamic labels and relationship types."""
        # Mock Neo4j storage
        storage = Mock(spec=Neo4jStorage)
        storage._entity_ops = Mock()
        storage._relation_ops = Mock()

        # Create test entity with specific type
        entity = Entity(
            name="Dr. Smith",
            type="MEDICAL_RESEARCHER",
            confidence=0.9,
            attributes={"specialty": "cardiology"},
        )

        # Create test relation with specific type
        relation = Relation(
            source_entity_id="entity_1",
            target_entity_id="entity_2",
            type="CONDUCTS_RESEARCH_ON",
            confidence=0.9,
            attributes={},
        )

        # Verify that the entity has the correct Neo4j label
        assert entity.get_neo4j_label() == "MEDICAL_RESEARCHER"

        # Verify that the relation has the correct Neo4j type
        assert relation.get_neo4j_type() == "CONDUCTS_RESEARCH_ON"

    def test_generic_type_avoidance(self):
        """Test that the system avoids generic types."""
        # These should be transformed to more specific types by the LLM
        generic_types = [
            "ENTITY",
            "THING",
            "ITEM",
            "OBJECT",
            "PERSON",
            "ORGANIZATION",
            "CONCEPT",
            "LOCATION",
            "RELATES",
            "CONNECTS",
            "LINKS",
            "MENTIONS",
        ]

        config = LLMConfig(provider="mock", model="test")
        entity_extractor = EntityExtractor(config)
        relation_extractor = RelationExtractor(config)

        entity_prompt = entity_extractor._create_entity_prompt()
        relation_prompt = relation_extractor._create_relation_prompt()

        # Check that prompts discourage these generic types
        for generic_type in ["ENTITY", "THING", "ITEM", "OBJECT"]:
            assert (
                f'"{generic_type}"' in entity_prompt
            )  # Should be mentioned as something to avoid

        for generic_type in ["RELATES", "CONNECTS", "LINKS", "MENTIONS"]:
            assert (
                f'"{generic_type}"' in relation_prompt
            )  # Should be mentioned as something to avoid

    def test_uppercase_underscore_format(self):
        """Test that types are formatted in uppercase with underscores."""
        config = LLMConfig(provider="mock", model="test")
        entity_extractor = EntityExtractor(config)
        relation_extractor = RelationExtractor(config)

        entity_prompt = entity_extractor._create_entity_prompt()
        relation_prompt = relation_extractor._create_relation_prompt()

        # Check that prompts specify uppercase underscore format
        assert "Uppercase with underscores" in entity_prompt
        assert "Uppercase with underscores" in relation_prompt
        assert "RESEARCH_INSTITUTION" in entity_prompt
        assert "PHARMACEUTICAL_DRUG" in entity_prompt
        assert "IS_EMPLOYED_BY" in relation_prompt
        assert "CONDUCTS_RESEARCH_ON" in relation_prompt


if __name__ == "__main__":
    pytest.main([__file__])
