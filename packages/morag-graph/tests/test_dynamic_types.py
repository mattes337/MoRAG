"""Comprehensive tests for dynamic entity and relation types functionality.

These tests ensure 100% coverage of the dynamic type system including:
- Default type behavior
- Custom type specification
- Empty type handling
- System prompt generation
- Type validation
- Edge cases and error handling
"""

from typing import Dict, List, Optional
from unittest.mock import AsyncMock, Mock, patch

import pytest
from morag_graph.extraction import EntityExtractor, RelationExtractor
from morag_graph.models import Entity, Relation

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


class TestEntityExtractorDynamicTypes:
    """Test dynamic entity types functionality."""

    def test_default_entity_types_initialization(self):
        """Test that EntityExtractor initializes with dynamic types by default."""
        config = LLMConfig(provider="mock", model="test")
        extractor = EntityExtractor(config)

        # Should use dynamic types by default
        assert extractor.dynamic_types == True
        # Should use empty types for pure dynamic mode
        assert extractor.entity_types == {}
        assert len(extractor.entity_types) == 0

    def test_custom_entity_types_initialization(self):
        """Test that EntityExtractor uses custom types when provided."""
        config = LLMConfig(provider="mock", model="test")
        custom_types = {
            "DISEASE": "Medical condition or illness",
            "TREATMENT": "Medical intervention or therapy",
        }
        extractor = EntityExtractor(config, entity_types=custom_types)

        # Should use only custom types, not defaults
        assert extractor.entity_types == custom_types
        assert len(extractor.entity_types) == 2
        assert "DISEASE" in extractor.entity_types
        assert "TREATMENT" in extractor.entity_types
        assert "PERSON" not in extractor.entity_types  # Default should not be included

    def test_empty_entity_types_initialization(self):
        """Test that EntityExtractor handles empty types dictionary."""
        config = LLMConfig(provider="mock", model="test")
        extractor = EntityExtractor(config, entity_types={})

        # Should use empty types, not defaults
        assert extractor.entity_types == {}
        assert len(extractor.entity_types) == 0

    def test_none_entity_types_uses_dynamic(self):
        """Test that None entity_types parameter uses dynamic mode."""
        config = LLMConfig(provider="mock", model="test")
        extractor = EntityExtractor(config, entity_types=None)

        # Should use dynamic types by default
        assert extractor.dynamic_types == True
        # None should use empty types for pure dynamic mode
        assert extractor.entity_types == {}

    def test_entity_types_system_prompt_generation(self):
        """Test that system prompt correctly includes specified entity types."""
        config = LLMConfig(provider="mock", model="test")
        custom_types = {
            "DISEASE": "Medical condition",
            "SYMPTOM": "Observable sign of disease",
        }
        extractor = EntityExtractor(config, entity_types=custom_types)

        system_prompt = extractor.get_system_prompt()

        # Should include custom types in prompt
        assert "DISEASE: Medical condition" in system_prompt
        assert "SYMPTOM: Observable sign of disease" in system_prompt
        # Should not include default types
        assert "PERSON:" not in system_prompt
        assert "ORGANIZATION:" not in system_prompt

    def test_empty_entity_types_system_prompt(self):
        """Test system prompt generation with empty entity types."""
        config = LLMConfig(provider="mock", model="test")
        extractor = EntityExtractor(config, entity_types={})

        system_prompt = extractor.get_system_prompt()

        # Should handle empty types gracefully
        assert system_prompt is not None
        assert len(system_prompt) > 0
        # Should not contain any type definitions
        assert "PERSON:" not in system_prompt
        assert "DISEASE:" not in system_prompt

    def test_dynamic_entity_types_mode(self):
        """Test that dynamic entity types mode works correctly."""
        config = LLMConfig(provider="mock", model="test")
        extractor = EntityExtractor(config, dynamic_types=True)

        # Should be in dynamic mode
        assert extractor.dynamic_types == True
        assert extractor.entity_types == {}

        # System prompt should indicate dynamic mode
        prompt = extractor.get_system_prompt()
        assert "semantic meaning" in prompt
        assert "not limit yourself" in prompt

    def test_entity_types_parameter_validation(self):
        """Test validation of entity_types parameter."""
        config = LLMConfig(provider="mock", model="test")

        # Valid dictionary should work
        valid_types = {"TYPE1": "Description 1", "TYPE2": "Description 2"}
        extractor = EntityExtractor(config, entity_types=valid_types)
        assert extractor.entity_types == valid_types

        # Empty dictionary should work
        extractor = EntityExtractor(config, entity_types={})
        assert extractor.entity_types == {}

        # None should work (uses pure dynamic mode)
        extractor = EntityExtractor(config, entity_types=None)
        assert extractor.dynamic_types == True
        assert extractor.entity_types == {}


class TestRelationExtractorDynamicTypes:
    """Test dynamic relation types functionality."""

    def test_default_relation_types_initialization(self):
        """Test that RelationExtractor initializes with dynamic types by default."""
        config = LLMConfig(provider="mock", model="test")
        extractor = RelationExtractor(config)

        # Should use dynamic types by default
        assert extractor.dynamic_types == True
        # Should use empty types for pure dynamic mode
        assert extractor.relation_types == {}
        assert len(extractor.relation_types) == 0

    def test_custom_relation_types_initialization(self):
        """Test that RelationExtractor uses custom types when provided."""
        config = LLMConfig(provider="mock", model="test")
        custom_types = {
            "CAUSES": "Pathogen causes disease",
            "TREATS": "Treatment treats condition",
        }
        extractor = RelationExtractor(config, relation_types=custom_types)

        # Should use only custom types, not defaults
        assert extractor.relation_types == custom_types
        assert len(extractor.relation_types) == 2
        assert "CAUSES" in extractor.relation_types
        assert "TREATS" in extractor.relation_types
        assert (
            "WORKS_FOR" not in extractor.relation_types
        )  # Default should not be included

    def test_empty_relation_types_initialization(self):
        """Test that RelationExtractor handles empty types dictionary."""
        config = LLMConfig(provider="mock", model="test")
        extractor = RelationExtractor(config, relation_types={})

        # Should use empty types, not defaults
        assert extractor.relation_types == {}
        assert len(extractor.relation_types) == 0

    def test_none_relation_types_uses_dynamic(self):
        """Test that None relation_types parameter uses dynamic mode."""
        config = LLMConfig(provider="mock", model="test")
        extractor = RelationExtractor(config, relation_types=None)

        # Should use dynamic types by default
        assert extractor.dynamic_types == True
        # None should use empty types for pure dynamic mode
        assert extractor.relation_types == {}

    def test_relation_types_system_prompt_generation(self):
        """Test that system prompt correctly includes specified relation types."""
        config = LLMConfig(provider="mock", model="test")
        custom_types = {
            "CAUSES": "Direct causal relationship",
            "PREVENTS": "Prevention relationship",
        }
        extractor = RelationExtractor(config, relation_types=custom_types)

        system_prompt = extractor.get_system_prompt()

        # Should include custom types in prompt
        assert "CAUSES: Direct causal relationship" in system_prompt
        assert "PREVENTS: Prevention relationship" in system_prompt
        # Should not include default types
        assert "WORKS_FOR:" not in system_prompt
        assert "LOCATED_IN:" not in system_prompt

    def test_empty_relation_types_system_prompt(self):
        """Test system prompt generation with empty relation types."""
        config = LLMConfig(provider="mock", model="test")
        extractor = RelationExtractor(config, relation_types={})

        system_prompt = extractor.get_system_prompt()

        # Should handle empty types gracefully
        assert system_prompt is not None
        assert len(system_prompt) > 0
        # Should not contain any type definitions
        assert "WORKS_FOR:" not in system_prompt
        assert "CAUSES:" not in system_prompt

    def test_dynamic_relation_types_mode(self):
        """Test that dynamic relation types mode works correctly."""
        config = LLMConfig(provider="mock", model="test")
        extractor = RelationExtractor(config, dynamic_types=True)

        # Should be in dynamic mode
        assert extractor.dynamic_types == True
        assert extractor.relation_types == {}

        # System prompt should indicate dynamic mode
        prompt = extractor.get_system_prompt()
        assert "semantic meaning" in prompt
        assert "not limit yourself" in prompt

    def test_relation_types_parameter_validation(self):
        """Test validation of relation_types parameter."""
        config = LLMConfig(provider="mock", model="test")

        # Valid dictionary should work
        valid_types = {"TYPE1": "Description 1", "TYPE2": "Description 2"}
        extractor = RelationExtractor(config, relation_types=valid_types)
        assert extractor.relation_types == valid_types

        # Empty dictionary should work
        extractor = RelationExtractor(config, relation_types={})
        assert extractor.relation_types == {}

        # None should work (uses pure dynamic mode)
        extractor = RelationExtractor(config, relation_types=None)
        assert extractor.dynamic_types == True
        assert extractor.relation_types == {}


class TestDynamicTypesIntegration:
    """Test integration scenarios for dynamic types."""

    def test_domain_specific_medical_types(self):
        """Test medical domain-specific type configuration."""
        config = LLMConfig(provider="mock", model="test")

        medical_entity_types = {
            "DISEASE": "Medical condition or illness",
            "TREATMENT": "Medical intervention or therapy",
            "SYMPTOM": "Observable sign of disease",
            "MEDICATION": "Pharmaceutical drug",
        }

        medical_relation_types = {
            "CAUSES": "Pathogen causes disease",
            "TREATS": "Treatment treats condition",
            "MANIFESTS_AS": "Disease manifests as symptom",
            "PRESCRIBED_FOR": "Medication prescribed for condition",
        }

        entity_extractor = EntityExtractor(config, entity_types=medical_entity_types)
        relation_extractor = RelationExtractor(
            config, relation_types=medical_relation_types
        )

        # Verify medical types are used
        assert entity_extractor.entity_types == medical_entity_types
        assert relation_extractor.relation_types == medical_relation_types

        # Verify system prompts contain medical types
        entity_prompt = entity_extractor.get_system_prompt()
        relation_prompt = relation_extractor.get_system_prompt()

        assert "DISEASE: Medical condition" in entity_prompt
        assert "CAUSES: Pathogen causes" in relation_prompt

    def test_minimal_types_configuration(self):
        """Test configuration with minimal type sets."""
        config = LLMConfig(provider="mock", model="test")

        minimal_entity_types = {"PERSON": "Individual person"}
        minimal_relation_types = {"KNOWS": "Person knows another person"}

        entity_extractor = EntityExtractor(config, entity_types=minimal_entity_types)
        relation_extractor = RelationExtractor(
            config, relation_types=minimal_relation_types
        )

        # Should use only specified types
        assert len(entity_extractor.entity_types) == 1
        assert len(relation_extractor.relation_types) == 1
        assert entity_extractor.entity_types["PERSON"] == "Individual person"
        assert (
            relation_extractor.relation_types["KNOWS"] == "Person knows another person"
        )

    def test_maximum_control_empty_types(self):
        """Test maximum control scenario with empty type dictionaries."""
        config = LLMConfig(provider="mock", model="test")

        entity_extractor = EntityExtractor(config, entity_types={})
        relation_extractor = RelationExtractor(config, relation_types={})

        # Should have no predefined types
        assert len(entity_extractor.entity_types) == 0
        assert len(relation_extractor.relation_types) == 0

        # System prompts should handle empty types
        entity_prompt = entity_extractor.get_system_prompt()
        relation_prompt = relation_extractor.get_system_prompt()

        assert entity_prompt is not None
        assert relation_prompt is not None

    def test_backward_compatibility_none_types(self):
        """Test backward compatibility when types are not specified."""
        config = LLMConfig(provider="mock", model="test")

        # Test with None (explicit) - should use pure dynamic mode
        entity_extractor = EntityExtractor(config, entity_types=None)
        relation_extractor = RelationExtractor(config, relation_types=None)

        assert entity_extractor.dynamic_types == True
        assert entity_extractor.entity_types == {}
        assert relation_extractor.dynamic_types == True
        assert relation_extractor.relation_types == {}

        # Test without specifying types parameter (implicit None) - should use pure dynamic mode
        entity_extractor2 = EntityExtractor(config)
        relation_extractor2 = RelationExtractor(config)

        assert entity_extractor2.dynamic_types == True
        assert entity_extractor2.entity_types == {}
        assert relation_extractor2.dynamic_types == True
        assert relation_extractor2.relation_types == {}

    def test_type_descriptions_in_prompts(self):
        """Test that type descriptions are properly included in system prompts."""
        config = LLMConfig(provider="mock", model="test")

        entity_types = {
            "CUSTOM_TYPE_1": "This is a detailed description of custom type 1",
            "CUSTOM_TYPE_2": "This is a detailed description of custom type 2",
        }

        relation_types = {
            "CUSTOM_REL_1": "This is a detailed description of custom relation 1",
            "CUSTOM_REL_2": "This is a detailed description of custom relation 2",
        }

        entity_extractor = EntityExtractor(config, entity_types=entity_types)
        relation_extractor = RelationExtractor(config, relation_types=relation_types)

        entity_prompt = entity_extractor.get_system_prompt()
        relation_prompt = relation_extractor.get_system_prompt()

        # Check that full descriptions are included
        assert (
            "CUSTOM_TYPE_1: This is a detailed description of custom type 1"
            in entity_prompt
        )
        assert (
            "CUSTOM_TYPE_2: This is a detailed description of custom type 2"
            in entity_prompt
        )
        assert (
            "CUSTOM_REL_1: This is a detailed description of custom relation 1"
            in relation_prompt
        )
        assert (
            "CUSTOM_REL_2: This is a detailed description of custom relation 2"
            in relation_prompt
        )


class TestDynamicTypesEdgeCases:
    """Test edge cases and error scenarios for dynamic types."""

    def test_special_characters_in_type_names(self):
        """Test handling of special characters in type names and descriptions."""
        config = LLMConfig(provider="mock", model="test")

        special_entity_types = {
            "TYPE_WITH_UNDERSCORE": "Description with special chars: !@#$%",
            "TYPE-WITH-DASH": 'Description with quotes: "quoted text"',
            "TYPE.WITH.DOTS": "Description with newlines:\nSecond line",
        }

        entity_extractor = EntityExtractor(config, entity_types=special_entity_types)

        # Should handle special characters without errors
        assert entity_extractor.entity_types == special_entity_types

        # System prompt should be generated without errors
        system_prompt = entity_extractor.get_system_prompt()
        assert system_prompt is not None
        assert "TYPE_WITH_UNDERSCORE" in system_prompt

    def test_very_long_type_descriptions(self):
        """Test handling of very long type descriptions."""
        config = LLMConfig(provider="mock", model="test")

        long_description = "This is a very long description that goes on and on " * 50
        long_entity_types = {"LONG_TYPE": long_description}

        entity_extractor = EntityExtractor(config, entity_types=long_entity_types)

        # Should handle long descriptions
        assert entity_extractor.entity_types["LONG_TYPE"] == long_description

        # System prompt should include the long description
        system_prompt = entity_extractor.get_system_prompt()
        assert long_description in system_prompt

    def test_unicode_characters_in_types(self):
        """Test handling of Unicode characters in type names and descriptions."""
        config = LLMConfig(provider="mock", model="test")

        unicode_entity_types = {
            "PERSÖN": "Person with Unicode characters: äöü",
            "组织": "Organization in Chinese: 公司",
            "ÉMOTIONS": "Emotions with accents: café, naïve",
        }

        entity_extractor = EntityExtractor(config, entity_types=unicode_entity_types)

        # Should handle Unicode characters
        assert entity_extractor.entity_types == unicode_entity_types

        # System prompt should be generated without errors
        system_prompt = entity_extractor.get_system_prompt()
        assert system_prompt is not None

    def test_empty_string_descriptions(self):
        """Test handling of empty string descriptions."""
        config = LLMConfig(provider="mock", model="test")

        empty_desc_types = {
            "TYPE_WITH_EMPTY_DESC": "",
            "TYPE_WITH_SPACE_DESC": "   ",
            "NORMAL_TYPE": "Normal description",
        }

        entity_extractor = EntityExtractor(config, entity_types=empty_desc_types)

        # Should handle empty descriptions
        assert entity_extractor.entity_types == empty_desc_types

        # System prompt should be generated
        system_prompt = entity_extractor.get_system_prompt()
        assert system_prompt is not None
        assert "TYPE_WITH_EMPTY_DESC:" in system_prompt
        assert "NORMAL_TYPE: Normal description" in system_prompt

    def test_large_number_of_types(self):
        """Test handling of a large number of custom types."""
        config = LLMConfig(provider="mock", model="test")

        # Create 100 custom types
        large_entity_types = {
            f"TYPE_{i:03d}": f"Description for type {i}" for i in range(100)
        }

        entity_extractor = EntityExtractor(config, entity_types=large_entity_types)

        # Should handle large number of types
        assert len(entity_extractor.entity_types) == 100
        assert entity_extractor.entity_types == large_entity_types

        # System prompt should be generated
        system_prompt = entity_extractor.get_system_prompt()
        assert system_prompt is not None
        assert "TYPE_000: Description for type 0" in system_prompt
        assert "TYPE_099: Description for type 99" in system_prompt


class TestDynamicTypesDocumentation:
    """Test that dynamic types functionality matches documentation."""

    def test_documented_examples_work(self):
        """Test that examples from documentation actually work."""
        config = LLMConfig(provider="mock", model="test")

        # Example 1: Dynamic types (general purpose) - pure dynamic mode
        extractor1 = EntityExtractor(config)
        assert extractor1.dynamic_types == True
        assert extractor1.entity_types == {}  # Should be empty for pure dynamic mode

        # Example 2: Custom types (domain-specific) in static mode
        medical_types = {
            "DISEASE": "Medical condition or illness",
            "TREATMENT": "Medical intervention or therapy",
            "SYMPTOM": "Observable sign of disease",
        }
        extractor2 = EntityExtractor(
            config, entity_types=medical_types, dynamic_types=False
        )
        assert extractor2.dynamic_types == False
        assert extractor2.entity_types == medical_types

        # Example 3: Minimal types (highly focused) in static mode
        minimal = {"PERSON": "Individual person"}
        extractor3 = EntityExtractor(config, entity_types=minimal, dynamic_types=False)
        assert extractor3.dynamic_types == False
        assert extractor3.entity_types == minimal

        # Example 4: No types (maximum control)
        extractor4 = EntityExtractor(config, entity_types={})
        assert extractor4.entity_types == {}

    def test_documented_relation_examples_work(self):
        """Test that relation extractor examples from documentation work."""
        config = LLMConfig(provider="mock", model="test")

        # Example 1: Dynamic types - pure dynamic mode
        extractor1 = RelationExtractor(config)
        assert extractor1.dynamic_types == True
        assert extractor1.relation_types == {}  # Should be empty for pure dynamic mode

        # Example 2: Medical domain in static mode
        medical_relations = {
            "CAUSES": "Pathogen causes disease",
            "TREATS": "Treatment treats condition",
        }
        extractor2 = RelationExtractor(
            config, relation_types=medical_relations, dynamic_types=False
        )
        assert extractor2.dynamic_types == False
        assert extractor2.relation_types == medical_relations

        # Example 3: Minimal types
        minimal = {"CAUSES": "Causal relationship"}
        extractor3 = RelationExtractor(config, relation_types=minimal)
        assert extractor3.relation_types == minimal

        # Example 4: No types
        extractor4 = RelationExtractor(config, relation_types={})
        assert extractor4.relation_types == {}
