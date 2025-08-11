"""Edge case tests for dynamic entity and relation types functionality.

These tests cover unusual scenarios, boundary conditions, and integration
aspects to ensure robust handling of dynamic types.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Optional
import json

from morag_graph.extraction import EntityExtractor, RelationExtractor
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


class TestDynamicTypesEdgeCases:
    """Test edge cases for dynamic types functionality."""
    
    def test_none_vs_empty_dict_entity_types(self):
        """Test the distinction between None and empty dict for entity types."""
        config = LLMConfig(provider="mock", model="test")
        
        # None should use pure dynamic mode
        extractor_none = EntityExtractor(config, entity_types=None)
        assert extractor_none.entity_types == {}
        
        # Empty dict should use no types
        extractor_empty = EntityExtractor(config, entity_types={})
        assert extractor_empty.entity_types == {}
        
        # Verify both use dynamic prompts (no predefined types)
        prompt_none = extractor_none.get_system_prompt()
        prompt_empty = extractor_empty.get_system_prompt()

        # Both should use dynamic prompts without predefined types
        assert "semantic meaning" in prompt_none or "not limit yourself" in prompt_none
        assert "semantic meaning" in prompt_empty or "not limit yourself" in prompt_empty
    
    def test_none_vs_empty_dict_relation_types(self):
        """Test the distinction between None and empty dict for relation types."""
        config = LLMConfig(provider="mock", model="test")
        
        # None should use pure dynamic mode
        extractor_none = RelationExtractor(config, relation_types=None)
        assert extractor_none.relation_types == {}
        
        # Empty dict should use no types
        extractor_empty = RelationExtractor(config, relation_types={})
        assert extractor_empty.relation_types == {}
        
        # Verify both use dynamic prompts (no predefined types)
        prompt_none = extractor_none.get_system_prompt()
        prompt_empty = extractor_empty.get_system_prompt()

        # Both should use dynamic prompts without predefined types
        assert "semantic meaning" in prompt_none or "not limit yourself" in prompt_none
        assert "semantic meaning" in prompt_empty or "not limit yourself" in prompt_empty
    
    def test_special_characters_in_type_names(self):
        """Test handling of special characters in type names and descriptions."""
        config = LLMConfig(provider="mock", model="test")
        
        special_entity_types = {
            "TYPE_WITH_UNDERSCORE": "Type with underscore",
            "TYPE-WITH-DASH": "Type with dash",
            "TYPE.WITH.DOT": "Type with dot",
            "TYPE WITH SPACE": "Type with space",
            "TYPE_WITH_UNICODE_🚀": "Type with unicode emoji",
            "TYPE_WITH_QUOTES": 'Description with "quotes" and \'apostrophes\'',
            "TYPE_WITH_NEWLINE": "Description with\nnewline",
            "TYPE_WITH_SPECIAL_CHARS": "Description with @#$%^&*()+={}[]|\\:;\"'<>?,./"
        }
        
        # Should handle special characters without errors
        extractor = EntityExtractor(config, entity_types=special_entity_types)
        assert len(extractor.entity_types) == len(special_entity_types)
        
        # System prompt should be generated successfully
        prompt = extractor.get_system_prompt()
        assert prompt is not None
        
        # Should contain the special type names
        assert "TYPE_WITH_UNDERSCORE" in prompt
        assert "TYPE-WITH-DASH" in prompt
        assert "TYPE.WITH.DOT" in prompt
    
    def test_very_long_type_descriptions(self):
        """Test handling of very long type descriptions."""
        config = LLMConfig(provider="mock", model="test")
        
        long_description = "A" * 1000  # 1000 character description
        very_long_description = "B" * 5000  # 5000 character description
        
        entity_types = {
            "LONG_TYPE": long_description,
            "VERY_LONG_TYPE": very_long_description,
            "NORMAL_TYPE": "Normal description"
        }
        
        # Should handle long descriptions without errors
        extractor = EntityExtractor(config, entity_types=entity_types)
        assert len(extractor.entity_types) == 3
        
        # System prompt should be generated successfully
        prompt = extractor.get_system_prompt()
        assert prompt is not None
        assert long_description in prompt
        assert very_long_description in prompt
    
    def test_duplicate_type_names_case_sensitivity(self):
        """Test handling of duplicate type names with different cases."""
        config = LLMConfig(provider="mock", model="test")
        
        # Python dicts will naturally handle case-sensitive keys
        entity_types = {
            "PERSON": "Lowercase person",
            "Person": "Titlecase person",
            "person": "Lowercase person",
            "PERSON_TYPE": "Person type variant"
        }
        
        extractor = EntityExtractor(config, entity_types=entity_types)
        assert len(extractor.entity_types) == 4  # All should be preserved
        
        prompt = extractor.get_system_prompt()
        assert "PERSON: Lowercase person" in prompt
        assert "Person: Titlecase person" in prompt
        assert "person: Lowercase person" in prompt
    
    def test_numeric_type_names(self):
        """Test handling of numeric and mixed alphanumeric type names."""
        config = LLMConfig(provider="mock", model="test")
        
        entity_types = {
            "123": "Numeric type name",
            "TYPE_123": "Mixed alphanumeric",
            "123_TYPE": "Numeric prefix",
            "TYPE_123_SUFFIX": "Numeric in middle",
            "42": "Answer to everything"
        }
        
        extractor = EntityExtractor(config, entity_types=entity_types)
        assert len(extractor.entity_types) == 5
        
        prompt = extractor.get_system_prompt()
        assert "123: Numeric type name" in prompt
        assert "42: Answer to everything" in prompt
    
    def test_empty_type_descriptions(self):
        """Test handling of empty type descriptions."""
        config = LLMConfig(provider="mock", model="test")
        
        entity_types = {
            "EMPTY_DESC": "",
            "WHITESPACE_DESC": "   ",
            "TAB_DESC": "\t",
            "NEWLINE_DESC": "\n",
            "NORMAL_DESC": "Normal description"
        }
        
        extractor = EntityExtractor(config, entity_types=entity_types)
        assert len(extractor.entity_types) == 5
        
        prompt = extractor.get_system_prompt()
        assert "EMPTY_DESC:" in prompt
        assert "NORMAL_DESC: Normal description" in prompt
    
    def test_single_type_configurations(self):
        """Test configurations with only a single type."""
        config = LLMConfig(provider="mock", model="test")
        
        # Single entity type
        single_entity_type = {"ONLY_TYPE": "The only entity type"}
        entity_extractor = EntityExtractor(config, entity_types=single_entity_type)
        
        assert len(entity_extractor.entity_types) == 1
        assert "ONLY_TYPE" in entity_extractor.entity_types
        
        entity_prompt = entity_extractor.get_system_prompt()
        assert "ONLY_TYPE: The only entity type" in entity_prompt
        
        # Single relation type
        single_relation_type = {"ONLY_RELATION": "The only relation type"}
        relation_extractor = RelationExtractor(config, relation_types=single_relation_type)
        
        assert len(relation_extractor.relation_types) == 1
        assert "ONLY_RELATION" in relation_extractor.relation_types
        
        relation_prompt = relation_extractor.get_system_prompt()
        assert "ONLY_RELATION: The only relation type" in relation_prompt
    
    def test_type_name_conflicts_with_defaults(self):
        """Test custom types that have same names as default types."""
        config = LLMConfig(provider="mock", model="test")
        
        # Override default types with custom descriptions
        custom_entity_types = {
            "PERSON": "Custom person definition",
            "ORGANIZATION": "Custom organization definition",
            "CUSTOM_TYPE": "Completely new type"
        }
        
        extractor = EntityExtractor(config, entity_types=custom_entity_types)
        
        # Should use custom descriptions, not defaults
        assert extractor.entity_types["PERSON"] == "Custom person definition"
        assert extractor.entity_types["ORGANIZATION"] == "Custom organization definition"
        
        prompt = extractor.get_system_prompt()
        assert "PERSON: Custom person definition" in prompt
        assert "PERSON: Names of people" not in prompt  # Default description should not appear
    
    def test_mixed_default_and_custom_types(self):
        """Test mixing some default types with custom types."""
        config = LLMConfig(provider="mock", model="test")
        
        # Mix of default and custom types
        mixed_types = {
            "PERSON": "Names of people, individuals, or human beings",  # Default PERSON
            "CUSTOM_ENTITY": "Custom entity type",
            "LOCATION": "Places, cities, countries, or geographical locations",  # Default LOCATION
            "ANOTHER_CUSTOM": "Another custom type"
        }
        
        extractor = EntityExtractor(config, entity_types=mixed_types)
        
        # Should contain exactly what was specified
        assert len(extractor.entity_types) == 4
        assert "ORGANIZATION" not in extractor.entity_types  # Default type not included
        assert "CUSTOM_ENTITY" in extractor.entity_types
        
        prompt = extractor.get_system_prompt()
        assert "CUSTOM_ENTITY: Custom entity type" in prompt
        assert "PERSON: Names of people, individuals" in prompt


class TestDynamicTypesIntegration:
    """Test integration scenarios for dynamic types."""
    
    def test_entity_and_relation_extractors_with_matching_types(self):
        """Test using entity and relation extractors with complementary types."""
        config = LLMConfig(provider="mock", model="test")
        
        # Medical domain types
        medical_entity_types = {
            "PATIENT": "Person receiving medical care",
            "DOCTOR": "Medical professional",
            "CONDITION": "Medical condition or disease",
            "MEDICATION": "Pharmaceutical drug"
        }
        
        medical_relation_types = {
            "TREATS": "Doctor treats patient",
            "DIAGNOSED_WITH": "Patient diagnosed with condition",
            "PRESCRIBED": "Doctor prescribes medication",
            "TAKES": "Patient takes medication"
        }
        
        entity_extractor = EntityExtractor(config, entity_types=medical_entity_types)
        relation_extractor = RelationExtractor(config, relation_types=medical_relation_types)
        
        # Verify both extractors have the expected types
        assert len(entity_extractor.entity_types) == 4
        assert len(relation_extractor.relation_types) == 4
        
        # Verify system prompts are domain-specific
        entity_prompt = entity_extractor.get_system_prompt()
        relation_prompt = relation_extractor.get_system_prompt()
        
        assert "PATIENT: Person receiving medical care" in entity_prompt
        assert "TREATS: Doctor treats patient" in relation_prompt
        
        # Should not contain default types
        assert "ORGANIZATION: Companies" not in entity_prompt
        assert "WORKS_FOR: Person works for" not in relation_prompt
    
    def test_extractors_with_overlapping_type_names(self):
        """Test extractors where entity and relation types have overlapping names."""
        config = LLMConfig(provider="mock", model="test")
        
        # Intentionally overlapping names (though semantically different)
        entity_types = {
            "CONNECTION": "A network connection or link",
            "PROCESS": "A running system process"
        }
        
        relation_types = {
            "CONNECTION": "One entity connects to another",
            "PROCESS": "One entity processes another"
        }
        
        entity_extractor = EntityExtractor(config, entity_types=entity_types)
        relation_extractor = RelationExtractor(config, relation_types=relation_types)
        
        # Both should work independently
        entity_prompt = entity_extractor.get_system_prompt()
        relation_prompt = relation_extractor.get_system_prompt()
        
        assert "CONNECTION: A network connection" in entity_prompt
        assert "CONNECTION: One entity connects" in relation_prompt
        assert "PROCESS: A running system" in entity_prompt
        assert "PROCESS: One entity processes" in relation_prompt
    
    def test_progressive_type_refinement(self):
        """Test scenario where types are progressively refined."""
        config = LLMConfig(provider="mock", model="test")
        
        # Start with broad types
        broad_types = {
            "ENTITY": "Any entity",
            "THING": "Any thing"
        }
        
        # Refine to specific types
        specific_types = {
            "PROTEIN": "Biological protein",
            "GENE": "Genetic sequence",
            "PATHWAY": "Biological pathway",
            "DISEASE": "Medical condition"
        }
        
        # Very specific types
        ultra_specific_types = {
            "KINASE_PROTEIN": "Protein kinase enzyme",
            "TUMOR_SUPPRESSOR_GENE": "Gene that prevents tumor formation"
        }
        
        # Each extractor should work with its level of specificity
        broad_extractor = EntityExtractor(config, entity_types=broad_types)
        specific_extractor = EntityExtractor(config, entity_types=specific_types)
        ultra_specific_extractor = EntityExtractor(config, entity_types=ultra_specific_types)
        
        broad_prompt = broad_extractor.get_system_prompt()
        specific_prompt = specific_extractor.get_system_prompt()
        ultra_specific_prompt = ultra_specific_extractor.get_system_prompt()
        
        # Each should contain only its own types
        assert "ENTITY: Any entity" in broad_prompt
        assert "PROTEIN: Biological protein" in specific_prompt
        assert "KINASE_PROTEIN: Protein kinase" in ultra_specific_prompt
        
        # Cross-contamination should not occur
        assert "PROTEIN:" not in broad_prompt
        assert "ENTITY:" not in specific_prompt
        
        # Cross-contamination should not occur
        # Check that GENE is not listed as a separate entity type (not just in descriptions)
        # The issue was that "TUMOR_SUPPRESSOR_GENE: Gene that prevents tumor formation" contains "GENE:"
        # We need to check for "- GENE:" (as a list item) instead of just "GENE:"
        assert "- GENE:" not in ultra_specific_prompt
    
    def test_type_inheritance_simulation(self):
        """Test simulating type inheritance through naming conventions."""
        config = LLMConfig(provider="mock", model="test")
        
        # Simulate inheritance through naming
        hierarchical_types = {
            "PERSON": "Any person",
            "PERSON_EMPLOYEE": "Person who is an employee",
            "PERSON_EMPLOYEE_MANAGER": "Employee who manages others",
            "PERSON_CUSTOMER": "Person who is a customer",
            "ORGANIZATION": "Any organization",
            "ORGANIZATION_COMPANY": "Commercial organization",
            "ORGANIZATION_NONPROFIT": "Non-profit organization"
        }
        
        extractor = EntityExtractor(config, entity_types=hierarchical_types)
        
        assert len(extractor.entity_types) == 7
        
        prompt = extractor.get_system_prompt()
        
        # All hierarchical types should be present
        assert "PERSON: Any person" in prompt
        assert "PERSON_EMPLOYEE: Person who is an employee" in prompt
        assert "PERSON_EMPLOYEE_MANAGER: Employee who manages" in prompt
        assert "ORGANIZATION_COMPANY: Commercial organization" in prompt
    
    def test_domain_switching(self):
        """Test switching between different domain-specific type sets."""
        config = LLMConfig(provider="mock", model="test")
        
        # Legal domain
        legal_types = {
            "PLAINTIFF": "Party bringing a lawsuit",
            "DEFENDANT": "Party being sued",
            "COURT": "Legal court or tribunal",
            "STATUTE": "Legal statute or law"
        }
        
        # Financial domain
        financial_types = {
            "INVESTOR": "Person or entity that invests",
            "COMPANY": "Business entity",
            "STOCK": "Financial security",
            "TRANSACTION": "Financial transaction"
        }
        
        # Technical domain
        technical_types = {
            "SERVER": "Computer server",
            "DATABASE": "Data storage system",
            "API": "Application programming interface",
            "USER": "System user"
        }
        
        # Create extractors for each domain
        legal_extractor = EntityExtractor(config, entity_types=legal_types)
        financial_extractor = EntityExtractor(config, entity_types=financial_types)
        technical_extractor = EntityExtractor(config, entity_types=technical_types)
        
        # Each should be domain-specific
        legal_prompt = legal_extractor.get_system_prompt()
        financial_prompt = financial_extractor.get_system_prompt()
        technical_prompt = technical_extractor.get_system_prompt()
        
        # Domain separation should be maintained
        assert "PLAINTIFF:" in legal_prompt
        assert "PLAINTIFF:" not in financial_prompt
        assert "PLAINTIFF:" not in technical_prompt
        
        assert "INVESTOR:" in financial_prompt
        assert "INVESTOR:" not in legal_prompt
        assert "INVESTOR:" not in technical_prompt
        
        assert "SERVER:" in technical_prompt
        assert "SERVER:" not in legal_prompt
        assert "SERVER:" not in financial_prompt
    
    def test_backward_compatibility_with_existing_code(self):
        """Test that existing code using defaults still works."""
        config = LLMConfig(provider="mock", model="test")
        
        # Old way: no types specified (should use defaults)
        old_entity_extractor = EntityExtractor(config)
        old_relation_extractor = RelationExtractor(config)
        
        # New way: explicitly None (should also use defaults)
        new_entity_extractor = EntityExtractor(config, entity_types=None)
        new_relation_extractor = RelationExtractor(config, relation_types=None)
        
        # Both should have the same default types
        assert old_entity_extractor.entity_types == new_entity_extractor.entity_types
        assert old_relation_extractor.relation_types == new_relation_extractor.relation_types
        
        # Both should generate the same prompts
        assert old_entity_extractor.get_system_prompt() == new_entity_extractor.get_system_prompt()
        assert old_relation_extractor.get_system_prompt() == new_relation_extractor.get_system_prompt()
        
        # Should use dynamic prompts (no predefined types)
        entity_prompt = old_entity_extractor.get_system_prompt()
        relation_prompt = old_relation_extractor.get_system_prompt()

        # Both should use dynamic prompts without predefined types
        assert "semantic meaning" in entity_prompt or "not limit yourself" in entity_prompt
        assert "semantic meaning" in relation_prompt or "not limit yourself" in relation_prompt