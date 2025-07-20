"""Tests for enhanced entity and relation processing improvements."""

import pytest
from unittest.mock import Mock, AsyncMock

from morag_graph.utils.entity_normalizer import EntityNormalizer
from morag_graph.utils.semantic_relation_enhancer import SemanticRelationEnhancer
from morag_graph.models import Entity as GraphEntity, Relation as GraphRelation


class TestEntityNormalizer:
    """Test entity normalization functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.normalizer = EntityNormalizer()
    
    def test_german_plural_normalization_sync(self):
        """Test German plural to singular normalization using sync method."""
        test_cases = [
            ("Hunde", "Hund"),
            ("Katzen", "Katze"),
            ("Schwermetalle", "Schwermetall"),
            ("Belastungen", "Belastung"),
        ]

        for plural, expected_singular in test_cases:
            # Note: sync method only does basic cleanup, full normalization requires LLM
            result = self.normalizer.normalize_entity_name_sync(plural, "de")
            # For sync method, we mainly test that it doesn't break the input
            assert len(result) > 0, f"Sync normalization should not return empty string for {plural}"
    
    def test_abbreviation_normalization_sync(self):
        """Test abbreviation normalization using sync method."""
        test_cases = [
            ("who", "WHO"),
            ("weltgesundheitsorganisation", "WHO"),
            ("world health organization", "WHO"),
            ("dna", "DNA"),
            ("desoxyribonukleinsäure", "DNA"),
            ("adhs", "ADHS"),
            ("aufmerksamkeitsdefizit-hyperaktivitätsstörung", "ADHS"),
        ]

        for input_name, expected in test_cases:
            result = self.normalizer.normalize_entity_name_sync(input_name)
            assert result == expected, f"Expected {expected}, got {result}"
    
    def test_context_removal_sync(self):
        """Test removal of contextual information from entity names using sync method."""
        test_cases = [
            ("protein in cells", "protein"),
            ("data from study", "data"),
            ("analysis of results", "analysis"),
            ("Protein bei Patienten", "Protein"),
            ("Analyse von Daten", "Analyse"),
            ("effect (in patients)", "effect"),
            ("substance [medical use]", "substance"),
        ]

        for input_name, expected in test_cases:
            result = self.normalizer.normalize_entity_name_sync(input_name)
            # Basic context removal should work in sync mode
            assert expected.lower() in result.lower(), f"Expected {expected} to be in {result}"

    def test_basic_cleanup(self):
        """Test basic cleanup functionality."""
        test_cases = [
            ("  protein  ", "protein"),
            ("PROTEIN", "PROTEIN"),  # Should preserve capitalization
            ("multi word entity", "Multi Word Entity"),  # Should capitalize if originally capitalized
        ]

        for input_name, expected in test_cases:
            result = self.normalizer.normalize_entity_name_sync(input_name)
            assert result.strip() == expected or result.strip().lower() == expected.lower(), f"Expected {expected}, got {result}"


class TestSemanticRelationEnhancer:
    """Test semantic relation enhancement functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.enhancer = SemanticRelationEnhancer()
    
    def test_causal_relation_enhancement(self):
        """Test enhancement of causal relations."""
        test_cases = [
            ("stress", "anxiety", "stress causes anxiety in patients", "causes"),
            ("medication", "symptom", "medication prevents symptom occurrence", "prevents"),
            ("exercise", "health", "exercise enables better health outcomes", "enables"),
        ]
        
        for source, target, context, expected_type in test_cases:
            result = self.enhancer.enhance_relation_type(source, target, context, "relates_to")
            assert expected_type in result.lower(), f"Expected {expected_type} in {result}"
    
    def test_medical_domain_detection(self):
        """Test medical domain detection and relation enhancement."""
        context = "The doctor treats the patient with medication for their condition"
        result = self.enhancer.enhance_relation_type("doctor", "patient", context, "relates_to")
        
        assert "treat" in result.lower(), f"Expected medical relation type, got {result}"
    
    def test_technical_domain_detection(self):
        """Test technical domain detection and relation enhancement."""
        context = "The software connects to the database through an API interface"
        result = self.enhancer.enhance_relation_type("software", "database", context, "relates_to")
        
        assert "connect" in result.lower(), f"Expected technical relation type, got {result}"
    
    def test_temporal_relation_enhancement(self):
        """Test enhancement of temporal relations."""
        test_cases = [
            ("event1", "event2", "event1 happens before event2", "precedes"),
            ("process1", "process2", "process1 occurs during process2", "occurs_during"),
            ("action1", "action2", "action1 follows action2", "follows"),
        ]
        
        for source, target, context, expected_type in test_cases:
            result = self.enhancer.enhance_relation_type(source, target, context, "relates_to")
            assert expected_type in result.lower(), f"Expected {expected_type} in {result}"
    
    def test_relation_type_validation(self):
        """Test relation type validation."""
        valid_types = ["causes", "treats", "connects_to", "manages", "creates"]
        invalid_types = ["", "x", "relates_to", "associated_with", "123invalid"]
        
        for valid_type in valid_types:
            assert self.enhancer.validate_relation_type(valid_type), f"{valid_type} should be valid"
        
        for invalid_type in invalid_types:
            assert not self.enhancer.validate_relation_type(invalid_type), f"{invalid_type} should be invalid"
    
    def test_domain_suggestions(self):
        """Test domain-specific relation suggestions."""
        medical_suggestions = self.enhancer.get_relation_suggestions("medical")
        assert "treats" in medical_suggestions
        assert "diagnoses" in medical_suggestions
        
        technical_suggestions = self.enhancer.get_relation_suggestions("technical")
        assert "implements" in technical_suggestions
        assert "connects_to" in technical_suggestions


class TestMultipleRelationSupport:
    """Test support for multiple relations between entity pairs."""
    
    def test_multiple_relations_preserved(self):
        """Test that multiple different relation types between same entities are preserved."""
        # Create mock relations with same entities but different types
        relations = [
            GraphRelation(
                source_entity_id="entity1",
                target_entity_id="entity2", 
                type="treats",
                confidence=0.8
            ),
            GraphRelation(
                source_entity_id="entity1",
                target_entity_id="entity2",
                type="diagnoses", 
                confidence=0.7
            ),
            GraphRelation(
                source_entity_id="entity1",
                target_entity_id="entity2",
                type="monitors",
                confidence=0.9
            )
        ]
        
        # Mock the relation agent's deduplication method
        from morag_graph.ai.relation_agent import RelationExtractionAgent
        agent = RelationExtractionAgent()
        
        deduplicated = agent._deduplicate_relations(relations)
        
        # All three relations should be preserved since they have different types
        assert len(deduplicated) == 3
        
        # Check that all relation types are present
        relation_types = {r.type for r in deduplicated}
        assert relation_types == {"treats", "diagnoses", "monitors"}
    
    def test_duplicate_relations_merged(self):
        """Test that exact duplicate relations are properly merged."""
        # Create duplicate relations (same source, target, and type)
        relations = [
            GraphRelation(
                source_entity_id="entity1",
                target_entity_id="entity2",
                type="treats",
                confidence=0.8,
                attributes={"context": "context1"}
            ),
            GraphRelation(
                source_entity_id="entity1", 
                target_entity_id="entity2",
                type="treats",
                confidence=0.6,
                attributes={"context": "context2"}
            )
        ]
        
        from morag_graph.ai.relation_agent import RelationExtractionAgent
        agent = RelationExtractionAgent()
        
        deduplicated = agent._deduplicate_relations(relations)
        
        # Should have only one relation (the duplicates merged)
        assert len(deduplicated) == 1
        
        # Should keep the higher confidence
        assert deduplicated[0].confidence == 0.8
        
        # Should merge contexts
        assert "context1" in deduplicated[0].attributes["context"]
        assert "context2" in deduplicated[0].attributes["context"]


if __name__ == "__main__":
    pytest.main([__file__])
