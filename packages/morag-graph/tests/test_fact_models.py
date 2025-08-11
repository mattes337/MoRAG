"""Tests for fact models and validation (no LLM dependencies)."""

import pytest
from datetime import datetime

from morag_graph.models.fact import Fact, FactRelation, FactType, FactRelationType
from morag_graph.extraction.fact_validator import FactValidator


class TestFactModel:
    """Test the Fact model."""
    
    def test_fact_creation(self):
        """Test basic fact creation."""
        fact = Fact(
            subject="Machine Learning",
            object="image classification accuracy",
            approach="convolutional neural networks",
            solution="95% accuracy on ImageNet dataset",
            source_chunk_id="chunk_123",
            source_document_id="doc_456",
            extraction_confidence=0.9,
            fact_type=FactType.RESEARCH
        )
        
        assert fact.subject == "Machine Learning"
        assert fact.object == "image classification accuracy"
        assert fact.approach == "convolutional neural networks"
        assert fact.solution == "95% accuracy on ImageNet dataset"
        assert fact.extraction_confidence == 0.9
        assert fact.fact_type == FactType.RESEARCH
        assert fact.id.startswith("fact_")
    
    def test_fact_completeness(self):
        """Test fact completeness checking."""
        # Complete fact
        complete_fact = Fact(
            subject="Python",
            object="web development",
            approach="Django framework",
            solution="rapid application development",
            source_chunk_id="chunk_1",
            source_document_id="doc_1",
            extraction_confidence=0.8,
            fact_type=FactType.PROCESS
        )
        assert complete_fact.is_complete()
        
        # Incomplete fact (no approach or solution)
        incomplete_fact = Fact(
            subject="Python",
            object="web development",
            source_chunk_id="chunk_1",
            source_document_id="doc_1",
            extraction_confidence=0.8,
            fact_type=FactType.PROCESS
        )
        assert not incomplete_fact.is_complete()
    
    def test_fact_neo4j_properties(self):
        """Test Neo4j properties conversion."""
        fact = Fact(
            subject="Test Subject",
            object="Test Object",
            approach="Test Approach",
            source_chunk_id="chunk_1",
            source_document_id="doc_1",
            extraction_confidence=0.85,
            fact_type=FactType.DEFINITION,
            keywords=["test", "example"]
        )
        
        props = fact.get_neo4j_properties()
        
        assert props["subject"] == "Test Subject"
        assert props["object"] == "Test Object"
        assert props["approach"] == "Test Approach"
        assert props["confidence"] == 0.85
        assert props["fact_type"] == FactType.DEFINITION
        assert props["keywords"] == "test,example"
    
    def test_fact_search_text(self):
        """Test search text generation."""
        fact = Fact(
            subject="Machine Learning",
            object="data analysis",
            approach="neural networks",
            solution="improved accuracy",
            remarks="requires large datasets",
            keywords=["AI", "ML"],
            source_chunk_id="chunk_1",
            source_document_id="doc_1",
            extraction_confidence=0.9,
            fact_type=FactType.RESEARCH
        )
        
        search_text = fact.get_search_text()
        assert "Machine Learning" in search_text
        assert "data analysis" in search_text
        assert "neural networks" in search_text
        assert "improved accuracy" in search_text
        assert "requires large datasets" in search_text
        assert "AI" in search_text
        assert "ML" in search_text
    
    def test_fact_display_text(self):
        """Test display text generation."""
        fact = Fact(
            subject="Python",
            object="web development",
            approach="Django framework",
            solution="rapid prototyping",
            source_chunk_id="chunk_1",
            source_document_id="doc_1",
            extraction_confidence=0.8,
            fact_type=FactType.PROCESS
        )
        
        display_text = fact.get_display_text()
        assert "Subject: Python" in display_text
        assert "Object: web development" in display_text
        assert "Approach: Django framework" in display_text
        assert "Solution: rapid prototyping" in display_text


class TestFactValidator:
    """Test the FactValidator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = FactValidator(min_confidence=0.7)
    
    def test_valid_fact(self):
        """Test validation of a valid fact."""
        fact = Fact(
            subject="Machine Learning algorithms",
            object="pattern recognition in large datasets",
            approach="supervised learning with neural networks",
            solution="improved accuracy and reduced processing time",
            source_chunk_id="chunk_1",
            source_document_id="doc_1",
            extraction_confidence=0.9,
            fact_type=FactType.RESEARCH
        )
        
        is_valid, issues = self.validator.validate_fact(fact)
        assert is_valid
        assert len(issues) == 0
    
    def test_invalid_fact_generic_subject(self):
        """Test validation fails for generic subject."""
        fact = Fact(
            subject="it",
            object="pattern recognition",
            approach="machine learning",
            source_chunk_id="chunk_1",
            source_document_id="doc_1",
            extraction_confidence=0.8,
            fact_type=FactType.RESEARCH
        )
        
        is_valid, issues = self.validator.validate_fact(fact)
        assert not is_valid
        assert any("Generic subject" in issue for issue in issues)
    
    def test_invalid_fact_low_confidence(self):
        """Test validation fails for low confidence."""
        fact = Fact(
            subject="Machine Learning",
            object="pattern recognition",
            approach="neural networks",
            source_chunk_id="chunk_1",
            source_document_id="doc_1",
            extraction_confidence=0.5,  # Below threshold
            fact_type=FactType.RESEARCH
        )
        
        is_valid, issues = self.validator.validate_fact(fact)
        assert not is_valid
        assert any("Confidence" in issue and "below threshold" in issue for issue in issues)
    
    def test_invalid_fact_missing_actionable_content(self):
        """Test validation fails for missing actionable content."""
        fact = Fact(
            subject="Something",
            object="something else",
            source_chunk_id="chunk_1",
            source_document_id="doc_1",
            extraction_confidence=0.8,
            fact_type=FactType.RESEARCH
        )
        
        is_valid, issues = self.validator.validate_fact(fact)
        assert not is_valid
        assert any("Missing both approach and solution" in issue for issue in issues)
    
    def test_batch_validation(self):
        """Test batch validation of multiple facts."""
        facts = [
            Fact(
                subject="Valid fact",
                object="valid content",
                approach="valid approach",
                source_chunk_id="chunk_1",
                source_document_id="doc_1",
                extraction_confidence=0.9,
                fact_type=FactType.RESEARCH
            ),
            Fact(
                subject="it",  # Invalid - generic subject
                object="invalid content",
                approach="some approach",
                source_chunk_id="chunk_2",
                source_document_id="doc_1",
                extraction_confidence=0.8,
                fact_type=FactType.RESEARCH
            )
        ]
        
        result = self.validator.validate_facts_batch(facts)
        
        assert result['total_facts'] == 2
        assert result['valid_facts'] == 1
        assert result['invalid_facts'] == 1
        assert result['validation_rate'] == 0.5
    
    def test_quality_score(self):
        """Test quality score calculation."""
        # High quality fact
        good_fact = Fact(
            subject="Machine Learning",
            object="image classification",
            approach="convolutional neural networks",
            solution="95% accuracy",
            remarks="tested on ImageNet",
            keywords=["ML", "CNN", "classification"],
            source_chunk_id="chunk_1",
            source_document_id="doc_1",
            extraction_confidence=0.9,
            fact_type=FactType.RESEARCH
        )
        
        score = self.validator.get_quality_score(good_fact)
        assert score > 0.9  # Should get bonus for completeness
        
        # Low quality fact
        bad_fact = Fact(
            subject="it",  # Generic subject
            object="something",
            source_chunk_id="chunk_1",
            source_document_id="doc_1",
            extraction_confidence=0.8,
            fact_type=FactType.RESEARCH
        )
        
        score = self.validator.get_quality_score(bad_fact)
        assert score < 0.8  # Should be penalized


class TestFactRelation:
    """Test the FactRelation model."""
    
    def test_fact_relation_creation(self):
        """Test basic fact relation creation."""
        relation = FactRelation(
            source_fact_id="fact_1",
            target_fact_id="fact_2",
            relation_type=FactRelationType.SUPPORTS,
            confidence=0.8,
            context="Fact 1 provides evidence for Fact 2"
        )
        
        assert relation.source_fact_id == "fact_1"
        assert relation.target_fact_id == "fact_2"
        assert relation.relation_type == FactRelationType.SUPPORTS
        assert relation.confidence == 0.8
        assert relation.id.startswith("fact_rel_")
    
    def test_fact_relation_neo4j_properties(self):
        """Test Neo4j properties conversion."""
        relation = FactRelation(
            source_fact_id="fact_1",
            target_fact_id="fact_2",
            relation_type=FactRelationType.ELABORATES,
            confidence=0.75,
            context="Additional details"
        )
        
        props = relation.get_neo4j_properties()
        
        assert props["relation_type"] == FactRelationType.ELABORATES
        assert props["confidence"] == 0.75
        assert props["context"] == "Additional details"


class TestFactTypes:
    """Test fact type constants."""
    
    def test_fact_types(self):
        """Test fact type constants."""
        all_types = FactType.all_types()
        
        assert FactType.RESEARCH in all_types
        assert FactType.PROCESS in all_types
        assert FactType.DEFINITION in all_types
        assert FactType.CAUSAL in all_types
        assert FactType.COMPARATIVE in all_types
        assert FactType.TEMPORAL in all_types
        assert FactType.STATISTICAL in all_types
        assert FactType.METHODOLOGICAL in all_types
    
    def test_fact_relation_types(self):
        """Test fact relation type constants."""
        all_types = FactRelationType.all_types()
        
        assert FactRelationType.SUPPORTS in all_types
        assert FactRelationType.CONTRADICTS in all_types
        assert FactRelationType.ELABORATES in all_types
        assert FactRelationType.SEQUENCE in all_types
        assert FactRelationType.COMPARISON in all_types
        assert FactRelationType.CAUSATION in all_types
        assert FactRelationType.TEMPORAL_ORDER in all_types


if __name__ == "__main__":
    pytest.main([__file__])
