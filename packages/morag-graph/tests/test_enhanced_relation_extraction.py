"""Tests for enhanced relation extraction capabilities."""

import pytest
import asyncio
from typing import List, Dict

from morag_graph.ai.enhanced_relation_agent import EnhancedRelationExtractionAgent
from morag_graph.ai.semantic_analyzer import SemanticRelationAnalyzer
from morag_graph.ai.domain_extractors import DomainExtractorFactory
from morag_graph.ai.multi_pass_extractor import MultiPassRelationExtractor
from morag_graph.models import Entity, Relation
from morag_core.ai import Relation as CoreRelation


class TestEnhancedRelationExtraction:
    """Test enhanced relation extraction capabilities."""

    @pytest.fixture
    def sample_medical_entities(self):
        """Create sample medical entities for testing."""
        return [
            Entity(
                name="Aspirin",
                type="SUBSTANCE",
                confidence=0.9,
                source_doc_id="test_doc"
            ),
            Entity(
                name="Heart Disease",
                type="CONCEPT",
                confidence=0.85,
                source_doc_id="test_doc"
            ),
            Entity(
                name="Dr. Smith",
                type="PERSON",
                confidence=0.9,
                source_doc_id="test_doc"
            ),
            Entity(
                name="Mayo Clinic",
                type="ORGANIZATION",
                confidence=0.88,
                source_doc_id="test_doc"
            )
        ]

    @pytest.fixture
    def sample_technical_entities(self):
        """Create sample technical entities for testing."""
        return [
            Entity(
                name="Python",
                type="TECHNOLOGY",
                confidence=0.9,
                source_doc_id="test_doc"
            ),
            Entity(
                name="Django",
                type="SOFTWARE",
                confidence=0.85,
                source_doc_id="test_doc"
            ),
            Entity(
                name="REST API",
                type="CONCEPT",
                confidence=0.8,
                source_doc_id="test_doc"
            ),
            Entity(
                name="Database",
                type="SYSTEM",
                confidence=0.9,
                source_doc_id="test_doc"
            )
        ]

    @pytest.fixture
    def medical_text(self):
        """Sample medical text with rich relationships."""
        return """
        Dr. Smith at Mayo Clinic has conducted extensive research on heart disease prevention.
        His studies show that Aspirin significantly reduces the risk of heart attacks by preventing
        blood clots. However, Aspirin can cause stomach bleeding in some patients, which
        contraindicated its use in people with ulcers. The research demonstrates that
        low-dose Aspirin treatment leads to a 25% reduction in cardiovascular events.
        Dr. Smith's methodology involves analyzing patient data over 10 years to establish
        causal relationships between medication and outcomes.
        """

    @pytest.fixture
    def technical_text(self):
        """Sample technical text with rich relationships."""
        return """
        Python is a programming language that enables rapid development of web applications.
        Django, a Python framework, implements the Model-View-Controller pattern and
        integrates seamlessly with various database systems. The framework provides
        built-in support for REST API development, which allows applications to
        communicate with external services. Django's ORM layer abstracts database
        operations and transforms Python objects into SQL queries. This architecture
        enables developers to build scalable applications that can handle millions of requests.
        """

    @pytest.mark.asyncio
    async def test_semantic_analyzer(self):
        """Test semantic relation analyzer."""
        analyzer = SemanticRelationAnalyzer()
        
        # Create a test relation
        relation = CoreRelation(
            source_entity="Aspirin",
            target_entity="Heart Disease",
            relation_type="RELATED_TO",
            confidence=0.7,
            context="Aspirin prevents heart disease by reducing blood clots"
        )
        
        # Analyze the relation
        enhancement = analyzer.analyze_relation_context(
            relation,
            "Aspirin prevents heart disease by reducing blood clots and inflammation",
            None
        )
        
        # Verify enhancement
        assert enhancement.original_type == "RELATED_TO"
        assert enhancement.enhanced_type in ["PREVENTS", "CAUSES", "INFLUENCES"]
        assert len(enhancement.semantic_signals) > 0
        assert enhancement.confidence_adjustment != 0

    @pytest.mark.asyncio
    async def test_domain_extractor_medical(self):
        """Test medical domain extractor."""
        extractor = DomainExtractorFactory.create_extractor("medical")
        assert extractor is not None
        
        # Test with medical entities and text
        entities = [
            Entity(name="Aspirin", type="SUBSTANCE", confidence=0.9, source_doc_id="test"),
            Entity(name="Heart Disease", type="CONCEPT", confidence=0.9, source_doc_id="test")
        ]
        
        text = "Aspirin treats heart disease by preventing blood clots"
        relations = extractor.extract_domain_relations(text, entities, [])
        
        # Should find treatment relationship
        assert len(relations) > 0
        treatment_relations = [r for r in relations if r.relation_type == "TREATS"]
        assert len(treatment_relations) > 0

    @pytest.mark.asyncio
    async def test_domain_extractor_technical(self):
        """Test technical domain extractor."""
        extractor = DomainExtractorFactory.create_extractor("technical")
        assert extractor is not None
        
        # Test with technical entities and text
        entities = [
            Entity(name="Django", type="SOFTWARE", confidence=0.9, source_doc_id="test"),
            Entity(name="Python", type="TECHNOLOGY", confidence=0.9, source_doc_id="test")
        ]
        
        text = "Django extends Python functionality and integrates with databases"
        relations = extractor.extract_domain_relations(text, entities, [])
        
        # Should find technical relationships
        assert len(relations) > 0
        tech_relations = [r for r in relations if r.relation_type in ["EXTENDS", "INTEGRATES_WITH"]]
        assert len(tech_relations) > 0

    @pytest.mark.asyncio
    async def test_multi_pass_extractor_medical(self, sample_medical_entities, medical_text):
        """Test multi-pass extractor with medical content."""
        extractor = MultiPassRelationExtractor(
            min_confidence=0.5,
            enable_semantic_analysis=True,
            enable_domain_extraction=True
        )
        
        result = await extractor.extract_relations_multi_pass(
            text=medical_text,
            entities=sample_medical_entities,
            source_doc_id="test_medical",
            domain_hint="medical"
        )
        
        # Verify results
        assert result.domain == "medical"
        assert len(result.final_relations) > 0
        assert result.statistics['total_passes'] >= 3
        
        # Check for meaningful medical relations
        relation_types = [r.type for r in result.final_relations]
        meaningful_types = ["TREATS", "PREVENTS", "CAUSES", "CONDUCTS", "RESEARCHES"]
        assert any(rt in meaningful_types for rt in relation_types)
        
        # Verify confidence distribution
        high_confidence = result.confidence_distribution.get('high', 0) + \
                         result.confidence_distribution.get('very_high', 0)
        assert high_confidence > 0

    @pytest.mark.asyncio
    async def test_multi_pass_extractor_technical(self, sample_technical_entities, technical_text):
        """Test multi-pass extractor with technical content."""
        extractor = MultiPassRelationExtractor(
            min_confidence=0.5,
            enable_semantic_analysis=True,
            enable_domain_extraction=True
        )
        
        result = await extractor.extract_relations_multi_pass(
            text=technical_text,
            entities=sample_technical_entities,
            source_doc_id="test_technical",
            domain_hint="technical"
        )
        
        # Verify results
        assert result.domain == "technical"
        assert len(result.final_relations) > 0
        
        # Check for meaningful technical relations
        relation_types = [r.type for r in result.final_relations]
        meaningful_types = ["IMPLEMENTS", "INTEGRATES_WITH", "ENABLES", "TRANSFORMS", "PROVIDES"]
        assert any(rt in meaningful_types for rt in relation_types)

    @pytest.mark.asyncio
    async def test_relation_quality_improvement(self, sample_medical_entities, medical_text):
        """Test that enhanced extraction produces higher quality relations."""
        # Test with basic extraction (simulated)
        basic_relations = [
            "MENTIONS", "RELATED_TO", "USES"  # Typical shallow relations
        ]
        
        # Test with enhanced extraction
        extractor = MultiPassRelationExtractor(min_confidence=0.6)
        result = await extractor.extract_relations_multi_pass(
            text=medical_text,
            entities=sample_medical_entities,
            domain_hint="medical"
        )
        
        enhanced_types = [r.type for r in result.final_relations]
        
        # Enhanced extraction should produce more specific relation types
        specific_types = [
            "TREATS", "PREVENTS", "CAUSES", "CONDUCTS", "RESEARCHES",
            "DEMONSTRATES", "ANALYZES", "ESTABLISHES"
        ]
        
        specific_count = sum(1 for rt in enhanced_types if rt in specific_types)
        generic_count = sum(1 for rt in enhanced_types if rt in ["MENTIONS", "RELATED_TO"])
        
        # Should have more specific relations than generic ones
        assert specific_count > generic_count

    def test_relation_category_mapping(self):
        """Test relation category mapping."""
        from morag_graph.ai.enhanced_relation_agent import EnhancedRelationExtractionAgent
        
        agent = EnhancedRelationExtractionAgent()
        
        # Test causal relations
        assert agent._get_relation_category("CAUSES") == "causal"
        assert agent._get_relation_category("PREVENTS") == "causal"
        
        # Test temporal relations
        assert agent._get_relation_category("PRECEDES") == "temporal"
        assert agent._get_relation_category("FOLLOWS") == "temporal"
        
        # Test hierarchical relations
        assert agent._get_relation_category("MANAGES") == "hierarchical"
        assert agent._get_relation_category("BELONGS_TO") == "hierarchical"

    def test_domain_detection(self):
        """Test domain detection capabilities."""
        extractor = MultiPassRelationExtractor()
        
        # Test medical domain detection
        medical_text = "The patient was prescribed medication for treatment"
        medical_entities = {"patient": "PERSON", "medication": "SUBSTANCE"}
        
        # This would be tested in the actual domain detection method
        # For now, we verify the domain patterns exist
        assert "medical" in extractor.enhanced_agent.domain_patterns
        assert "technical" in extractor.enhanced_agent.domain_patterns

    @pytest.mark.asyncio
    async def test_confidence_scoring(self, sample_medical_entities, medical_text):
        """Test confidence scoring improvements."""
        extractor = MultiPassRelationExtractor(min_confidence=0.4)  # Lower threshold to see all relations
        
        result = await extractor.extract_relations_multi_pass(
            text=medical_text,
            entities=sample_medical_entities,
            domain_hint="medical"
        )
        
        # Check confidence distribution
        confidences = [r.confidence for r in result.final_relations]
        
        # Should have a range of confidences
        assert min(confidences) >= 0.4
        assert max(confidences) <= 1.0
        
        # High-evidence relations should have higher confidence
        high_conf_relations = [r for r in result.final_relations if r.confidence > 0.8]
        assert len(high_conf_relations) > 0

    def test_extraction_summary(self, sample_medical_entities, medical_text):
        """Test extraction summary generation."""
        extractor = MultiPassRelationExtractor()
        
        # Create a mock result
        from morag_graph.ai.multi_pass_extractor import MultiPassResult, ExtractionResult, ExtractionPass
        
        mock_result = MultiPassResult(
            final_relations=[],
            pass_results=[],
            statistics={
                'total_passes': 5,
                'total_relations': 10,
                'relations_by_pass': {'basic': 3, 'semantic': 4, 'domain': 2, 'validation': 1},
                'avg_confidence': 0.75,
                'relation_types': {'TREATS': 3, 'PREVENTS': 2, 'CAUSES': 1}
            },
            domain="medical",
            confidence_distribution={'high': 5, 'medium': 3, 'low': 2}
        )
        
        summary = extractor.get_extraction_summary(mock_result)
        
        # Verify summary contains key information
        assert "medical" in summary
        assert "10" in summary  # total relations
        assert "5" in summary   # total passes
        assert "0.750" in summary  # average confidence
        assert "TREATS" in summary  # top relation type
