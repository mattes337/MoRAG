"""Tests for fact relevance scoring."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from morag_reasoning.fact_scorer import (
    FactRelevanceScorer,
    ScoredFact,
    ScoringDimensions,
    ScoringStrategy
)
from morag_reasoning.graph_fact_extractor import ExtractedFact, FactType
from morag_reasoning.llm import LLMClient


class TestFactRelevanceScorer:
    """Test the fact relevance scorer."""
    
    @pytest.fixture
    def mock_llm_client(self):
        """Mock LLM client for testing."""
        return MagicMock(spec=LLMClient)
    
    @pytest.fixture
    def sample_facts(self):
        """Sample extracted facts for testing."""
        return [
            ExtractedFact(
                fact_id="fact_001",
                content="Einstein developed the theory of relativity",
                fact_type=FactType.DIRECT,
                confidence=0.9,
                source_entities=["ent_einstein", "ent_relativity"],
                source_relations=["rel_developed"],
                source_documents=["doc_physics"],
                extraction_path=["ent_einstein", "ent_relativity"],
                context={"relation_type": "DEVELOPED"},
                metadata={"extraction_method": "direct_triplet"}
            ),
            ExtractedFact(
                fact_id="fact_002",
                content="Princeton University is located in New Jersey",
                fact_type=FactType.DIRECT,
                confidence=0.8,
                source_entities=["ent_princeton", "ent_new_jersey"],
                source_relations=["rel_located_in"],
                source_documents=["doc_geography"],
                extraction_path=["ent_princeton", "ent_new_jersey"],
                context={"relation_type": "LOCATED_IN"},
                metadata={"extraction_method": "direct_triplet"}
            ),
            ExtractedFact(
                fact_id="fact_003",
                content="Einstein worked at Princeton University and developed relativity theory there",
                fact_type=FactType.CHAIN,
                confidence=0.7,
                source_entities=["ent_einstein", "ent_princeton", "ent_relativity"],
                source_relations=["rel_worked_at", "rel_developed"],
                source_documents=["doc_physics", "doc_biography"],
                extraction_path=["ent_einstein", "ent_princeton", "ent_relativity"],
                context={"chain_length": 3},
                metadata={"extraction_method": "relationship_chain"}
            )
        ]
    
    @pytest.mark.asyncio
    async def test_scorer_initialization(self, mock_llm_client):
        """Test scorer initialization with different configurations."""
        config = {
            'scoring_strategy': 'relevance_focused',
            'min_score_threshold': 0.3,
            'llm_enabled': False,  # Disable LLM for testing
            'semantic_enabled': False  # Disable semantic similarity for testing
        }
        
        scorer = FactRelevanceScorer(mock_llm_client, config)
        
        assert scorer.scoring_strategy == ScoringStrategy.RELEVANCE_FOCUSED
        assert scorer.min_score_threshold == 0.3
        assert not scorer.llm_enabled
        assert not scorer.semantic_enabled
    
    @pytest.mark.asyncio
    async def test_basic_fact_scoring(self, mock_llm_client, sample_facts):
        """Test basic fact scoring functionality."""
        config = {
            'llm_enabled': False,
            'semantic_enabled': False,
            'scoring_strategy': 'balanced'
        }
        
        scorer = FactRelevanceScorer(mock_llm_client, config)
        
        query = "What did Einstein develop?"
        scored_facts = await scorer.score_facts(sample_facts, query)
        
        assert len(scored_facts) > 0
        assert all(isinstance(sf, ScoredFact) for sf in scored_facts)
        assert all(0.0 <= sf.overall_score <= 1.0 for sf in scored_facts)
        assert all(isinstance(sf.scoring_dimensions, ScoringDimensions) for sf in scored_facts)
    
    @pytest.mark.asyncio
    async def test_relevance_focused_scoring(self, mock_llm_client, sample_facts):
        """Test relevance-focused scoring strategy."""
        config = {
            'llm_enabled': False,
            'semantic_enabled': False,
            'scoring_strategy': 'relevance_focused'
        }
        
        scorer = FactRelevanceScorer(mock_llm_client, config)
        
        query = "What did Einstein develop?"
        scored_facts = await scorer.score_facts(sample_facts, query)
        
        # The first fact should score highest as it directly answers the query
        assert len(scored_facts) > 0
        
        # Find the Einstein relativity fact
        einstein_fact = next(
            (sf for sf in scored_facts if "Einstein" in sf.fact.content and "relativity" in sf.fact.content),
            None
        )
        assert einstein_fact is not None
        assert einstein_fact.overall_score > 0.0
    
    @pytest.mark.asyncio
    async def test_scoring_dimensions(self, mock_llm_client, sample_facts):
        """Test individual scoring dimensions."""
        config = {
            'llm_enabled': False,
            'semantic_enabled': False
        }
        
        scorer = FactRelevanceScorer(mock_llm_client, config)
        
        query = "What did Einstein develop?"
        scored_facts = await scorer.score_facts(sample_facts, query)
        
        for scored_fact in scored_facts:
            dims = scored_fact.scoring_dimensions
            
            # All dimensions should be between 0 and 1
            assert 0.0 <= dims.query_relevance <= 1.0
            assert 0.0 <= dims.source_quality <= 1.0
            assert 0.0 <= dims.confidence <= 1.0
            assert 0.0 <= dims.recency <= 1.0
            assert 0.0 <= dims.completeness <= 1.0
            assert 0.0 <= dims.specificity <= 1.0
    
    @pytest.mark.asyncio
    async def test_confidence_scoring(self, mock_llm_client, sample_facts):
        """Test that confidence scores are properly used."""
        config = {
            'llm_enabled': False,
            'semantic_enabled': False,
            'scoring_strategy': 'confidence_focused'
        }
        
        scorer = FactRelevanceScorer(mock_llm_client, config)
        
        query = "Tell me about Einstein"
        scored_facts = await scorer.score_facts(sample_facts, query)
        
        # Facts with higher confidence should generally score higher
        # (though other factors also matter)
        high_confidence_facts = [sf for sf in scored_facts if sf.fact.confidence >= 0.8]
        low_confidence_facts = [sf for sf in scored_facts if sf.fact.confidence < 0.8]
        
        if high_confidence_facts and low_confidence_facts:
            avg_high_score = sum(sf.overall_score for sf in high_confidence_facts) / len(high_confidence_facts)
            avg_low_score = sum(sf.overall_score for sf in low_confidence_facts) / len(low_confidence_facts)
            
            # High confidence facts should generally score better
            assert avg_high_score >= avg_low_score
    
    @pytest.mark.asyncio
    async def test_fact_type_scoring(self, mock_llm_client, sample_facts):
        """Test that different fact types are scored appropriately."""
        config = {
            'llm_enabled': False,
            'semantic_enabled': False
        }
        
        scorer = FactRelevanceScorer(mock_llm_client, config)
        
        query = "Einstein and relativity"
        scored_facts = await scorer.score_facts(sample_facts, query)
        
        # Check that direct facts and chain facts are both scored
        direct_facts = [sf for sf in scored_facts if sf.fact.fact_type == FactType.DIRECT]
        chain_facts = [sf for sf in scored_facts if sf.fact.fact_type == FactType.CHAIN]
        
        assert len(direct_facts) > 0
        assert len(chain_facts) > 0
        
        # All facts should have reasonable scores
        for sf in scored_facts:
            assert sf.overall_score > 0.0
    
    @pytest.mark.asyncio
    async def test_keyword_relevance_calculation(self, mock_llm_client):
        """Test keyword-based relevance calculation."""
        config = {
            'llm_enabled': False,
            'semantic_enabled': False
        }
        
        scorer = FactRelevanceScorer(mock_llm_client, config)
        
        # Test exact match
        relevance = scorer._calculate_keyword_relevance(
            "Einstein developed relativity", 
            "Einstein relativity"
        )
        assert relevance > 0.5
        
        # Test partial match
        relevance = scorer._calculate_keyword_relevance(
            "Einstein worked at Princeton", 
            "Einstein"
        )
        assert relevance > 0.0
        
        # Test no match
        relevance = scorer._calculate_keyword_relevance(
            "Completely unrelated content", 
            "Einstein"
        )
        assert relevance == 0.0
    
    @pytest.mark.asyncio
    async def test_source_quality_calculation(self, mock_llm_client, sample_facts):
        """Test source quality calculation."""
        config = {
            'llm_enabled': False,
            'semantic_enabled': False
        }
        
        scorer = FactRelevanceScorer(mock_llm_client, config)
        
        for fact in sample_facts:
            quality_score = scorer._calculate_source_quality(fact)
            assert 0.0 <= quality_score <= 1.0
            
            # Direct facts should have higher quality than chain facts
            if fact.fact_type == FactType.DIRECT:
                assert quality_score >= 0.5
    
    @pytest.mark.asyncio
    async def test_empty_facts_handling(self, mock_llm_client):
        """Test handling of empty fact lists."""
        config = {
            'llm_enabled': False,
            'semantic_enabled': False
        }
        
        scorer = FactRelevanceScorer(mock_llm_client, config)
        
        scored_facts = await scorer.score_facts([], "test query")
        assert scored_facts == []
    
    @pytest.mark.asyncio
    async def test_scoring_threshold_filtering(self, mock_llm_client, sample_facts):
        """Test filtering by minimum score threshold."""
        config = {
            'llm_enabled': False,
            'semantic_enabled': False,
            'min_score_threshold': 0.8  # High threshold
        }
        
        scorer = FactRelevanceScorer(mock_llm_client, config)
        
        query = "Completely unrelated query about cooking recipes"
        scored_facts = await scorer.score_facts(sample_facts, query)
        
        # Should filter out low-relevance facts
        assert all(sf.overall_score >= 0.8 for sf in scored_facts)
    
    @pytest.mark.asyncio
    async def test_scoring_reasoning_generation(self, mock_llm_client, sample_facts):
        """Test that scoring reasoning is generated."""
        config = {
            'llm_enabled': False,
            'semantic_enabled': False
        }
        
        scorer = FactRelevanceScorer(mock_llm_client, config)
        
        query = "What did Einstein develop?"
        scored_facts = await scorer.score_facts(sample_facts, query)
        
        for scored_fact in scored_facts:
            assert scored_fact.reasoning
            assert isinstance(scored_fact.reasoning, str)
            assert len(scored_fact.reasoning) > 0
    
    @pytest.mark.asyncio
    async def test_different_scoring_strategies(self, mock_llm_client, sample_facts):
        """Test different scoring strategies."""
        strategies = [
            ScoringStrategy.BALANCED,
            ScoringStrategy.RELEVANCE_FOCUSED,
            ScoringStrategy.QUALITY_FOCUSED,
            ScoringStrategy.CONFIDENCE_FOCUSED,
            ScoringStrategy.ADAPTIVE
        ]
        
        query = "What did Einstein develop?"
        
        for strategy in strategies:
            config = {
                'llm_enabled': False,
                'semantic_enabled': False,
                'scoring_strategy': strategy.value
            }
            
            scorer = FactRelevanceScorer(mock_llm_client, config)
            scored_facts = await scorer.score_facts(sample_facts, query)
            
            # Should work with all strategies
            assert isinstance(scored_facts, list)
            if scored_facts:  # May be empty due to filtering
                assert all(isinstance(sf, ScoredFact) for sf in scored_facts)
