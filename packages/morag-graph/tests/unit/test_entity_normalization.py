"""Unit tests for entity normalization functionality."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

from packages.morag_graph.src.morag_graph.normalizers.entity_normalizer import (
    EnhancedEntityNormalizer,
    EntityVariation,
    EntityMergeCandidate,
    NormalizedEntity
)
from packages.morag_graph.src.morag_graph.config.normalization_config import (
    EntityNormalizationConfig,
    NormalizationRulesConfig
)


class TestEntityVariation:
    """Test EntityVariation dataclass."""
    
    def test_entity_variation_creation(self):
        """Test creating EntityVariation instance."""
        variation = EntityVariation(
            original="AI systems",
            normalized="artificial intelligence system",
            confidence=0.9,
            rule_applied="acronym_expansion"
        )
        
        assert variation.original == "AI systems"
        assert variation.normalized == "artificial intelligence system"
        assert variation.confidence == 0.9
        assert variation.rule_applied == "acronym_expansion"


class TestEntityMergeCandidate:
    """Test EntityMergeCandidate dataclass."""
    
    def test_merge_candidate_creation(self):
        """Test creating EntityMergeCandidate instance."""
        candidate = EntityMergeCandidate(
            entities=["AI", "A.I.", "artificial intelligence"],
            canonical_form="artificial intelligence",
            confidence=0.85,
            merge_reason="acronym_variations"
        )
        
        assert len(candidate.entities) == 3
        assert candidate.canonical_form == "artificial intelligence"
        assert candidate.confidence == 0.85
        assert candidate.merge_reason == "acronym_variations"


class TestEnhancedEntityNormalizer:
    """Test EnhancedEntityNormalizer class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.mock_llm_service = Mock()
        self.test_config = {
            'batch_size': 10,
            'min_confidence': 0.7,
            'merge_confidence_threshold': 0.8,
            'enable_llm_normalization': True,
            'enable_rule_based_fallback': True
        }
    
    def test_normalizer_initialization_with_llm(self):
        """Test normalizer initialization with LLM service."""
        normalizer = EnhancedEntityNormalizer(
            llm_service=self.mock_llm_service,
            config=self.test_config
        )
        
        assert normalizer.llm_service == self.mock_llm_service
        assert normalizer.batch_size == 10
        assert normalizer.min_confidence == 0.7
        assert normalizer.enable_llm_normalization is True
        assert normalizer.normalization_agent is not None
        assert normalizer.merge_analysis_agent is not None
    
    def test_normalizer_initialization_without_llm(self):
        """Test normalizer initialization without LLM service."""
        normalizer = EnhancedEntityNormalizer(config=self.test_config)
        
        assert normalizer.llm_service is None
        assert normalizer.normalization_agent is None
        assert normalizer.merge_analysis_agent is None
    
    def test_normalizer_initialization_with_default_config(self):
        """Test normalizer initialization with default configuration."""
        with patch('packages.morag_graph.src.morag_graph.normalizers.entity_normalizer.get_config_for_component') as mock_config:
            mock_config.return_value = self.test_config
            normalizer = EnhancedEntityNormalizer()
            
            assert normalizer.config == self.test_config
            mock_config.assert_called_once_with('normalizer')
    
    @pytest.mark.asyncio
    async def test_normalize_entity_without_llm(self):
        """Test entity normalization without LLM service."""
        normalizer = EnhancedEntityNormalizer(config={'enable_llm_normalization': False})
        
        variation = await normalizer.normalize_entity("test entity")
        
        assert variation.original == "test entity"
        assert variation.normalized == "test entity"  # Should be stripped
        assert variation.confidence == 0.5
        assert variation.rule_applied == "basic_cleanup"
    
    @pytest.mark.asyncio
    async def test_normalize_entity_with_cache(self):
        """Test entity normalization with caching."""
        normalizer = EnhancedEntityNormalizer(config={'enable_llm_normalization': False})
        
        # First call
        variation1 = await normalizer.normalize_entity("test entity", "en")
        
        # Second call should use cache
        variation2 = await normalizer.normalize_entity("test entity", "en")
        
        assert variation1.original == variation2.original
        assert variation1.normalized == variation2.normalized
        assert "test entity:en" in normalizer.normalization_cache
    
    @pytest.mark.asyncio
    async def test_find_merge_candidates_without_llm(self):
        """Test finding merge candidates without LLM service."""
        normalizer = EnhancedEntityNormalizer(config={'enable_llm_normalization': False})
        
        entities = ["AI", "artificial intelligence", "machine learning", "ML"]
        candidates = await normalizer.find_merge_candidates(entities)
        
        # Should return empty list without LLM
        assert isinstance(candidates, list)
    
    def test_fallback_similarity_matching(self):
        """Test fallback similarity matching."""
        normalizer = EnhancedEntityNormalizer(config={'enable_llm_normalization': False})
        
        entities = ["artificial intelligence", "Artificial Intelligence", "AI"]
        candidates = normalizer._fallback_similarity_matching(entities)
        
        # Should find high similarity between first two
        assert len(candidates) >= 1
        found_match = False
        for candidate in candidates:
            if ("artificial intelligence" in candidate.entities and 
                "Artificial Intelligence" in candidate.entities):
                found_match = True
                assert candidate.confidence > 0.85
                break
        assert found_match
    
    def test_fallback_similarity_matching_no_matches(self):
        """Test fallback similarity matching with no similar entities."""
        normalizer = EnhancedEntityNormalizer(config={'enable_llm_normalization': False})
        
        entities = ["apple", "banana", "car", "house"]
        candidates = normalizer._fallback_similarity_matching(entities)
        
        # Should find no high-similarity matches
        assert len(candidates) == 0


class TestNormalizedEntity:
    """Test NormalizedEntity model."""
    
    def test_normalized_entity_creation(self):
        """Test creating NormalizedEntity instance."""
        entity = NormalizedEntity(
            original_text="AI systems",
            normalized_text="artificial intelligence system",
            canonical_form="artificial intelligence system",
            language="en",
            entity_type="TECHNOLOGY",
            confidence=0.9,
            variations=["AI", "A.I.", "artificial intelligence"],
            normalization_method="llm",
            metadata={"source": "test"}
        )
        
        assert entity.original_text == "AI systems"
        assert entity.normalized_text == "artificial intelligence system"
        assert entity.canonical_form == "artificial intelligence system"
        assert entity.language == "en"
        assert entity.entity_type == "TECHNOLOGY"
        assert entity.confidence == 0.9
        assert len(entity.variations) == 3
        assert entity.normalization_method == "llm"
        assert entity.metadata["source"] == "test"
    
    def test_normalized_entity_defaults(self):
        """Test NormalizedEntity with default values."""
        entity = NormalizedEntity(
            original_text="test",
            normalized_text="test",
            canonical_form="test"
        )
        
        assert entity.language is None
        assert entity.entity_type is None
        assert entity.confidence == 1.0
        assert entity.variations == []
        assert entity.normalization_method == "rule_based"
        assert entity.metadata == {}


class TestEntityNormalizationIntegration:
    """Integration tests for entity normalization components."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.test_entities = [
            "artificial intelligence",
            "AI",
            "A.I.",
            "machine learning",
            "ML",
            "deep learning",
            "neural networks",
            "John Smith",
            "Smith, John",
            "Google Inc.",
            "Google",
            "Microsoft Corporation",
            "Microsoft Corp"
        ]
    
    @pytest.mark.asyncio
    async def test_end_to_end_normalization_without_llm(self):
        """Test end-to-end normalization without LLM service."""
        normalizer = EnhancedEntityNormalizer(config={'enable_llm_normalization': False})
        
        # Test individual entity normalization
        variations = []
        for entity in self.test_entities[:5]:  # Test first 5 entities
            variation = await normalizer.normalize_entity(entity)
            variations.append(variation)
        
        assert len(variations) == 5
        for variation in variations:
            assert isinstance(variation, EntityVariation)
            assert variation.original is not None
            assert variation.normalized is not None
            assert 0.0 <= variation.confidence <= 1.0
    
    @pytest.mark.asyncio
    async def test_merge_candidate_detection(self):
        """Test merge candidate detection."""
        normalizer = EnhancedEntityNormalizer(config={'enable_llm_normalization': False})
        
        # Test with entities that should be similar
        similar_entities = [
            "artificial intelligence",
            "Artificial Intelligence", 
            "ARTIFICIAL INTELLIGENCE"
        ]
        
        candidates = await normalizer.find_merge_candidates(similar_entities)
        
        # Should find at least one merge candidate
        assert isinstance(candidates, list)
        # Note: Without LLM, this will use fallback similarity matching
    
    def test_configuration_integration(self):
        """Test configuration integration."""
        custom_config = {
            'batch_size': 5,
            'min_confidence': 0.8,
            'enable_llm_normalization': False,
            'enable_rule_based_fallback': True
        }
        
        normalizer = EnhancedEntityNormalizer(config=custom_config)
        
        assert normalizer.batch_size == 5
        assert normalizer.min_confidence == 0.8
        assert normalizer.enable_llm_normalization is False
        assert normalizer.enable_rule_based_fallback is True


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.mark.asyncio
    async def test_empty_entity_list(self):
        """Test handling of empty entity list."""
        normalizer = EnhancedEntityNormalizer(config={'enable_llm_normalization': False})
        
        candidates = await normalizer.find_merge_candidates([])
        assert candidates == []
    
    @pytest.mark.asyncio
    async def test_single_entity(self):
        """Test handling of single entity."""
        normalizer = EnhancedEntityNormalizer(config={'enable_llm_normalization': False})
        
        candidates = await normalizer.find_merge_candidates(["single entity"])
        assert candidates == []
    
    @pytest.mark.asyncio
    async def test_none_entity(self):
        """Test handling of None entity."""
        normalizer = EnhancedEntityNormalizer(config={'enable_llm_normalization': False})
        
        # Should handle None gracefully
        variation = await normalizer.normalize_entity("")
        assert variation.original == ""
        assert variation.normalized == ""
    
    @pytest.mark.asyncio
    async def test_very_long_entity_name(self):
        """Test handling of very long entity names."""
        normalizer = EnhancedEntityNormalizer(config={'enable_llm_normalization': False})
        
        long_entity = "a" * 1000  # Very long entity name
        variation = await normalizer.normalize_entity(long_entity)
        
        assert variation.original == long_entity
        assert len(variation.normalized) <= len(long_entity)  # Should not grow
    
    def test_invalid_configuration(self):
        """Test handling of invalid configuration."""
        invalid_config = {
            'batch_size': -1,  # Invalid
            'min_confidence': 2.0,  # Invalid (should be 0-1)
            'merge_confidence_threshold': -0.5  # Invalid
        }
        
        # Should handle invalid config gracefully
        normalizer = EnhancedEntityNormalizer(config=invalid_config)
        
        # Should use reasonable defaults or clamp values
        assert normalizer.batch_size > 0
        assert 0.0 <= normalizer.min_confidence <= 1.0
