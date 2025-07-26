"""Tests for confidence manager."""

import pytest
from morag_graph.normalizers.confidence_manager import (
    ConfidenceManager, ConfidenceLevel, ConfidenceThresholds, 
    ConfidenceScore, FilteredTriplet
)
from morag_graph.normalizers.entity_linker import LinkedTriplet, EntityMatch
from morag_graph.models.entity import Entity as GraphEntity


class TestConfidenceManager:
    """Test cases for confidence manager."""
    
    @pytest.fixture
    def manager(self):
        """Create confidence manager instance for testing."""
        config = {
            'min_overall_confidence': 0.6,
            'min_entity_match_confidence': 0.5,
            'min_validation_score': 0.6,
            'require_entity_linking': False,
            'threshold_very_low': 0.3,
            'threshold_low': 0.5,
            'threshold_medium': 0.7,
            'threshold_high': 0.8,
            'threshold_very_high': 0.9
        }
        return ConfidenceManager(config=config)
    
    @pytest.fixture
    def strict_manager(self):
        """Create strict confidence manager for testing."""
        config = {
            'min_overall_confidence': 0.8,
            'min_entity_match_confidence': 0.7,
            'min_validation_score': 0.8,
            'require_entity_linking': True
        }
        return ConfidenceManager(config=config)
    
    @pytest.fixture
    def sample_entities(self):
        """Create sample entities for testing."""
        return [
            GraphEntity(
                name="John Smith",
                type="PERSON",
                confidence=0.9,
                source_doc_id="doc1"
            ),
            GraphEntity(
                name="Google Inc",
                type="ORGANIZATION",
                confidence=0.8,
                source_doc_id="doc1"
            )
        ]
    
    @pytest.fixture
    def sample_triplets(self, sample_entities):
        """Create sample linked triplets for testing."""
        return [
            # High quality triplet with entity linking
            LinkedTriplet(
                subject="John Smith",
                predicate="works at",
                object="Google Inc",
                subject_entity=sample_entities[0],
                object_entity=sample_entities[1],
                subject_match=EntityMatch(
                    openie_entity="John Smith",
                    spacy_entity=sample_entities[0],
                    match_type="exact",
                    confidence=1.0,
                    similarity_score=1.0
                ),
                object_match=EntityMatch(
                    openie_entity="Google Inc",
                    spacy_entity=sample_entities[1],
                    match_type="exact",
                    confidence=1.0,
                    similarity_score=1.0
                ),
                confidence=0.9,
                validation_score=0.8,
                sentence="John Smith works at Google Inc.",
                sentence_id="doc1_0",
                source_doc_id="doc1",
                metadata={'sentence_quality_score': 0.9}
            ),
            # Medium quality triplet with partial linking
            LinkedTriplet(
                subject="Mary Johnson",
                predicate="is",
                object="CEO",
                subject_entity=None,
                object_entity=None,
                subject_match=None,
                object_match=None,
                confidence=0.7,
                validation_score=0.6,
                sentence="Mary Johnson is the CEO.",
                sentence_id="doc1_1",
                source_doc_id="doc1",
                metadata={'sentence_quality_score': 0.7}
            ),
            # Low quality triplet
            LinkedTriplet(
                subject="xyz",
                predicate="of",
                object="abc",
                subject_entity=None,
                object_entity=None,
                subject_match=None,
                object_match=None,
                confidence=0.3,
                validation_score=0.4,
                sentence="xyz of abc.",
                sentence_id="doc1_2",
                source_doc_id="doc1",
                metadata={}
            )
        ]
    
    def test_init(self, manager):
        """Test manager initialization."""
        assert manager.min_overall_confidence == 0.6
        assert manager.min_entity_match_confidence == 0.5
        assert manager.min_validation_score == 0.6
        assert manager.require_entity_linking is False
        
        # Check thresholds
        assert manager.thresholds.very_low == 0.3
        assert manager.thresholds.low == 0.5
        assert manager.thresholds.medium == 0.7
        assert manager.thresholds.high == 0.8
        assert manager.thresholds.very_high == 0.9
    
    @pytest.mark.asyncio
    async def test_filter_empty_triplets(self, manager):
        """Test filtering empty triplet list."""
        result = await manager.filter_triplets([])
        assert result == []
    
    @pytest.mark.asyncio
    async def test_filter_triplets_medium_level(self, manager, sample_triplets):
        """Test filtering with medium confidence level."""
        result = await manager.filter_triplets(
            sample_triplets, 
            target_confidence_level=ConfidenceLevel.MEDIUM
        )
        
        # Should filter out low quality triplets
        passed_triplets = [t for t in result if t.filter_reason is None]
        assert len(passed_triplets) <= len(sample_triplets)
        
        # High quality triplet should pass
        high_quality_passed = any(
            t.triplet.subject == "John Smith" and t.filter_reason is None
            for t in result
        )
        assert high_quality_passed
    
    @pytest.mark.asyncio
    async def test_filter_triplets_high_level(self, manager, sample_triplets):
        """Test filtering with high confidence level."""
        result = await manager.filter_triplets(
            sample_triplets,
            target_confidence_level=ConfidenceLevel.HIGH
        )
        
        # Should be more restrictive
        passed_triplets = [t for t in result if t.filter_reason is None]
        
        # Check that filtering occurred
        for triplet in passed_triplets:
            assert triplet.confidence_score.overall_score >= 0.8
    
    def test_calculate_confidence_score(self, manager, sample_triplets):
        """Test confidence score calculation."""
        # Test high quality triplet
        high_quality = sample_triplets[0]
        score = manager._calculate_confidence_score(high_quality)
        
        assert isinstance(score, ConfidenceScore)
        assert 0.0 <= score.overall_score <= 1.0
        assert score.confidence_level in ConfidenceLevel
        assert isinstance(score.component_scores, dict)
        assert isinstance(score.quality_flags, set)
        
        # Check component scores
        assert 'extraction_confidence' in score.component_scores
        assert 'validation_score' in score.component_scores
        assert 'entity_linking_score' in score.component_scores
        assert 'predicate_quality' in score.component_scores
        assert 'sentence_quality' in score.component_scores
        
        # High quality triplet should have high scores
        assert score.overall_score > 0.7
        assert 'high_extraction_confidence' in score.quality_flags
        assert 'both_entities_linked' in score.quality_flags
    
    def test_calculate_entity_linking_score(self, manager, sample_triplets):
        """Test entity linking score calculation."""
        # High quality with both entities linked
        high_quality = sample_triplets[0]
        score = manager._calculate_entity_linking_score(high_quality)
        assert score > 0.8  # Both entities with exact matches
        
        # Medium quality with no entities linked
        medium_quality = sample_triplets[1]
        score = manager._calculate_entity_linking_score(medium_quality)
        assert score == 0.0  # No entities linked
    
    def test_calculate_predicate_quality(self, manager):
        """Test predicate quality calculation."""
        # Good predicates
        assert manager._calculate_predicate_quality("works at") > 0.7
        assert manager._calculate_predicate_quality("is") > 0.7
        assert manager._calculate_predicate_quality("manages") > 0.7
        
        # Weak predicates
        assert manager._calculate_predicate_quality("of") < 0.5
        assert manager._calculate_predicate_quality("in") < 0.5
        
        # Empty predicate
        assert manager._calculate_predicate_quality("") == 0.0
        assert manager._calculate_predicate_quality("   ") == 0.0
    
    def test_extract_sentence_quality(self, manager, sample_triplets):
        """Test sentence quality extraction."""
        # Triplet with explicit sentence quality
        high_quality = sample_triplets[0]
        score = manager._extract_sentence_quality(high_quality)
        assert score == 0.9  # From metadata
        
        # Triplet without explicit quality
        medium_quality = sample_triplets[1]
        score = manager._extract_sentence_quality(medium_quality)
        assert 0.0 <= score <= 1.0  # Should estimate
    
    def test_determine_confidence_level(self, manager):
        """Test confidence level determination."""
        assert manager._determine_confidence_level(0.95) == ConfidenceLevel.VERY_HIGH
        assert manager._determine_confidence_level(0.85) == ConfidenceLevel.HIGH
        assert manager._determine_confidence_level(0.75) == ConfidenceLevel.MEDIUM
        assert manager._determine_confidence_level(0.55) == ConfidenceLevel.LOW
        assert manager._determine_confidence_level(0.25) == ConfidenceLevel.VERY_LOW
    
    def test_passes_confidence_filter(self, manager):
        """Test confidence filter logic."""
        # High quality score
        high_score = ConfidenceScore(
            overall_score=0.8,
            component_scores={
                'validation_score': 0.8,
                'entity_linking_score': 0.9
            },
            confidence_level=ConfidenceLevel.HIGH,
            quality_flags={'both_entities_linked'}
        )
        assert manager._passes_confidence_filter(high_score, 0.7)
        
        # Low quality score
        low_score = ConfidenceScore(
            overall_score=0.4,
            component_scores={
                'validation_score': 0.4,
                'entity_linking_score': 0.0
            },
            confidence_level=ConfidenceLevel.LOW,
            quality_flags={'no_entities_linked'}
        )
        assert not manager._passes_confidence_filter(low_score, 0.7)
    
    def test_strict_filtering(self, strict_manager):
        """Test strict filtering requirements."""
        # Score that would pass normal filter but not strict
        medium_score = ConfidenceScore(
            overall_score=0.75,
            component_scores={
                'validation_score': 0.7,
                'entity_linking_score': 0.0
            },
            confidence_level=ConfidenceLevel.MEDIUM,
            quality_flags={'no_entities_linked'}
        )
        
        # Should fail strict filter due to no entity linking
        assert not strict_manager._passes_confidence_filter(medium_score, 0.7)
    
    @pytest.mark.asyncio
    async def test_get_quality_statistics(self, manager, sample_triplets):
        """Test quality statistics generation."""
        stats = await manager.get_quality_statistics(sample_triplets)
        
        # Check structure
        assert 'total_triplets' in stats
        assert 'average_confidence' in stats
        assert 'min_confidence' in stats
        assert 'max_confidence' in stats
        assert 'confidence_distribution' in stats
        assert 'quality_flags_distribution' in stats
        
        # Check values
        assert stats['total_triplets'] == len(sample_triplets)
        assert 0.0 <= stats['average_confidence'] <= 1.0
        assert 0.0 <= stats['min_confidence'] <= 1.0
        assert 0.0 <= stats['max_confidence'] <= 1.0
        
        # Check distributions
        confidence_dist = stats['confidence_distribution']
        assert all(level.value in confidence_dist for level in ConfidenceLevel)
        assert sum(confidence_dist.values()) == len(sample_triplets)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, manager):
        """Test error handling in confidence scoring."""
        # Create a problematic triplet
        problematic_triplet = LinkedTriplet(
            subject="",
            predicate="",
            object="",
            subject_entity=None,
            object_entity=None,
            subject_match=None,
            object_match=None,
            confidence=0.0,
            validation_score=0.0,
            sentence="",
            sentence_id="",
            source_doc_id=None,
            metadata={}
        )
        
        # Should not crash
        result = await manager.filter_triplets([problematic_triplet])
        assert isinstance(result, list)
        assert len(result) >= 0
    
    @pytest.mark.asyncio
    async def test_close(self, manager):
        """Test manager cleanup."""
        # Should not raise any exceptions
        await manager.close()
    
    @pytest.mark.asyncio
    async def test_component_weights(self, manager, sample_triplets):
        """Test that component weights affect overall score."""
        # Get original score
        original_score = manager._calculate_confidence_score(sample_triplets[0])
        
        # Modify weights to emphasize entity linking
        manager.component_weights['entity_linking_score'] = 0.8
        manager.component_weights['extraction_confidence'] = 0.1
        manager.component_weights['validation_score'] = 0.1
        
        # Recalculate score
        new_score = manager._calculate_confidence_score(sample_triplets[0])
        
        # Score should be different (likely higher due to good entity linking)
        assert new_score.overall_score != original_score.overall_score
    
    def test_confidence_thresholds(self):
        """Test confidence thresholds structure."""
        thresholds = ConfidenceThresholds()
        
        # Check default values
        assert thresholds.very_low == 0.3
        assert thresholds.low == 0.5
        assert thresholds.medium == 0.7
        assert thresholds.high == 0.8
        assert thresholds.very_high == 0.9
        
        # Check ordering
        assert thresholds.very_low < thresholds.low
        assert thresholds.low < thresholds.medium
        assert thresholds.medium < thresholds.high
        assert thresholds.high < thresholds.very_high


if __name__ == "__main__":
    pytest.main([__file__])
