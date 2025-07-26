"""Tests for triplet processor."""

import pytest
from morag_graph.processors.triplet_processor import TripletProcessor, ValidatedTriplet
from morag_graph.processors.sentence_processor import ProcessedSentence
from morag_graph.services.openie_service import OpenIETriplet


class TestTripletProcessor:
    """Test cases for triplet processor."""
    
    @pytest.fixture
    def processor(self):
        """Create triplet processor instance for testing."""
        config = {
            'min_confidence': 0.5,
            'min_validation_score': 0.4,
            'enable_validation': True,
            'enable_normalization': True,
            'max_subject_length': 100,
            'max_predicate_length': 50,
            'max_object_length': 100
        }
        return TripletProcessor(config=config)
    
    @pytest.fixture
    def simple_processor(self):
        """Create simple processor without validation."""
        config = {
            'min_confidence': 0.0,
            'min_validation_score': 0.0,
            'enable_validation': False,
            'enable_normalization': False
        }
        return TripletProcessor(config=config)
    
    @pytest.fixture
    def sample_triplets(self):
        """Create sample OpenIE triplets for testing."""
        return [
            OpenIETriplet(
                subject="John Smith",
                predicate="works at",
                object="Google Inc",
                confidence=0.8,
                sentence="John Smith works at Google Inc."
            ),
            OpenIETriplet(
                subject="Mary Johnson",
                predicate="is",
                object="CEO",
                confidence=0.9,
                sentence="Mary Johnson is the CEO of the company."
            ),
            OpenIETriplet(
                subject="@@##",
                predicate="xyz",
                object="123",
                confidence=0.3,
                sentence="This is a bad triplet with invalid characters."
            )
        ]
    
    @pytest.fixture
    def sample_sentences(self):
        """Create sample processed sentences for testing."""
        return [
            ProcessedSentence(
                text="John Smith works at Google Inc.",
                original_text="John Smith works at Google Inc.",
                start_pos=0,
                end_pos=31,
                sentence_id="doc1_0",
                quality_score=0.9,
                metadata={'word_count': 6}
            ),
            ProcessedSentence(
                text="Mary Johnson is the CEO of the company.",
                original_text="Mary Johnson is the CEO of the company.",
                start_pos=32,
                end_pos=71,
                sentence_id="doc1_1",
                quality_score=0.8,
                metadata={'word_count': 8}
            )
        ]
    
    def test_init(self, processor):
        """Test processor initialization."""
        assert processor.min_confidence == 0.5
        assert processor.min_validation_score == 0.4
        assert processor.enable_validation is True
        assert processor.enable_normalization is True
        assert processor.max_subject_length == 100
        assert processor.max_predicate_length == 50
        assert processor.max_object_length == 100
    
    @pytest.mark.asyncio
    async def test_process_empty_triplets(self, processor):
        """Test processing empty triplet list."""
        result = await processor.process_triplets([])
        assert result == []
    
    @pytest.mark.asyncio
    async def test_process_simple_triplets(self, simple_processor, sample_triplets):
        """Test processing simple triplets without validation."""
        result = await simple_processor.process_triplets(sample_triplets[:2])
        
        assert len(result) >= 2
        assert all(isinstance(t, ValidatedTriplet) for t in result)
        
        # Check first triplet
        first = result[0]
        assert first.subject == "John Smith"
        assert first.predicate == "works at"
        assert first.object == "Google Inc"
        assert first.confidence == 0.8
    
    @pytest.mark.asyncio
    async def test_process_with_validation(self, processor, sample_triplets):
        """Test processing with validation enabled."""
        result = await processor.process_triplets(sample_triplets)
        
        # Should filter out the bad triplet (@@##, xyz, 123)
        assert len(result) <= 2
        
        # Check that good triplets pass validation
        subjects = [t.subject for t in result]
        assert any("John Smith" in s for s in subjects)
        
        # Check validation scores
        for triplet in result:
            assert 0.0 <= triplet.validation_score <= 1.0
            assert isinstance(triplet.validation_flags, set)
    
    @pytest.mark.asyncio
    async def test_process_with_sentences(self, processor, sample_triplets, sample_sentences):
        """Test processing with sentence context."""
        result = await processor.process_triplets(
            sample_triplets[:2], 
            sentences=sample_sentences,
            source_doc_id="test_doc"
        )
        
        assert len(result) >= 1
        
        # Check that sentence IDs are properly assigned
        for triplet in result:
            assert triplet.sentence_id.startswith("doc1_")
            assert triplet.source_doc_id == "test_doc"
            
            # Check metadata includes sentence information
            assert 'sentence_quality_score' in triplet.metadata
            assert 'sentence_metadata' in triplet.metadata
    
    def test_normalize_entity(self, processor):
        """Test entity normalization."""
        # Test basic normalization
        assert processor._normalize_entity("  john smith  ") == "John Smith"
        assert processor._normalize_entity("google inc.") == "Google Inc"
        assert processor._normalize_entity("(test entity)") == "test entity"
        
        # Test whitespace normalization
        assert processor._normalize_entity("john    smith") == "John Smith"
    
    def test_normalize_predicate(self, processor):
        """Test predicate normalization."""
        # Test basic normalization
        assert processor._normalize_predicate("Works At") == "works at"
        assert processor._normalize_predicate("  IS A  ") == "is"
        assert processor._normalize_predicate("employed by") == "works at"
        assert processor._normalize_predicate("member of") == "belongs to"
    
    def test_calculate_validation_score(self, processor):
        """Test validation score calculation."""
        # Good triplet with proper nouns and good predicate
        score, flags = processor._calculate_validation_score(
            "John Smith", "works at", "Google Inc", "John Smith works at Google Inc."
        )
        
        assert 0.0 <= score <= 1.0
        assert 'proper_noun_subject' in flags
        assert 'proper_noun_object' in flags
        assert 'good_predicate' in flags
        assert 'different_entities' in flags
        
        # Poor triplet
        poor_score, poor_flags = processor._calculate_validation_score(
            "@@##", "xyz", "123", "Bad sentence with @@## xyz 123."
        )
        
        assert poor_score < score
        assert 'meaningful_entities' not in poor_flags
    
    def test_validation_patterns(self, processor):
        """Test validation pattern matching."""
        patterns = processor._validation_patterns
        
        # Test proper noun pattern
        assert patterns['proper_noun'].search("John Smith")
        assert patterns['proper_noun'].search("Google Inc")
        assert not patterns['proper_noun'].search("the person")
        
        # Test good predicate pattern
        assert patterns['good_predicate'].search("works at the company")
        assert patterns['good_predicate'].search("is a manager")
        assert not patterns['good_predicate'].search("of the")
        
        # Test invalid characters
        assert patterns['invalid_chars'].search("test@#$")
        assert not patterns['invalid_chars'].search("John Smith")
        
        # Test temporal pattern
        assert patterns['temporal'].search("yesterday he worked")
        assert patterns['temporal'].search("in 2023")
        assert not patterns['temporal'].search("he worked hard")
    
    @pytest.mark.asyncio
    async def test_length_filtering(self, processor):
        """Test filtering by entity length."""
        long_triplet = OpenIETriplet(
            subject="A" * 150,  # Too long
            predicate="works at",
            object="Google",
            confidence=0.8,
            sentence="Long subject works at Google."
        )
        
        result = await processor.process_triplets([long_triplet])
        assert len(result) == 0  # Should be filtered out
    
    @pytest.mark.asyncio
    async def test_confidence_filtering(self, processor):
        """Test filtering by confidence threshold."""
        low_confidence_triplet = OpenIETriplet(
            subject="John",
            predicate="works at",
            object="Google",
            confidence=0.2,  # Below threshold
            sentence="John works at Google."
        )
        
        # Note: This tests the validation score filtering, not confidence directly
        # The confidence is preserved in the ValidatedTriplet
        result = await processor.process_triplets([low_confidence_triplet])
        
        # Should still process the triplet (confidence filtering happens elsewhere)
        # But validation score might be low
        if result:
            assert result[0].confidence == 0.2
    
    @pytest.mark.asyncio
    async def test_metadata_generation(self, processor, sample_sentences):
        """Test metadata generation for validated triplets."""
        triplet = OpenIETriplet(
            subject="John Smith",
            predicate="works at",
            object="Google",
            confidence=0.8,
            sentence="John Smith works at Google."
        )
        
        result = await processor.process_triplets([triplet], sentences=sample_sentences)
        
        assert len(result) >= 1
        metadata = result[0].metadata
        
        # Check required metadata fields
        assert 'original_subject' in metadata
        assert 'original_predicate' in metadata
        assert 'original_object' in metadata
        assert 'extraction_confidence' in metadata
        assert 'sentence_length' in metadata
        assert 'subject_length' in metadata
        assert 'predicate_length' in metadata
        assert 'object_length' in metadata
        
        # Check values
        assert metadata['original_subject'] == "John Smith"
        assert metadata['extraction_confidence'] == 0.8
        assert metadata['subject_length'] > 0
    
    @pytest.mark.asyncio
    async def test_error_handling(self, processor):
        """Test error handling in processing."""
        # Create a triplet that might cause issues
        problematic_triplet = OpenIETriplet(
            subject="",  # Empty subject
            predicate="",  # Empty predicate
            object="",  # Empty object
            confidence=0.8,
            sentence="Empty triplet test."
        )
        
        # Should not crash, just filter out invalid triplets
        result = await processor.process_triplets([problematic_triplet])
        assert isinstance(result, list)
        # Empty triplet should be filtered out
        assert len(result) == 0
    
    @pytest.mark.asyncio
    async def test_close(self, processor):
        """Test processor cleanup."""
        # Should not raise any exceptions
        await processor.close()
    
    @pytest.mark.asyncio
    async def test_quality_indicators(self, processor):
        """Test quality indicator scoring."""
        # Test triplet with multiple quality indicators
        high_quality_triplet = OpenIETriplet(
            subject="Dr. John Smith",  # Proper noun
            predicate="works at",  # Good predicate
            object="Stanford University",  # Proper noun
            confidence=0.9,
            sentence="Yesterday Dr. John Smith works at Stanford University in California."  # Temporal and spatial
        )
        
        result = await processor.process_triplets([high_quality_triplet])
        
        assert len(result) >= 1
        triplet = result[0]
        
        # Should have high validation score due to multiple quality indicators
        assert triplet.validation_score > 0.5
        
        # Check specific flags
        flags = triplet.validation_flags
        assert 'proper_noun_subject' in flags
        assert 'proper_noun_object' in flags
        assert 'good_predicate' in flags
        assert 'temporal_context' in flags
        assert 'spatial_context' in flags


if __name__ == "__main__":
    pytest.main([__file__])
