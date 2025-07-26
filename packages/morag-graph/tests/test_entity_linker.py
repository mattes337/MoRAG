"""Tests for entity linker."""

import pytest
from morag_graph.normalizers.entity_linker import EntityLinker, EntityMatch, LinkedTriplet
from morag_graph.processors.triplet_processor import ValidatedTriplet
from morag_graph.models.entity import Entity as GraphEntity


class TestEntityLinker:
    """Test cases for entity linker."""
    
    @pytest.fixture
    def linker(self):
        """Create entity linker instance for testing."""
        config = {
            'min_match_confidence': 0.6,
            'enable_fuzzy_matching': True,
            'fuzzy_threshold': 0.8,
            'max_edit_distance': 2,
            'enable_type_filtering': True
        }
        return EntityLinker(config=config)
    
    @pytest.fixture
    def simple_linker(self):
        """Create simple linker without fuzzy matching."""
        config = {
            'min_match_confidence': 0.5,
            'enable_fuzzy_matching': False,
            'enable_type_filtering': False
        }
        return EntityLinker(config=config)
    
    @pytest.fixture
    def sample_spacy_entities(self):
        """Create sample spaCy entities for testing."""
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
            ),
            GraphEntity(
                name="Microsoft Corporation",
                type="ORGANIZATION",
                confidence=0.85,
                source_doc_id="doc1"
            ),
            GraphEntity(
                name="New York",
                type="LOCATION",
                confidence=0.9,
                source_doc_id="doc1"
            )
        ]
    
    @pytest.fixture
    def sample_triplets(self):
        """Create sample validated triplets for testing."""
        return [
            ValidatedTriplet(
                subject="John Smith",
                predicate="works at",
                object="Google Inc",
                confidence=0.8,
                sentence="John Smith works at Google Inc.",
                sentence_id="doc1_0",
                validation_score=0.9,
                validation_flags={'proper_noun_subject', 'proper_noun_object'},
                source_doc_id="doc1"
            ),
            ValidatedTriplet(
                subject="Microsoft Corp",  # Slight variation
                predicate="is located in",
                object="New York City",  # Slight variation
                confidence=0.7,
                sentence="Microsoft Corp is located in New York City.",
                sentence_id="doc1_1",
                validation_score=0.8,
                validation_flags={'proper_noun_subject', 'proper_noun_object'},
                source_doc_id="doc1"
            )
        ]
    
    def test_init(self, linker):
        """Test linker initialization."""
        assert linker.min_match_confidence == 0.6
        assert linker.enable_fuzzy_matching is True
        assert linker.fuzzy_threshold == 0.8
        assert linker.max_edit_distance == 2
        assert linker.enable_type_filtering is True
    
    @pytest.mark.asyncio
    async def test_link_empty_triplets(self, linker, sample_spacy_entities):
        """Test linking empty triplet list."""
        result = await linker.link_triplets([], sample_spacy_entities)
        assert result == []
    
    @pytest.mark.asyncio
    async def test_exact_matching(self, simple_linker, sample_spacy_entities, sample_triplets):
        """Test exact entity matching."""
        # Use the first triplet which has exact matches
        result = await simple_linker.link_triplets([sample_triplets[0]], sample_spacy_entities)
        
        assert len(result) == 1
        linked_triplet = result[0]
        
        # Check that entities were linked
        assert linked_triplet.subject_entity is not None
        assert linked_triplet.object_entity is not None
        assert linked_triplet.subject_entity.name == "John Smith"
        assert linked_triplet.object_entity.name == "Google Inc"
        
        # Check match information
        assert linked_triplet.subject_match.match_type == 'exact'
        assert linked_triplet.object_match.match_type == 'exact'
        assert linked_triplet.subject_match.confidence == 1.0
        assert linked_triplet.object_match.confidence == 1.0
    
    @pytest.mark.asyncio
    async def test_fuzzy_matching(self, linker, sample_spacy_entities, sample_triplets):
        """Test fuzzy entity matching."""
        # Use the second triplet which has variations
        result = await linker.link_triplets([sample_triplets[1]], sample_spacy_entities)
        
        assert len(result) == 1
        linked_triplet = result[0]
        
        # Check that subject was linked (Microsoft Corp -> Microsoft Corporation)
        assert linked_triplet.subject_entity is not None
        assert linked_triplet.subject_entity.name == "Microsoft Corporation"
        
        # Check match type
        if linked_triplet.subject_match:
            assert linked_triplet.subject_match.match_type in ['fuzzy', 'normalized']
            assert linked_triplet.subject_match.confidence >= 0.6
    
    def test_normalize_entity_text(self, linker):
        """Test entity text normalization."""
        # Test basic normalization
        assert linker._normalize_entity_text("  John Smith  ") == "john smith"
        assert linker._normalize_entity_text("The Google Inc.") == "google"
        assert linker._normalize_entity_text("Microsoft Corporation") == "microsoft"
        
        # Test punctuation removal
        assert linker._normalize_entity_text("Smith, John") == "smith john"
        assert linker._normalize_entity_text("U.S.A.") == "usa"
        
        # Test prefix/suffix removal
        assert linker._normalize_entity_text("The New York Times") == "new york times"
        assert linker._normalize_entity_text("Apple Inc") == "apple"
    
    def test_generate_entity_variations(self, linker):
        """Test entity variation generation."""
        variations = linker._generate_entity_variations("Google Inc")
        
        # Should include original and variations
        assert "Google Inc" in variations
        assert "Google Inc".title() in variations
        assert "Google Inc".upper() in variations
        
        # Should include version without suffix
        assert any("Google" in var for var in variations)
        
        # Should not have duplicates
        assert len(variations) == len(set(variations))
    
    def test_levenshtein_distance(self, linker):
        """Test Levenshtein distance calculation."""
        # Identical strings
        assert linker._levenshtein_distance("test", "test") == 0
        
        # Single character difference
        assert linker._levenshtein_distance("test", "best") == 1
        
        # Multiple differences
        assert linker._levenshtein_distance("kitten", "sitting") == 3
        
        # Empty strings
        assert linker._levenshtein_distance("", "") == 0
        assert linker._levenshtein_distance("test", "") == 4
    
    def test_levenshtein_ratio(self, linker):
        """Test Levenshtein ratio calculation."""
        # Identical strings
        assert linker._levenshtein_ratio("test", "test") == 1.0
        
        # Similar strings
        ratio = linker._levenshtein_ratio("Google Inc", "Google Corp")
        assert 0.5 < ratio < 1.0
        
        # Very different strings
        ratio = linker._levenshtein_ratio("Apple", "Microsoft")
        assert ratio < 0.5
        
        # Empty strings
        assert linker._levenshtein_ratio("", "") == 1.0
        assert linker._levenshtein_ratio("test", "") == 0.0
    
    def test_calculate_similarity(self, linker):
        """Test similarity calculation."""
        # Identical strings
        assert linker._calculate_similarity("test", "test") == 1.0
        
        # Similar strings
        similarity = linker._calculate_similarity("Microsoft Corp", "Microsoft Corporation")
        assert similarity > 0.7
        
        # Different strings
        similarity = linker._calculate_similarity("Apple", "Google")
        assert similarity < 0.5
        
        # Empty strings
        assert linker._calculate_similarity("", "") == 1.0
        assert linker._calculate_similarity("test", "") == 0.0
    
    @pytest.mark.asyncio
    async def test_build_entity_lookup(self, linker, sample_spacy_entities):
        """Test entity lookup structure building."""
        lookup = await linker._build_entity_lookup(sample_spacy_entities)
        
        # Check structure
        assert 'exact_matches' in lookup
        assert 'normalized_matches' in lookup
        assert 'type_groups' in lookup
        assert 'all_entities' in lookup
        
        # Check exact matches
        assert 'john smith' in lookup['exact_matches']
        assert 'google inc' in lookup['exact_matches']
        
        # Check type groups
        assert 'PERSON' in lookup['type_groups']
        assert 'ORGANIZATION' in lookup['type_groups']
        assert len(lookup['type_groups']['ORGANIZATION']) == 2  # Google Inc and Microsoft Corporation
    
    @pytest.mark.asyncio
    async def test_find_entity_match(self, linker, sample_spacy_entities):
        """Test finding entity matches."""
        lookup = await linker._build_entity_lookup(sample_spacy_entities)
        
        # Test exact match
        match = linker._find_entity_match("John Smith", lookup)
        assert match is not None
        assert match.match_type == 'exact'
        assert match.confidence == 1.0
        assert match.spacy_entity.name == "John Smith"
        
        # Test normalized match
        match = linker._find_entity_match("The Google Inc", lookup)
        if match:  # Might be found through normalization
            assert match.match_type in ['exact', 'normalized']
            assert match.spacy_entity.name == "Google Inc"
        
        # Test no match
        match = linker._find_entity_match("Unknown Entity", lookup)
        # Might be None or a fuzzy match depending on configuration
        if match:
            assert match.confidence < 1.0
    
    @pytest.mark.asyncio
    async def test_metadata_generation(self, linker, sample_spacy_entities, sample_triplets):
        """Test metadata generation in linked triplets."""
        result = await linker.link_triplets([sample_triplets[0]], sample_spacy_entities)
        
        assert len(result) >= 1
        linked_triplet = result[0]
        metadata = linked_triplet.metadata
        
        # Check required metadata fields
        assert 'subject_linked' in metadata
        assert 'object_linked' in metadata
        assert 'subject_match_type' in metadata
        assert 'object_match_type' in metadata
        assert 'subject_match_confidence' in metadata
        assert 'object_match_confidence' in metadata
        
        # Check values for successful linking
        if linked_triplet.subject_entity:
            assert metadata['subject_linked'] is True
            assert metadata['subject_match_confidence'] > 0.0
        
        if linked_triplet.object_entity:
            assert metadata['object_linked'] is True
            assert metadata['object_match_confidence'] > 0.0
    
    @pytest.mark.asyncio
    async def test_no_matches(self, linker, sample_spacy_entities):
        """Test handling of triplets with no entity matches."""
        no_match_triplet = ValidatedTriplet(
            subject="Unknown Person",
            predicate="works at",
            object="Unknown Company",
            confidence=0.8,
            sentence="Unknown Person works at Unknown Company.",
            sentence_id="doc1_2",
            validation_score=0.7,
            validation_flags=set(),
            source_doc_id="doc1"
        )
        
        result = await linker.link_triplets([no_match_triplet], sample_spacy_entities)
        
        assert len(result) == 1
        linked_triplet = result[0]
        
        # Should still create a linked triplet even without matches
        assert linked_triplet.subject == "Unknown Person"
        assert linked_triplet.object == "Unknown Company"
        # Entities might be None if no matches found
        assert linked_triplet.metadata['subject_linked'] == (linked_triplet.subject_entity is not None)
        assert linked_triplet.metadata['object_linked'] == (linked_triplet.object_entity is not None)
    
    @pytest.mark.asyncio
    async def test_close(self, linker):
        """Test linker cleanup."""
        # Should not raise any exceptions
        await linker.close()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, linker, sample_spacy_entities):
        """Test error handling in linking."""
        # Create a problematic triplet
        problematic_triplet = ValidatedTriplet(
            subject="",  # Empty subject
            predicate="works at",
            object="Google",
            confidence=0.8,
            sentence="Empty subject works at Google.",
            sentence_id="doc1_3",
            validation_score=0.7,
            validation_flags=set(),
            source_doc_id="doc1"
        )
        
        # Should not crash
        result = await linker.link_triplets([problematic_triplet], sample_spacy_entities)
        assert isinstance(result, list)


if __name__ == "__main__":
    pytest.main([__file__])
