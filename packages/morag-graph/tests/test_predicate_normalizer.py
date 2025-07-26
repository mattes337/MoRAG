"""Tests for predicate normalizer."""

import pytest
from morag_graph.normalizers.predicate_normalizer import (
    PredicateNormalizer, NormalizedPredicate, RelationshipType
)


class TestPredicateNormalizer:
    """Test cases for predicate normalizer."""
    
    @pytest.fixture
    def normalizer(self):
        """Create predicate normalizer instance for testing."""
        config = {
            'enable_llm_normalization': False,  # Disable for testing
            'enable_rule_based_fallback': True,
            'supported_languages': ['en', 'es', 'de'],
            'batch_size': 10,
            'min_confidence': 0.7
        }
        return PredicateNormalizer(config=config)
    
    @pytest.fixture
    def llm_normalizer(self):
        """Create normalizer with LLM enabled for testing."""
        config = {
            'enable_llm_normalization': True,
            'enable_rule_based_fallback': True,
            'supported_languages': ['en', 'es', 'de'],
            'batch_size': 5,
            'min_confidence': 0.6
        }
        return PredicateNormalizer(config=config)
    
    def test_init(self, normalizer):
        """Test normalizer initialization."""
        assert normalizer.enable_llm_normalization is False
        assert normalizer.enable_rule_based_fallback is True
        assert normalizer.supported_languages == ['en', 'es', 'de']
        assert normalizer.batch_size == 10
        assert normalizer.min_confidence == 0.7
    
    @pytest.mark.asyncio
    async def test_normalize_empty_predicates(self, normalizer):
        """Test normalizing empty predicate list."""
        result = await normalizer.normalize_predicates([])
        assert result == []
    
    @pytest.mark.asyncio
    async def test_normalize_english_predicates(self, normalizer):
        """Test normalizing English predicates."""
        predicates = [
            "works for",
            "is a",
            "located in",
            "member of",
            "creates"
        ]
        
        result = await normalizer.normalize_predicates(predicates)
        
        assert len(result) == 5
        
        # Check specific normalizations
        normalized_forms = [pred.canonical_form for pred in result]
        assert "works_at" in normalized_forms  # works for -> works_at
        assert "is" in normalized_forms        # is a -> is
        assert "located_in" in normalized_forms # located in -> located_in
        assert "member_of" in normalized_forms  # member of -> member_of
        
        # Check that all results are NormalizedPredicate objects
        for pred in result:
            assert isinstance(pred, NormalizedPredicate)
            assert pred.original
            assert pred.normalized
            assert pred.canonical_form
            assert isinstance(pred.relationship_type, RelationshipType)
            assert 0.0 <= pred.confidence <= 1.0
    
    @pytest.mark.asyncio
    async def test_normalize_multilingual_predicates(self, normalizer):
        """Test normalizing predicates in multiple languages."""
        predicates = [
            "works at",      # English
            "trabaja en",    # Spanish
            "arbeitet bei",  # German
            "is",            # English
            "es",            # Spanish
            "ist"            # German
        ]
        
        result = await normalizer.normalize_predicates(predicates)
        
        assert len(result) == 6
        
        # Check that different languages are detected
        languages = [pred.language for pred in result]
        assert 'en' in languages
        assert 'es' in languages or 'de' in languages  # At least one non-English
        
        # Check that similar predicates are normalized to same canonical form
        canonical_forms = [pred.canonical_form for pred in result]
        
        # Should have work-related predicates normalized consistently
        work_predicates = [cf for cf in canonical_forms if 'work' in cf]
        assert len(work_predicates) >= 2  # Multiple work predicates should be normalized
    
    def test_detect_language(self, normalizer):
        """Test language detection."""
        # English (default)
        assert normalizer._detect_language("works at") == 'en'
        assert normalizer._detect_language("creates") == 'en'
        
        # German
        assert normalizer._detect_language("arbeitet bei") == 'de'
        assert normalizer._detect_language("ist ein") == 'de'
        assert normalizer._detect_language("befindet sich") == 'de'
        
        # Spanish
        assert normalizer._detect_language("trabaja en") == 'es'
        assert normalizer._detect_language("es un") == 'es'
        assert normalizer._detect_language("tiene") == 'es'
    
    def test_clean_predicate(self, normalizer):
        """Test predicate cleaning."""
        # Remove filler words
        assert normalizer._clean_predicate("is a manager") == "is manager"
        assert normalizer._clean_predicate("works at the company") == "works at company"
        
        # Normalize whitespace
        assert normalizer._clean_predicate("  works   at  ") == "works at"
        
        # Remove punctuation
        assert normalizer._clean_predicate("works, at!") == "works at"
        
        # Convert to lowercase
        assert normalizer._clean_predicate("WORKS AT") == "works at"
    
    def test_apply_generic_rules(self, normalizer):
        """Test generic normalization rules."""
        # Verb form normalization
        assert normalizer._apply_generic_rules("working") == "work"
        assert normalizer._apply_generic_rules("worked") == "work"
        assert normalizer._apply_generic_rules("creates") == "create"
        
        # Pattern-based normalization
        assert normalizer._apply_generic_rules("work at") == "works_at"
        assert normalizer._apply_generic_rules("locate in") == "located_in"
        assert normalizer._apply_generic_rules("member of") == "member_of"
        assert normalizer._apply_generic_rules("part of") == "part_of"
    
    def test_normalize_single_predicate(self, normalizer):
        """Test single predicate normalization."""
        # Exact match
        result = normalizer._normalize_single_predicate("works for", "en")
        assert result.canonical_form == "works_at"
        assert result.confidence >= 0.8
        
        # Partial match
        result = normalizer._normalize_single_predicate("working for", "en")
        assert result.confidence > 0.0
        
        # No match - generic rules
        result = normalizer._normalize_single_predicate("unknown predicate", "en")
        assert result.canonical_form == "unknown predicate"
        assert result.confidence > 0.0
    
    def test_relationship_type_mapping(self, normalizer):
        """Test relationship type mapping."""
        mappings = normalizer._relationship_mappings
        
        # Identity
        assert mappings.get("is") == RelationshipType.IDENTITY
        assert mappings.get("are") == RelationshipType.IDENTITY
        
        # Employment
        assert mappings.get("works_at") == RelationshipType.EMPLOYMENT
        assert mappings.get("employed_by") == RelationshipType.EMPLOYMENT
        
        # Location
        assert mappings.get("located_in") == RelationshipType.LOCATION
        assert mappings.get("based_in") == RelationshipType.LOCATION
        
        # Possession
        assert mappings.get("has") == RelationshipType.POSSESSION
        assert mappings.get("owns") == RelationshipType.POSSESSION
        
        # Creation
        assert mappings.get("creates") == RelationshipType.CREATION
        assert mappings.get("develops") == RelationshipType.CREATION
    
    def test_calculate_rule_confidence(self, normalizer):
        """Test confidence calculation for rule-based normalization."""
        # No change - high confidence
        confidence = normalizer._calculate_rule_confidence("works", "works")
        assert confidence == 1.0
        
        # Shortened predicate - good change
        confidence = normalizer._calculate_rule_confidence("works for", "works_at")
        assert confidence > 0.7
        
        # Standardized compound - good change
        confidence = normalizer._calculate_rule_confidence("member of", "member_of")
        assert confidence > 0.7
        
        # All confidences should be between 0 and 1
        assert 0.0 <= confidence <= 1.0
    
    def test_generate_predicate_variations(self, normalizer):
        """Test predicate variation generation."""
        variations = normalizer._generate_predicate_variations("works_at", "en")
        
        # Should include original and variations
        assert "works_at" in variations
        assert "works at" in variations  # Space version
        
        # Should include English verb forms
        assert any("work" in var for var in variations)
        
        # Should not have duplicates
        assert len(variations) == len(set(variations))
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, normalizer):
        """Test batch processing of predicates."""
        # Create a list larger than batch size
        predicates = [f"predicate{i}" for i in range(25)]
        
        result = await normalizer.normalize_predicates(predicates)
        
        # Should process all predicates
        assert len(result) == 25
        
        # Check that all are processed
        for pred in result:
            assert pred.normalized
            assert pred.confidence > 0.0
    
    @pytest.mark.asyncio
    async def test_error_handling(self, normalizer):
        """Test error handling in normalization."""
        # Should not crash on problematic input
        predicates = ["", "   ", "normal predicate"]
        
        result = await normalizer.normalize_predicates(predicates)
        
        # Should return results for all inputs
        assert len(result) == 3
        
        # Check that empty predicates are handled
        for pred in result:
            assert isinstance(pred, NormalizedPredicate)
            assert pred.confidence >= 0.0
    
    @pytest.mark.asyncio
    async def test_metadata_generation(self, normalizer):
        """Test metadata generation."""
        predicates = ["works for", "creates"]
        
        result = await normalizer.normalize_predicates(predicates, source_doc_id="test_doc")
        
        assert len(result) >= 1
        
        for pred in result:
            # Check metadata structure
            assert isinstance(pred.metadata, dict)
            assert 'method' in pred.metadata
            assert 'detected_language' in pred.metadata
            assert 'changes_made' in pred.metadata
            
            # Check language detection
            assert pred.language in ['en', 'es', 'de', None]
            
            # Check variations
            assert isinstance(pred.variations, list)
            assert len(pred.variations) > 0
            
            # Check relationship type
            assert isinstance(pred.relationship_type, RelationshipType)
    
    @pytest.mark.asyncio
    async def test_close(self, normalizer):
        """Test normalizer cleanup."""
        # Should not raise any exceptions
        await normalizer.close()
    
    def test_normalization_mappings(self, normalizer):
        """Test normalization mappings structure."""
        mappings = normalizer._normalization_mappings
        
        # Check that all supported languages have mappings
        assert 'en' in mappings
        assert 'es' in mappings
        assert 'de' in mappings
        
        # Check English mappings
        en_mappings = mappings['en']
        assert 'works for' in en_mappings
        assert 'is a' in en_mappings
        assert 'located in' in en_mappings
        
        # Check that mappings are consistent
        assert en_mappings['works for'] == 'works_at'
        assert en_mappings['is a'] == 'is'
    
    def test_relationship_types_enum(self):
        """Test relationship types enumeration."""
        # Check that all expected types exist
        assert RelationshipType.IDENTITY
        assert RelationshipType.EMPLOYMENT
        assert RelationshipType.LOCATION
        assert RelationshipType.POSSESSION
        assert RelationshipType.MEMBERSHIP
        assert RelationshipType.CREATION
        assert RelationshipType.MANAGEMENT
        assert RelationshipType.EDUCATION
        assert RelationshipType.COMMUNICATION
        assert RelationshipType.TEMPORAL
        assert RelationshipType.CAUSAL
        assert RelationshipType.ACTION
        assert RelationshipType.RELATIONSHIP
        assert RelationshipType.OTHER
        
        # Check that values are strings
        assert isinstance(RelationshipType.IDENTITY.value, str)
        assert isinstance(RelationshipType.EMPLOYMENT.value, str)


if __name__ == "__main__":
    pytest.main([__file__])
