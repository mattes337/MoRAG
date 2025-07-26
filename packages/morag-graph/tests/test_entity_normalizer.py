"""Tests for entity normalizer."""

import pytest
from morag_graph.normalizers.entity_normalizer import EntityNormalizer, NormalizedEntity


class TestEntityNormalizer:
    """Test cases for entity normalizer."""
    
    @pytest.fixture
    def normalizer(self):
        """Create entity normalizer instance for testing."""
        config = {
            'enable_llm_normalization': False,  # Disable for testing
            'enable_rule_based_fallback': True,
            'supported_languages': ['en', 'es', 'de'],
            'batch_size': 10,
            'min_confidence': 0.7
        }
        return EntityNormalizer(config=config)
    
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
        return EntityNormalizer(config=config)
    
    def test_init(self, normalizer):
        """Test normalizer initialization."""
        assert normalizer.enable_llm_normalization is False
        assert normalizer.enable_rule_based_fallback is True
        assert normalizer.supported_languages == ['en', 'es', 'de']
        assert normalizer.batch_size == 10
        assert normalizer.min_confidence == 0.7
    
    @pytest.mark.asyncio
    async def test_normalize_empty_entities(self, normalizer):
        """Test normalizing empty entity list."""
        result = await normalizer.normalize_entities([])
        assert result == []
    
    @pytest.mark.asyncio
    async def test_normalize_english_entities(self, normalizer):
        """Test normalizing English entities."""
        entities = [
            "companies",
            "running",
            "children",
            "the Apple Inc",
            "worked"
        ]
        
        result = await normalizer.normalize_entities(entities)
        
        assert len(result) == 5
        
        # Check specific normalizations
        normalized_texts = [entity.normalized_text for entity in result]
        assert "company" in normalized_texts  # companies -> company
        assert "child" in normalized_texts    # children -> child
        
        # Check that all results are NormalizedEntity objects
        for entity in result:
            assert isinstance(entity, NormalizedEntity)
            assert entity.original_text
            assert entity.normalized_text
            assert entity.canonical_form
            assert 0.0 <= entity.confidence <= 1.0
            assert entity.normalization_method == "rule_based"
    
    @pytest.mark.asyncio
    async def test_normalize_with_types(self, normalizer):
        """Test normalizing entities with types."""
        entities = ["companies", "John Smith", "running"]
        entity_types = ["ORGANIZATION", "PERSON", "ACTION"]
        
        result = await normalizer.normalize_entities(entities, entity_types)
        
        assert len(result) == 3
        
        # Check that types are preserved
        for i, entity in enumerate(result):
            assert entity.entity_type == entity_types[i]
    
    def test_detect_language(self, normalizer):
        """Test language detection."""
        # English (default)
        assert normalizer._detect_language("Apple Inc") == 'en'
        assert normalizer._detect_language("running companies") == 'en'
        
        # German
        assert normalizer._detect_language("die Unternehmen") == 'de'
        assert normalizer._detect_language("Müller") == 'de'
        assert normalizer._detect_language("das Haus") == 'de'
        
        # Spanish
        assert normalizer._detect_language("las empresas") == 'es'
        assert normalizer._detect_language("niño") == 'es'
        assert normalizer._detect_language("el perro") == 'es'
    
    def test_normalize_english(self, normalizer):
        """Test English-specific normalization."""
        # Plurals
        assert normalizer._normalize_english("companies") == "company"
        assert normalizer._normalize_english("boxes") == "box"
        assert normalizer._normalize_english("cities") == "city"
        
        # Irregular plurals
        assert normalizer._normalize_english("children") == "child"
        assert normalizer._normalize_english("people") == "person"
        assert normalizer._normalize_english("mice") == "mouse"
        
        # Verb forms
        assert normalizer._normalize_english("running") == "run"
        assert normalizer._normalize_english("worked") == "work"
        
        # Articles
        assert normalizer._normalize_english("the Apple Inc") == "Apple Inc"
        assert normalizer._normalize_english("a company") == "company"
        
        # No change needed
        assert normalizer._normalize_english("John Smith") == "John Smith"
    
    def test_normalize_spanish(self, normalizer):
        """Test Spanish-specific normalization."""
        # Plurals
        assert normalizer._normalize_spanish("empresas") == "empresa"
        assert normalizer._normalize_spanish("niños") == "niño"
        
        # Verb forms
        assert normalizer._normalize_spanish("corriendo") == "corr"  # Simple rule
        assert normalizer._normalize_spanish("trabajando") == "trabaj"
        
        # Articles
        assert normalizer._normalize_spanish("las empresas") == "empresa"
        assert normalizer._normalize_spanish("el perro") == "perro"
    
    def test_normalize_german(self, normalizer):
        """Test German-specific normalization."""
        # Plurals
        assert normalizer._normalize_german("Unternehmen") == "Unternehm"  # Simple rule
        assert normalizer._normalize_german("Häuser") == "Häus"
        
        # Articles
        assert normalizer._normalize_german("die Katze") == "Katze"
        assert normalizer._normalize_german("das Haus") == "Haus"
    
    def test_generate_variations(self, normalizer):
        """Test variation generation."""
        variations = normalizer._generate_variations("company", "en")
        
        # Should include original and case variations
        assert "company" in variations
        assert "Company" in variations
        assert "COMPANY" in variations
        
        # Should include English plurals
        assert "companies" in variations
        assert "companys" in variations  # Even if incorrect, for matching
        
        # Should not have duplicates
        assert len(variations) == len(set(variations))
    
    def test_calculate_rule_confidence(self, normalizer):
        """Test confidence calculation for rule-based normalization."""
        # No change - high confidence
        confidence = normalizer._calculate_rule_confidence("Apple", "Apple")
        assert confidence == 1.0
        
        # Plural removal - good change
        confidence = normalizer._calculate_rule_confidence("companies", "company")
        assert confidence > 0.8
        
        # Case change only - high confidence
        confidence = normalizer._calculate_rule_confidence("APPLE", "apple")
        assert confidence > 0.8
        
        # All confidences should be between 0 and 1
        assert 0.0 <= confidence <= 1.0
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, normalizer):
        """Test batch processing of entities."""
        # Create a list larger than batch size
        entities = [f"company{i}" for i in range(25)]
        
        result = await normalizer.normalize_entities(entities)
        
        # Should process all entities
        assert len(result) == 25
        
        # Check that all are processed
        for entity in result:
            assert entity.normalized_text
            assert entity.confidence > 0.0
    
    @pytest.mark.asyncio
    async def test_error_handling(self, normalizer):
        """Test error handling in normalization."""
        # Should not crash on problematic input
        entities = ["", "   ", "normal entity"]
        
        result = await normalizer.normalize_entities(entities)
        
        # Should return results for all inputs
        assert len(result) == 3
        
        # Check that empty entities are handled
        for entity in result:
            assert isinstance(entity, NormalizedEntity)
            assert entity.confidence >= 0.0
    
    @pytest.mark.asyncio
    async def test_metadata_generation(self, normalizer):
        """Test metadata generation."""
        entities = ["companies", "Apple Inc"]
        
        result = await normalizer.normalize_entities(entities, source_doc_id="test_doc")
        
        assert len(result) >= 1
        
        for entity in result:
            # Check metadata structure
            assert isinstance(entity.metadata, dict)
            assert 'detected_language' in entity.metadata
            assert 'changes_made' in entity.metadata
            
            # Check language detection
            assert entity.language in ['en', 'es', 'de', None]
            
            # Check variations
            assert isinstance(entity.variations, list)
            assert len(entity.variations) > 0
    
    @pytest.mark.asyncio
    async def test_close(self, normalizer):
        """Test normalizer cleanup."""
        # Should not raise any exceptions
        await normalizer.close()
    
    @pytest.mark.asyncio
    async def test_multilingual_batch(self, normalizer):
        """Test processing entities in multiple languages."""
        entities = [
            "companies",      # English
            "empresas",       # Spanish
            "Unternehmen",    # German
            "John Smith",     # Proper noun (English)
            "las casas",      # Spanish with article
            "die Katzen"      # German with article
        ]
        
        result = await normalizer.normalize_entities(entities)
        
        assert len(result) == 6
        
        # Check that different languages are detected
        languages = [entity.language for entity in result]
        assert 'en' in languages
        assert 'es' in languages or 'de' in languages  # At least one non-English
        
        # Check that normalization occurred
        normalized_texts = [entity.normalized_text for entity in result]
        
        # Should have some normalized forms
        assert any(original != normalized for original, normalized in 
                  zip(entities, normalized_texts))
    
    def test_normalization_patterns(self, normalizer):
        """Test compiled normalization patterns."""
        patterns = normalizer._normalization_patterns
        
        # Test article pattern
        assert patterns['articles'].search("the company")
        assert patterns['articles'].search("el perro")
        assert patterns['articles'].search("die Katze")
        
        # Test plural patterns
        assert patterns['en_plural_s'].search("companies")
        assert patterns['es_plural_es'].search("empresas")
        
        # Test whitespace pattern
        assert patterns['whitespace'].search("  multiple   spaces  ")


if __name__ == "__main__":
    pytest.main([__file__])
