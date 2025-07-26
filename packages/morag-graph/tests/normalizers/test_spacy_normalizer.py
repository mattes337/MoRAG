"""Tests for SpaCy entity normalizer."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List

from morag_graph.normalizers.spacy_normalizer import SpacyNormalizer
from morag_graph.normalizers.entity_normalizer import NormalizedEntity
from morag_core.exceptions import ProcessingError


class TestSpacyNormalizer:
    """Test cases for SpaCy entity normalizer."""
    
    @pytest.fixture
    def mock_spacy_available(self):
        """Mock SpaCy availability."""
        with patch('morag_graph.normalizers.spacy_normalizer.SPACY_AVAILABLE', True):
            with patch('morag_graph.normalizers.spacy_normalizer.spacy') as mock_spacy:
                yield mock_spacy
    
    @pytest.fixture
    def sample_spacy_doc(self):
        """Create a mock SpaCy document for normalization."""
        mock_doc = Mock()
        
        # Create mock tokens
        mock_token1 = Mock()
        mock_token1.text = "companies"
        mock_token1.lemma_ = "company"
        mock_token1.pos_ = "NOUN"
        mock_token1.is_space = False
        mock_token1.is_punct = False
        
        mock_token2 = Mock()
        mock_token2.text = "running"
        mock_token2.lemma_ = "run"
        mock_token2.pos_ = "VERB"
        mock_token2.is_space = False
        mock_token2.is_punct = False
        
        mock_doc.__iter__ = Mock(return_value=iter([mock_token1, mock_token2]))
        mock_doc.__len__ = Mock(return_value=2)
        return mock_doc
    
    @pytest.fixture
    def proper_noun_doc(self):
        """Create a mock SpaCy document with proper nouns."""
        mock_doc = Mock()
        
        mock_token = Mock()
        mock_token.text = "Einstein"
        mock_token.lemma_ = "Einstein"
        mock_token.pos_ = "PROPN"
        mock_token.is_space = False
        mock_token.is_punct = False
        
        mock_doc.__iter__ = Mock(return_value=iter([mock_token]))
        mock_doc.__len__ = Mock(return_value=1)
        return mock_doc
    
    def test_init_without_spacy(self):
        """Test initialization when SpaCy is not available."""
        with patch('morag_graph.normalizers.spacy_normalizer.SPACY_AVAILABLE', False):
            with pytest.raises(ProcessingError, match="SpaCy is not available"):
                SpacyNormalizer()
    
    def test_init_with_spacy(self, mock_spacy_available):
        """Test successful initialization with SpaCy."""
        normalizer = SpacyNormalizer(
            supported_languages=['en', 'de'],
            fallback_language='en',
            preserve_proper_nouns=True
        )
        
        assert normalizer.supported_languages == ['en', 'de']
        assert normalizer.fallback_language == 'en'
        assert normalizer.preserve_proper_nouns is True
    
    def test_get_model_success(self, mock_spacy_available):
        """Test successful model loading."""
        mock_model = Mock()
        mock_spacy_available.load.return_value = mock_model
        
        normalizer = SpacyNormalizer()
        model = normalizer._get_model('en')
        
        assert model == mock_model
        assert 'en' in normalizer._loaded_models
    
    def test_get_model_failure(self, mock_spacy_available):
        """Test model loading failure."""
        mock_spacy_available.load.side_effect = OSError("Model not found")
        
        normalizer = SpacyNormalizer()
        model = normalizer._get_model('en')
        
        assert model is None
        assert 'en' not in normalizer._loaded_models
    
    @pytest.mark.asyncio
    async def test_normalize_entities_empty(self, mock_spacy_available):
        """Test normalization with empty entity list."""
        normalizer = SpacyNormalizer()
        
        result = await normalizer.normalize_entities([])
        assert result == []
    
    @pytest.mark.asyncio
    async def test_normalize_entities_success(self, mock_spacy_available, sample_spacy_doc):
        """Test successful entity normalization."""
        mock_model = Mock()
        mock_model.return_value = sample_spacy_doc
        mock_spacy_available.load.return_value = mock_model
        
        normalizer = SpacyNormalizer()
        
        entities = ["companies", "running"]
        entity_types = ["ORGANIZATION", "ACTION"]
        
        result = await normalizer.normalize_entities(
            entities,
            entity_types=entity_types,
            language='en',
            source_doc_id="test_doc"
        )
        
        assert len(result) == 2
        
        # Check first entity normalization
        normalized1 = result[0]
        assert normalized1.original_text == "companies"
        assert normalized1.normalized_text == "company"  # Should be lemmatized
        assert normalized1.language == "en"
        assert normalized1.entity_type == "ORGANIZATION"
        assert normalized1.normalization_method == "spacy_linguistic"
        assert normalized1.confidence == 0.9
    
    @pytest.mark.asyncio
    async def test_normalize_proper_nouns(self, mock_spacy_available, proper_noun_doc):
        """Test normalization of proper nouns (should be preserved)."""
        mock_model = Mock()
        mock_model.return_value = proper_noun_doc
        mock_spacy_available.load.return_value = mock_model
        
        normalizer = SpacyNormalizer(preserve_proper_nouns=True)
        
        entities = ["Einstein"]
        entity_types = ["PERSON"]
        
        result = await normalizer.normalize_entities(
            entities,
            entity_types=entity_types,
            language='en'
        )
        
        assert len(result) == 1
        normalized = result[0]
        assert normalized.original_text == "Einstein"
        assert normalized.normalized_text == "Einstein"  # Should be preserved
        assert normalized.normalization_method == "spacy_preserved"
        assert normalized.confidence == 0.95
    
    def test_should_preserve_entity_by_type(self, mock_spacy_available):
        """Test entity preservation based on type."""
        normalizer = SpacyNormalizer()
        
        # Mock document
        mock_doc = Mock()
        mock_doc.__len__ = Mock(return_value=1)
        
        # Test preservation for PERSON type
        should_preserve = normalizer._should_preserve_entity(mock_doc, "PERSON", "en")
        assert should_preserve is True
        
        # Test non-preservation for common noun
        should_preserve = normalizer._should_preserve_entity(mock_doc, "CONCEPT", "en")
        assert should_preserve is False
    
    def test_should_preserve_entity_by_pos(self, mock_spacy_available):
        """Test entity preservation based on POS tags."""
        normalizer = SpacyNormalizer()
        
        # Mock document with proper noun
        mock_token = Mock()
        mock_token.pos_ = "PROPN"
        mock_doc = Mock()
        mock_doc.__iter__ = Mock(return_value=iter([mock_token]))
        mock_doc.__len__ = Mock(return_value=1)
        
        should_preserve = normalizer._should_preserve_entity(mock_doc, None, "en")
        assert should_preserve is True
    
    def test_clean_proper_noun(self, mock_spacy_available):
        """Test proper noun cleaning."""
        normalizer = SpacyNormalizer()
        
        # Test cleaning with extra whitespace and punctuation
        cleaned = normalizer._clean_proper_noun("  John Doe  ")
        assert cleaned == "John Doe"
        
        cleaned = normalizer._clean_proper_noun("!@#John Doe$%^")
        assert cleaned == "John Doe"
    
    def test_apply_linguistic_normalization(self, mock_spacy_available, sample_spacy_doc):
        """Test linguistic normalization."""
        normalizer = SpacyNormalizer()
        
        normalized = normalizer._apply_linguistic_normalization(sample_spacy_doc, "en")
        assert "company" in normalized  # Should contain lemmatized form
        assert "run" in normalized
    
    def test_clean_lemma_english(self, mock_spacy_available):
        """Test lemma cleaning for English."""
        normalizer = SpacyNormalizer()
        
        # Test article removal
        cleaned = normalizer._clean_lemma("the company", "en")
        assert cleaned == "company"
        
        cleaned = normalizer._clean_lemma("a person", "en")
        assert cleaned == "person"
    
    def test_clean_lemma_german(self, mock_spacy_available):
        """Test lemma cleaning for German."""
        normalizer = SpacyNormalizer()
        
        # Test German article removal
        cleaned = normalizer._clean_lemma("der Unternehmen", "de")
        assert cleaned == "Unternehmen"
        
        cleaned = normalizer._clean_lemma("die Firma", "de")
        assert cleaned == "Firma"
    
    def test_clean_lemma_spanish(self, mock_spacy_available):
        """Test lemma cleaning for Spanish."""
        normalizer = SpacyNormalizer()
        
        # Test Spanish article removal
        cleaned = normalizer._clean_lemma("el doctor", "es")
        assert cleaned == "doctor"
        
        cleaned = normalizer._clean_lemma("la empresa", "es")
        assert cleaned == "empresa"
    
    def test_generate_spacy_variations(self, mock_spacy_available, sample_spacy_doc):
        """Test variation generation."""
        normalizer = SpacyNormalizer()
        
        variations = normalizer._generate_spacy_variations("Company", sample_spacy_doc, "en")
        
        assert "Company" in variations
        assert "company" in variations
        assert "COMPANY" in variations
        assert len(variations) > 1
        assert len(set(variations)) == len(variations)  # No duplicates
    
    @pytest.mark.asyncio
    async def test_normalize_with_rules_fallback(self, mock_spacy_available):
        """Test fallback to rule-based normalization."""
        # Mock model loading failure
        mock_spacy_available.load.side_effect = OSError("Model not found")
        
        normalizer = SpacyNormalizer()
        
        entities = ["the companies", "las empresas"]
        entity_types = ["ORGANIZATION", "ORGANIZATION"]
        
        result = await normalizer.normalize_entities(
            entities,
            entity_types=entity_types,
            language='en'
        )
        
        assert len(result) == 2
        
        # Should use rule-based fallback
        for normalized in result:
            assert normalized.normalization_method == "rule_based_fallback"
            assert normalized.confidence == 0.7
    
    def test_normalize_with_rules_english(self, mock_spacy_available):
        """Test rule-based normalization for English."""
        normalizer = SpacyNormalizer()
        
        result = normalizer._normalize_with_rules(
            [{"text": "the companies", "type": "ORGANIZATION"}],
            None,
            "en"
        )
        
        assert len(result) == 1
        normalized = result[0]
        assert normalized.original_text == "the companies"
        assert "companies" in normalized.normalized_text  # Article removed
        assert normalized.language == "en"
    
    def test_normalize_with_rules_spanish(self, mock_spacy_available):
        """Test rule-based normalization for Spanish."""
        normalizer = SpacyNormalizer()
        
        result = normalizer._normalize_with_rules(
            [{"text": "las empresas", "type": "ORGANIZATION"}],
            None,
            "es"
        )
        
        assert len(result) == 1
        normalized = result[0]
        assert normalized.original_text == "las empresas"
        assert "empresas" in normalized.normalized_text  # Article removed
        assert normalized.language == "es"
    
    def test_normalize_with_rules_german(self, mock_spacy_available):
        """Test rule-based normalization for German."""
        normalizer = SpacyNormalizer()
        
        result = normalizer._normalize_with_rules(
            [{"text": "die Unternehmen", "type": "ORGANIZATION"}],
            None,
            "de"
        )
        
        assert len(result) == 1
        normalized = result[0]
        assert normalized.original_text == "die Unternehmen"
        assert "Unternehmen" in normalized.normalized_text  # Article removed
        assert normalized.language == "de"
    
    @pytest.mark.asyncio
    async def test_close_cleanup(self, mock_spacy_available):
        """Test resource cleanup."""
        normalizer = SpacyNormalizer()
        
        # Mock the executor
        normalizer._executor = Mock()
        
        await normalizer.close()
        
        normalizer._executor.shutdown.assert_called_once_with(wait=True)
        assert len(normalizer._loaded_models) == 0
