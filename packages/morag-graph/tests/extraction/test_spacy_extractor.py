"""Tests for SpaCy entity extractor."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List

from morag_graph.extraction.spacy_extractor import SpacyEntityExtractor
from morag_graph.models import Entity
from morag_core.exceptions import ProcessingError


class TestSpacyEntityExtractor:
    """Test cases for SpaCy entity extractor."""
    
    @pytest.fixture
    def mock_spacy_available(self):
        """Mock SpaCy availability."""
        with patch('morag_graph.extraction.spacy_extractor.SPACY_AVAILABLE', True):
            with patch('morag_graph.extraction.spacy_extractor.spacy') as mock_spacy:
                mock_spacy.util.is_package.return_value = True
                yield mock_spacy
    
    @pytest.fixture
    def mock_langdetect_available(self):
        """Mock langdetect availability."""
        with patch('morag_graph.extraction.spacy_extractor.LANGDETECT_AVAILABLE', True):
            with patch('morag_graph.extraction.spacy_extractor.detect') as mock_detect:
                mock_detect.return_value = 'en'
                yield mock_detect
    
    @pytest.fixture
    def sample_spacy_doc(self):
        """Create a mock SpaCy document with entities."""
        mock_doc = Mock()
        
        # Create mock entities
        mock_ent1 = Mock()
        mock_ent1.text = "John Doe"
        mock_ent1.label_ = "PERSON"
        mock_ent1.start_char = 0
        mock_ent1.end_char = 8
        mock_ent1.start = 0
        mock_ent1.end = 2
        
        mock_ent2 = Mock()
        mock_ent2.text = "Microsoft"
        mock_ent2.label_ = "ORG"
        mock_ent2.start_char = 20
        mock_ent2.end_char = 29
        mock_ent2.start = 4
        mock_ent2.end = 5
        
        mock_doc.ents = [mock_ent1, mock_ent2]
        return mock_doc
    
    def test_init_without_spacy(self):
        """Test initialization when SpaCy is not available."""
        with patch('morag_graph.extraction.spacy_extractor.SPACY_AVAILABLE', False):
            with pytest.raises(ProcessingError, match="SpaCy is not available"):
                SpacyEntityExtractor()
    
    def test_init_with_spacy(self, mock_spacy_available):
        """Test successful initialization with SpaCy."""
        extractor = SpacyEntityExtractor(
            min_confidence=0.7,
            supported_languages=['en', 'de'],
            fallback_language='en'
        )
        
        assert extractor.min_confidence == 0.7
        assert extractor.supported_languages == ['en', 'de']
        assert extractor.fallback_language == 'en'
        assert extractor.enable_language_detection is True
    
    def test_language_detection(self, mock_spacy_available, mock_langdetect_available):
        """Test language detection functionality."""
        extractor = SpacyEntityExtractor()
        
        # Test English detection
        mock_langdetect_available.return_value = 'en'
        lang = extractor._detect_language("Hello world")
        assert lang == 'en'
        
        # Test fallback for unsupported language
        mock_langdetect_available.return_value = 'fr'
        lang = extractor._detect_language("Bonjour monde")
        assert lang == 'en'  # Should fallback to English
    
    def test_get_model_success(self, mock_spacy_available):
        """Test successful model loading."""
        mock_model = Mock()
        mock_spacy_available.load.return_value = mock_model
        
        extractor = SpacyEntityExtractor()
        model = extractor._get_model('en')
        
        assert model == mock_model
        assert 'en' in extractor._loaded_models
        mock_spacy_available.load.assert_called_with('en_core_web_lg')
    
    def test_get_model_failure(self, mock_spacy_available):
        """Test model loading failure."""
        mock_spacy_available.load.side_effect = OSError("Model not found")
        
        extractor = SpacyEntityExtractor()
        model = extractor._get_model('en')
        
        assert model is None
        assert 'en' not in extractor._loaded_models
    
    def test_confidence_calculation(self, mock_spacy_available):
        """Test confidence score calculation."""
        extractor = SpacyEntityExtractor()
        
        # Mock entity and document
        mock_ent = Mock()
        mock_ent.text = "John Doe"
        mock_ent.label_ = "PERSON"
        mock_doc = Mock()
        
        confidence = extractor._calculate_confidence(mock_ent, mock_doc)
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be reasonably high for PERSON entity
    
    @pytest.mark.asyncio
    async def test_extract_empty_text(self, mock_spacy_available):
        """Test extraction with empty text."""
        extractor = SpacyEntityExtractor()
        
        entities = await extractor.extract("")
        assert entities == []
        
        entities = await extractor.extract("   ")
        assert entities == []
    
    @pytest.mark.asyncio
    async def test_extract_success(self, mock_spacy_available, mock_langdetect_available, sample_spacy_doc):
        """Test successful entity extraction."""
        mock_model = Mock()
        mock_model.return_value = sample_spacy_doc
        mock_spacy_available.load.return_value = mock_model
        
        extractor = SpacyEntityExtractor()
        
        entities = await extractor.extract(
            "John Doe works at Microsoft",
            source_doc_id="test_doc"
        )
        
        assert len(entities) == 2
        
        # Check first entity (John Doe)
        person_entity = entities[0]
        assert person_entity.name == "John Doe"
        assert person_entity.type == "PERSON"
        assert person_entity.source_doc_id == "test_doc"
        assert person_entity.confidence >= 0.6
        assert person_entity.attributes['extraction_method'] == 'spacy'
        assert person_entity.attributes['spacy_label'] == 'PERSON'
        
        # Check second entity (Microsoft)
        org_entity = entities[1]
        assert org_entity.name == "Microsoft"
        assert org_entity.type == "ORGANIZATION"
        assert org_entity.source_doc_id == "test_doc"
    
    @pytest.mark.asyncio
    async def test_extract_with_context(self, mock_spacy_available, mock_langdetect_available, sample_spacy_doc):
        """Test extraction with context."""
        mock_model = Mock()
        mock_model.return_value = sample_spacy_doc
        mock_spacy_available.load.return_value = mock_model
        
        extractor = SpacyEntityExtractor()
        
        entities = await extractor.extract_with_context(
            text="John Doe works at Microsoft",
            context="This is about a software engineer.",
            source_doc_id="test_doc"
        )
        
        assert len(entities) == 2
        # Entities should have adjusted positions since context was prepended
        for entity in entities:
            assert entity.attributes.get('start_pos', 0) >= 0
    
    def test_get_available_languages(self, mock_spacy_available):
        """Test getting available languages."""
        mock_spacy_available.util.is_package.side_effect = lambda x: x in ['en_core_web_lg', 'de_core_news_lg']
        
        extractor = SpacyEntityExtractor(supported_languages=['en', 'de', 'es'])
        available = extractor.get_available_languages()
        
        assert 'en' in available
        assert 'de' in available
        assert 'es' not in available
    
    def test_get_model_info(self, mock_spacy_available):
        """Test getting model information."""
        mock_model = Mock()
        mock_model.meta = {'name': 'en_core_web_lg', 'version': '3.7.0', 'lang': 'en'}
        mock_spacy_available.load.return_value = mock_model
        mock_spacy_available.util.is_package.return_value = True
        
        extractor = SpacyEntityExtractor()
        extractor._get_model('en')  # Load the model
        
        info = extractor.get_model_info()
        
        assert info['available_languages'] == ['en', 'de', 'es']
        assert 'en' in info['loaded_models']
        assert info['min_confidence'] == 0.6
        assert 'en_model_info' in info
    
    @pytest.mark.asyncio
    async def test_extract_processing_error(self, mock_spacy_available, mock_langdetect_available):
        """Test handling of processing errors."""
        mock_spacy_available.load.side_effect = Exception("Unexpected error")
        
        extractor = SpacyEntityExtractor()
        
        with pytest.raises(ProcessingError, match="SpaCy entity extraction failed"):
            await extractor.extract("Some text")
    
    @pytest.mark.asyncio
    async def test_close_cleanup(self, mock_spacy_available):
        """Test resource cleanup."""
        extractor = SpacyEntityExtractor()
        
        # Mock the executor
        extractor._executor = Mock()
        
        await extractor.close()
        
        extractor._executor.shutdown.assert_called_once_with(wait=True)
        assert len(extractor._loaded_models) == 0
    
    def test_entity_type_mapping(self, mock_spacy_available):
        """Test entity type mapping from SpaCy to our schema."""
        extractor = SpacyEntityExtractor()
        
        # Test various SpaCy entity types
        assert extractor.ENTITY_TYPE_MAPPING['PERSON'] == 'PERSON'
        assert extractor.ENTITY_TYPE_MAPPING['ORG'] == 'ORGANIZATION'
        assert extractor.ENTITY_TYPE_MAPPING['GPE'] == 'LOCATION'
        assert extractor.ENTITY_TYPE_MAPPING['MONEY'] == 'MONETARY'
        assert extractor.ENTITY_TYPE_MAPPING['DATE'] == 'TEMPORAL'
    
    def test_language_model_mapping(self, mock_spacy_available):
        """Test language to model mapping."""
        extractor = SpacyEntityExtractor()
        
        assert extractor.LANGUAGE_MODELS['en'] == 'en_core_web_lg'
        assert extractor.LANGUAGE_MODELS['de'] == 'de_core_news_lg'
        assert extractor.LANGUAGE_MODELS['es'] == 'es_core_news_lg'
