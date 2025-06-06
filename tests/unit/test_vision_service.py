"""Unit tests for vision service."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
from PIL import Image

from morag_image.services import VisionService, vision_service
from morag_core.exceptions import ExternalServiceError

class TestVisionService:
    """Test cases for VisionService."""
    
    @pytest.fixture
    def service(self):
        """Create vision service instance."""
        return VisionService()
    
    @pytest.fixture
    def mock_image_file(self, tmp_path):
        """Create mock image file."""
        image_file = tmp_path / "test_image.jpg"
        test_image = Image.new('RGB', (100, 100), color='blue')
        test_image.save(image_file, 'JPEG')
        return image_file
    
    @pytest.mark.asyncio
    @patch('google.generativeai.GenerativeModel')
    @patch('morag.core.config.settings')
    async def test_generate_caption_success(self, mock_settings, mock_model_class, service, mock_image_file):
        """Test successful caption generation."""
        mock_settings.gemini_api_key = "test_api_key"
        
        # Mock Gemini response
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "A blue square image with vibrant colors"
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        with patch('asyncio.to_thread', return_value=mock_response):
            caption = await service.generate_caption(mock_image_file)
            
            assert caption == "A blue square image with vibrant colors"
            mock_model_class.assert_called_once_with('gemini-1.5-flash')
    
    @pytest.mark.asyncio
    @patch('morag.core.config.settings')
    async def test_generate_caption_no_api_key(self, mock_settings, service, mock_image_file):
        """Test caption generation without API key."""
        mock_settings.gemini_api_key = None
        
        with pytest.raises(ExternalServiceError, match="Gemini API key not configured"):
            await service.generate_caption(mock_image_file)
    
    @pytest.mark.asyncio
    @patch('google.generativeai.GenerativeModel')
    @patch('morag.core.config.settings')
    async def test_generate_caption_with_custom_prompt(self, mock_settings, mock_model_class, service, mock_image_file):
        """Test caption generation with custom prompt."""
        mock_settings.gemini_api_key = "test_api_key"
        
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Custom analysis result"
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        custom_prompt = "Describe this image in one word."
        
        with patch('asyncio.to_thread', return_value=mock_response):
            caption = await service.generate_caption(mock_image_file, custom_prompt=custom_prompt)
            
            assert caption == "Custom analysis result"
    
    @pytest.mark.asyncio
    @patch('google.generativeai.GenerativeModel')
    @patch('morag.core.config.settings')
    async def test_generate_caption_api_error(self, mock_settings, mock_model_class, service, mock_image_file):
        """Test caption generation with API error."""
        mock_settings.gemini_api_key = "test_api_key"
        
        mock_model_class.side_effect = Exception("API connection failed")
        
        with pytest.raises(ExternalServiceError, match="Caption generation failed"):
            await service.generate_caption(mock_image_file)
    
    @pytest.mark.asyncio
    @patch('google.generativeai.GenerativeModel')
    @patch('morag.core.config.settings')
    async def test_analyze_image_content_success(self, mock_settings, mock_model_class, service, mock_image_file):
        """Test successful image content analysis."""
        mock_settings.gemini_api_key = "test_api_key"
        
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = """
        1. OBJECTS: Blue square, geometric shape
        2. SETTING: Digital canvas, solid background
        3. ACTIVITIES: None
        4. STYLE: Minimalist, digital art
        5. TEXT: None
        6. EMOTIONS: Calm, neutral
        7. TECHNICAL: High quality, sharp edges
        """
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        with patch('asyncio.to_thread', return_value=mock_response):
            analysis = await service.analyze_image_content(mock_image_file)
            
            assert isinstance(analysis, dict)
            assert "objects" in analysis
            assert "setting" in analysis
            assert "raw_analysis" in analysis
            assert analysis["raw_analysis"] == mock_response.text.strip()
    
    @pytest.mark.asyncio
    @patch('google.generativeai.GenerativeModel')
    @patch('morag.core.config.settings')
    async def test_classify_image_type_success(self, mock_settings, mock_model_class, service, mock_image_file):
        """Test successful image type classification."""
        mock_settings.gemini_api_key = "test_api_key"
        
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "artwork"
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        with patch('asyncio.to_thread', return_value=mock_response):
            classification = await service.classify_image_type(mock_image_file)
            
            assert classification == "artwork"
    
    @pytest.mark.asyncio
    @patch('google.generativeai.GenerativeModel')
    @patch('morag.core.config.settings')
    async def test_classify_image_type_invalid_response(self, mock_settings, mock_model_class, service, mock_image_file):
        """Test image classification with invalid response."""
        mock_settings.gemini_api_key = "test_api_key"
        
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "invalid_category"
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        with patch('asyncio.to_thread', return_value=mock_response):
            classification = await service.classify_image_type(mock_image_file)
            
            # Should fallback to "other" for invalid categories
            assert classification == "other"
    
    @pytest.mark.asyncio
    @patch('google.generativeai.GenerativeModel')
    @patch('morag.core.config.settings')
    async def test_classify_image_type_error(self, mock_settings, mock_model_class, service, mock_image_file):
        """Test image classification with error."""
        mock_settings.gemini_api_key = "test_api_key"
        
        mock_model_class.side_effect = Exception("Classification failed")
        
        # Should return "other" on error, not raise exception
        classification = await service.classify_image_type(mock_image_file)
        assert classification == "other"
    
    @pytest.mark.asyncio
    @patch('google.generativeai.GenerativeModel')
    @patch('morag.core.config.settings')
    async def test_extract_text_content_success(self, mock_settings, mock_model_class, service, mock_image_file):
        """Test successful text content extraction."""
        mock_settings.gemini_api_key = "test_api_key"
        
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Hello World\nWelcome to the test"
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        with patch('asyncio.to_thread', return_value=mock_response):
            text = await service.extract_text_content(mock_image_file)
            
            assert text == "Hello World\nWelcome to the test"
    
    @pytest.mark.asyncio
    @patch('google.generativeai.GenerativeModel')
    @patch('morag.core.config.settings')
    async def test_extract_text_content_no_text(self, mock_settings, mock_model_class, service, mock_image_file):
        """Test text extraction when no text is detected."""
        mock_settings.gemini_api_key = "test_api_key"
        
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "No text detected"
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        with patch('asyncio.to_thread', return_value=mock_response):
            text = await service.extract_text_content(mock_image_file)
            
            # Should return empty string when no text detected
            assert text == ""
    
    def test_parse_content_analysis_structured(self, service):
        """Test parsing of structured content analysis."""
        analysis_text = """
        1. OBJECTS: Car, tree, building
        2. SETTING: Urban street scene
        3. ACTIVITIES: Traffic, pedestrians walking
        4. STYLE: Realistic photography
        5. TEXT: Street signs visible
        6. EMOTIONS: Busy, energetic
        7. TECHNICAL: Good lighting, sharp focus
        """
        
        result = service._parse_content_analysis(analysis_text)
        
        assert "objects" in result
        assert "Car, tree, building" in result["objects"]
        assert "Urban street scene" in result["setting"]
        assert "Traffic, pedestrians walking" in result["activities"]
        assert result["raw_analysis"] == analysis_text
    
    def test_parse_content_analysis_unstructured(self, service):
        """Test parsing of unstructured content analysis."""
        analysis_text = "This is a simple description without structure."
        
        result = service._parse_content_analysis(analysis_text)
        
        assert isinstance(result, dict)
        assert "raw_analysis" in result
        assert result["raw_analysis"] == analysis_text
        # Other fields should be empty
        assert result["objects"] == ""
        assert result["setting"] == ""
    
    def test_parse_content_analysis_error(self, service):
        """Test parsing with malformed input."""
        # This should not raise an exception
        result = service._parse_content_analysis(None)
        
        assert isinstance(result, dict)
        assert "raw_analysis" in result
    
    @pytest.mark.asyncio
    @patch('morag.core.config.settings')
    async def test_extract_text_content_no_api_key(self, mock_settings, service, mock_image_file):
        """Test text extraction without API key."""
        mock_settings.gemini_api_key = None
        
        with pytest.raises(ExternalServiceError, match="Gemini API key not configured"):
            await service.extract_text_content(mock_image_file)
    
    @pytest.mark.asyncio
    @patch('morag.core.config.settings')
    async def test_analyze_image_content_no_api_key(self, mock_settings, service, mock_image_file):
        """Test content analysis without API key."""
        mock_settings.gemini_api_key = None
        
        with pytest.raises(ExternalServiceError, match="Gemini API key not configured"):
            await service.analyze_image_content(mock_image_file)
    
    @pytest.mark.asyncio
    @patch('morag.core.config.settings')
    async def test_classify_image_type_no_api_key(self, mock_settings, service, mock_image_file):
        """Test image classification without API key."""
        mock_settings.gemini_api_key = None
        
        with pytest.raises(ExternalServiceError, match="Gemini API key not configured"):
            await service.classify_image_type(mock_image_file)

def test_global_service_instance():
    """Test that global service instance is available."""
    assert vision_service is not None
    assert isinstance(vision_service, VisionService)
