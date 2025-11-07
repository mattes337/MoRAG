"""Unit tests for image processor."""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
from PIL import Image

from morag_image import ImageProcessor, ImageConfig, ImageMetadata, ImageProcessingResult
from morag_core.exceptions import ProcessingError

class TestImageProcessor:
    """Test cases for ImageProcessor."""

    @pytest.fixture
    def image_processor(self):
        """Create image processor instance."""
        return ImageProcessor()

    @pytest.fixture
    def mock_image_file(self, tmp_path):
        """Create mock image file."""
        image_file = tmp_path / "test_image.jpg"
        # Create a simple test image
        test_image = Image.new('RGB', (100, 100), color='red')
        test_image.save(image_file, 'JPEG')
        return image_file

    @pytest.fixture
    def image_config(self):
        """Create image configuration."""
        return ImageConfig(
            generate_caption=True,
            extract_text=True,
            extract_metadata=True,
            resize_max_dimension=512
        )

    @pytest.fixture
    def mock_metadata(self):
        """Create mock image metadata."""
        return ImageMetadata(
            width=100,
            height=100,
            format="JPEG",
            mode="RGB",
            file_size=1000,
            has_exif=False,
            exif_data={},
            creation_time=None,
            camera_make=None,
            camera_model=None
        )

    @pytest.mark.asyncio
    async def test_process_image_success(self, image_processor, mock_image_file, image_config, mock_metadata):
        """Test successful image processing."""
        with patch.object(image_processor, '_extract_metadata', return_value=mock_metadata), \
             patch.object(image_processor, '_preprocess_image', return_value=mock_image_file), \
             patch.object(image_processor, '_generate_caption', return_value=("Test caption", 0.9)), \
             patch.object(image_processor, '_extract_text', return_value=("Test text", 0.8)):

            result = await image_processor.process_image(mock_image_file, image_config)

            assert isinstance(result, ImageProcessingResult)
            assert result.metadata == mock_metadata
            assert result.caption == "Test caption"
            assert result.extracted_text == "Test text"
            assert result.confidence_scores["caption"] == 0.9
            assert result.confidence_scores["ocr"] == 0.8
            assert result.processing_time > 0

    @pytest.mark.asyncio
    async def test_process_image_file_not_found(self, image_processor, image_config):
        """Test image processing with non-existent file."""
        non_existent_file = Path("/non/existent/file.jpg")

        with pytest.raises(ProcessingError, match="Image file not found"):
            await image_processor.process_image(non_existent_file, image_config)

    @pytest.mark.asyncio
    async def test_extract_metadata_success(self, image_processor, mock_image_file):
        """Test successful metadata extraction."""
        metadata = await image_processor._extract_metadata(mock_image_file)

        assert metadata.width == 100
        assert metadata.height == 100
        assert metadata.format == "JPEG"
        assert metadata.mode == "RGB"
        assert metadata.file_size > 0
        assert metadata.has_exif is False

    @pytest.mark.asyncio
    async def test_extract_metadata_with_exif(self, image_processor, tmp_path):
        """Test metadata extraction with EXIF data."""
        # Create image with EXIF data
        image_file = tmp_path / "test_with_exif.jpg"
        test_image = Image.new('RGB', (200, 150), color='blue')

        # Add some fake EXIF data
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
        test_image.save(image_file, 'JPEG')

        metadata = await image_processor._extract_metadata(image_file)

        assert metadata.width == 200
        assert metadata.height == 150
        assert metadata.format == "JPEG"

    @pytest.mark.asyncio
    async def test_preprocess_image_no_resize_needed(self, image_processor, mock_image_file):
        """Test image preprocessing when no resize is needed."""
        config = ImageConfig(resize_max_dimension=1024)  # Larger than test image

        result_path = await image_processor._preprocess_image(mock_image_file, config)

        # Should return original file since no resize needed
        assert result_path == mock_image_file

    @pytest.mark.asyncio
    async def test_preprocess_image_with_resize(self, image_processor, tmp_path):
        """Test image preprocessing with resizing."""
        # Create large image
        large_image_file = tmp_path / "large_image.jpg"
        large_image = Image.new('RGB', (2000, 1500), color='green')
        large_image.save(large_image_file, 'JPEG')

        config = ImageConfig(resize_max_dimension=1024)

        result_path = await image_processor._preprocess_image(large_image_file, config)

        # Should return different path (resized image)
        assert result_path != large_image_file
        assert result_path.exists()

        # Check new dimensions
        with Image.open(result_path) as resized_img:
            assert max(resized_img.width, resized_img.height) <= 1024

        # Clean up
        result_path.unlink()

    @pytest.mark.asyncio
    @patch('google.generativeai.GenerativeModel')
    async def test_generate_caption_success(self, mock_model_class, image_processor, mock_image_file):
        """Test successful caption generation."""
        # Mock Gemini response
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "A red square image"
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model

        with patch('asyncio.to_thread', return_value=mock_response):
            # Create a config with the new model
            from morag_image.processor import ImageConfig
            config = ImageConfig()
            caption, confidence = await image_processor._generate_caption(mock_image_file, config)

            assert caption == "A red square image"
            assert confidence > 0

    @pytest.mark.asyncio
    async def test_generate_caption_failure(self, image_processor, mock_image_file):
        """Test caption generation failure."""
        with patch('google.generativeai.GenerativeModel', side_effect=Exception("API error")):
            # Create a config with the new model
            from morag_image.processor import ImageConfig
            config = ImageConfig()
            caption, confidence = await image_processor._generate_caption(mock_image_file, config)

            assert caption == ""
            assert confidence == 0.0

    @pytest.mark.asyncio
    async def test_extract_text_tesseract(self, image_processor, mock_image_file):
        """Test text extraction with Tesseract."""
        with patch.object(image_processor, '_extract_text_tesseract', return_value=("Extracted text", 0.85)):
            text, confidence = await image_processor._extract_text(mock_image_file, "tesseract")

            assert text == "Extracted text"
            assert confidence == 0.85

    @pytest.mark.asyncio
    async def test_extract_text_easyocr(self, image_processor, mock_image_file):
        """Test text extraction with EasyOCR."""
        with patch.object(image_processor, '_extract_text_easyocr', return_value=("EasyOCR text", 0.92)):
            text, confidence = await image_processor._extract_text(mock_image_file, "easyocr")

            assert text == "EasyOCR text"
            assert confidence == 0.92

    @pytest.mark.asyncio
    async def test_extract_text_unknown_engine(self, image_processor, mock_image_file):
        """Test text extraction with unknown OCR engine."""
        with patch.object(image_processor, '_extract_text_tesseract', return_value=("Fallback text", 0.7)):
            text, confidence = await image_processor._extract_text(mock_image_file, "unknown_engine")

            # Should fallback to tesseract
            assert text == "Fallback text"
            assert confidence == 0.7

    @pytest.mark.asyncio
    @patch('pytesseract.image_to_data')
    async def test_extract_text_tesseract_success(self, mock_tesseract, image_processor, mock_image_file):
        """Test successful Tesseract text extraction."""
        # Mock Tesseract response
        mock_tesseract.return_value = {
            'text': ['', 'Hello', 'World', ''],
            'conf': [0, 85, 90, 0]
        }

        with patch('asyncio.to_thread', return_value=mock_tesseract.return_value):
            text, confidence = await image_processor._extract_text_tesseract(mock_image_file)

            assert "Hello World" in text
            assert confidence > 0

    @pytest.mark.asyncio
    async def test_extract_text_tesseract_not_available(self, image_processor, mock_image_file):
        """Test Tesseract text extraction when not available."""
        with patch('pytesseract.image_to_data', side_effect=ImportError("pytesseract not available")):
            text, confidence = await image_processor._extract_text_tesseract(mock_image_file)

            assert text == ""
            assert confidence == 0.0

    @pytest.mark.asyncio
    @patch('easyocr.Reader')
    async def test_extract_text_easyocr_success(self, mock_reader_class, image_processor, mock_image_file):
        """Test successful EasyOCR text extraction."""
        # Mock EasyOCR response
        mock_reader = Mock()
        mock_reader.readtext.return_value = [
            ([[0, 0], [100, 0], [100, 50], [0, 50]], 'Hello', 0.95),
            ([[0, 60], [100, 60], [100, 100], [0, 100]], 'World', 0.88)
        ]
        mock_reader_class.return_value = mock_reader

        with patch('asyncio.to_thread', return_value=mock_reader.readtext.return_value):
            text, confidence = await image_processor._extract_text_easyocr(mock_image_file)

            assert "Hello World" in text
            assert confidence > 0

    @pytest.mark.asyncio
    async def test_extract_text_easyocr_not_available(self, image_processor, mock_image_file):
        """Test EasyOCR text extraction when not available."""
        with patch('easyocr.Reader', side_effect=ImportError("easyocr not available")):
            text, confidence = await image_processor._extract_text_easyocr(mock_image_file)

            assert text == ""
            assert confidence == 0.0

    def test_cleanup_temp_files(self, image_processor, tmp_path):
        """Test temporary file cleanup."""
        # Create temporary files
        temp_files = []
        for i in range(3):
            temp_file = tmp_path / f"temp_{i}.txt"
            temp_file.write_text("test content")
            temp_files.append(temp_file)

        # Verify files exist
        assert all(f.exists() for f in temp_files)

        # Clean up
        image_processor.cleanup_temp_files(temp_files)

        # Verify files are deleted
        assert all(not f.exists() for f in temp_files)

    def test_cleanup_temp_files_missing_file(self, image_processor, tmp_path):
        """Test cleanup with missing files."""
        non_existent_file = tmp_path / "non_existent.txt"

        # Should not raise exception
        image_processor.cleanup_temp_files([non_existent_file])

    @pytest.mark.asyncio
    async def test_process_image_no_caption(self, image_processor, mock_image_file, mock_metadata):
        """Test image processing without caption generation."""
        config = ImageConfig(generate_caption=False, extract_text=True)

        with patch.object(image_processor, '_extract_metadata', return_value=mock_metadata), \
             patch.object(image_processor, '_preprocess_image', return_value=mock_image_file), \
             patch.object(image_processor, '_extract_text', return_value=("Test text", 0.8)):

            result = await image_processor.process_image(mock_image_file, config)

            assert result.caption is None
            assert result.extracted_text == "Test text"

    @pytest.mark.asyncio
    async def test_process_image_no_text_extraction(self, image_processor, mock_image_file, mock_metadata):
        """Test image processing without text extraction."""
        config = ImageConfig(generate_caption=True, extract_text=False)

        with patch.object(image_processor, '_extract_metadata', return_value=mock_metadata), \
             patch.object(image_processor, '_preprocess_image', return_value=mock_image_file), \
             patch.object(image_processor, '_generate_caption', return_value=("Test caption", 0.9)):

            result = await image_processor.process_image(mock_image_file, config)

            assert result.caption == "Test caption"
            assert result.extracted_text is None
