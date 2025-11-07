"""Tests for the image processor module."""

import asyncio
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from morag_image.processor import ImageConfig, ImageProcessingResult, ImageProcessor

# Skip tests if API key is not available
pytestmark = pytest.mark.skipif(
    os.environ.get("GOOGLE_API_KEY") is None,
    reason="GOOGLE_API_KEY environment variable not set",
)


# Fixture for test image path
@pytest.fixture
def test_image_path():
    """Path to a test image file."""
    # This is a placeholder - in a real test, you would use a real test image
    return Path(__file__).parent / "resources" / "test_image.jpg"


# Fixture for image processor
@pytest.fixture
def image_processor():
    """Create an ImageProcessor instance for testing."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    return ImageProcessor(api_key=api_key)


# Test extracting metadata
def test_extract_metadata(image_processor, test_image_path, monkeypatch):
    """Test extracting metadata from an image."""
    # Mock PIL.Image.open to avoid needing a real image file
    mock_image = MagicMock()
    mock_image.width = 800
    mock_image.height = 600
    mock_image.format = "JPEG"
    mock_image.mode = "RGB"
    mock_image.info = {}
    mock_image._getexif = MagicMock(return_value=None)

    with patch("PIL.Image.open", return_value=mock_image):
        with patch("os.path.getsize", return_value=1024):
            metadata = image_processor._extract_metadata(test_image_path)

    assert metadata.width == 800
    assert metadata.height == 600
    assert metadata.format == "JPEG"
    assert metadata.mode == "RGB"
    assert metadata.file_size == 1024
    assert metadata.has_exif is False


# Test preprocessing image
def test_preprocess_image(image_processor, test_image_path, monkeypatch):
    """Test image preprocessing (resizing)."""
    # Mock PIL.Image.open to avoid needing a real image file
    mock_image = MagicMock()
    mock_image.width = 2000
    mock_image.height = 1500
    mock_image.resize.return_value = MagicMock(width=1024, height=768)

    with patch("PIL.Image.open", return_value=mock_image):
        resized_image = image_processor._preprocess_image(
            test_image_path, max_dimension=1024
        )

    # Check that resize was called with correct parameters
    mock_image.resize.assert_called_once()
    assert resized_image.width == 1024
    assert resized_image.height == 768


# Test OCR with Tesseract
@pytest.mark.asyncio
async def test_extract_text_tesseract(image_processor, test_image_path):
    """Test extracting text using Tesseract OCR."""
    # Mock pytesseract.image_to_string to avoid needing Tesseract installed
    with patch("pytesseract.image_to_string", return_value="Sample text from image"):
        with patch("PIL.Image.open"):
            text = await image_processor._extract_text_tesseract(test_image_path)

    assert text == "Sample text from image"


# Test OCR with EasyOCR
@pytest.mark.asyncio
async def test_extract_text_easyocr(image_processor, test_image_path):
    """Test extracting text using EasyOCR."""
    # Mock EasyOCR Reader
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = [
        ([0, 0, 100, 100], "Sample", 0.95),
        ([0, 100, 100, 200], "text", 0.90),
        ([100, 0, 200, 100], "from", 0.85),
        ([100, 100, 200, 200], "image", 0.80),
    ]

    with patch("easyocr.Reader", return_value=mock_reader):
        with patch("PIL.Image.open"):
            text, confidence = await image_processor._extract_text_easyocr(
                test_image_path
            )

    assert text == "Sample text from image"
    assert confidence == 0.875  # Average of confidence scores


# Test generating caption with Gemini
@pytest.mark.asyncio
async def test_generate_caption(image_processor, test_image_path):
    """Test generating image caption using Gemini."""
    # Mock Gemini API response
    mock_response = MagicMock()
    mock_response.text = "A beautiful landscape with mountains and a lake"

    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response

    with patch("google.generativeai.GenerativeModel", return_value=mock_model):
        with patch("PIL.Image.open"):
            caption = await image_processor._generate_caption(test_image_path)

    assert caption == "A beautiful landscape with mountains and a lake"


# Test full image processing
@pytest.mark.asyncio
async def test_process_image(image_processor, test_image_path):
    """Test the full image processing pipeline."""
    # Mock all the individual processing functions
    with patch.object(
        image_processor, "_extract_metadata"
    ) as mock_extract_metadata, patch.object(
        image_processor, "_preprocess_image"
    ) as mock_preprocess, patch.object(
        image_processor, "_generate_caption"
    ) as mock_generate_caption, patch.object(
        image_processor, "_extract_text_tesseract"
    ) as mock_extract_text:
        # Set up mock returns
        mock_metadata = MagicMock()
        mock_metadata.width = 800
        mock_metadata.height = 600
        mock_extract_metadata.return_value = mock_metadata

        mock_preprocess.return_value = MagicMock()
        mock_generate_caption.return_value = "Test caption"
        mock_extract_text.return_value = "Test extracted text"

        # Create test config
        config = ImageConfig(
            generate_caption=True,
            extract_text=True,
            extract_metadata=True,
            ocr_engine="tesseract",
        )

        # Process image
        result = await image_processor.process_image(test_image_path, config)

    # Check result
    assert isinstance(result, ImageProcessingResult)
    assert result.caption == "Test caption"
    assert result.extracted_text == "Test extracted text"
    assert result.metadata == mock_metadata
    assert result.processing_time > 0
