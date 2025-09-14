"""Tests for the image service module."""

import os
import pytest
from pathlib import Path
import asyncio
from unittest.mock import patch, MagicMock

from morag_image.service import ImageService
from morag_image.processor import ImageConfig, ImageProcessingResult, ImageMetadata
from morag_core.exceptions import ProcessingError

# Skip tests if API key is not available
pytestmark = pytest.mark.skipif(
    os.environ.get("GOOGLE_API_KEY") is None,
    reason="GOOGLE_API_KEY environment variable not set"
)

# Fixture for test image path
@pytest.fixture
def test_image_path():
    """Path to a test image file."""
    # This is a placeholder - in a real test, you would use a real test image
    return Path(__file__).parent / "resources" / "test_image.jpg"

# Fixture for image service
@pytest.fixture
def image_service():
    """Create an ImageService instance for testing."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    return ImageService(api_key=api_key)

# Fixture for mock processing result
@pytest.fixture
def mock_processing_result():
    """Create a mock ImageProcessingResult for testing."""
    metadata = ImageMetadata(
        width=800,
        height=600,
        format="JPEG",
        mode="RGB",
        file_size=1024,
        has_exif=False,
        exif_data={},
        creation_time=None,
        camera_make=None,
        camera_model=None
    )
    
    return ImageProcessingResult(
        caption="A test image caption",
        extracted_text="Sample text from image",
        metadata=metadata,
        processing_time=0.5,
        confidence_scores={"ocr": 0.9}
    )

# Test processing a single image
@pytest.mark.asyncio
async def test_process_image(image_service, test_image_path, mock_processing_result):
    """Test processing a single image."""
    # Mock the processor's process_image method
    with patch.object(image_service.processor, "process_image", 
                     return_value=mock_processing_result):
        
        # Process image with default config
        result = await image_service.process_image(test_image_path)
    
    # Check result
    assert isinstance(result, dict)
    assert result["caption"] == "A test image caption"
    assert result["extracted_text"] == "Sample text from image"
    assert result["metadata"]["width"] == 800
    assert result["metadata"]["height"] == 600
    assert result["processing_time"] == 0.5
    assert result["confidence_scores"]["ocr"] == 0.9

# Test processing a single image with custom config
@pytest.mark.asyncio
async def test_process_image_with_config(image_service, test_image_path, mock_processing_result):
    """Test processing a single image with custom configuration."""
    # Mock the processor's process_image method
    with patch.object(image_service.processor, "process_image", 
                     return_value=mock_processing_result) as mock_process:
        
        # Process image with custom config
        config = {
            "generate_caption": True,
            "extract_text": False,
            "ocr_engine": "easyocr",
            "resize_max_dimension": 800
        }
        
        result = await image_service.process_image(test_image_path, config)
    
    # Check that processor was called with correct config
    args, kwargs = mock_process.call_args
    assert args[0] == test_image_path
    assert isinstance(args[1], ImageConfig)
    assert args[1].generate_caption is True
    assert args[1].extract_text is False
    assert args[1].ocr_engine == "easyocr"
    assert args[1].resize_max_dimension == 800

# Test processing multiple images
@pytest.mark.asyncio
async def test_process_batch(image_service, mock_processing_result):
    """Test processing multiple images."""
    # Create test image paths
    test_paths = [
        Path("/path/to/image1.jpg"),
        Path("/path/to/image2.jpg"),
        Path("/path/to/image3.jpg")
    ]
    
    # Mock the processor's process_images method
    with patch.object(image_service.processor, "process_images", 
                     return_value=[mock_processing_result] * 3):
        
        # Process images
        results = await image_service.process_batch(test_paths)
    
    # Check results
    assert isinstance(results, list)
    assert len(results) == 3
    
    for result in results:
        assert isinstance(result, dict)
        assert result["caption"] == "A test image caption"
        assert result["extracted_text"] == "Sample text from image"

# Test error handling
@pytest.mark.asyncio
async def test_error_handling(image_service, test_image_path):
    """Test error handling in the service."""
    # Mock the processor's process_image method to raise an exception
    with patch.object(image_service.processor, "process_image", 
                     side_effect=Exception("Test error")):
        
        # Process image should raise ProcessingError
        with pytest.raises(ProcessingError) as excinfo:
            await image_service.process_image(test_image_path)
        
        # Check error message
        assert "Image processing failed: Test error" in str(excinfo.value)

# Test result conversion
def test_result_to_dict(image_service, mock_processing_result):
    """Test conversion of ImageProcessingResult to dictionary."""
    # Convert result to dictionary
    result_dict = image_service._result_to_dict(mock_processing_result)
    
    # Check dictionary structure
    assert isinstance(result_dict, dict)
    assert "caption" in result_dict
    assert "extracted_text" in result_dict
    assert "metadata" in result_dict
    assert "processing_time" in result_dict
    assert "confidence_scores" in result_dict
    
    # Check metadata conversion
    metadata = result_dict["metadata"]
    assert metadata["width"] == 800
    assert metadata["height"] == 600
    assert metadata["format"] == "JPEG"
    assert metadata["mode"] == "RGB"
    assert metadata["file_size"] == 1024
    assert metadata["has_exif"] is False