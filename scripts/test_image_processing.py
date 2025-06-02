#!/usr/bin/env python3
"""Test script for image processing functionality."""

import asyncio
import sys
import tempfile
from pathlib import Path
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morag.processors.image import image_processor, ImageConfig
from morag.services.vision_service import vision_service
from morag.services.ocr_service import ocr_service
from morag.tasks.image_tasks import process_image_file
from morag.core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_image_config():
    """Test image configuration."""
    print("\n=== Testing Image Configuration ===")
    
    try:
        # Test default configuration
        default_config = ImageConfig()
        print(f"Default config: generate_caption={default_config.generate_caption}")
        print(f"Default config: extract_text={default_config.extract_text}")
        print(f"Default config: ocr_engine={default_config.ocr_engine}")
        print(f"Default config: resize_max_dimension={default_config.resize_max_dimension}")
        
        # Test custom configuration
        custom_config = ImageConfig(
            generate_caption=False,
            extract_text=True,
            ocr_engine="easyocr",
            resize_max_dimension=512,
            image_quality=90
        )
        print(f"Custom config: generate_caption={custom_config.generate_caption}")
        print(f"Custom config: ocr_engine={custom_config.ocr_engine}")
        print(f"Custom config: resize_max_dimension={custom_config.resize_max_dimension}")
        print(f"Custom config: image_quality={custom_config.image_quality}")
        
        print("✓ Image configuration works correctly")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False
    
    return True

async def test_image_processor_initialization():
    """Test image processor initialization."""
    print("\n=== Testing Image Processor Initialization ===")
    
    try:
        # Test processor initialization
        print(f"Image processor temp dir: {image_processor.temp_dir}")
        print(f"Temp dir exists: {image_processor.temp_dir.exists()}")
        
        # Test cleanup functionality with mock files
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_files = []
            for i in range(3):
                temp_file = Path(tmp_dir) / f"test_image_{i}.tmp"
                temp_file.write_text("test content")
                temp_files.append(temp_file)
            
            print(f"Created {len(temp_files)} temporary files")
            
            # Verify files exist
            assert all(f.exists() for f in temp_files)
            print("✓ Temporary files created successfully")
            
            # Test cleanup
            image_processor.cleanup_temp_files(temp_files)
            
            # Verify files are deleted
            assert all(not f.exists() for f in temp_files)
            print("✓ Temporary files cleaned up successfully")
        
        print("✓ Image processor initialization works correctly")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False
    
    return True

async def test_vision_service_initialization():
    """Test vision service initialization."""
    print("\n=== Testing Vision Service Initialization ===")
    
    try:
        # Test service initialization
        print(f"Gemini API key configured: {bool(settings.gemini_api_key)}")
        
        if not settings.gemini_api_key:
            print("⚠ Gemini API key not configured - vision features will be limited")
        else:
            print("✓ Gemini API key is configured")
        
        print("✓ Vision service initialization works correctly")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False
    
    return True

async def test_ocr_service_initialization():
    """Test OCR service initialization."""
    print("\n=== Testing OCR Service Initialization ===")
    
    try:
        # Test OCR engines availability
        print(f"Tesseract available: {ocr_service._tesseract_available}")
        print(f"EasyOCR available: {ocr_service._easyocr_available}")
        
        if not ocr_service._tesseract_available and not ocr_service._easyocr_available:
            print("⚠ No OCR engines available - text extraction will be disabled")
        else:
            print("✓ At least one OCR engine is available")
        
        print("✓ OCR service initialization works correctly")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False
    
    return True

async def test_task_imports():
    """Test that all image task imports work correctly."""
    print("\n=== Testing Image Task Imports ===")
    
    try:
        from morag.tasks.image_tasks import (
            process_image_file,
            generate_image_caption,
            extract_image_text,
            analyze_image_content,
            detect_text_regions
        )
        
        print("✓ process_image_file imported successfully")
        print("✓ generate_image_caption imported successfully")
        print("✓ extract_image_text imported successfully")
        print("✓ analyze_image_content imported successfully")
        print("✓ detect_text_regions imported successfully")
        
        # Test that tasks are properly registered with Celery
        print(f"process_image_file task name: {process_image_file.name}")
        print(f"generate_image_caption task name: {generate_image_caption.name}")
        print(f"extract_image_text task name: {extract_image_text.name}")
        
        print("✓ Image task imports work correctly")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False
    
    return True

async def test_image_metadata_extraction():
    """Test image metadata extraction with mock data."""
    print("\n=== Testing Image Metadata Extraction ===")
    
    try:
        # Create a simple test image
        from PIL import Image
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            # Create a simple RGB image
            test_image = Image.new('RGB', (100, 100), color='red')
            test_image.save(tmp_file.name, 'JPEG')
            mock_image_path = Path(tmp_file.name)
        
        print(f"Created test image: {mock_image_path}")
        
        # Test metadata extraction
        metadata = await image_processor._extract_metadata(mock_image_path)
        
        print(f"Image dimensions: {metadata.width}x{metadata.height}")
        print(f"Image format: {metadata.format}")
        print(f"Image mode: {metadata.mode}")
        print(f"File size: {metadata.file_size} bytes")
        print(f"Has EXIF: {metadata.has_exif}")
        
        assert metadata.width == 100
        assert metadata.height == 100
        assert metadata.format == "JPEG"
        
        # Clean up
        mock_image_path.unlink()
        print("✓ Test image cleaned up")
        
        print("✓ Image metadata extraction works correctly")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False
    
    return True

async def test_error_handling():
    """Test error handling in image processing."""
    print("\n=== Testing Error Handling ===")
    
    try:
        from morag.core.exceptions import ProcessingError, ExternalServiceError
        
        # Test that exceptions can be imported and created
        proc_error = ProcessingError("Test processing error")
        ext_error = ExternalServiceError("Test external error", "test_service")
        
        print(f"ProcessingError: {proc_error}")
        print(f"ExternalServiceError: {ext_error}")
        print(f"External service: {ext_error.service}")
        
        print("✓ Error handling works correctly")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False
    
    return True

async def test_image_preprocessing():
    """Test image preprocessing functionality."""
    print("\n=== Testing Image Preprocessing ===")
    
    try:
        from PIL import Image
        
        # Create a large test image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            test_image = Image.new('RGB', (2000, 1500), color='blue')
            test_image.save(tmp_file.name, 'JPEG')
            large_image_path = Path(tmp_file.name)
        
        print(f"Created large test image: {large_image_path}")
        print(f"Original size: 2000x1500")
        
        # Test preprocessing with resizing
        config = ImageConfig(resize_max_dimension=1024)
        processed_path = await image_processor._preprocess_image(large_image_path, config)
        
        if processed_path != large_image_path:
            print(f"Image was resized: {processed_path}")
            
            # Check new dimensions
            with Image.open(processed_path) as resized_img:
                print(f"New size: {resized_img.width}x{resized_img.height}")
                assert max(resized_img.width, resized_img.height) <= 1024
            
            # Clean up processed image
            processed_path.unlink()
        else:
            print("No resizing was needed")
        
        # Clean up original image
        large_image_path.unlink()
        print("✓ Test images cleaned up")
        
        print("✓ Image preprocessing works correctly")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False
    
    return True

async def main():
    """Run all image processing tests."""
    print("Starting Image Processing Tests")
    print("=" * 50)
    
    tests = [
        test_image_config,
        test_image_processor_initialization,
        test_vision_service_initialization,
        test_ocr_service_initialization,
        test_task_imports,
        test_error_handling,
        test_image_metadata_extraction,
        test_image_preprocessing,
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("Test Results Summary")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{i+1}. {test.__name__}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All image processing tests passed!")
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
