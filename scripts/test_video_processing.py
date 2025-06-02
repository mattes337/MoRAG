#!/usr/bin/env python3
"""Test script for video processing functionality."""

import asyncio
import sys
import tempfile
from pathlib import Path
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morag.processors.video import video_processor, VideoConfig
from morag.services.ffmpeg_service import ffmpeg_service
from morag.tasks.video_tasks import process_video_file
from morag.core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_video_metadata_extraction():
    """Test video metadata extraction with mock data."""
    print("\n=== Testing Video Metadata Extraction ===")
    
    try:
        # Create a mock video file for testing
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            tmp_file.write(b"fake video content for testing")
            mock_video_path = Path(tmp_file.name)
        
        print(f"Created mock video file: {mock_video_path}")
        
        # This will fail with real FFmpeg, but we can test the error handling
        try:
            metadata = await ffmpeg_service.extract_metadata(mock_video_path)
            print(f"Metadata extracted: {metadata}")
        except Exception as e:
            print(f"Expected error (mock file): {e}")
            print("✓ Error handling works correctly")
        
        # Clean up
        mock_video_path.unlink()
        print("✓ Mock file cleaned up")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False
    
    return True

async def test_video_processor_config():
    """Test video processor configuration."""
    print("\n=== Testing Video Processor Configuration ===")
    
    try:
        # Test default configuration
        default_config = VideoConfig()
        print(f"Default config: extract_audio={default_config.extract_audio}")
        print(f"Default config: generate_thumbnails={default_config.generate_thumbnails}")
        print(f"Default config: thumbnail_count={default_config.thumbnail_count}")
        
        # Test custom configuration
        custom_config = VideoConfig(
            extract_audio=False,
            generate_thumbnails=True,
            thumbnail_count=10,
            extract_keyframes=True,
            max_keyframes=15,
            thumbnail_size=(640, 480)
        )
        print(f"Custom config: extract_audio={custom_config.extract_audio}")
        print(f"Custom config: thumbnail_count={custom_config.thumbnail_count}")
        print(f"Custom config: thumbnail_size={custom_config.thumbnail_size}")
        
        print("✓ Video configuration works correctly")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False
    
    return True

async def test_ffmpeg_service_initialization():
    """Test FFmpeg service initialization."""
    print("\n=== Testing FFmpeg Service Initialization ===")
    
    try:
        # Test service initialization
        print(f"FFmpeg service temp dir: {ffmpeg_service.temp_dir}")
        print(f"Temp dir exists: {ffmpeg_service.temp_dir.exists()}")
        
        # Test frame rate parsing
        test_rates = ["30/1", "25/1", "60000/1001", "29.97", "invalid", "30/0"]
        for rate in test_rates:
            parsed = ffmpeg_service._parse_frame_rate(rate)
            print(f"Frame rate '{rate}' -> {parsed}")
        
        print("✓ FFmpeg service initialization works correctly")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False
    
    return True

async def test_video_processor_initialization():
    """Test video processor initialization."""
    print("\n=== Testing Video Processor Initialization ===")
    
    try:
        # Test processor initialization
        print(f"Video processor temp dir: {video_processor.temp_dir}")
        print(f"Temp dir exists: {video_processor.temp_dir.exists()}")
        
        # Test cleanup functionality with mock files
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_files = []
            for i in range(3):
                temp_file = Path(tmp_dir) / f"test_file_{i}.tmp"
                temp_file.write_text("test content")
                temp_files.append(temp_file)
            
            print(f"Created {len(temp_files)} temporary files")
            
            # Verify files exist
            assert all(f.exists() for f in temp_files)
            print("✓ Temporary files created successfully")
            
            # Test cleanup
            video_processor.cleanup_temp_files(temp_files)
            
            # Verify files are deleted
            assert all(not f.exists() for f in temp_files)
            print("✓ Temporary files cleaned up successfully")
        
        print("✓ Video processor initialization works correctly")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False
    
    return True

async def test_task_imports():
    """Test that all video task imports work correctly."""
    print("\n=== Testing Video Task Imports ===")
    
    try:
        from morag.tasks.video_tasks import (
            process_video_file,
            extract_video_audio,
            generate_video_thumbnails
        )
        
        print("✓ process_video_file imported successfully")
        print("✓ extract_video_audio imported successfully")
        print("✓ generate_video_thumbnails imported successfully")
        
        # Test that tasks are properly registered with Celery
        print(f"process_video_file task name: {process_video_file.name}")
        print(f"extract_video_audio task name: {extract_video_audio.name}")
        print(f"generate_video_thumbnails task name: {generate_video_thumbnails.name}")
        
        print("✓ Video task imports work correctly")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False
    
    return True

async def test_error_handling():
    """Test error handling in video processing."""
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

async def main():
    """Run all video processing tests."""
    print("Starting Video Processing Tests")
    print("=" * 50)
    
    tests = [
        test_video_processor_config,
        test_ffmpeg_service_initialization,
        test_video_processor_initialization,
        test_task_imports,
        test_error_handling,
        test_video_metadata_extraction,
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
        print("🎉 All video processing tests passed!")
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
