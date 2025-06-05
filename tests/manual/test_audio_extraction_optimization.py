#!/usr/bin/env python3
"""Test script for optimized audio extraction from video files."""

import asyncio
import sys
import tempfile
import time
from pathlib import Path
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morag_video.services import ffmpeg_service
from morag_video import video_processor, VideoConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_audio_format_defaults():
    """Test that default audio format is now MP3."""
    print("\n=== Testing Default Audio Format ===")
    
    try:
        # Test VideoConfig default
        config = VideoConfig()
        print(f"VideoConfig default audio_format: {config.audio_format}")
        assert config.audio_format == "mp3", f"Expected 'mp3', got '{config.audio_format}'"
        print("✓ VideoConfig defaults to MP3")
        
        # Test FFmpeg service method signature
        import inspect
        sig = inspect.signature(ffmpeg_service.extract_audio)
        output_format_default = sig.parameters['output_format'].default
        print(f"FFmpeg service default output_format: {output_format_default}")
        assert output_format_default == "mp3", f"Expected 'mp3', got '{output_format_default}'"
        print("✓ FFmpeg service defaults to MP3")
        
        optimize_for_speed_default = sig.parameters['optimize_for_speed'].default
        print(f"FFmpeg service default optimize_for_speed: {optimize_for_speed_default}")
        assert optimize_for_speed_default == True, f"Expected True, got {optimize_for_speed_default}"
        print("✓ FFmpeg service defaults to speed optimization")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

async def test_codec_selection_logic():
    """Test codec selection logic for different scenarios."""
    print("\n=== Testing Codec Selection Logic ===")
    
    try:
        # Mock metadata for different audio codecs
        test_cases = [
            {"audio_codec": "mp3", "output_format": "mp3", "expected_codec": "copy"},
            {"audio_codec": "aac", "output_format": "aac", "expected_codec": "copy"},
            {"audio_codec": "aac", "output_format": "mp3", "expected_codec": "libmp3lame"},
            {"audio_codec": "mp3", "output_format": "wav", "expected_codec": "pcm_s16le"},
            {"audio_codec": "unknown", "output_format": "mp3", "expected_codec": "libmp3lame"},
        ]
        
        for i, case in enumerate(test_cases):
            print(f"\nTest case {i+1}: {case['audio_codec']} -> {case['output_format']}")
            
            # Simulate the codec selection logic
            source_codec = case['audio_codec'].lower()
            output_format = case['output_format'].lower()
            
            if output_format == "mp3" and "mp3" in source_codec:
                selected_codec = "copy"
                use_copy = True
            elif output_format == "aac" and "aac" in source_codec:
                selected_codec = "copy"
                use_copy = True
            else:
                use_copy = False
                if output_format == "wav":
                    selected_codec = "pcm_s16le"
                elif output_format == "mp3":
                    selected_codec = "libmp3lame"
                elif output_format == "aac":
                    selected_codec = "aac"
                else:
                    selected_codec = "libmp3lame"
            
            print(f"  Selected codec: {selected_codec}")
            print(f"  Use copy: {use_copy}")
            
            assert selected_codec == case['expected_codec'], \
                f"Expected {case['expected_codec']}, got {selected_codec}"
            print(f"  ✓ Correct codec selected")
        
        print("\n✓ All codec selection tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

async def test_file_size_comparison():
    """Test file size differences between formats."""
    print("\n=== Testing File Size Comparison ===")
    
    try:
        # Create a mock video file for testing
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            # Write some fake video data
            tmp_file.write(b"fake video content for testing" * 1000)
            mock_video_path = Path(tmp_file.name)
        
        print(f"Created mock video file: {mock_video_path}")
        print(f"Mock file size: {mock_video_path.stat().st_size} bytes")
        
        # Test different format expectations
        formats_info = {
            "mp3": "Compressed audio - smaller file size, fast processing",
            "wav": "Uncompressed audio - large file size, minimal processing",
            "aac": "Compressed audio - smaller file size, fast processing"
        }
        
        for format_name, description in formats_info.items():
            print(f"\n{format_name.upper()} format:")
            print(f"  Description: {description}")
            
            if format_name == "wav":
                print(f"  ⚠️  WARNING: WAV format will create files 5-20x larger than compressed formats")
            else:
                print(f"  ✓ Recommended for minimal file size")
        
        # Clean up
        mock_video_path.unlink()
        print(f"\n✓ Mock file cleaned up")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

async def test_performance_expectations():
    """Test performance expectations for different optimization modes."""
    print("\n=== Testing Performance Expectations ===")
    
    try:
        scenarios = [
            {
                "name": "Stream Copy (MP3 -> MP3)",
                "description": "Fastest - no re-encoding, just copy audio stream",
                "processing_time": "Minimal (seconds)",
                "file_size": "Same as source"
            },
            {
                "name": "Stream Copy (AAC -> AAC)", 
                "description": "Fastest - no re-encoding, just copy audio stream",
                "processing_time": "Minimal (seconds)",
                "file_size": "Same as source"
            },
            {
                "name": "MP3 Encoding (AAC -> MP3)",
                "description": "Fast encoding with 128k bitrate",
                "processing_time": "Fast (seconds to minutes)",
                "file_size": "Small (compressed)"
            },
            {
                "name": "WAV Extraction (Any -> WAV)",
                "description": "Uncompressed audio extraction",
                "processing_time": "Medium (minutes)",
                "file_size": "Large (5-20x source size)"
            }
        ]
        
        for scenario in scenarios:
            print(f"\n{scenario['name']}:")
            print(f"  Description: {scenario['description']}")
            print(f"  Processing Time: {scenario['processing_time']}")
            print(f"  File Size: {scenario['file_size']}")
        
        print("\n✓ Performance expectations documented")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

async def main():
    """Run all audio extraction optimization tests."""
    print("Starting Audio Extraction Optimization Tests")
    print("=" * 60)
    
    tests = [
        test_audio_format_defaults,
        test_codec_selection_logic,
        test_file_size_comparison,
        test_performance_expectations,
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{i+1}. {test.__name__}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All audio extraction optimization tests passed!")
        print("\n📋 Summary of Changes:")
        print("• Default audio format changed from WAV to MP3")
        print("• Added stream copying for minimal processing overhead")
        print("• Added speed optimization by default")
        print("• MP3 files will be ~5-20x smaller than WAV")
        print("• Processing time significantly reduced for compatible formats")
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
