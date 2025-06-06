#!/usr/bin/env python3
"""Test script to verify API fixes for MoRAG."""

import asyncio
import json
import sys
from pathlib import Path

# Add the packages to the path
sys.path.insert(0, str(Path(__file__).parent / "packages" / "morag" / "src"))
sys.path.insert(0, str(Path(__file__).parent / "packages" / "morag-services" / "src"))
sys.path.insert(0, str(Path(__file__).parent / "packages" / "morag-core" / "src"))

from morag import MoRAGAPI
from morag_services import ContentType


async def test_content_type_detection():
    """Test content type detection for different file types."""
    print("🔍 Testing Content Type Detection")
    print("=" * 50)
    
    api = MoRAGAPI()
    
    # Test image detection
    test_cases = [
        ("test.jpg", "image"),
        ("test.png", "image"),
        ("test.gif", "image"),
        ("test.pdf", "document"),
        ("test.mp3", "audio"),
        ("test.mp4", "video"),
        ("https://example.com", "web"),
        ("https://youtube.com/watch?v=123", "youtube"),
    ]
    
    for test_input, expected in test_cases:
        if test_input.startswith("http"):
            detected = api._detect_content_type(test_input)
        else:
            detected = api._detect_content_type_from_file(Path(test_input))
        
        status = "✅" if detected == expected else "❌"
        print(f"{status} {test_input} -> {detected} (expected: {expected})")
    
    await api.cleanup()


async def test_content_type_enum():
    """Test ContentType enum validation."""
    print("\n🔧 Testing ContentType Enum")
    print("=" * 50)
    
    test_types = ["document", "audio", "video", "image", "web", "youtube"]
    
    for content_type in test_types:
        try:
            enum_value = ContentType(content_type)
            print(f"✅ {content_type} -> {enum_value}")
        except Exception as e:
            print(f"❌ {content_type} -> Error: {e}")


async def test_audio_config():
    """Test audio configuration defaults."""
    print("\n🎵 Testing Audio Configuration")
    print("=" * 50)
    
    try:
        from morag_audio.processor import AudioConfig
        
        config = AudioConfig()
        print(f"✅ Diarization enabled: {config.enable_diarization}")
        print(f"✅ Topic segmentation enabled: {config.enable_topic_segmentation}")
        print(f"✅ Model size: {config.model_size}")
        print(f"✅ Device: {config.device}")
        
    except Exception as e:
        print(f"❌ Audio config test failed: {e}")


async def test_json_conversion():
    """Test JSON conversion for audio."""
    print("\n📄 Testing JSON Conversion")
    print("=" * 50)
    
    try:
        from morag_audio.converters.audio_converter import AudioConverter
        from morag_audio.processor import AudioProcessingResult, AudioSegment
        
        # Create mock audio result
        segments = [
            AudioSegment(
                start=0.0,
                end=5.0,
                text="Hello, this is a test.",
                confidence=0.95
            ),
            AudioSegment(
                start=5.0,
                end=10.0,
                text="This is another segment.",
                confidence=0.92
            )
        ]
        
        result = AudioProcessingResult(
            transcript="Hello, this is a test. This is another segment.",
            segments=segments,
            metadata={"duration": 10.0, "language": "en"},
            file_path="test.mp3",
            processing_time=2.5,
            success=True
        )
        
        converter = AudioConverter()
        json_result = await converter.convert_to_json(result)
        
        print("✅ JSON conversion successful:")
        print(json.dumps(json_result, indent=2))
        
    except Exception as e:
        print(f"❌ JSON conversion test failed: {e}")


async def main():
    """Run all tests."""
    print("🚀 MoRAG API Fixes Test Suite")
    print("=" * 60)
    
    try:
        await test_content_type_detection()
        await test_content_type_enum()
        await test_audio_config()
        await test_json_conversion()
        
        print("\n🎉 All tests completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
