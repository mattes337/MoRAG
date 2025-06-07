#!/usr/bin/env python3
"""
Test script to verify the video converter format fix.
This script tests that the video converter produces the correct format:
- No "## Processing Details" section
- Audio transcript formatted as topic with timestamps and speaker labels
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morag_video import VideoConverter
from morag_core.interfaces.converter import ConversionOptions


async def test_video_format():
    """Test the video converter format."""
    print("🔄 Testing video converter format...")
    
    converter = VideoConverter()
    
    # Create mock video result
    video_result = Mock()
    video_result.metadata = {
        'filename': 'test_video.mp4',
        'duration': 120,
        'resolution': '1920x1080',
        'fps': 30,
        'format': 'mp4'
    }
    
    # Mock audio transcript
    video_result.audio_transcript = "Hello everyone. Welcome to this video. Today we'll discuss important topics."
    video_result.summary = "A video about important topics."
    
    # Mock keyframes (optional)
    video_result.keyframes = [
        {
            'timestamp': 10.0,
            'description': 'Opening scene',
            'objects': ['person', 'microphone'],
            'text_content': 'Welcome'
        }
    ]
    
    # Mock scenes (optional)
    video_result.scenes = [
        {
            'start_time': 0.0,
            'end_time': 60.0,
            'description': 'Introduction segment',
            'activity': 'speaking'
        }
    ]
    
    # Create conversion options
    options = ConversionOptions(
        include_metadata=True,
        extract_images=True,
        format_options={'include_timestamps': True}
    )
    
    # Generate markdown
    markdown = await converter._create_structured_markdown(video_result, options)
    
    print("📝 Generated markdown:")
    print("-" * 60)
    print(markdown)
    print("-" * 60)
    
    # Verify format
    checks = []
    
    # Check that unwanted sections are NOT present
    if "## Processing Details" not in markdown:
        checks.append("✅ No '## Processing Details' section")
    else:
        checks.append("❌ Found unwanted '## Processing Details' section")
    
    # Check that audio transcript is formatted as topic with timestamp
    if "# Audio Content [" in markdown and "]" in markdown:
        checks.append("✅ Audio content formatted as topic with timestamp")
    else:
        checks.append("❌ Audio content not properly formatted")
    
    # Check for speaker labels in transcript
    if "Speaker_00:" in markdown:
        checks.append("✅ Speaker labels added to transcript")
    else:
        checks.append("❌ Missing speaker labels in transcript")
    
    # Check basic structure
    if "# Video Analysis:" in markdown:
        checks.append("✅ Correct document header")
    else:
        checks.append("❌ Missing document header")
    
    # Check that other sections are preserved
    if "## Video Information" in markdown:
        checks.append("✅ Video information section preserved")
    else:
        checks.append("❌ Missing video information section")
    
    if "## Visual Timeline" in markdown:
        checks.append("✅ Visual timeline section preserved")
    else:
        checks.append("❌ Missing visual timeline section")
    
    print("\n🔍 Format validation:")
    for check in checks:
        print(f"  {check}")
    
    # Count failures
    failures = [c for c in checks if c.startswith("❌")]
    if failures:
        print(f"\n❌ {len(failures)} format issues found!")
        return False
    else:
        print(f"\n✅ All format checks passed!")
        return True


async def test_video_format_without_transcript():
    """Test video format when no audio transcript is available."""
    print("\n🔄 Testing video format without transcript...")
    
    converter = VideoConverter()
    
    # Create mock video result without transcript
    video_result = Mock()
    video_result.metadata = {
        'filename': 'silent_video.mp4',
        'duration': 60,
        'resolution': '1280x720',
        'fps': 24,
        'format': 'mp4'
    }
    
    # No audio transcript
    video_result.audio_transcript = None
    video_result.transcript = None
    video_result.summary = "A silent video."
    video_result.keyframes = []
    video_result.scenes = []
    
    # Create conversion options
    options = ConversionOptions(
        include_metadata=True,
        extract_images=False
    )
    
    # Generate markdown
    markdown = await converter._create_structured_markdown(video_result, options)
    
    print("📝 Generated markdown:")
    print("-" * 60)
    print(markdown)
    print("-" * 60)
    
    # Verify format
    checks = []
    
    # Check that unwanted sections are NOT present
    if "## Processing Details" not in markdown:
        checks.append("✅ No '## Processing Details' section")
    else:
        checks.append("❌ Found unwanted '## Processing Details' section")
    
    # Should not have audio content section when no transcript
    if "# Audio Content" not in markdown:
        checks.append("✅ No audio content section when no transcript")
    else:
        checks.append("❌ Unexpected audio content section")
    
    # Check basic structure
    if "# Video Analysis:" in markdown:
        checks.append("✅ Correct document header")
    else:
        checks.append("❌ Missing document header")
    
    print("\n🔍 Format validation:")
    for check in checks:
        print(f"  {check}")
    
    # Count failures
    failures = [c for c in checks if c.startswith("❌")]
    if failures:
        print(f"\n❌ {len(failures)} format issues found!")
        return False
    else:
        print(f"\n✅ All format checks passed!")
        return True


async def main():
    """Run all video format tests."""
    print("🚀 Testing Video Converter Format Fix")
    print("=" * 60)
    
    try:
        with_transcript_success = await test_video_format()
        without_transcript_success = await test_video_format_without_transcript()
        
        if with_transcript_success and without_transcript_success:
            print("\n🎉 All tests passed! Video converter format is fixed.")
            return True
        else:
            print("\n💥 Some tests failed. Format needs more work.")
            return False
            
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
