#!/usr/bin/env python3
"""
Test script to verify the audio converter format fix.
This script tests that the audio converter produces the correct format:
- No "## Speakers" section
- No "## Transcript" section  
- No "## Processing Details" section
- Topics have timestamps in headers
- Content uses speaker-labeled dialogue format
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morag_audio import AudioConverter
from morag_core.interfaces.converter import ConversionOptions


async def test_enhanced_format():
    """Test the enhanced audio format with speaker diarization."""
    print("🔄 Testing enhanced audio format...")
    
    converter = AudioConverter()
    
    # Create mock enhanced result
    enhanced_result = Mock()
    enhanced_result.transcript = "Hello there. How are you today? I'm doing well, thank you."
    enhanced_result.summary = "A brief conversation between two people."
    enhanced_result.metadata = {
        'filename': 'test_audio.wav',
        'duration': 180,
        'language': 'en',
        'model_used': 'whisper-large-v3',
        'num_speakers': 2,
        'num_topics': 2,
        'diarization_used': True,
        'topic_segmentation_used': True
    }
    
    # Mock speakers
    enhanced_result.speakers = [
        {'id': 'SPEAKER_00', 'total_speaking_time': 90, 'segments_count': 3},
        {'id': 'SPEAKER_01', 'total_speaking_time': 90, 'segments_count': 2}
    ]
    
    # Mock segments with timing
    enhanced_result.segments = [
        Mock(text="Hello there.", start_time=0.0, end_time=2.0),
        Mock(text="How are you today?", start_time=2.5, end_time=5.0),
        Mock(text="I'm doing well, thank you.", start_time=5.5, end_time=8.0)
    ]
    
    # Mock speaker segments
    enhanced_result.speaker_segments = [
        {'speaker': 'SPEAKER_00', 'start_time': 0.0, 'end_time': 2.0},
        {'speaker': 'SPEAKER_01', 'start_time': 2.5, 'end_time': 5.0},
        {'speaker': 'SPEAKER_00', 'start_time': 5.5, 'end_time': 8.0}
    ]
    
    # Mock topics
    enhanced_result.topics = [
        {
            'topic': 'Greeting',
            'sentences': ['Hello there.', 'How are you today?'],
            'start_sentence': 0
        },
        {
            'topic': 'Response',
            'sentences': ["I'm doing well, thank you."],
            'start_sentence': 2
        }
    ]
    
    # Create conversion options
    options = ConversionOptions(
        include_metadata=True,
        format_options={
            'include_speaker_info': True,
            'include_topic_info': True,
            'include_timestamps': True
        }
    )
    
    # Generate markdown
    markdown = await converter._create_enhanced_structured_markdown(enhanced_result, options)
    
    print("📝 Generated markdown:")
    print("-" * 60)
    print(markdown)
    print("-" * 60)
    
    # Verify format
    checks = []
    
    # Check that unwanted sections are NOT present
    if "## Speakers" not in markdown:
        checks.append("✅ No '## Speakers' section")
    else:
        checks.append("❌ Found unwanted '## Speakers' section")
    
    if "## Transcript" not in markdown:
        checks.append("✅ No '## Transcript' section")
    else:
        checks.append("❌ Found unwanted '## Transcript' section")
    
    if "## Processing Details" not in markdown:
        checks.append("✅ No '## Processing Details' section")
    else:
        checks.append("❌ Found unwanted '## Processing Details' section")
    
    # Check that topics have timestamps
    if "# Greeting [" in markdown and "]" in markdown:
        checks.append("✅ Topics have timestamps in headers")
    else:
        checks.append("❌ Topics missing timestamps in headers")
    
    # Check for speaker-labeled dialogue
    if "SPEAKER_00:" in markdown or "Speaker_00:" in markdown:
        checks.append("✅ Speaker-labeled dialogue format")
    else:
        checks.append("❌ Missing speaker-labeled dialogue")
    
    # Check basic structure
    if "# Audio Transcription:" in markdown:
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


async def test_basic_format():
    """Test the basic audio format without speaker diarization."""
    print("\n🔄 Testing basic audio format...")
    
    converter = AudioConverter()
    
    # Create mock basic result
    audio_result = Mock()
    audio_result.transcript = "This is a simple transcript without speaker diarization."
    audio_result.summary = None
    audio_result.metadata = {
        'filename': 'simple_audio.wav',
        'duration': 60,
        'language': 'en',
        'model_used': 'whisper-base'
    }
    
    # Mock segments
    audio_result.segments = [
        Mock(text="This is a simple transcript without speaker diarization.", start_time=0.0, end_time=5.0)
    ]
    
    # Create conversion options
    options = ConversionOptions(
        include_metadata=True,
        format_options={'include_timestamps': True}
    )
    
    # Generate markdown
    markdown = await converter._create_structured_markdown(audio_result, options)
    
    print("📝 Generated markdown:")
    print("-" * 60)
    print(markdown)
    print("-" * 60)
    
    # Verify format
    checks = []
    
    # Check that unwanted sections are NOT present
    if "## Transcript" not in markdown:
        checks.append("✅ No '## Transcript' section")
    else:
        checks.append("❌ Found unwanted '## Transcript' section")
    
    if "## Processing Details" not in markdown:
        checks.append("✅ No '## Processing Details' section")
    else:
        checks.append("❌ Found unwanted '## Processing Details' section")
    
    # Check for topic format
    if "# Main Content [" in markdown:
        checks.append("✅ Content formatted as topic with timestamp")
    else:
        checks.append("❌ Missing topic format")
    
    # Check for speaker labels
    if "Speaker_00:" in markdown:
        checks.append("✅ Speaker labels added")
    else:
        checks.append("❌ Missing speaker labels")
    
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
    """Run all format tests."""
    print("🚀 Testing Audio Converter Format Fix")
    print("=" * 60)
    
    try:
        enhanced_success = await test_enhanced_format()
        basic_success = await test_basic_format()
        
        if enhanced_success and basic_success:
            print("\n🎉 All tests passed! Audio converter format is fixed.")
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
