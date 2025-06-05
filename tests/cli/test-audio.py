#!/usr/bin/env python3
"""
MoRAG Audio Processing Test Script

Usage: python test-audio.py <audio_file>

Examples:
    python test-audio.py my-audio.mp3
    python test-audio.py recording.wav
    python test-audio.py video.mp4  # Extract audio from video
"""

import sys
import asyncio
import json
from pathlib import Path
from typing import Optional

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from morag_audio import AudioProcessor, AudioConfig
    from morag_core.models import ProcessingConfig
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you have installed the MoRAG packages:")
    print("  pip install -e packages/morag-core")
    print("  pip install -e packages/morag-audio")
    sys.exit(1)


def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'-'*40}")
    print(f"  {title}")
    print(f"{'-'*40}")


def print_result(key: str, value: str, indent: int = 0):
    """Print a formatted key-value result."""
    spaces = "  " * indent
    print(f"{spaces}📋 {key}: {value}")


async def test_audio_processing(audio_file: Path) -> bool:
    """Test audio processing functionality."""
    print_header("MoRAG Audio Processing Test")
    
    if not audio_file.exists():
        print(f"❌ Error: Audio file not found: {audio_file}")
        return False
    
    print_result("Input File", str(audio_file))
    print_result("File Size", f"{audio_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    try:
        # Initialize audio configuration
        config = AudioConfig(
            model_size="base",  # Use base model for faster processing
            device="auto",  # Auto-detect best available device
            enable_diarization=False,  # Disable for faster processing
            enable_topic_segmentation=False,  # Disable for faster processing
            vad_filter=True,
            word_timestamps=True,
            include_metadata=True
        )
        print_result("Audio Configuration", "✅ Created successfully")

        # Initialize audio processor
        processor = AudioProcessor(config)
        print_result("Audio Processor", "✅ Initialized successfully")

        print_section("Processing Audio File")
        print("🔄 Starting audio processing...")

        # Process the audio file
        result = await processor.process(audio_file)
        
        if result.success:
            print("✅ Audio processing completed successfully!")

            print_section("Processing Results")
            print_result("Status", "✅ Success")
            print_result("Processing Time", f"{result.processing_time:.2f} seconds")
            print_result("Transcript Length", f"{len(result.transcript)} characters")
            print_result("Segments Count", f"{len(result.segments)}")

            if result.metadata:
                print_section("Metadata")
                for key, value in result.metadata.items():
                    if isinstance(value, (dict, list)):
                        print_result(key, json.dumps(value, indent=2))
                    else:
                        print_result(key, str(value))

            if result.transcript:
                print_section("Transcript Preview")
                transcript_preview = result.transcript[:500] + "..." if len(result.transcript) > 500 else result.transcript
                print(f"📄 Transcript ({len(result.transcript)} characters):")
                print(transcript_preview)

            if result.segments:
                print_section("Segments Preview (first 3)")
                for i, segment in enumerate(result.segments[:3]):
                    print(f"  Segment {i+1}: [{segment.start:.2f}s - {segment.end:.2f}s]")
                    print(f"    Text: {segment.text[:100]}{'...' if len(segment.text) > 100 else ''}")
                    if segment.speaker:
                        print(f"    Speaker: {segment.speaker}")
                    if segment.confidence:
                        print(f"    Confidence: {segment.confidence:.3f}")

            # Save results to file
            output_file = audio_file.parent / f"{audio_file.stem}_test_result.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'success': result.success,
                    'processing_time': result.processing_time,
                    'transcript': result.transcript,
                    'segments': [
                        {
                            'start': seg.start,
                            'end': seg.end,
                            'text': seg.text,
                            'speaker': seg.speaker,
                            'confidence': seg.confidence,
                            'topic_id': seg.topic_id,
                            'topic_label': seg.topic_label
                        } for seg in result.segments
                    ],
                    'metadata': result.metadata,
                    'file_path': result.file_path,
                    'error_message': result.error_message
                }, f, indent=2, ensure_ascii=False)

            print_section("Output")
            print_result("Results saved to", str(output_file))

            return True

        else:
            print("❌ Audio processing failed!")
            print_result("Error", result.error_message or "Unknown error")
            return False
            
    except Exception as e:
        print(f"❌ Error during audio processing: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python test-audio.py <audio_file>")
        print()
        print("Examples:")
        print("  python test-audio.py my-audio.mp3")
        print("  python test-audio.py recording.wav")
        print("  python test-audio.py video.mp4  # Extract audio from video")
        sys.exit(1)
    
    audio_file = Path(sys.argv[1])
    
    try:
        success = asyncio.run(test_audio_processing(audio_file))
        if success:
            print("\n🎉 Audio processing test completed successfully!")
            sys.exit(0)
        else:
            print("\n💥 Audio processing test failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
