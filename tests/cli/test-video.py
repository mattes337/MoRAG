#!/usr/bin/env python3
"""
MoRAG Video Processing Test Script

Usage: python test-video.py <video_file>

Examples:
    python test-video.py my-video.mp4
    python test-video.py recording.avi
    python test-video.py presentation.mov
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
    from morag_video import VideoProcessor, VideoConfig
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you have installed the MoRAG packages:")
    print("  pip install -e packages/morag-core")
    print("  pip install -e packages/morag-video")
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


async def test_video_processing(video_file: Path) -> bool:
    """Test video processing functionality."""
    print_header("MoRAG Video Processing Test")
    
    if not video_file.exists():
        print(f"❌ Error: Video file not found: {video_file}")
        return False
    
    print_result("Input File", str(video_file))
    print_result("File Size", f"{video_file.stat().st_size / 1024 / 1024:.2f} MB")
    print_result("File Extension", video_file.suffix.lower())
    
    try:
        # Initialize video configuration
        config = VideoConfig(
            extract_audio=True,
            generate_thumbnails=True,
            thumbnail_count=3,  # Fewer thumbnails for faster processing
            extract_keyframes=False,  # Disable for faster processing
            enable_enhanced_audio=True,
            enable_speaker_diarization=False,  # Disable for faster processing
            enable_topic_segmentation=False,  # Disable for faster processing
            audio_model_size="base",  # Use base model for faster processing
            enable_ocr=False  # Disable for faster processing
        )
        print_result("Video Configuration", "✅ Created successfully")

        # Initialize video processor
        processor = VideoProcessor(config)
        print_result("Video Processor", "✅ Initialized successfully")

        print_section("Processing Video File")
        print("🔄 Starting video processing...")
        print("   This may take a while for large videos...")

        # Process the video file
        result = await processor.process_video(video_file)

        print("✅ Video processing completed successfully!")

        print_section("Processing Results")
        print_result("Status", "✅ Success")
        print_result("Processing Time", f"{result.processing_time:.2f} seconds")

        print_section("Video Metadata")
        metadata = result.metadata
        print_result("Duration", f"{metadata.duration:.2f} seconds")
        print_result("Resolution", f"{metadata.width}x{metadata.height}")
        print_result("FPS", f"{metadata.fps:.2f}")
        print_result("Codec", metadata.codec)
        print_result("Format", metadata.format)
        print_result("Has Audio", "✅ Yes" if metadata.has_audio else "❌ No")
        print_result("File Size", f"{metadata.file_size / 1024 / 1024:.2f} MB")

        if result.audio_path:
            print_section("Audio Processing")
            print_result("Audio Extracted", "✅ Yes")
            print_result("Audio Path", str(result.audio_path))

            if result.audio_processing_result:
                audio_result = result.audio_processing_result
                print_result("Transcript Length", f"{len(audio_result.transcript)} characters")
                print_result("Segments Count", f"{len(audio_result.segments)}")

                if audio_result.transcript:
                    print_section("Transcript Preview")
                    transcript_preview = audio_result.transcript[:500] + "..." if len(audio_result.transcript) > 500 else audio_result.transcript
                    print(f"📄 Transcript ({len(audio_result.transcript)} characters):")
                    print(transcript_preview)

        if result.thumbnails:
            print_section("Thumbnails")
            print_result("Thumbnails Generated", f"{len(result.thumbnails)}")
            for i, thumb in enumerate(result.thumbnails):
                print_result(f"Thumbnail {i+1}", str(thumb))

        if result.keyframes:
            print_section("Keyframes")
            print_result("Keyframes Generated", f"{len(result.keyframes)}")
            for i, frame in enumerate(result.keyframes):
                print_result(f"Keyframe {i+1}", str(frame))

        if result.ocr_results:
            print_section("OCR Results")
            print_result("OCR Performed", "✅ Yes")
            print_result("OCR Data", json.dumps(result.ocr_results, indent=2))

        # Save results to file
        output_file = video_file.parent / f"{video_file.stem}_test_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'processing_time': result.processing_time,
                'metadata': {
                    'duration': metadata.duration,
                    'width': metadata.width,
                    'height': metadata.height,
                    'fps': metadata.fps,
                    'codec': metadata.codec,
                    'format': metadata.format,
                    'has_audio': metadata.has_audio,
                    'file_size': metadata.file_size
                },
                'audio_path': str(result.audio_path) if result.audio_path else None,
                'thumbnails': [str(t) for t in result.thumbnails],
                'keyframes': [str(k) for k in result.keyframes],
                'audio_processing_result': {
                    'transcript': result.audio_processing_result.transcript if result.audio_processing_result else None,
                    'segments_count': len(result.audio_processing_result.segments) if result.audio_processing_result else 0
                } if result.audio_processing_result else None,
                'ocr_results': result.ocr_results,
                'temp_files': [str(f) for f in result.temp_files]
            }, f, indent=2, ensure_ascii=False)

        print_section("Output")
        print_result("Results saved to", str(output_file))

        return True
            
    except Exception as e:
        print(f"❌ Error during video processing: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python test-video.py <video_file>")
        print()
        print("Examples:")
        print("  python test-video.py my-video.mp4")
        print("  python test-video.py recording.avi")
        print("  python test-video.py presentation.mov")
        print()
        print("Note: Video processing may take several minutes for large files.")
        sys.exit(1)
    
    video_file = Path(sys.argv[1])
    
    try:
        success = asyncio.run(test_video_processing(video_file))
        if success:
            print("\n🎉 Video processing test completed successfully!")
            sys.exit(0)
        else:
            print("\n💥 Video processing test failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
