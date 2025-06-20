#!/usr/bin/env python3
"""
MoRAG Video Processing Test Script

Supports both processing (immediate results) and ingestion (background + storage) modes.

Usage:
    python test-video.py <video_file> [options]

Processing Mode (immediate results):
    python test-video.py my-video.mp4
    python test-video.py recording.avi --thumbnails
    python test-video.py presentation.mov --enable-ocr

Ingestion Mode (background processing + storage):
    python test-video.py my-video.mp4 --ingest
    python test-video.py recording.avi --ingest --metadata '{"type": "meeting"}'
    python test-video.py presentation.mov --ingest --webhook-url https://my-app.com/webhook

Options:
    --ingest                    Enable ingestion mode (background processing + storage)
    --webhook-url URL          Webhook URL for completion notifications (ingestion mode only)
    --metadata JSON            Additional metadata as JSON string (ingestion mode only)
    --thumbnails               Generate thumbnails (opt-in, default: false)
    --thumbnail-count N        Number of thumbnails to generate (default: 3)
    --enable-ocr               Enable OCR on video frames
    --help                     Show this help message
"""

import sys
import asyncio
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
import requests

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


async def test_video_processing(video_file: Path, generate_thumbnails: bool = False,
                               thumbnail_count: int = 3, enable_ocr: bool = False) -> bool:
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
            generate_thumbnails=generate_thumbnails,
            thumbnail_count=thumbnail_count,
            extract_keyframes=False,  # Disable for faster processing
            enable_enhanced_audio=True,
            enable_speaker_diarization=False,  # Disable for faster processing
            enable_topic_segmentation=False,  # Disable for faster processing
            audio_model_size="base",  # Use base model for faster processing
            enable_ocr=enable_ocr
        )
        print_result("Video Configuration", "✅ Created successfully")
        print_result("Generate Thumbnails", "✅ Enabled" if generate_thumbnails else "❌ Disabled")
        print_result("Thumbnail Count", str(thumbnail_count) if generate_thumbnails else "N/A")
        print_result("OCR Enabled", "✅ Enabled" if enable_ocr else "❌ Disabled")

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
                'mode': 'processing',
                'processing_time': result.processing_time,
                'generate_thumbnails': generate_thumbnails,
                'thumbnail_count': thumbnail_count,
                'enable_ocr': enable_ocr,
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


async def test_video_ingestion(
    video_file: Path,
    webhook_url: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Test video ingestion via direct API calls.
    
    Args:
        video_file: Path to video file
        webhook_url: Optional webhook URL for completion notifications
        metadata: Optional metadata dictionary
        
    Returns:
        True if ingestion was successful, False otherwise
    """
    print_header("MoRAG Video Ingestion Test")

    if not video_file.exists():
        print(f"❌ Error: Video file not found: {video_file}")
        return False

    print_result("Input File", str(video_file))
    print_result("File Size", f"{video_file.stat().st_size / 1024 / 1024:.2f} MB")
    print_result("File Extension", video_file.suffix.lower())
    print_result("Webhook URL", webhook_url or "None")
    print_result("Metadata", json.dumps(metadata, indent=2) if metadata else "None")

    try:
        from morag.api import MoRAGAPI
        from graph_extraction import extract_and_ingest
        import uuid
        
        print_section("Processing Video File")
        print("🔄 Starting video ingestion...")
        print("   This may take a while for large videos...")
        
        # Initialize the API
        api = MoRAGAPI()
        
        # Prepare options
        options = {
            'store_in_vector_db': True,
            'metadata': metadata or {},
            'webhook_url': webhook_url
        }
        
        # Process the video file
        result = await api.process_file(str(video_file), 'video', options)
        
        if result.success:
            print("✅ Video processing completed successfully!")
            
            # Generate a task ID for compatibility
            task_id = str(uuid.uuid4())
            
            # Perform ingestion to vector database and graph database
            if result.text_content:
                print_section("Ingesting to Databases")
                print("📊 Ingesting content to databases...")
                await extract_and_ingest(
                    text_content=result.text_content,
                    doc_id=task_id,
                    metadata=metadata or {},
                    use_qdrant=True,
                    use_neo4j=True
                )
                print("✅ Content ingested to vector and graph databases!")
            
            print_section("Ingestion Results")
            print_result("Status", "✅ Success")
            print_result("Task ID", task_id)
            print_result("Content Length", f"{len(result.text_content or '')} characters")
            print_result("Processing Time", f"{result.processing_time:.2f} seconds")
            
            if webhook_url:
                print_result("Webhook URL", f"Would notify: {webhook_url}")
            
            # Save ingestion result
            ingestion_result = {
                'mode': 'ingestion',
                'task_id': task_id,
                'success': True,
                'content_length': len(result.text_content or ''),
                'metadata': result.metadata,
                'processing_time': result.processing_time,
                'webhook_url': webhook_url,
                'file_path': str(video_file)
            }
            
            output_file = video_file.parent / f"{video_file.stem}_ingest_result.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(ingestion_result, f, indent=2, ensure_ascii=False)
            
            print_section("Output")
            print_result("Ingestion result saved to", str(output_file))
            
            return True
        else:
            print("❌ Video ingestion failed!")
            if result.error_message:
                print_result("Error", result.error_message)
            return False
            
    except Exception as e:
        print(f"❌ Error during video ingestion: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="MoRAG Video Processing Test Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Processing Mode (immediate results):
    python test-video.py my-video.mp4
    python test-video.py recording.avi --thumbnails --thumbnail-count 5
    python test-video.py presentation.mov --enable-ocr

  Ingestion Mode (background processing + storage):
    python test-video.py my-video.mp4 --ingest
    python test-video.py recording.avi --ingest --metadata '{"type": "meeting"}'
    python test-video.py presentation.mov --ingest --webhook-url https://my-app.com/webhook

Note: Video processing may take several minutes for large files.
        """
    )

    parser.add_argument('video_file', help='Path to video file')
    parser.add_argument('--ingest', action='store_true',
                       help='Enable ingestion mode (background processing + storage)')
    parser.add_argument('--qdrant', action='store_true',
                       help='Store in Qdrant vector database (ingestion mode only)')
    parser.add_argument('--neo4j', action='store_true',
                       help='Store in Neo4j graph database (ingestion mode only)')
    parser.add_argument('--webhook-url', help='Webhook URL for completion notifications (ingestion mode only)')
    parser.add_argument('--metadata', help='Additional metadata as JSON string (ingestion mode only)')
    parser.add_argument('--thumbnails', action='store_true',
                       help='Generate thumbnails (opt-in, default: false)')
    parser.add_argument('--thumbnail-count', type=int, default=3,
                       help='Number of thumbnails to generate (default: 3)')
    parser.add_argument('--enable-ocr', action='store_true',
                       help='Enable OCR on video frames')

    args = parser.parse_args()

    video_file = Path(args.video_file)

    # Parse metadata if provided
    metadata = None
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError as e:
            print(f"❌ Error: Invalid JSON in metadata: {e}")
            sys.exit(1)

    try:
        if args.ingest:
            # Ingestion mode
            success = asyncio.run(test_video_ingestion(
                video_file,
                webhook_url=args.webhook_url,
                metadata=metadata
            ))
            if success:
                print("\n🎉 Video ingestion test completed successfully!")
                print("💡 Use the task ID to monitor progress and retrieve results.")
                sys.exit(0)
            else:
                print("\n💥 Video ingestion test failed!")
                sys.exit(1)
        else:
            # Processing mode
            success = asyncio.run(test_video_processing(
                video_file,
                generate_thumbnails=args.thumbnails,
                thumbnail_count=args.thumbnail_count,
                enable_ocr=args.enable_ocr
            ))
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
