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
    --enable-diarization       Enable speaker diarization for audio track
    --enable-topics            Enable topic segmentation for audio track
    --model-size SIZE          Whisper model size: tiny, base, small, medium, large (default: base)
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
                               thumbnail_count: int = 3, enable_ocr: bool = False,
                               enable_diarization: bool = False, enable_topics: bool = False,
                               model_size: str = "base") -> bool:
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
            enable_enhanced_audio=True,  # Always enable for transcription in processing mode
            enable_speaker_diarization=enable_diarization,
            enable_topic_segmentation=enable_topics,
            audio_model_size=model_size,
            enable_ocr=enable_ocr
        )
        print_result("Video Configuration", "✅ Created successfully")
        print_result("Generate Thumbnails", "✅ Enabled" if generate_thumbnails else "❌ Disabled")
        print_result("Thumbnail Count", str(thumbnail_count) if generate_thumbnails else "N/A")
        print_result("OCR Enabled", "✅ Enabled" if enable_ocr else "❌ Disabled")
        print_result("Speaker Diarization", "✅ Enabled" if enable_diarization else "❌ Disabled")
        print_result("Topic Segmentation", "✅ Enabled" if enable_topics else "❌ Disabled")
        print_result("Audio Model Size", model_size)

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
                print_result("Segments Count", f"{len(audio_result.segments) if hasattr(audio_result, 'segments') and audio_result.segments else 0}")

                # Show speaker diarization results
                if enable_diarization and audio_result.metadata:
                    speakers_info = audio_result.metadata.get('speakers', [])
                    num_speakers = audio_result.metadata.get('num_speakers', 0)

                    print_section("Speaker Diarization Results")
                    print_result("Total Speakers", str(num_speakers))

                    if speakers_info:
                        print("📊 Speaker Information:")
                        for speaker in speakers_info:
                            if isinstance(speaker, dict):
                                speaker_id = speaker.get('id', 'Unknown')
                                speaker_name = speaker.get('name', f'Speaker {speaker_id}')
                                print(f"  {speaker_name}")
                            else:
                                # Handle case where speaker is just a string ID
                                print(f"  Speaker {speaker}")

                    # Count segments with speaker information
                    segments_with_speakers = len([seg for seg in audio_result.segments if seg.speaker]) if audio_result.segments else 0
                    print_result("Segments with Speaker Info", f"{segments_with_speakers}/{len(audio_result.segments) if audio_result.segments else 0}")

                # Show topic segmentation results
                if enable_topics and audio_result.metadata:
                    topics_info = audio_result.metadata.get('topics', [])
                    num_topics = audio_result.metadata.get('num_topics', 0)

                    print_section("Topic Segmentation Results")
                    print_result("Total Topics", str(num_topics))

                    if topics_info:
                        print("📋 Topic Overview:")
                        for topic in topics_info[:3]:  # Show first 3 topics
                            topic_id = topic.get('id', 'Unknown')
                            title = topic.get('title', f'Topic {topic_id}')
                            duration = topic.get('duration', 0)
                            duration_str = f"{duration:.1f}s" if duration else "N/A"
                            print(f"  {title}: {duration_str}")

                    # Count segments with topic information
                    segments_with_topics = len([seg for seg in audio_result.segments if seg.topic_id is not None]) if audio_result.segments else 0
                    print_result("Segments with Topic Info", f"{segments_with_topics}/{len(audio_result.segments) if audio_result.segments else 0}")

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

        # Prepare detailed audio processing result
        audio_result_data = None
        if result.audio_processing_result:
            audio_result = result.audio_processing_result
            audio_result_data = {
                'transcript': audio_result.transcript,
                'segments': [
                    {
                        'start': seg.start,
                        'end': seg.end,
                        'text': seg.text,
                        'speaker': seg.speaker,
                        'confidence': seg.confidence,
                        'topic_id': seg.topic_id,
                        'topic_label': seg.topic_label
                    } for seg in audio_result.segments
                ] if hasattr(audio_result, 'segments') and audio_result.segments else [],
                'metadata': audio_result.metadata if hasattr(audio_result, 'metadata') else {},
                'processing_time': audio_result.processing_time if hasattr(audio_result, 'processing_time') else 0.0,
                'file_path': audio_result.file_path if hasattr(audio_result, 'file_path') else None,
                'success': audio_result.success if hasattr(audio_result, 'success') else True,
                'error_message': audio_result.error_message if hasattr(audio_result, 'error_message') else None
            }

            # Extract speaker information from metadata and segments
            if audio_result.metadata and enable_diarization:
                speakers_info = audio_result.metadata.get('speakers', [])
                num_speakers = audio_result.metadata.get('num_speakers', 0)

                if speakers_info or num_speakers > 0:
                    audio_result_data['speaker_diarization'] = {
                        'enabled': True,
                        'total_speakers': num_speakers,
                        'speakers': speakers_info,
                        'segments_with_speakers': len([seg for seg in audio_result.segments if seg.speaker]) if audio_result.segments else 0
                    }
                else:
                    audio_result_data['speaker_diarization'] = {
                        'enabled': True,
                        'total_speakers': 0,
                        'speakers': [],
                        'segments_with_speakers': 0,
                        'note': 'Speaker diarization was enabled but no speakers were detected'
                    }
            else:
                audio_result_data['speaker_diarization'] = {
                    'enabled': False,
                    'note': 'Speaker diarization was not enabled'
                }

            # Extract topic information from metadata and segments
            if audio_result.metadata and enable_topics:
                topics_info = audio_result.metadata.get('topics', [])
                num_topics = audio_result.metadata.get('num_topics', 0)

                if topics_info or num_topics > 0:
                    audio_result_data['topic_segmentation'] = {
                        'enabled': True,
                        'total_topics': num_topics,
                        'topics': topics_info,
                        'segments_with_topics': len([seg for seg in audio_result.segments if seg.topic_id is not None]) if audio_result.segments else 0
                    }
                else:
                    audio_result_data['topic_segmentation'] = {
                        'enabled': True,
                        'total_topics': 0,
                        'topics': [],
                        'segments_with_topics': 0,
                        'note': 'Topic segmentation was enabled but no topics were detected'
                    }
            else:
                audio_result_data['topic_segmentation'] = {
                    'enabled': False,
                    'note': 'Topic segmentation was not enabled'
                }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'mode': 'processing',
                'processing_time': result.processing_time,
                'configuration': {
                    'generate_thumbnails': generate_thumbnails,
                    'thumbnail_count': thumbnail_count,
                    'enable_ocr': enable_ocr,
                    'enable_diarization': enable_diarization,
                    'enable_topics': enable_topics,
                    'model_size': model_size
                },
                'video_metadata': {
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
                'audio_processing_result': audio_result_data,
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


async def test_video_ingestion(video_file: Path, webhook_url: Optional[str] = None,
                              metadata: Optional[Dict[str, Any]] = None) -> bool:
    """Test video ingestion functionality."""
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
        print_section("Submitting Ingestion Task")
        print("🔄 Starting video ingestion...")
        print("   This may take a while for large videos...")

        # Prepare form data
        files = {'file': open(video_file, 'rb')}
        data = {'source_type': 'video'}

        if webhook_url:
            data['webhook_url'] = webhook_url
        if metadata:
            data['metadata'] = json.dumps(metadata)

        # Submit to ingestion API
        response = requests.post(
            'http://localhost:8000/api/v1/ingest/file',
            files=files,
            data=data,
            timeout=30
        )

        files['file'].close()

        if response.status_code == 200:
            result = response.json()
            print("✅ Video ingestion task submitted successfully!")

            print_section("Ingestion Results")
            print_result("Status", "✅ Success")
            print_result("Task ID", result.get('task_id', 'Unknown'))
            print_result("Message", result.get('message', 'Task created'))
            print_result("Estimated Time", f"{result.get('estimated_time', 'Unknown')} seconds")

            # Save ingestion result
            output_file = video_file.parent / f"{video_file.stem}_ingest_result.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'mode': 'ingestion',
                    'task_id': result.get('task_id'),
                    'status': result.get('status'),
                    'message': result.get('message'),
                    'estimated_time': result.get('estimated_time'),
                    'webhook_url': webhook_url,
                    'metadata': metadata,
                    'file_path': str(video_file)
                }, f, indent=2, ensure_ascii=False)

            print_section("Output")
            print_result("Ingestion result saved to", str(output_file))
            print_result("Monitor task status", f"curl http://localhost:8000/api/v1/status/{result.get('task_id')}")

            return True
        else:
            print("❌ Video ingestion failed!")
            print_result("Status Code", str(response.status_code))
            print_result("Error", response.text)
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
    python test-video.py meeting.mp4 --enable-diarization --enable-topics
    python test-video.py lecture.mp4 --enable-diarization --model-size large

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
    parser.add_argument('--webhook-url', help='Webhook URL for completion notifications (ingestion mode only)')
    parser.add_argument('--metadata', help='Additional metadata as JSON string (ingestion mode only)')
    parser.add_argument('--thumbnails', action='store_true',
                       help='Generate thumbnails (opt-in, default: false)')
    parser.add_argument('--thumbnail-count', type=int, default=3,
                       help='Number of thumbnails to generate (default: 3)')
    parser.add_argument('--enable-ocr', action='store_true',
                       help='Enable OCR on video frames')
    parser.add_argument('--enable-diarization', action='store_true',
                       help='Enable speaker diarization for audio track')
    parser.add_argument('--enable-topics', action='store_true',
                       help='Enable topic segmentation for audio track')
    parser.add_argument('--model-size', choices=['tiny', 'base', 'small', 'medium', 'large'],
                       default='base', help='Whisper model size (default: base)')

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
                enable_ocr=args.enable_ocr,
                enable_diarization=args.enable_diarization,
                enable_topics=args.enable_topics,
                model_size=args.model_size
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
