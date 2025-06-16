#!/usr/bin/env python3
"""
MoRAG YouTube Processing Test Script

Supports both processing (immediate results) and ingestion (background + storage) modes.

Usage:
    python test-youtube.py <youtube_url> [options]

Processing Mode (immediate results):
    python test-youtube.py https://www.youtube.com/watch?v=dQw4w9WgXcQ
    python test-youtube.py https://youtu.be/dQw4w9WgXcQ

Ingestion Mode (background processing + storage):
    python test-youtube.py https://www.youtube.com/watch?v=VIDEO_ID --ingest
    python test-youtube.py https://youtu.be/VIDEO_ID --ingest --webhook-url https://my-app.com/webhook
    python test-youtube.py https://www.youtube.com/watch?v=VIDEO_ID --ingest --metadata '{"category": "education"}'

Options:
    --ingest                    Enable ingestion mode (background processing + storage)
    --webhook-url URL          Webhook URL for completion notifications (ingestion mode only)
    --metadata JSON            Additional metadata as JSON string (ingestion mode only)
    --enable-diarization       Enable speaker diarization for audio track
    --enable-topics            Enable topic segmentation for audio track
    --model-size SIZE          Whisper model size: tiny, base, small, medium, large (default: base)
    --extract-audio            Extract and process audio (default: false for faster testing)
    --help                     Show this help message
"""

import sys
import asyncio
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
import re
import requests

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from morag_youtube import YouTubeProcessor, YouTubeConfig
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you have installed the MoRAG packages:")
    print("  pip install -e packages/morag-core")
    print("  pip install -e packages/morag-youtube")
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


def validate_youtube_url(url: str) -> bool:
    """Validate YouTube URL format."""
    youtube_patterns = [
        r'https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+',
        r'https?://youtu\.be/[\w-]+',
        r'https?://(?:www\.)?youtube\.com/embed/[\w-]+',
    ]
    
    for pattern in youtube_patterns:
        if re.match(pattern, url):
            return True
    return False


async def test_youtube_processing(url: str, extract_audio: bool = False,
                                  enable_diarization: bool = False, enable_topics: bool = False,
                                  model_size: str = "base") -> bool:
    """Test YouTube processing functionality."""
    print_header("MoRAG YouTube Processing Test")
    
    if not validate_youtube_url(url):
        print(f"❌ Error: Invalid YouTube URL format: {url}")
        print("Please provide a valid YouTube URL like:")
        print("  https://www.youtube.com/watch?v=VIDEO_ID")
        print("  https://youtu.be/VIDEO_ID")
        return False
    
    print_result("YouTube URL", url)
    print_result("Extract Audio", "✅ Enabled" if extract_audio else "❌ Disabled")
    print_result("Speaker Diarization", "✅ Enabled" if enable_diarization else "❌ Disabled")
    print_result("Topic Segmentation", "✅ Enabled" if enable_topics else "❌ Disabled")
    print_result("Audio Model Size", model_size)

    try:
        # Initialize YouTube processor
        processor = YouTubeProcessor()
        print_result("YouTube Processor", "✅ Initialized successfully")

        # Create YouTube configuration
        config = YouTubeConfig(
            quality="best",
            extract_audio=extract_audio,
            download_subtitles=False,  # Disable for faster testing
            download_thumbnails=False,  # Disable for faster testing
            extract_metadata_only=not extract_audio  # Only extract metadata if not processing audio
        )
        print_result("YouTube Config", "✅ Created successfully")

        print_section("Processing YouTube Video")
        print("🔄 Starting YouTube video processing...")
        if extract_audio:
            print("   Downloading video and extracting audio...")
            if enable_diarization or enable_topics:
                print("   Enhanced audio processing enabled...")
        else:
            print("   Extracting metadata only for faster testing...")

        # Process the YouTube URL
        result = await processor.process_url(url, config)
        
        if result.success:
            print("✅ YouTube processing completed successfully!")

            # Process audio if extracted (always process for transcription in processing mode)
            audio_processing_result = None
            if result.audio_path and extract_audio:
                try:
                    print_section("Enhanced Audio Processing")
                    print("🔄 Processing extracted audio with enhanced features...")

                    # Import audio processor
                    from morag_audio import AudioProcessor, AudioConfig

                    # Create audio config
                    audio_config = AudioConfig(
                        model_size=model_size,
                        device="auto",
                        enable_diarization=enable_diarization,
                        enable_topic_segmentation=enable_topics,
                        vad_filter=True,
                        word_timestamps=True,
                        include_metadata=True
                    )

                    # Initialize audio processor
                    audio_processor = AudioProcessor(audio_config)
                    print_result("Audio Processor", "✅ Initialized successfully")

                    # Process the audio
                    audio_processing_result = await audio_processor.process(result.audio_path)

                    if audio_processing_result.success:
                        print("✅ Audio processing completed successfully!")
                        print_result("Audio Processing Time", f"{audio_processing_result.processing_time:.2f} seconds")
                        print_result("Transcript Length", f"{len(audio_processing_result.transcript)} characters")
                        print_result("Segments Count", f"{len(audio_processing_result.segments)}")

                        # Show speaker diarization results
                        if enable_diarization and audio_processing_result.metadata:
                            speakers_info = audio_processing_result.metadata.get('speakers', [])
                            num_speakers = audio_processing_result.metadata.get('num_speakers', 0)

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
                                        print(f"  Speaker {speaker}")

                            segments_with_speakers = len([seg for seg in audio_processing_result.segments if seg.speaker]) if audio_processing_result.segments else 0
                            print_result("Segments with Speaker Info", f"{segments_with_speakers}/{len(audio_processing_result.segments) if audio_processing_result.segments else 0}")

                        # Show topic segmentation results
                        if enable_topics and audio_processing_result.metadata:
                            topics_info = audio_processing_result.metadata.get('topics', [])
                            num_topics = audio_processing_result.metadata.get('num_topics', 0)

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

                            segments_with_topics = len([seg for seg in audio_processing_result.segments if seg.topic_id is not None]) if audio_processing_result.segments else 0
                            print_result("Segments with Topic Info", f"{segments_with_topics}/{len(audio_processing_result.segments) if audio_processing_result.segments else 0}")

                        # Show transcript preview
                        if audio_processing_result.transcript:
                            print_section("Transcript Preview")
                            transcript_preview = audio_processing_result.transcript[:500] + "..." if len(audio_processing_result.transcript) > 500 else audio_processing_result.transcript
                            print(f"📄 Transcript ({len(audio_processing_result.transcript)} characters):")
                            print(transcript_preview)
                    else:
                        print("❌ Audio processing failed!")
                        print_result("Audio Error", audio_processing_result.error_message or "Unknown error")

                except Exception as e:
                    print(f"❌ Error during audio processing: {e}")
                    import traceback
                    traceback.print_exc()

            print_section("Processing Results")
            print_result("Status", "✅ Success")
            print_result("Processing Time", f"{result.processing_time:.2f} seconds")
            print_result("File Size", f"{result.file_size / 1024 / 1024:.2f} MB")

            if result.metadata:
                print_section("Video Metadata")
                metadata = result.metadata
                print_result("Title", metadata.title)
                print_result("Channel", metadata.channel)
                print_result("Duration", f"{metadata.duration} seconds")
                print_result("View Count", f"{metadata.view_count:,}")
                print_result("Upload Date", metadata.upload_date)
                print_result("Video ID", metadata.video_id)

                if metadata.description:
                    print_section("Description Preview")
                    desc_preview = metadata.description[:300] + "..." if len(metadata.description) > 300 else metadata.description
                    print(f"📄 Description ({len(metadata.description)} characters):")
                    print(desc_preview)

                if metadata.tags:
                    print_section("Tags (first 10)")
                    for i, tag in enumerate(metadata.tags[:10]):
                        print_result(f"Tag {i+1}", tag)

            if result.video_path:
                print_section("Downloaded Files")
                print_result("Video Path", str(result.video_path))

            if result.audio_path:
                print_result("Audio Path", str(result.audio_path))

            if result.subtitle_paths:
                print_result("Subtitles", f"{len(result.subtitle_paths)} files")
                for i, subtitle in enumerate(result.subtitle_paths):
                    print_result(f"Subtitle {i+1}", str(subtitle))

            if result.thumbnail_paths:
                print_result("Thumbnails", f"{len(result.thumbnail_paths)} files")
                for i, thumbnail in enumerate(result.thumbnail_paths):
                    print_result(f"Thumbnail {i+1}", str(thumbnail))

            # Extract video ID for filename
            video_id = re.search(r'(?:v=|youtu\.be/)([^&\n?#]+)', url)
            safe_filename = video_id.group(1) if video_id else "youtube_video"

            # Prepare detailed audio processing result
            audio_result_data = None
            if audio_processing_result:
                audio_result_data = {
                    'transcript': audio_processing_result.transcript,
                    'segments': [
                        {
                            'start': seg.start,
                            'end': seg.end,
                            'text': seg.text,
                            'speaker': seg.speaker,
                            'confidence': seg.confidence,
                            'topic_id': seg.topic_id,
                            'topic_label': seg.topic_label
                        } for seg in audio_processing_result.segments
                    ] if hasattr(audio_processing_result, 'segments') and audio_processing_result.segments else [],
                    'metadata': audio_processing_result.metadata if hasattr(audio_processing_result, 'metadata') else {},
                    'processing_time': audio_processing_result.processing_time if hasattr(audio_processing_result, 'processing_time') else 0.0,
                    'file_path': audio_processing_result.file_path if hasattr(audio_processing_result, 'file_path') else None,
                    'success': audio_processing_result.success if hasattr(audio_processing_result, 'success') else True,
                    'error_message': audio_processing_result.error_message if hasattr(audio_processing_result, 'error_message') else None
                }

                # Extract speaker information from metadata and segments
                if audio_processing_result.metadata and enable_diarization:
                    speakers_info = audio_processing_result.metadata.get('speakers', [])
                    num_speakers = audio_processing_result.metadata.get('num_speakers', 0)

                    if speakers_info or num_speakers > 0:
                        audio_result_data['speaker_diarization'] = {
                            'enabled': True,
                            'total_speakers': num_speakers,
                            'speakers': speakers_info,
                            'segments_with_speakers': len([seg for seg in audio_processing_result.segments if seg.speaker]) if audio_processing_result.segments else 0
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
                if audio_processing_result.metadata and enable_topics:
                    topics_info = audio_processing_result.metadata.get('topics', [])
                    num_topics = audio_processing_result.metadata.get('num_topics', 0)

                    if topics_info or num_topics > 0:
                        audio_result_data['topic_segmentation'] = {
                            'enabled': True,
                            'total_topics': num_topics,
                            'topics': topics_info,
                            'segments_with_topics': len([seg for seg in audio_processing_result.segments if seg.topic_id is not None]) if audio_processing_result.segments else 0
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

            # Save results to file
            output_file = Path(f"uploads/youtube_{safe_filename}_test_result.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'mode': 'processing',
                    'url': url,
                    'success': result.success,
                    'processing_time': result.processing_time,
                    'file_size': result.file_size,
                    'configuration': {
                        'extract_audio': extract_audio,
                        'enable_diarization': enable_diarization,
                        'enable_topics': enable_topics,
                        'model_size': model_size
                    },
                    'youtube_metadata': {
                        'title': metadata.title if result.metadata else None,
                        'channel': metadata.channel if result.metadata else None,
                        'duration': metadata.duration if result.metadata else None,
                        'view_count': metadata.view_count if result.metadata else None,
                        'upload_date': metadata.upload_date if result.metadata else None,
                        'video_id': metadata.video_id if result.metadata else None,
                        'description': metadata.description if result.metadata else None,
                        'tags': metadata.tags if result.metadata else []
                    } if result.metadata else {},
                    'video_path': str(result.video_path) if result.video_path else None,
                    'audio_path': str(result.audio_path) if result.audio_path else None,
                    'audio_processing_result': audio_result_data,
                    'subtitle_paths': [str(p) for p in result.subtitle_paths],
                    'thumbnail_paths': [str(p) for p in result.thumbnail_paths],
                    'temp_files': [str(f) for f in result.temp_files],
                    'error_message': result.error_message
                }, f, indent=2, ensure_ascii=False)

            print_section("Output")
            print_result("Results saved to", str(output_file))

            return True

        else:
            print("❌ YouTube processing failed!")
            print_result("Error", result.error_message or "Unknown error")
            return False

    except Exception as e:
        print(f"❌ Error during YouTube processing: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_youtube_ingestion(url: str, webhook_url: Optional[str] = None,
                                metadata: Optional[Dict[str, Any]] = None) -> bool:
    """Test YouTube ingestion functionality."""
    print_header("MoRAG YouTube Ingestion Test")

    if not validate_youtube_url(url):
        print(f"❌ Error: Invalid YouTube URL format: {url}")
        print("Please provide a valid YouTube URL like:")
        print("  https://www.youtube.com/watch?v=VIDEO_ID")
        print("  https://youtu.be/VIDEO_ID")
        return False

    print_result("YouTube URL", url)
    print_result("Webhook URL", webhook_url or "None")
    print_result("Metadata", json.dumps(metadata, indent=2) if metadata else "None")

    try:
        print_section("Submitting Ingestion Task")
        print("🔄 Starting YouTube ingestion...")
        print("   This may take several minutes for long videos...")

        # Prepare request data
        data = {
            'source_type': 'youtube',
            'url': url
        }

        if webhook_url:
            data['webhook_url'] = webhook_url
        if metadata:
            data['metadata'] = metadata

        # Submit to ingestion API
        response = requests.post(
            'http://localhost:8000/api/v1/ingest/url',
            json=data,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            print("✅ YouTube ingestion task submitted successfully!")

            print_section("Ingestion Results")
            print_result("Status", "✅ Success")
            print_result("Task ID", result.get('task_id', 'Unknown'))
            print_result("Message", result.get('message', 'Task created'))
            print_result("Estimated Time", f"{result.get('estimated_time', 'Unknown')} seconds")

            # Extract video ID for filename
            video_id = re.search(r'(?:v=|youtu\.be/)([^&\n?#]+)', url)
            safe_filename = video_id.group(1) if video_id else "youtube_video"

            # Save ingestion result
            output_file = Path(f"uploads/youtube_{safe_filename}_ingest_result.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'mode': 'ingestion',
                    'task_id': result.get('task_id'),
                    'status': result.get('status'),
                    'message': result.get('message'),
                    'estimated_time': result.get('estimated_time'),
                    'webhook_url': webhook_url,
                    'metadata': metadata,
                    'url': url
                }, f, indent=2, ensure_ascii=False)

            print_section("Output")
            print_result("Ingestion result saved to", str(output_file))
            print_result("Monitor task status", f"curl http://localhost:8000/api/v1/status/{result.get('task_id')}")

            return True
        else:
            print("❌ YouTube ingestion failed!")
            print_result("Status Code", str(response.status_code))
            print_result("Error", response.text)
            return False

    except Exception as e:
        print(f"❌ Error during YouTube ingestion: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="MoRAG YouTube Processing Test Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Processing Mode (immediate results):
    python test-youtube.py https://www.youtube.com/watch?v=dQw4w9WgXcQ
    python test-youtube.py https://youtu.be/dQw4w9WgXcQ --extract-audio
    python test-youtube.py https://www.youtube.com/watch?v=VIDEO_ID --extract-audio --enable-diarization --enable-topics
    python test-youtube.py https://youtu.be/VIDEO_ID --extract-audio --enable-diarization --model-size large

  Ingestion Mode (background processing + storage):
    python test-youtube.py https://www.youtube.com/watch?v=VIDEO_ID --ingest
    python test-youtube.py https://youtu.be/VIDEO_ID --ingest --webhook-url https://my-app.com/webhook
    python test-youtube.py https://www.youtube.com/watch?v=VIDEO_ID --ingest --metadata '{"category": "education"}'

Note: Processing may take several minutes for long videos.
Make sure you have a stable internet connection.
        """
    )

    parser.add_argument('youtube_url', help='YouTube URL to process')
    parser.add_argument('--ingest', action='store_true',
                       help='Enable ingestion mode (background processing + storage)')
    parser.add_argument('--webhook-url', help='Webhook URL for completion notifications (ingestion mode only)')
    parser.add_argument('--metadata', help='Additional metadata as JSON string (ingestion mode only)')
    parser.add_argument('--enable-diarization', action='store_true',
                       help='Enable speaker diarization for audio track')
    parser.add_argument('--enable-topics', action='store_true',
                       help='Enable topic segmentation for audio track')
    parser.add_argument('--model-size', choices=['tiny', 'base', 'small', 'medium', 'large'],
                       default='base', help='Whisper model size (default: base)')
    parser.add_argument('--extract-audio', action='store_true',
                       help='Extract and process audio (default: false for faster testing)')

    args = parser.parse_args()

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
            success = asyncio.run(test_youtube_ingestion(
                args.youtube_url,
                webhook_url=args.webhook_url,
                metadata=metadata
            ))
            if success:
                print("\n🎉 YouTube ingestion test completed successfully!")
                print("💡 Use the task ID to monitor progress and retrieve results.")
                sys.exit(0)
            else:
                print("\n💥 YouTube ingestion test failed!")
                sys.exit(1)
        else:
            # Processing mode
            success = asyncio.run(test_youtube_processing(
                args.youtube_url,
                extract_audio=args.extract_audio,
                enable_diarization=args.enable_diarization,
                enable_topics=args.enable_topics,
                model_size=args.model_size
            ))
            if success:
                print("\n🎉 YouTube processing test completed successfully!")
                sys.exit(0)
            else:
                print("\n💥 YouTube processing test failed!")
                sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
