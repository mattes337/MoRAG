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
    --help                     Show this help message
"""

import sys
import os
import asyncio
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
import re
import requests

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from the project root
from dotenv import load_dotenv
env_path = project_root / '.env'
load_dotenv(env_path)

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


async def test_youtube_processing(url: str) -> bool:
    """Test YouTube processing functionality."""
    print_header("MoRAG YouTube Processing Test")
    
    if not validate_youtube_url(url):
        print(f"❌ Error: Invalid YouTube URL format: {url}")
        print("Please provide a valid YouTube URL like:")
        print("  https://www.youtube.com/watch?v=VIDEO_ID")
        print("  https://youtu.be/VIDEO_ID")
        return False
    
    print_result("YouTube URL", url)
    
    try:
        # Initialize YouTube processor
        processor = YouTubeProcessor()
        print_result("YouTube Processor", "✅ Initialized successfully")

        # Create YouTube configuration (metadata only for faster testing)
        config = YouTubeConfig(
            quality="best",
            extract_audio=False,  # Disable for faster testing
            download_subtitles=False,  # Disable for faster testing
            download_thumbnails=False,  # Disable for faster testing
            extract_metadata_only=True  # Only extract metadata for testing
        )
        print_result("YouTube Config", "✅ Created successfully")

        print_section("Processing YouTube Video")
        print("🔄 Starting YouTube video processing...")
        print("   Extracting metadata only for faster testing...")

        # Process the YouTube URL
        result = await processor.process_url(url, config)
        
        if result.success:
            print("✅ YouTube processing completed successfully!")

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

            # Save results to file
            output_file = Path(f"uploads/youtube_{safe_filename}_test_result.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'mode': 'processing',
                    'url': url,
                    'success': result.success,
                    'processing_time': result.processing_time,
                    'file_size': result.file_size,
                    'metadata': {
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
                                metadata: Optional[Dict[str, Any]] = None,
                                qdrant_collection_name: Optional[str] = None,
                                neo4j_database_name: Optional[str] = None) -> bool:
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
        if qdrant_collection_name:
            data['qdrant_collection'] = qdrant_collection_name
        if neo4j_database_name:
            data['neo4j_database'] = neo4j_database_name

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
    python test-youtube.py https://youtu.be/dQw4w9WgXcQ

  Ingestion Mode (background processing + storage):
    python test-youtube.py https://www.youtube.com/watch?v=VIDEO_ID --ingest
    python test-youtube.py https://youtu.be/VIDEO_ID --ingest --webhook-url https://my-app.com/webhook
    python test-youtube.py https://www.youtube.com/watch?v=VIDEO_ID --ingest --metadata '{"category": "education"}'

  Resume from Process Result:
    python test-youtube.py https://www.youtube.com/watch?v=VIDEO_ID --use-process-result my-youtube.process_result.json

  Resume from Ingestion Data:
    python test-youtube.py https://www.youtube.com/watch?v=VIDEO_ID --use-ingestion-data my-youtube.ingest_data.json

Note: Processing may take several minutes for long videos.
Make sure you have a stable internet connection.
        """
    )

    parser.add_argument('youtube_url', help='YouTube URL to process')
    parser.add_argument('--ingest', action='store_true',
                       help='Enable ingestion mode (background processing + storage)')
    parser.add_argument('--qdrant', action='store_true',
                       help='Store in Qdrant vector database (ingestion mode only)')
    parser.add_argument('--qdrant-collection', help='Qdrant collection name (default: from environment or morag_youtube)')
    parser.add_argument('--neo4j', action='store_true',
                       help='Store in Neo4j graph database (ingestion mode only)')
    parser.add_argument('--neo4j-database', help='Neo4j database name (default: from environment or neo4j)')
    parser.add_argument('--webhook-url', help='Webhook URL for completion notifications (ingestion mode only)')
    parser.add_argument('--metadata', help='Additional metadata as JSON string (ingestion mode only)')
    parser.add_argument('--language', help='Language code for processing (auto-detect if not specified)')
    parser.add_argument('--use-process-result', help='Skip processing and use existing process result file (e.g., my-file.process_result.json)')
    parser.add_argument('--use-ingestion-data', help='Skip processing and ingestion calculation, use existing ingestion data file (e.g., my-file.ingest_data.json)')

    args = parser.parse_args()

    # Parse metadata if provided
    metadata = None
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError as e:
            print(f"❌ Error: Invalid JSON in metadata: {e}")
            sys.exit(1)

    # Handle resume arguments
    from resume_utils import handle_resume_arguments
    handle_resume_arguments(args, args.youtube_url, 'youtube', metadata)

    try:
        if args.ingest:
            # Ingestion mode
            success = asyncio.run(test_youtube_ingestion(
                args.youtube_url,
                webhook_url=args.webhook_url,
                metadata=metadata,
                qdrant_collection_name=args.qdrant_collection,
                neo4j_database_name=args.neo4j_database
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
            success = asyncio.run(test_youtube_processing(args.youtube_url))
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
