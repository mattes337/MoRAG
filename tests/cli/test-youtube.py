#!/usr/bin/env python3
"""
MoRAG YouTube Processing Test Script

Usage: python test-youtube.py <youtube_url>

Examples:
    python test-youtube.py https://www.youtube.com/watch?v=dQw4w9WgXcQ
    python test-youtube.py https://youtu.be/dQw4w9WgXcQ
"""

import sys
import asyncio
import json
from pathlib import Path
from typing import Optional
import re

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


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python test-youtube.py <youtube_url>")
        print()
        print("Examples:")
        print("  python test-youtube.py https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        print("  python test-youtube.py https://youtu.be/dQw4w9WgXcQ")
        print()
        print("Note: Processing may take several minutes for long videos.")
        print("Make sure you have a stable internet connection.")
        sys.exit(1)
    
    url = sys.argv[1]
    
    try:
        success = asyncio.run(test_youtube_processing(url))
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
