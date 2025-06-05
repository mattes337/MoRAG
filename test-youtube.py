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
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from morag_youtube import YouTubeProcessor
    from morag_services import ServiceConfig, ContentType
    from morag_core.models import ProcessingConfig
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
        # Initialize configuration
        config = ServiceConfig()
        print_result("Configuration", "✅ Loaded successfully")

        # Initialize YouTube processor
        processor = YouTubeProcessor(config)
        print_result("YouTube Processor", "✅ Initialized successfully")

        # Create processing configuration
        processing_config = ProcessingConfig(
            max_file_size=1024 * 1024 * 1024,  # 1GB
            timeout=900.0,  # 15 minutes
            extract_metadata=True
        )
        print_result("Processing Config", "✅ Created successfully")
        
        print_section("Processing YouTube Video")
        print("🔄 Starting YouTube video processing...")
        print("   This may take several minutes depending on video length...")
        
        # Process the YouTube URL
        result = await processor.process_url(url, processing_config)
        
        if result.success:
            print("✅ YouTube processing completed successfully!")
            
            print_section("Processing Results")
            print_result("Status", "✅ Success")
            print_result("Content Type", result.content_type)
            print_result("Processing Time", f"{result.processing_time:.2f} seconds")
            
            if result.metadata:
                print_section("Metadata")
                for key, value in result.metadata.items():
                    if isinstance(value, (dict, list)):
                        print_result(key, json.dumps(value, indent=2))
                    else:
                        print_result(key, str(value))
            
            if result.content:
                print_section("Content Preview")
                content_preview = result.content[:1000] + "..." if len(result.content) > 1000 else result.content
                print(f"📄 Content ({len(result.content)} characters):")
                print(content_preview)
                
                # Check for specific content sections
                if "## Video Information" in result.content:
                    print_result("Video Information", "✅ Found")
                if "## Audio Transcription" in result.content:
                    print_result("Audio Transcription", "✅ Found")
                if "## Video Description" in result.content:
                    print_result("Video Description", "✅ Found")
            
            if result.summary:
                print_section("Summary")
                print(f"📝 {result.summary}")
            
            # Extract video ID for filename
            video_id = re.search(r'(?:v=|youtu\.be/)([^&\n?#]+)', url)
            safe_filename = video_id.group(1) if video_id else "youtube_video"
            
            # Save results to file
            output_file = Path(f"youtube_{safe_filename}_test_result.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'url': url,
                    'success': result.success,
                    'content_type': result.content_type,
                    'processing_time': result.processing_time,
                    'metadata': result.metadata,
                    'content': result.content,
                    'summary': result.summary,
                    'error': result.error
                }, f, indent=2, ensure_ascii=False)
            
            # Also save markdown content
            markdown_file = Path(f"youtube_{safe_filename}_converted.md")
            with open(markdown_file, 'w', encoding='utf-8') as f:
                f.write(result.content)
            
            print_section("Output")
            print_result("Results saved to", str(output_file))
            print_result("Markdown saved to", str(markdown_file))
            
            return True
            
        else:
            print("❌ YouTube processing failed!")
            print_result("Error", result.error or "Unknown error")
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
