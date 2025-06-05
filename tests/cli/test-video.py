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
    from morag_video import VideoProcessor
    from morag_services import ServiceConfig, ContentType
    from morag_core.models import ProcessingConfig
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
        # Initialize configuration
        config = ServiceConfig()
        print_result("Configuration", "✅ Loaded successfully")

        # Initialize video processor
        processor = VideoProcessor(config)
        print_result("Video Processor", "✅ Initialized successfully")

        # Create processing configuration
        processing_config = ProcessingConfig(
            max_file_size=500 * 1024 * 1024,  # 500MB
            timeout=600.0,
            extract_metadata=True
        )
        print_result("Processing Config", "✅ Created successfully")
        
        print_section("Processing Video File")
        print("🔄 Starting video processing...")
        print("   This may take a while for large videos...")
        
        # Process the video file
        result = await processor.process_file(video_file, processing_config)
        
        if result.success:
            print("✅ Video processing completed successfully!")
            
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
                
                # Check for transcription content
                if "## Audio Transcription" in result.content:
                    print_result("Audio Transcription", "✅ Found")
                if "## Visual Content" in result.content:
                    print_result("Visual Content", "✅ Found")
                if "## Keyframes" in result.content:
                    print_result("Keyframes", "✅ Found")
            
            if result.summary:
                print_section("Summary")
                print(f"📝 {result.summary}")
            
            # Save results to file
            output_file = video_file.parent / f"{video_file.stem}_test_result.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'success': result.success,
                    'content_type': result.content_type,
                    'processing_time': result.processing_time,
                    'metadata': result.metadata,
                    'content': result.content,
                    'summary': result.summary,
                    'error': result.error
                }, f, indent=2, ensure_ascii=False)
            
            # Also save markdown content
            markdown_file = video_file.parent / f"{video_file.stem}_converted.md"
            with open(markdown_file, 'w', encoding='utf-8') as f:
                f.write(result.content)
            
            print_section("Output")
            print_result("Results saved to", str(output_file))
            print_result("Markdown saved to", str(markdown_file))
            
            return True
            
        else:
            print("❌ Video processing failed!")
            print_result("Error", result.error or "Unknown error")
            return False
            
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
