#!/usr/bin/env python3
"""
Quick Video Test - Test video conversion with timeout to see logging

This script tests video conversion with a timeout to capture logging output
and identify where the duplication issue occurs.
"""

import asyncio
import sys
import time
import signal
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morag.converters import DocumentConverter, ConversionOptions, ChunkingStrategy
import structlog

# Configure detailed logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Conversion timed out")


async def quick_video_test(video_path: Path, timeout_seconds: int = 300):
    """Test video conversion with timeout."""
    print("🎥 Quick Video Conversion Test")
    print("=" * 50)
    print(f"📁 Input file: {video_path}")
    
    if not video_path.exists():
        print(f"❌ File not found: {video_path}")
        return 1
    
    print(f"📊 File size: {video_path.stat().st_size:,} bytes")
    
    # Create output path
    output_path = video_path.parent / f"{video_path.stem}_quick_test.md"
    
    # Initialize converter
    converter = DocumentConverter()
    
    # Create conversion options
    options = ConversionOptions.for_format('video')
    options.chunking_strategy = ChunkingStrategy.PAGE
    options.min_quality_threshold = 0.5
    
    print(f"\n⚙️  Starting conversion with {timeout_seconds}s timeout...")
    print("🔍 Watch for logging output to identify duplication source...")
    
    try:
        # Set up timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        
        start_time = time.time()
        result = await converter.convert_to_markdown(video_path, options)
        
        # Cancel timeout
        signal.alarm(0)
        
        processing_time = time.time() - start_time
        
        if result.success and result.content:
            # Save partial content
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result.content)
            
            print(f"\n📊 Conversion completed:")
            print(f"  • Processing time: {processing_time:.2f} seconds")
            print(f"  • Content length: {len(result.content):,} characters")
            print(f"  • Word count: {result.word_count:,} words")
            print(f"  • Output saved to: {output_path}")
            
            # Quick duplication check
            lines = result.content.split('\n')
            line_counts = {}
            for line in lines:
                stripped = line.strip()
                if stripped:
                    line_counts[stripped] = line_counts.get(stripped, 0) + 1
            
            duplicates = {line: count for line, count in line_counts.items() if count > 5}
            if duplicates:
                print(f"\n🚨 DUPLICATION DETECTED:")
                most_duplicated = max(duplicates.items(), key=lambda x: x[1])
                print(f"  • Most duplicated line: {most_duplicated[1]} times")
                print(f"  • Line: '{most_duplicated[0][:100]}{'...' if len(most_duplicated[0]) > 100 else ''}'")
            else:
                print(f"\n✅ No significant duplication detected")
            
            return 0
        else:
            print(f"❌ Conversion failed: {result.error_message or 'Unknown error'}")
            return 1
            
    except TimeoutError:
        print(f"\n⏰ Conversion timed out after {timeout_seconds} seconds")
        print("This suggests the duplication might be happening in an infinite loop")
        return 1
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Make sure to cancel any remaining alarm
        signal.alarm(0)


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python quick_video_test.py <video_file>")
        sys.exit(1)
    
    video_path = Path(sys.argv[1])
    
    try:
        exit_code = asyncio.run(quick_video_test(video_path, timeout_seconds=180))  # 3 minutes
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
