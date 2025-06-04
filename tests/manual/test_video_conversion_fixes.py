#!/usr/bin/env python3
"""
Test script for video conversion fixes.

This script tests the fixes for:
1. FFmpeg thumbnail generation warnings
2. Quality scoring improvements for video content
3. Fallback converter registration and functionality
4. Enhanced error handling and logging

Usage:
    python test_video_conversion_fixes.py <video_file_path>
"""

import asyncio
import sys
import time
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from morag.converters import DocumentConverter, ConversionOptions
from morag.converters.registry import document_converter
import structlog

# Configure logging
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


async def test_video_conversion_fixes(video_path: Path):
    """Test video conversion fixes."""
    print("🎬 Testing Video Conversion Fixes")
    print("=" * 60)
    print(f"📁 Video file: {video_path}")
    print(f"📊 File size: {video_path.stat().st_size:,} bytes")
    print()
    
    # Test 1: Check converter registration
    print("🔧 Test 1: Converter Registration")
    print("-" * 40)
    
    converter_info = document_converter.get_converter_info()
    if 'video' in converter_info:
        video_info = converter_info['video']
        print(f"✅ Primary converter: {video_info['primary_converter']}")
        print(f"✅ Fallback converters: {video_info['fallback_converters']}")
        print(f"✅ Total fallback options: {len(video_info['fallback_converters'])}")
    else:
        print("❌ No video converter registered")
        return
    print()
    
    # Test 2: Primary converter with quality threshold
    print("🎯 Test 2: Primary Converter with Quality Threshold")
    print("-" * 40)
    
    options = ConversionOptions(
        enable_fallback=True,
        min_quality_threshold=0.8,  # High threshold to trigger fallback
        include_metadata=True,
        format_options={
            'include_audio': True,
            'extract_keyframes': True,
            'thumbnail_count': 3,
            'keyframe_count': 5,
            'optimize_for_speed': True
        }
    )
    
    start_time = time.time()
    try:
        result = await document_converter.convert_to_markdown(video_path, options)
        processing_time = time.time() - start_time
        
        print(f"✅ Conversion successful: {result.success}")
        print(f"✅ Converter used: {result.converter_used}")
        print(f"✅ Fallback used: {result.fallback_used}")
        print(f"✅ Processing time: {processing_time:.2f} seconds")
        print(f"✅ Content length: {len(result.content):,} characters")
        print(f"✅ Word count: {result.word_count:,} words")
        
        if result.quality_score:
            print(f"✅ Quality score: {result.quality_score.overall_score:.2f}")
            print(f"  - Completeness: {result.quality_score.completeness_score:.2f}")
            print(f"  - Readability: {result.quality_score.readability_score:.2f}")
            print(f"  - Structure: {result.quality_score.structure_score:.2f}")
            print(f"  - Metadata: {result.quality_score.metadata_preservation:.2f}")
        
        if result.warnings:
            print(f"⚠️  Warnings ({len(result.warnings)}):")
            for warning in result.warnings:
                print(f"  - {warning}")
        
        if result.error_message:
            print(f"❌ Error: {result.error_message}")
        
        # Save result for inspection
        output_path = video_path.parent / f"{video_path.stem}_conversion_test.md"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result.content)
        print(f"✅ Output saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    
    # Test 3: Force fallback by using very high quality threshold
    print("🔄 Test 3: Force Fallback Converter")
    print("-" * 40)
    
    fallback_options = ConversionOptions(
        enable_fallback=True,
        min_quality_threshold=0.99,  # Very high threshold to force fallback
        include_metadata=True,
        format_options={
            'include_audio': True,
            'extract_keyframes': False,  # Disable for faster processing
            'thumbnail_count': 0,
            'optimize_for_speed': True
        }
    )
    
    start_time = time.time()
    try:
        result = await document_converter.convert_to_markdown(video_path, fallback_options)
        processing_time = time.time() - start_time
        
        print(f"✅ Fallback conversion successful: {result.success}")
        print(f"✅ Converter used: {result.converter_used}")
        print(f"✅ Fallback used: {result.fallback_used}")
        print(f"✅ Processing time: {processing_time:.2f} seconds")
        print(f"✅ Content length: {len(result.content):,} characters")
        print(f"✅ Word count: {result.word_count:,} words")
        
        if result.quality_score:
            print(f"✅ Quality score: {result.quality_score.overall_score:.2f}")
        
        # Save fallback result for comparison
        fallback_output_path = video_path.parent / f"{video_path.stem}_fallback_test.md"
        with open(fallback_output_path, 'w', encoding='utf-8') as f:
            f.write(result.content)
        print(f"✅ Fallback output saved to: {fallback_output_path}")
        
    except Exception as e:
        print(f"❌ Fallback conversion failed: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    
    # Test 4: Test individual converters
    print("🧪 Test 4: Individual Converter Testing")
    print("-" * 40)
    
    # Test primary video converter
    try:
        from morag.converters.video import VideoConverter
        video_converter = VideoConverter()
        
        start_time = time.time()
        result = await video_converter.convert(video_path, options)
        processing_time = time.time() - start_time
        
        print(f"✅ Primary VideoConverter: {result.success}")
        print(f"  - Processing time: {processing_time:.2f} seconds")
        print(f"  - Quality score: {result.quality_score.overall_score if result.quality_score else 'N/A'}")
        
    except Exception as e:
        print(f"❌ Primary VideoConverter failed: {e}")
    
    # Test simple video converter
    try:
        from morag.converters.simple_video import SimpleVideoConverter
        simple_converter = SimpleVideoConverter()
        
        start_time = time.time()
        result = await simple_converter.convert(video_path, options)
        processing_time = time.time() - start_time
        
        print(f"✅ SimpleVideoConverter: {result.success}")
        print(f"  - Processing time: {processing_time:.2f} seconds")
        print(f"  - Quality score: {result.quality_score.overall_score if result.quality_score else 'N/A'}")
        
    except Exception as e:
        print(f"❌ SimpleVideoConverter failed: {e}")
    
    # Test audio converter as fallback
    try:
        from morag.converters.audio import AudioConverter
        audio_converter = AudioConverter()
        
        start_time = time.time()
        result = await audio_converter.convert(video_path, options)
        processing_time = time.time() - start_time
        
        print(f"✅ AudioConverter (fallback): {result.success}")
        print(f"  - Processing time: {processing_time:.2f} seconds")
        print(f"  - Quality score: {result.quality_score.overall_score if result.quality_score else 'N/A'}")
        
    except Exception as e:
        print(f"❌ AudioConverter fallback failed: {e}")
    
    print()
    print("🎉 Video conversion fixes testing completed!")


async def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python test_video_conversion_fixes.py <video_file_path>")
        sys.exit(1)
    
    video_path = Path(sys.argv[1])
    
    if not video_path.exists():
        print(f"❌ Error: Video file not found: {video_path}")
        sys.exit(1)
    
    if not video_path.is_file():
        print(f"❌ Error: Path is not a file: {video_path}")
        sys.exit(1)
    
    try:
        await test_video_conversion_fixes(video_path)
    except KeyboardInterrupt:
        print("\n⏹️  Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
