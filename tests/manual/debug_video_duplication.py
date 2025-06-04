#!/usr/bin/env python3
"""
Debug Video Duplication Issue

This script helps debug the video conversion duplication issue by:
1. Converting a video file with extensive logging
2. Analyzing the output for duplication patterns
3. Providing detailed diagnostics

Usage:
    python debug_video_duplication.py <video_file_path>
"""

import asyncio
import argparse
import sys
import time
from pathlib import Path
from typing import Optional
import json
import re

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morag.converters import (
    DocumentConverter,
    ConversionOptions,
    ChunkingStrategy,
    ConversionError,
    UnsupportedFormatError
)
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


def analyze_duplication(content: str) -> dict:
    """Analyze the content for duplication patterns."""
    lines = content.split('\n')
    
    # Count line occurrences
    line_counts = {}
    for line in lines:
        stripped = line.strip()
        if stripped:
            line_counts[stripped] = line_counts.get(stripped, 0) + 1
    
    # Find duplicates
    duplicates = {line: count for line, count in line_counts.items() if count > 1}
    
    # Find the most duplicated line
    most_duplicated = None
    max_count = 0
    if duplicates:
        most_duplicated = max(duplicates.items(), key=lambda x: x[1])
        max_count = most_duplicated[1]
    
    # Analyze patterns
    patterns = {}
    
    # Look for repeated speaker patterns
    speaker_pattern = re.compile(r'^(SPEAKER_\d+): (.+)$')
    speaker_text_counts = {}
    
    for line in lines:
        match = speaker_pattern.match(line.strip())
        if match:
            speaker, text = match.groups()
            key = text.strip()
            speaker_text_counts[key] = speaker_text_counts.get(key, 0) + 1
    
    speaker_duplicates = {text: count for text, count in speaker_text_counts.items() if count > 1}
    
    return {
        'total_lines': len(lines),
        'unique_lines': len(line_counts),
        'duplicate_lines': len(duplicates),
        'duplicates': duplicates,
        'most_duplicated': most_duplicated,
        'max_duplication_count': max_count,
        'speaker_duplicates': speaker_duplicates,
        'speaker_duplicate_count': len(speaker_duplicates)
    }


def print_duplication_analysis(analysis: dict):
    """Print duplication analysis results."""
    print("\n🔍 DUPLICATION ANALYSIS")
    print("=" * 50)
    print(f"📊 Total lines: {analysis['total_lines']:,}")
    print(f"📊 Unique lines: {analysis['unique_lines']:,}")
    print(f"📊 Duplicate lines: {analysis['duplicate_lines']:,}")
    
    if analysis['most_duplicated']:
        line, count = analysis['most_duplicated']
        print(f"🔴 Most duplicated line ({count} times):")
        print(f"   '{line[:100]}{'...' if len(line) > 100 else ''}'")
    
    if analysis['speaker_duplicates']:
        print(f"\n🎤 Speaker text duplicates: {analysis['speaker_duplicate_count']}")
        # Show top 5 most duplicated speaker texts
        sorted_speaker_dupes = sorted(analysis['speaker_duplicates'].items(), 
                                    key=lambda x: x[1], reverse=True)[:5]
        for text, count in sorted_speaker_dupes:
            print(f"   {count}x: '{text[:80]}{'...' if len(text) > 80 else ''}'")
    
    # Duplication severity assessment
    duplication_ratio = analysis['duplicate_lines'] / analysis['unique_lines'] if analysis['unique_lines'] > 0 else 0
    if duplication_ratio > 0.5:
        print("🚨 SEVERE DUPLICATION DETECTED")
    elif duplication_ratio > 0.2:
        print("⚠️  MODERATE DUPLICATION DETECTED")
    elif duplication_ratio > 0.05:
        print("🟡 MINOR DUPLICATION DETECTED")
    else:
        print("✅ NO SIGNIFICANT DUPLICATION")


async def debug_video_conversion(video_path: Path):
    """Debug video conversion with detailed logging."""
    print("🐛 Video Duplication Debug Tool")
    print("=" * 50)
    print(f"📁 Input file: {video_path}")
    print(f"📊 File size: {video_path.stat().st_size:,} bytes")
    
    # Create output path
    output_path = video_path.parent / f"{video_path.stem}_debug_converted.md"
    metadata_path = video_path.parent / f"{video_path.stem}_debug_converted.json"
    
    # Initialize converter
    converter = DocumentConverter()
    
    # Create conversion options with debugging enabled
    options = ConversionOptions.for_format('video')
    options.chunking_strategy = ChunkingStrategy.PAGE
    options.include_metadata = True
    options.min_quality_threshold = 0.5  # Lower threshold to avoid fallbacks
    
    print("\n⚙️  Conversion Options:")
    print(f"  • Chunking strategy: {options.chunking_strategy.value}")
    print(f"  • Quality threshold: {options.min_quality_threshold}")
    print(f"  • Enable fallback: {options.enable_fallback}")
    
    try:
        print("\n🚀 Starting conversion with debug logging...")
        start_time = time.time()
        
        result = await converter.convert_to_markdown(video_path, options)
        
        processing_time = time.time() - start_time
        
        if result.success and result.content:
            # Save the content
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result.content)
            
            # Analyze for duplication
            analysis = analyze_duplication(result.content)
            
            # Save metadata with analysis
            metadata = {
                "conversion_info": {
                    "success": result.success,
                    "processing_time": processing_time,
                    "converter_used": result.converter_used,
                    "fallback_used": result.fallback_used,
                    "content_length": len(result.content),
                    "word_count": result.word_count
                },
                "duplication_analysis": analysis,
                "quality_score": {
                    "overall_score": result.quality_score.overall_score if result.quality_score else None,
                    "is_high_quality": result.is_high_quality
                },
                "warnings": result.warnings,
                "error_message": result.error_message
            }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            # Print results
            print(f"\n📊 Conversion Results:")
            print(f"  • Success: {'✅' if result.success else '❌'}")
            print(f"  • Processing time: {processing_time:.2f} seconds")
            print(f"  • Converter used: {result.converter_used}")
            print(f"  • Content length: {len(result.content):,} characters")
            print(f"  • Word count: {result.word_count:,} words")
            
            if result.quality_score:
                print(f"  • Quality score: {result.quality_score.overall_score:.2f}")
            
            # Print duplication analysis
            print_duplication_analysis(analysis)
            
            print(f"\n📄 Output saved to: {output_path}")
            print(f"📄 Metadata saved to: {metadata_path}")
            
            # If severe duplication detected, show sample
            if analysis['max_duplication_count'] > 10:
                print(f"\n🚨 SEVERE DUPLICATION DETECTED!")
                print(f"The line appears {analysis['max_duplication_count']} times:")
                print(f"'{analysis['most_duplicated'][0][:200]}{'...' if len(analysis['most_duplicated'][0]) > 200 else ''}'")
            
            return 0
        else:
            print(f"❌ Conversion failed: {result.error_message or 'Unknown error'}")
            return 1
            
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Debug video duplication issues")
    parser.add_argument("video_path", type=str, help="Path to video file")
    
    args = parser.parse_args()
    
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"❌ Error: File not found: {video_path}")
        sys.exit(1)
    
    if not video_path.is_file():
        print(f"❌ Error: Path is not a file: {video_path}")
        sys.exit(1)
    
    try:
        exit_code = asyncio.run(debug_video_conversion(video_path))
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️  Debug interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
