#!/usr/bin/env python3
"""
Universal Document Conversion Test Script

This script tests the universal document conversion pipeline by:
1. Taking a file path as argument
2. Detecting the document format automatically
3. Converting the document to markdown
4. Saving the output with a suffix
5. Displaying conversion metrics and quality scores

Usage:
    python test_universal_conversion.py <file_path> [options]

Examples:
    python test_universal_conversion.py document.pdf
    python test_universal_conversion.py audio.mp3 --chunking-strategy sentence
    python test_universal_conversion.py presentation.pptx --include-metadata
"""

import asyncio
import argparse
import sys
import time
from pathlib import Path
from typing import Optional
import json

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morag.converters import (
    DocumentConverter,
    ConversionOptions,
    ChunkingStrategy,
    ConversionError,
    UnsupportedFormatError
)
from morag.converters.config import get_conversion_config
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


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test universal document conversion pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s document.pdf
  %(prog)s audio.mp3 --chunking-strategy sentence
  %(prog)s presentation.pptx --include-metadata --output-suffix _converted
        """
    )
    
    parser.add_argument(
        "file_path",
        type=str,
        help="Path to the document to convert"
    )
    
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="_converted",
        help="Suffix to add to output filename (default: _converted)"
    )
    
    parser.add_argument(
        "--chunking-strategy",
        type=str,
        choices=[strategy.value for strategy in ChunkingStrategy],
        default="page",
        help="Chunking strategy to use (default: page)"
    )
    
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Include metadata in conversion"
    )
    
    parser.add_argument(
        "--extract-images",
        action="store_true",
        help="Extract images during conversion"
    )
    
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.7,
        help="Minimum quality threshold (default: 0.7)"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (default: same as input file)"
    )
    
    return parser.parse_args()


def create_output_path(input_path: Path, suffix: str, output_dir: Optional[str] = None) -> Path:
    """Create output file path with suffix."""
    if output_dir:
        output_directory = Path(output_dir)
        output_directory.mkdir(parents=True, exist_ok=True)
    else:
        output_directory = input_path.parent
    
    # Create output filename: original_name + suffix + .md
    stem = input_path.stem
    output_filename = f"{stem}{suffix}.md"
    
    return output_directory / output_filename


def print_conversion_info(converter: DocumentConverter, file_path: Path):
    """Print information about the converter and file."""
    print("🔄 Universal Document Conversion Test")
    print("=" * 50)
    print(f"📁 Input file: {file_path}")
    print(f"📊 File size: {file_path.stat().st_size:,} bytes")
    
    # Detect format
    detected_format = converter.detect_format(file_path)
    print(f"🔍 Detected format: {detected_format}")
    
    # Show supported formats
    supported_formats = converter.list_supported_formats()
    print(f"📋 Supported formats: {', '.join(supported_formats)}")
    
    # Check if format is supported
    if detected_format not in supported_formats:
        print(f"⚠️  Warning: Format '{detected_format}' may not be fully supported")
    
    print()


def print_conversion_options(options: ConversionOptions):
    """Print conversion options."""
    print("⚙️  Conversion Options:")
    print(f"  • Chunking strategy: {options.chunking_strategy.value}")
    print(f"  • Include metadata: {options.include_metadata}")
    print(f"  • Extract images: {options.extract_images}")
    print(f"  • Quality threshold: {options.min_quality_threshold}")
    print(f"  • Enable fallback: {options.enable_fallback}")
    
    if options.format_options:
        print("  • Format-specific options:")
        for key, value in options.format_options.items():
            print(f"    - {key}: {value}")
    print()


def print_conversion_result(result, processing_time: float, output_path: Path):
    """Print conversion results."""
    print("📊 Conversion Results:")
    print(f"  • Success: {'✅' if result.success else '❌'}")
    print(f"  • Processing time: {processing_time:.2f} seconds")
    print(f"  • Converter used: {result.converter_used or 'Unknown'}")
    print(f"  • Fallback used: {'Yes' if result.fallback_used else 'No'}")
    print(f"  • Content length: {len(result.content):,} characters")
    print(f"  • Word count: {result.word_count:,} words")
    
    if result.quality_score:
        print(f"  • Quality score: {result.quality_score.overall_score:.2f}")
        print(f"    - Completeness: {result.quality_score.completeness_score:.2f}")
        print(f"    - Readability: {result.quality_score.readability_score:.2f}")
        print(f"    - Structure: {result.quality_score.structure_score:.2f}")
        print(f"    - Metadata preservation: {result.quality_score.metadata_preservation:.2f}")
        
        # Quality interpretation
        if result.quality_score.overall_score >= 0.8:
            print("    🟢 High quality conversion")
        elif result.quality_score.overall_score >= 0.6:
            print("    🟡 Acceptable quality conversion")
        else:
            print("    🔴 Low quality conversion")
    
    if result.warnings:
        print(f"  • Warnings ({len(result.warnings)}):")
        for warning in result.warnings:
            print(f"    - {warning}")
    
    if result.error_message:
        print(f"  • Error: {result.error_message}")
    
    print(f"  • Output saved to: {output_path}")
    print()


def save_conversion_metadata(result, output_path: Path, processing_time: float):
    """Save conversion metadata to JSON file."""
    metadata_path = output_path.with_suffix('.json')
    
    metadata = {
        "conversion_info": {
            "success": result.success,
            "processing_time": processing_time,
            "converter_used": result.converter_used,
            "fallback_used": result.fallback_used,
            "original_format": result.original_format,
            "target_format": result.target_format,
            "timestamp": time.time()
        },
        "content_stats": {
            "content_length": len(result.content),
            "word_count": result.word_count,
            "line_count": len(result.content.splitlines()) if result.content else 0
        },
        "quality_metrics": {
            "overall_score": result.quality_score.overall_score if result.quality_score else None,
            "completeness_score": result.quality_score.completeness_score if result.quality_score else None,
            "readability_score": result.quality_score.readability_score if result.quality_score else None,
            "structure_score": result.quality_score.structure_score if result.quality_score else None,
            "metadata_preservation": result.quality_score.metadata_preservation if result.quality_score else None,
            "is_high_quality": result.is_high_quality
        },
        "document_metadata": result.metadata,
        "warnings": result.warnings,
        "error_message": result.error_message,
        "images": result.images
    }
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"📄 Metadata saved to: {metadata_path}")


async def test_conversion(args):
    """Test document conversion with the given arguments."""
    # Validate input file
    input_path = Path(args.file_path)
    if not input_path.exists():
        print(f"❌ Error: File not found: {input_path}")
        return 1
    
    if not input_path.is_file():
        print(f"❌ Error: Path is not a file: {input_path}")
        return 1
    
    # Create output path
    output_path = create_output_path(input_path, args.output_suffix, args.output_dir)
    
    # Initialize converter
    converter = DocumentConverter()
    
    # Print initial information
    print_conversion_info(converter, input_path)
    
    # Create conversion options
    try:
        detected_format = converter.detect_format(input_path)
        options = ConversionOptions.for_format(detected_format)
        
        # Apply command line options
        options.chunking_strategy = ChunkingStrategy(args.chunking_strategy)
        options.include_metadata = args.include_metadata
        options.extract_images = args.extract_images
        options.min_quality_threshold = args.quality_threshold
        
        print_conversion_options(options)
        
    except Exception as e:
        print(f"❌ Error creating conversion options: {e}")
        return 1
    
    # Perform conversion
    try:
        print("🚀 Starting conversion...")
        start_time = time.time()
        
        result = await converter.convert_to_markdown(input_path, options)
        
        processing_time = time.time() - start_time
        
        # Save markdown content
        if result.success and result.content:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result.content)
            
            # Save metadata
            save_conversion_metadata(result, output_path, processing_time)
            
            print_conversion_result(result, processing_time, output_path)
            
            if args.verbose and result.metadata:
                print("📋 Document Metadata:")
                for key, value in result.metadata.items():
                    print(f"  • {key}: {value}")
                print()
            
            return 0 if result.success else 1
        else:
            print(f"❌ Conversion failed: {result.error_message or 'Unknown error'}")
            return 1
            
    except UnsupportedFormatError as e:
        print(f"❌ Unsupported format: {e}")
        return 1
    except ConversionError as e:
        print(f"❌ Conversion error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


async def main():
    """Main function."""
    args = parse_arguments()
    
    try:
        exit_code = await test_conversion(args)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️  Conversion interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
