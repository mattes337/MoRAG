#!/usr/bin/env python3
"""
Universal Document Converter Demo

This script demonstrates the universal document conversion capabilities
implemented in Task 24. It shows how to convert various document formats
to structured markdown using the unified conversion framework.
"""

import asyncio
import sys
from pathlib import Path
import tempfile
import json

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morag_services import (
    DocumentConverter,
    ConversionOptions,
    ChunkingStrategy
)
from morag_core.models import get_conversion_config, create_sample_config


async def demo_basic_conversion():
    """Demonstrate basic document conversion."""
    print("🔄 Universal Document Converter Demo")
    print("=" * 50)
    
    # Initialize the converter
    converter = DocumentConverter()
    
    # Show supported formats
    print(f"📋 Supported formats: {', '.join(converter.list_supported_formats())}")
    print()
    
    # Show registered converters
    print("🔧 Registered converters:")
    converter_info = converter.get_converter_info()
    for format_type, info in converter_info.items():
        print(f"  • {format_type}: {info['primary_converter']}")
        if info['fallback_converters']:
            print(f"    Fallbacks: {', '.join(info['fallback_converters'])}")
    print()


async def demo_pdf_conversion():
    """Demonstrate PDF conversion with different options."""
    print("📄 PDF Conversion Demo")
    print("-" * 30)
    
    # Create a sample PDF file (placeholder)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(b"Sample PDF content")
        pdf_path = Path(tmp.name)
    
    try:
        converter = DocumentConverter()
        
        # Test format detection
        detected_format = converter.detect_format(pdf_path)
        print(f"Detected format: {detected_format}")
        
        # Create conversion options
        options = ConversionOptions.for_format('pdf')
        options.chunking_strategy = ChunkingStrategy.PAGE
        options.include_metadata = True
        options.include_toc = True
        
        print(f"Conversion options: {options.__dict__}")
        print()
        
        # Note: This would fail in demo because we don't have actual PDF processing
        # but shows the API usage
        print("Note: Actual conversion would require proper PDF file and MoRAG processors")
        
    finally:
        pdf_path.unlink()


async def demo_configuration():
    """Demonstrate configuration system."""
    print("⚙️  Configuration Demo")
    print("-" * 25)
    
    # Get current configuration
    config = get_conversion_config()
    
    print("Default options:")
    for key, value in config.default_options.items():
        print(f"  • {key}: {value}")
    print()
    
    print("Format-specific options:")
    for format_type, options in config.format_specific.items():
        print(f"  • {format_type}:")
        for key, value in options.items():
            print(f"    - {key}: {value}")
    print()
    
    # Create sample configuration file
    sample_config_path = Path("sample_conversion_config.yaml")
    create_sample_config(sample_config_path)
    print(f"📝 Created sample configuration: {sample_config_path}")
    
    # Clean up
    if sample_config_path.exists():
        sample_config_path.unlink()


async def demo_conversion_options():
    """Demonstrate different conversion options."""
    print("🎛️  Conversion Options Demo")
    print("-" * 30)
    
    # Show different chunking strategies
    print("Available chunking strategies:")
    for strategy in ChunkingStrategy:
        print(f"  • {strategy.value}")
    print()
    
    # Create options for different formats
    formats = ['pdf', 'audio', 'video', 'web']
    
    for format_type in formats:
        print(f"{format_type.upper()} options:")
        options = ConversionOptions.for_format(format_type)
        
        print(f"  • Chunking strategy: {options.chunking_strategy.value}")
        print(f"  • Quality threshold: {options.min_quality_threshold}")
        print(f"  • Enable fallback: {options.enable_fallback}")
        
        if options.format_options:
            print("  • Format-specific options:")
            for key, value in options.format_options.items():
                print(f"    - {key}: {value}")
        print()


async def demo_quality_assessment():
    """Demonstrate quality assessment features."""
    print("📊 Quality Assessment Demo")
    print("-" * 30)
    
    from src.morag.converters.quality import ConversionQualityValidator
    from morag_core.interfaces.converter import ConversionResult, QualityScore
    
    validator = ConversionQualityValidator()
    
    # Create a sample conversion result
    sample_content = """
# Sample Document

This is a sample document with proper structure.

## Introduction

This document demonstrates the quality assessment capabilities
of the universal document conversion system.

### Key Features

- Structured content
- Proper headings
- Readable text
- Good formatting

## Conclusion

The quality assessment system evaluates multiple factors
to determine conversion quality.
"""
    
    sample_result = ConversionResult(
        content=sample_content,
        metadata={
            'title': 'Sample Document',
            'word_count': len(sample_content.split()),
            'format_type': 'demo'
        },
        success=True
    )
    
    # Assess quality
    quality_score = validator.validate_conversion("demo_file.txt", sample_result)
    
    print("Quality assessment results:")
    print(f"  • Overall score: {quality_score.overall_score:.2f}")
    print(f"  • Completeness: {quality_score.completeness_score:.2f}")
    print(f"  • Readability: {quality_score.readability_score:.2f}")
    print(f"  • Structure: {quality_score.structure_score:.2f}")
    print(f"  • Metadata preservation: {quality_score.metadata_preservation:.2f}")
    print()
    
    # Show quality interpretation
    if quality_score.overall_score >= 0.8:
        print("✅ High quality conversion")
    elif quality_score.overall_score >= 0.6:
        print("⚠️  Acceptable quality conversion")
    else:
        print("❌ Low quality conversion - may need fallback")


async def main():
    """Run all demos."""
    print("🚀 Universal Document Conversion Framework Demo")
    print("=" * 60)
    print()
    
    try:
        await demo_basic_conversion()
        await demo_pdf_conversion()
        await demo_configuration()
        await demo_conversion_options()
        await demo_quality_assessment()
        
        print("✅ Demo completed successfully!")
        print()
        print("Next steps:")
        print("1. Implement enhanced converters for Tasks 25-29")
        print("2. Add real document processing capabilities")
        print("3. Integrate with MoRAG API endpoints")
        print("4. Add comprehensive testing")
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
