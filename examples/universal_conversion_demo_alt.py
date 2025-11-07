#!/usr/bin/env python3
"""
Universal Document Conversion Demo Script

This script demonstrates various usage examples of the universal document conversion pipeline.
It shows how to convert different file types with different options.

Usage:
    python demo_universal_conversion.py
"""

import asyncio
import sys
import tempfile
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morag_core.interfaces.converter import ChunkingStrategy, ConversionOptions
from morag_document.converters import DocumentConverter


async def demo_text_conversion():
    """Demonstrate text file conversion with different chunking strategies."""
    print("📝 Text Conversion Demo")
    print("=" * 30)

    # Create a sample text file
    sample_text = """Sample Document

This is a demonstration of text conversion. The system can handle various chunking strategies.

Features:
- Automatic format detection
- Quality assessment
- Multiple chunking options

This shows how the universal converter works with plain text files."""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(sample_text)
        text_file = Path(f.name)

    try:
        converter = DocumentConverter()

        # Test different chunking strategies
        strategies = ["page", "paragraph", "sentence"]

        for strategy in strategies:
            print(f"\n🔄 Converting with {strategy} chunking...")

            options = ConversionOptions(
                chunking_strategy=ChunkingStrategy(strategy), include_metadata=True
            )

            result = await converter.convert_to_markdown(text_file, options)

            if result.success:
                print(f"✅ Success! Quality: {result.quality_score.overall_score:.2f}")
                print(f"   Word count: {result.word_count}")
                print(f"   Content length: {len(result.content)} chars")
            else:
                print(f"❌ Failed: {result.error_message}")

    finally:
        text_file.unlink()


async def demo_pdf_conversion():
    """Demonstrate PDF conversion if files are available."""
    print("\n📄 PDF Conversion Demo")
    print("=" * 30)

    # Look for PDF files in uploads directory
    uploads_dir = Path(__file__).parent.parent / "uploads"
    pdf_files = list(uploads_dir.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in uploads directory")
        return

    # Use the first PDF file found
    pdf_file = pdf_files[0]
    print(f"Converting: {pdf_file.name}")

    converter = DocumentConverter()

    options = ConversionOptions(
        chunking_strategy=ChunkingStrategy.PAGE,
        include_metadata=True,
        extract_images=False,  # Skip images for demo
    )

    print("🔄 Converting PDF (this may take a while)...")
    result = await converter.convert_to_markdown(pdf_file, options)

    if result.success:
        print(f"✅ Success! Quality: {result.quality_score.overall_score:.2f}")
        print(f"   Processing time: {result.processing_time:.2f}s")
        print(f"   Word count: {result.word_count:,}")
        print(f"   Content length: {len(result.content):,} chars")
        print(f"   Converter used: {result.converter_used}")
        print(f"   Fallback used: {result.fallback_used}")
    else:
        print(f"❌ Failed: {result.error_message}")


async def demo_format_detection():
    """Demonstrate automatic format detection."""
    print("\n🔍 Format Detection Demo")
    print("=" * 30)

    converter = DocumentConverter()

    # Test various file extensions
    test_files = [
        "document.pdf",
        "spreadsheet.xlsx",
        "presentation.pptx",
        "audio.mp3",
        "video.mp4",
        "webpage.html",
        "text.txt",
        "markdown.md",
    ]

    print("Format detection results:")
    for filename in test_files:
        detected = converter.detect_format(filename)
        print(f"  {filename:<20} → {detected}")


async def demo_converter_info():
    """Show information about available converters."""
    print("\n🔧 Available Converters")
    print("=" * 30)

    converter = DocumentConverter()

    print("Supported formats:")
    formats = converter.list_supported_formats()
    for fmt in sorted(formats):
        print(f"  • {fmt}")

    print("\nConverter details:")
    info = converter.get_converter_info()
    for format_type, details in info.items():
        print(f"  {format_type}:")
        print(f"    Primary: {details['primary_converter']}")
        if details["fallback_converters"]:
            print(f"    Fallbacks: {', '.join(details['fallback_converters'])}")


async def demo_quality_assessment():
    """Demonstrate quality assessment features."""
    print("\n📊 Quality Assessment Demo")
    print("=" * 30)

    # Create sample content with different quality levels
    samples = [
        (
            "High Quality",
            "# Document\n\n## Section 1\n\nWell structured content with proper headings.\n\n## Section 2\n\nMore content here.",
        ),
        (
            "Medium Quality",
            "Document\n\nSome content without proper structure.\n\nMore text here.",
        ),
        ("Low Quality", "just some text without any structure at all"),
    ]

    converter = DocumentConverter()

    for name, content in samples:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            temp_file = Path(f.name)

        try:
            result = await converter.convert_to_markdown(temp_file)

            if result.success and result.quality_score:
                print(f"\n{name}:")
                print(f"  Overall: {result.quality_score.overall_score:.2f}")
                print(f"  Completeness: {result.quality_score.completeness_score:.2f}")
                print(f"  Readability: {result.quality_score.readability_score:.2f}")
                print(f"  Structure: {result.quality_score.structure_score:.2f}")

                if result.quality_score.overall_score >= 0.8:
                    print("  🟢 High quality")
                elif result.quality_score.overall_score >= 0.6:
                    print("  🟡 Acceptable quality")
                else:
                    print("  🔴 Low quality")

        finally:
            temp_file.unlink()


async def main():
    """Run all demonstrations."""
    print("🚀 Universal Document Conversion Demo")
    print("=" * 50)

    try:
        await demo_format_detection()
        await demo_converter_info()
        await demo_text_conversion()
        await demo_quality_assessment()
        await demo_pdf_conversion()

        print("\n✅ Demo completed successfully!")
        print("\nTo test with your own files, use:")
        print("  python test_universal_conversion.py <your_file> [options]")

    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
