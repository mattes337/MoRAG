#!/usr/bin/env python3
"""
Test script to debug PDF parsing issues.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_docling_import():
    """Test if docling can be imported."""
    print("🔍 Testing Docling Import")
    print("=" * 30)

    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        print("✅ Docling imports successful")
        return True
    except ImportError as e:
        print(f"❌ Docling import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error importing docling: {e}")
        return False


def test_unstructured_import():
    """Test if unstructured.io can be imported."""
    print("\n🔍 Testing Unstructured.io Import")
    print("=" * 35)

    try:
        from unstructured.partition.auto import partition
        from unstructured.partition.docx import partition_docx
        from unstructured.partition.md import partition_md
        from unstructured.partition.pdf import partition_pdf

        print("✅ Unstructured.io imports successful")
        return True
    except ImportError as e:
        print(f"❌ Unstructured.io import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error importing unstructured.io: {e}")
        return False


def test_document_processor():
    """Test the document processor configuration."""
    print("\n🔍 Testing Document Processor")
    print("=" * 32)

    try:
        from morag_document import UNSTRUCTURED_AVAILABLE, document_processor

        print(f"✅ Document processor imported")
        print(f"   - Unstructured available: {UNSTRUCTURED_AVAILABLE}")
        print(
            f"   - Supported types: {list(document_processor.supported_types.keys())}"
        )
        return True
    except Exception as e:
        print(f"❌ Document processor import failed: {e}")
        return False


def check_pdf_file():
    """Check if there's a test PDF file available."""
    print("\n🔍 Checking for Test PDF Files")
    print("=" * 33)

    # Look for PDF files in common locations
    test_locations = [".", "tests", "test_files", "samples"]

    pdf_files = []
    for location in test_locations:
        path = Path(location)
        if path.exists():
            pdfs = list(path.glob("*.pdf"))
            pdf_files.extend(pdfs)

    if pdf_files:
        print(f"✅ Found {len(pdf_files)} PDF file(s):")
        for pdf in pdf_files[:3]:  # Show first 3
            print(f"   - {pdf}")
        return pdf_files[0]  # Return first PDF for testing
    else:
        print("⚠️  No PDF files found for testing")
        return None


def test_pdf_parsing_with_file(pdf_file):
    """Test PDF parsing with an actual file."""
    print(f"\n🔍 Testing PDF Parsing with {pdf_file.name}")
    print("=" * 50)

    try:
        import asyncio

        from morag_document import document_processor

        async def parse_test():
            try:
                # Test with docling
                print("Testing with docling...")
                result = await document_processor.parse_document(
                    pdf_file, use_docling=True
                )
                print(f"✅ Docling parsing successful:")
                print(f"   - Chunks: {len(result.chunks)}")
                print(f"   - Pages: {result.total_pages}")
                print(f"   - Word count: {result.word_count}")
                print(f"   - Parser: {result.metadata.get('parser', 'unknown')}")

                if result.chunks:
                    first_chunk = result.chunks[0]
                    print(f"   - First chunk preview: {first_chunk.text[:100]}...")

                    # Check if it looks like binary/PDF object data
                    if "obj" in first_chunk.text and "stream" in first_chunk.text:
                        print("   ⚠️  WARNING: First chunk looks like PDF object data!")
                        return False
                    else:
                        print("   ✅ First chunk looks like readable text")
                        return True
                else:
                    print("   ⚠️  WARNING: No chunks extracted!")
                    return False

            except Exception as e:
                print(f"❌ PDF parsing failed: {e}")
                return False

        return asyncio.run(parse_test())

    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        return False


def main():
    """Main test function."""
    print("🧪 PDF Parsing Debug Tool")
    print("=" * 50)

    # Test imports
    docling_ok = test_docling_import()
    unstructured_ok = test_unstructured_import()
    processor_ok = test_document_processor()

    # Find test file
    test_pdf = check_pdf_file()

    # Test parsing if we have a file
    parsing_ok = False
    if test_pdf and processor_ok:
        parsing_ok = test_pdf_parsing_with_file(test_pdf)

    # Summary
    print("\n" + "=" * 50)
    print("🏁 Test Summary:")
    print(f"- Docling Import: {'✅' if docling_ok else '❌'}")
    print(f"- Unstructured.io Import: {'✅' if unstructured_ok else '❌'}")
    print(f"- Document Processor: {'✅' if processor_ok else '❌'}")
    print(f"- Test PDF Available: {'✅' if test_pdf else '❌'}")
    print(f"- PDF Parsing Test: {'✅' if parsing_ok else '❌'}")

    if not docling_ok:
        print("\n💡 To install docling:")
        print("   pip install docling")

    if not unstructured_ok:
        print("\n💡 To install unstructured.io:")
        print("   pip install unstructured[pdf]")

    if not test_pdf:
        print("\n💡 To test with a PDF file:")
        print("   Place a PDF file in the current directory")

    if not parsing_ok and test_pdf:
        print("\n💡 PDF parsing issues detected:")
        print("   1. Check if the PDF is corrupted or password-protected")
        print("   2. Try with a different PDF file")
        print("   3. Check the application logs for detailed error messages")
        print("   4. Verify docling and unstructured.io installations")


if __name__ == "__main__":
    main()
