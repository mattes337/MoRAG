#!/usr/bin/env python3
"""
Manual testing script for document processing functionality.
This script tests the document processor without requiring the full pipeline.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morag.processors.document import document_processor
from morag.core.config import settings

async def test_markdown_processing():
    """Test processing of a markdown document."""
    print("=" * 60)
    print("Testing Markdown Document Processing")
    print("=" * 60)
    
    # Create a sample markdown file
    test_content = """
# Sample Document for Testing

This is a sample document for testing the document processor.

## Introduction

The document processor handles various file formats including PDF, DOCX, and Markdown.
It extracts text, identifies different element types, and prepares content for embedding.

### Key Features

- **Document parsing**: Supports multiple formats
- **Text extraction**: Clean text extraction from documents
- **Metadata extraction**: Captures document metadata
- **Image detection**: Identifies images for separate processing

## Technical Details

The processor uses unstructured.io as the primary parsing library, with docling as an
alternative for PDF processing. Each document is broken down into chunks that can be
processed independently.

### Processing Pipeline

1. File validation
2. Document parsing
3. Element extraction
4. Chunk creation
5. Metadata association

## Tables Example

| Feature | Status | Notes |
|---------|--------|-------|
| PDF Support | ✅ | Via unstructured.io |
| DOCX Support | ✅ | Native support |
| Markdown | ✅ | Full support |
| Images | 🔄 | Detection only |

## Conclusion

This completes the sample document for testing purposes.
The processor should extract all text content and identify the table structure.
    """
    
    # Write to temporary file
    test_file = Path("test_sample.md")
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(test_content)
    
    try:
        print(f"Processing file: {test_file}")
        result = await document_processor.parse_document(test_file)
        
        print(f"\n📊 Processing Results:")
        print(f"   Chunks found: {len(result.chunks)}")
        print(f"   Word count: {result.word_count}")
        print(f"   Images found: {len(result.images)}")
        print(f"   Total pages: {result.total_pages}")
        print(f"   Parser used: {result.metadata.get('parser', 'unknown')}")
        
        print(f"\n📝 Document Metadata:")
        for key, value in result.metadata.items():
            print(f"   {key}: {value}")
        
        print(f"\n📄 Sample Chunks (first 3):")
        for i, chunk in enumerate(result.chunks[:3]):
            print(f"\n   Chunk {i+1}:")
            print(f"   Type: {chunk.chunk_type}")
            print(f"   Page: {chunk.page_number}")
            print(f"   Text: {chunk.text[:100]}{'...' if len(chunk.text) > 100 else ''}")
            if chunk.metadata:
                print(f"   Metadata: {chunk.metadata}")
        
        if len(result.chunks) > 3:
            print(f"\n   ... and {len(result.chunks) - 3} more chunks")
            
        return True
        
    except Exception as e:
        print(f"❌ Error processing markdown: {e}")
        return False
        
    finally:
        # Clean up
        if test_file.exists():
            test_file.unlink()

async def test_text_processing():
    """Test processing of a plain text document."""
    print("\n" + "=" * 60)
    print("Testing Text Document Processing")
    print("=" * 60)
    
    test_content = """
This is a plain text document for testing.

It contains multiple paragraphs with different types of content.
Some paragraphs are longer and contain more detailed information
about the testing process and what we expect to see.

Here's a list of items:
- First item
- Second item  
- Third item with more details

And here's some technical information:
The system should be able to handle various text formats and extract
meaningful chunks that can be processed by the embedding system.

Final paragraph with conclusion.
    """
    
    test_file = Path("test_sample.txt")
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(test_content)
    
    try:
        print(f"Processing file: {test_file}")
        result = await document_processor.parse_document(test_file)
        
        print(f"\n📊 Processing Results:")
        print(f"   Chunks found: {len(result.chunks)}")
        print(f"   Word count: {result.word_count}")
        print(f"   Parser used: {result.metadata.get('parser', 'unknown')}")
        
        print(f"\n📄 All Chunks:")
        for i, chunk in enumerate(result.chunks):
            print(f"\n   Chunk {i+1}:")
            print(f"   Type: {chunk.chunk_type}")
            print(f"   Text: {chunk.text}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error processing text: {e}")
        return False
        
    finally:
        if test_file.exists():
            test_file.unlink()

async def test_file_validation():
    """Test file validation functionality."""
    print("\n" + "=" * 60)
    print("Testing File Validation")
    print("=" * 60)
    
    # Test supported file types
    supported_files = ["test.pdf", "test.docx", "test.md", "test.txt"]
    
    print("✅ Testing supported file types:")
    for filename in supported_files:
        try:
            doc_type = document_processor.detect_document_type(filename)
            print(f"   {filename} -> {doc_type.value}")
        except Exception as e:
            print(f"   ❌ {filename} -> Error: {e}")
    
    # Test unsupported file types
    unsupported_files = ["test.xyz", "test.exe", "test.jpg"]
    
    print("\n❌ Testing unsupported file types:")
    for filename in unsupported_files:
        try:
            doc_type = document_processor.detect_document_type(filename)
            print(f"   ❌ {filename} -> {doc_type.value} (should have failed!)")
        except Exception as e:
            print(f"   ✅ {filename} -> Correctly rejected: {type(e).__name__}")
    
    # Test file existence validation
    print("\n📁 Testing file existence validation:")
    try:
        document_processor.validate_file("non_existent_file.pdf")
        print("   ❌ Non-existent file validation should have failed!")
    except Exception as e:
        print(f"   ✅ Non-existent file correctly rejected: {type(e).__name__}")
    
    return True

async def test_docling_fallback():
    """Test docling fallback functionality."""
    print("\n" + "=" * 60)
    print("Testing Docling Fallback")
    print("=" * 60)
    
    # Create a fake PDF file (actually text)
    test_content = "This is a fake PDF file for testing docling fallback."
    test_file = Path("test_fake.pdf")
    
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(test_content)
    
    try:
        print(f"Processing fake PDF with docling=True: {test_file}")
        result = await document_processor.parse_document(test_file, use_docling=True)
        
        print(f"\n📊 Processing Results:")
        print(f"   Chunks found: {len(result.chunks)}")
        print(f"   Parser used: {result.metadata.get('parser', 'unknown')}")
        print(f"   Fallback successful: {'Yes' if result.metadata.get('parser') == 'unstructured' else 'No'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing docling fallback: {e}")
        return False
        
    finally:
        if test_file.exists():
            test_file.unlink()

async def main():
    """Run all tests."""
    print("🚀 Starting Document Processor Manual Tests")
    print(f"📁 Working directory: {Path.cwd()}")
    
    # Check if Gemini API key is configured (not required for basic parsing)
    if settings.gemini_api_key:
        print("✅ Gemini API key configured")
    else:
        print("⚠️  Gemini API key not configured (not required for basic parsing tests)")
    
    tests = [
        ("File Validation", test_file_validation),
        ("Markdown Processing", test_markdown_processing),
        ("Text Processing", test_text_processing),
        ("Docling Fallback", test_docling_fallback),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            success = await test_func()
            results.append((test_name, success))
            if success:
                print(f"✅ {test_name} completed successfully")
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"💥 {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} {test_name}")
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Document processor is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
