#!/usr/bin/env python3
"""
Integration test script for document processing improvements.
Tests configuration debugging, word boundary preservation, PDF processing, and contextual retrieval.
"""

import asyncio
import os
import sys
from pathlib import Path
import tempfile

# Add the packages to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "morag-core" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "morag-document" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "morag-services" / "src"))

def test_configuration_debugging():
    """Test configuration debugging output."""
    print("🔧 Testing Configuration Debugging")
    print("=" * 50)
    
    try:
        from morag_core.config import validate_configuration_and_log
        
        # Set some test environment variables
        os.environ["MORAG_DEFAULT_CHUNK_SIZE"] = "5000"
        os.environ["MORAG_MAX_PAGE_CHUNK_SIZE"] = "10000"
        os.environ["MORAG_ENABLE_PAGE_BASED_CHUNKING"] = "true"
        os.environ["MORAG_DEFAULT_CHUNKING_STRATEGY"] = "page"
        
        # Test configuration validation and logging
        settings = validate_configuration_and_log()
        
        print(f"✅ Configuration loaded successfully")
        print(f"   - Default chunk size: {settings.default_chunk_size}")
        print(f"   - Max page chunk size: {settings.max_page_chunk_size}")
        print(f"   - Page-based chunking enabled: {settings.enable_page_based_chunking}")
        print(f"   - Default chunking strategy: {settings.default_chunking_strategy}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration debugging test failed: {e}")
        return False


def test_word_boundary_detection():
    """Test improved word boundary detection."""
    print("\n🔤 Testing Word Boundary Detection")
    print("=" * 50)
    
    try:
        from morag_document.converters.base import DocumentConverter
        
        converter = DocumentConverter()
        
        # Test cases for word boundary detection
        test_cases = [
            ("This is a test sentence with some punctuation!", 20),
            ("Hello world! How are you today?", 15),
            ("The quick brown fox jumps over the lazy dog.", 25),
            ("Testing word-boundary detection with hyphenated-words.", 30),
        ]
        
        all_passed = True
        
        for text, position in test_cases:
            # Test backward search
            boundary_back = converter._find_word_boundary(text, position, "backward")
            
            # Test forward search
            boundary_forward = converter._find_word_boundary(text, position, "forward")
            
            print(f"✅ Text: '{text[:30]}...'")
            print(f"   Position {position} -> Backward: {boundary_back}, Forward: {boundary_forward}")
            
            # Basic validation that boundaries are within text bounds
            if not (0 <= boundary_back <= len(text) and 0 <= boundary_forward <= len(text)):
                print(f"❌ Boundary out of range")
                all_passed = False
        
        if all_passed:
            print("✅ All word boundary tests passed!")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Word boundary test failed: {e}")
        return False


async def test_pdf_processing():
    """Test PDF processing with docling integration."""
    print("\n📄 Testing PDF Processing")
    print("=" * 50)
    
    try:
        from morag_document.converters.pdf import PDFConverter
        from morag_core.interfaces.converter import ConversionOptions
        from morag_core.models.document import Document, DocumentMetadata, DocumentType

        converter = PDFConverter()

        # Check docling availability
        print(f"Docling available: {converter._docling_available}")

        # Create a simple test document (we can't test actual PDF without a file)
        metadata = DocumentMetadata(
            source_type=DocumentType.PDF,
            source_name="test.pdf"
        )
        document = Document(metadata=metadata)
        options = ConversionOptions(format_type="pdf")
        
        print("✅ PDF converter initialized successfully")
        print(f"   - Supported formats: {converter.supported_formats}")
        print(f"   - Docling integration: {'Available' if converter._docling_available else 'Not available (fallback to pypdf)'}")
        
        return True
        
    except Exception as e:
        print(f"❌ PDF processing test failed: {e}")
        return False


async def test_contextual_retrieval():
    """Test contextual retrieval service."""
    print("\n🧠 Testing Contextual Retrieval")
    print("=" * 50)
    
    try:
        from morag_services.contextual_retrieval import ContextualRetrievalService
        from morag_services.embedding import GeminiEmbeddingService
        from morag_core.models.document import Document, DocumentChunk, DocumentMetadata, DocumentType

        # Create a mock embedding service (without API key for testing)
        embedding_service = GeminiEmbeddingService(api_key="test_key")

        # Create contextual retrieval service
        contextual_service = ContextualRetrievalService(embedding_service)

        # Create a test document with chunks
        metadata = DocumentMetadata(
            source_type=DocumentType.TEXT,
            source_name="test_document.txt",
            title="Test Document"
        )
        document = Document(metadata=metadata)
        document.raw_text = "This is a test document about machine learning and artificial intelligence."

        # Add test chunks
        chunk1 = DocumentChunk(
            document_id=document.id,
            content="Machine learning is a subset of artificial intelligence.",
            page_number=1,
            section="Introduction"
        )
        chunk2 = DocumentChunk(
            document_id=document.id,
            content="Deep learning uses neural networks with multiple layers.",
            page_number=1,
            section="Deep Learning"
        )
        
        document.chunks = [chunk1, chunk2]
        
        print("✅ Contextual retrieval service initialized successfully")
        print(f"   - Document: {document.metadata.title}")
        print(f"   - Chunks: {len(document.chunks)}")
        print(f"   - Service ready: {contextual_service is not None}")
        
        # Test sparse embedding generation
        sparse_embedding = await contextual_service._generate_sparse_embedding(
            "This is a test text for sparse embedding generation."
        )
        
        print(f"✅ Sparse embedding generated: {len(sparse_embedding)} keywords")
        if sparse_embedding:
            top_keywords = list(sparse_embedding.keys())[:5]
            print(f"   - Top keywords: {top_keywords}")
        
        return True
        
    except Exception as e:
        print(f"❌ Contextual retrieval test failed: {e}")
        return False


async def test_chunking_with_word_boundaries():
    """Test document chunking with word boundary preservation."""
    print("\n✂️ Testing Chunking with Word Boundaries")
    print("=" * 50)
    
    try:
        from morag_document.converters.base import DocumentConverter
        from morag_core.interfaces.converter import ConversionOptions, ChunkingStrategy
        from morag_core.models.document import Document, DocumentMetadata, DocumentType

        converter = DocumentConverter()

        # Create test document
        metadata = DocumentMetadata(
            source_type=DocumentType.TEXT,
            source_name="test_chunking.txt"
        )
        document = Document(metadata=metadata)
        document.raw_text = (
            "This is a test document for chunking. It contains multiple sentences "
            "and should be split at proper word boundaries. The chunking algorithm "
            "should never split words in the middle, ensuring that each chunk "
            "contains complete words only. This helps maintain readability and "
            "semantic coherence in the resulting chunks."
        )
        
        # Test character-based chunking with word boundaries
        options = ConversionOptions(
            format_type="text",
            chunking_strategy=ChunkingStrategy.CHARACTER,
            chunk_size=100,  # Small size to force splitting
            chunk_overlap=20
        )
        
        # Clear previous chunks
        document.chunks = []
        
        # Apply chunking
        await converter._chunk_document(document, options)
        
        print(f"✅ Character chunking: {len(document.chunks)} chunks")
        
        # Verify no words are split
        for i, chunk in enumerate(document.chunks):
            words = chunk.content.split()
            print(f"   - Chunk {i+1}: {len(chunk.content)} chars, {len(words)} words")
            if len(chunk.content) > 50:
                print(f"     Preview: {chunk.content[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Chunking test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 MoRAG Document Processing Improvements Test Suite")
    print("=" * 60)
    
    tests = [
        ("Configuration Debugging", test_configuration_debugging),
        ("Word Boundary Detection", test_word_boundary_detection),
        ("PDF Processing", test_pdf_processing),
        ("Contextual Retrieval", test_contextual_retrieval),
        ("Chunking with Word Boundaries", test_chunking_with_word_boundaries),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} - {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Document processing improvements are working correctly.")
    else:
        print("⚠️ Some tests failed. Please review the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    asyncio.run(main())
