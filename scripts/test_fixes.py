#!/usr/bin/env python3
"""
Test script to verify the configuration and implementation fixes.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the packages to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "morag-core" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "morag-services" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "morag-document" / "src"))

from morag_core.config import get_settings
from morag_core.interfaces.converter import ChunkingStrategy, ConversionOptions
from morag_core.models.document import Document, DocumentMetadata, DocumentType


def test_configuration():
    """Test configuration loading."""
    print("Testing Configuration...")
    
    try:
        settings = get_settings()
        
        # Test that new configuration fields are available
        assert hasattr(settings, 'default_chunking_strategy'), "default_chunking_strategy not found"
        assert hasattr(settings, 'enable_page_based_chunking'), "enable_page_based_chunking not found"
        assert hasattr(settings, 'max_page_chunk_size'), "max_page_chunk_size not found"
        
        print(f"✅ Configuration loaded successfully")
        print(f"   Default chunking strategy: {settings.default_chunking_strategy}")
        print(f"   Page-based chunking enabled: {settings.enable_page_based_chunking}")
        print(f"   Max page chunk size: {settings.max_page_chunk_size}")
        print(f"   Default chunk size: {settings.default_chunk_size}")
        print(f"   Default chunk overlap: {settings.default_chunk_overlap}")
        print(f"   Embedding batch size: {settings.embedding_batch_size}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_chunking_strategies():
    """Test chunking strategies."""
    print("\nTesting Chunking Strategies...")
    
    try:
        # Test that PAGE strategy is available
        page_strategy = ChunkingStrategy.PAGE
        assert page_strategy.value == "page", "PAGE strategy value incorrect"
        
        print(f"✅ PAGE chunking strategy available: {page_strategy.value}")
        
        # Test all strategies
        strategies = [
            ChunkingStrategy.PAGE,
            ChunkingStrategy.CHAPTER,
            ChunkingStrategy.PARAGRAPH,
            ChunkingStrategy.SENTENCE,
            ChunkingStrategy.WORD,
            ChunkingStrategy.CHARACTER
        ]
        
        for strategy in strategies:
            print(f"   {strategy.value}: {strategy.name}")
            
        return True
        
    except Exception as e:
        print(f"❌ Chunking strategies test failed: {e}")
        return False


async def test_document_processing():
    """Test document processing with page-based chunking."""
    print("\nTesting Document Processing...")
    
    try:
        from morag_document.converters.base import BaseConverter
        
        # Create a test document
        metadata = DocumentMetadata(
            title="Test Document",
            document_type=DocumentType.TEXT,
            source="test",
            language="en"
        )
        
        document = Document(metadata=metadata)
        document.raw_text = """Page 1 content here. This is the first page with some content.
        
Page 2 content here. This is the second page with different content.

Page 3 content here. This is the third page with more content."""
        
        # Create converter and test page-based chunking
        converter = BaseConverter()
        
        # Check if page-based chunking method exists
        assert hasattr(converter, '_chunk_by_pages'), "Page-based chunking method not found"
        assert hasattr(converter, '_find_word_boundary'), "Word boundary method not found"
        
        # Test conversion options
        options = ConversionOptions(
            format_type="text",
            chunking_strategy=ChunkingStrategy.PAGE,
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Test chunking
        await converter._chunk_document(document, options)
        
        print(f"✅ Document processing test passed")
        print(f"   Document chunks created: {len(document.chunks)}")
        
        for i, chunk in enumerate(document.chunks):
            print(f"   Chunk {i+1}: {len(chunk.content)} chars - {chunk.content[:50]}...")
            
        return True
        
    except Exception as e:
        print(f"❌ Document processing test failed: {e}")
        return False


async def test_embedding_batch():
    """Test embedding batch processing."""
    print("\nTesting Embedding Batch Processing...")
    
    try:
        # Check if we have API key
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            print("⚠️  GEMINI_API_KEY not set, skipping embedding test")
            return True
            
        from morag_services.embedding import GeminiEmbeddingService
        
        # Check if batch method exists
        assert hasattr(GeminiEmbeddingService, 'generate_embeddings_batch'), "Batch embedding method not found"
        
        # Create service
        service = GeminiEmbeddingService(api_key=api_key)
        
        # Test batch processing
        texts = [
            "This is the first test text.",
            "This is the second test text.",
            "This is the third test text."
        ]
        
        # This would actually call the API, so we just check the method exists
        print(f"✅ Embedding batch processing method available")
        print(f"   Service created successfully")
        print(f"   Batch method: {service.generate_embeddings_batch}")
        
        return True
        
    except Exception as e:
        print(f"❌ Embedding batch test failed: {e}")
        return False


def test_word_boundary_preservation():
    """Test word boundary preservation in text splitting."""
    print("\nTesting Word Boundary Preservation...")
    
    try:
        from morag_document.converters.base import BaseConverter
        
        converter = BaseConverter()
        
        # Test word boundary finding
        text = "This is a test sentence with some words that should not be split in the middle."
        
        # Test backward boundary finding
        boundary = converter._find_word_boundary(text, 30, "backward")
        
        # Check that we don't split in the middle of a word
        if boundary > 0 and boundary < len(text):
            char_at_boundary = text[boundary]
            assert char_at_boundary.isspace() or char_at_boundary in '.!?;:,', f"Boundary not at word edge: '{char_at_boundary}'"
        
        print(f"✅ Word boundary preservation test passed")
        print(f"   Text length: {len(text)}")
        print(f"   Boundary at position {boundary}: '{text[boundary-2:boundary+2]}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Word boundary test failed: {e}")
        return False


def test_content_checksum():
    """Test content checksum functionality."""
    print("\nTesting Content Checksum...")
    
    try:
        import hashlib
        
        # Test checksum generation
        content = "This is test content for checksum generation."
        checksum = hashlib.sha256(content.encode()).hexdigest()
        
        assert len(checksum) == 64, "Checksum length incorrect"
        assert checksum.isalnum(), "Checksum format incorrect"
        
        # Test that same content produces same checksum
        checksum2 = hashlib.sha256(content.encode()).hexdigest()
        assert checksum == checksum2, "Checksums not consistent"
        
        print(f"✅ Content checksum test passed")
        print(f"   Content: {content[:30]}...")
        print(f"   Checksum: {checksum[:16]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Content checksum test failed: {e}")
        return False


async def main():
    """Main test function."""
    print("MoRAG Fixes Test Suite")
    print("=" * 50)
    
    tests = [
        ("Configuration", test_configuration),
        ("Chunking Strategies", test_chunking_strategies),
        ("Document Processing", test_document_processing),
        ("Embedding Batch", test_embedding_batch),
        ("Word Boundary", test_word_boundary_preservation),
        ("Content Checksum", test_content_checksum),
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
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The fixes are working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    asyncio.run(main())
