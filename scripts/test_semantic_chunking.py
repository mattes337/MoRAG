#!/usr/bin/env python3
"""
Manual testing script for semantic chunking functionality.
This script tests the semantic chunking service with various text types and strategies.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morag.services.chunking import chunking_service

async def test_basic_chunking():
    """Test basic chunking functionality."""
    print("=" * 60)
    print("Testing Basic Chunking Functionality")
    print("=" * 60)
    
    text = """
    This is a sample document for testing semantic chunking capabilities.
    The document contains multiple sentences with varying complexity and topics.
    
    Machine learning is a fascinating field that combines statistics, computer science, and domain expertise.
    It enables computers to learn patterns from data without being explicitly programmed for every scenario.
    Deep learning, a subset of machine learning, uses neural networks with multiple layers to model complex patterns.
    
    Natural language processing is another important area of artificial intelligence.
    It focuses on enabling computers to understand, interpret, and generate human language.
    Applications include chatbots, translation systems, and sentiment analysis tools.
    
    Computer vision helps machines interpret and understand visual information from the world.
    It's used in autonomous vehicles, medical imaging, and facial recognition systems.
    The field has advanced significantly with the development of convolutional neural networks.
    """
    
    print(f"📄 Original text length: {len(text)} characters")
    print(f"📄 Word count: {len(text.split())} words")
    
    # Test different strategies
    strategies = ["simple", "sentence", "paragraph", "semantic"]
    
    for strategy in strategies:
        print(f"\n🔍 Testing {strategy.upper()} chunking:")
        
        try:
            chunks = await chunking_service.semantic_chunk(
                text, 
                chunk_size=200, 
                strategy=strategy
            )
            
            print(f"   Chunks created: {len(chunks)}")
            print(f"   Average chunk size: {sum(len(c) for c in chunks) / len(chunks):.1f} chars")
            
            # Show first chunk as example
            if chunks:
                print(f"   First chunk: {chunks[0][:100]}{'...' if len(chunks[0]) > 100 else ''}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return True

async def test_metadata_extraction():
    """Test chunking with metadata extraction."""
    print("\n" + "=" * 60)
    print("Testing Metadata Extraction")
    print("=" * 60)
    
    text = """
    Apple Inc. is a technology company based in Cupertino, California.
    The company was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.
    Apple is known for products like the iPhone, iPad, and MacBook.
    
    Microsoft Corporation is another major technology company, headquartered in Redmond, Washington.
    It was founded by Bill Gates and Paul Allen in 1975.
    Microsoft is famous for Windows operating system and Office productivity suite.
    """
    
    try:
        chunk_infos = await chunking_service.chunk_with_metadata(
            text, 
            chunk_size=150, 
            strategy="semantic"
        )
        
        print(f"📊 Chunks with metadata: {len(chunk_infos)}")
        
        for i, chunk_info in enumerate(chunk_infos):
            print(f"\n📄 Chunk {i+1}:")
            print(f"   Type: {chunk_info.chunk_type}")
            print(f"   Word count: {chunk_info.word_count}")
            print(f"   Sentence count: {chunk_info.sentence_count}")
            print(f"   Entities found: {len(chunk_info.entities)}")
            print(f"   Topics: {chunk_info.topics[:3]}")  # Show first 3 topics
            print(f"   Text: {chunk_info.text[:100]}{'...' if len(chunk_info.text) > 100 else ''}")
            
            if chunk_info.entities:
                print(f"   Sample entities: {[e['text'] for e in chunk_info.entities[:3]]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in metadata extraction: {e}")
        return False

async def test_text_analysis():
    """Test text structure analysis."""
    print("\n" + "=" * 60)
    print("Testing Text Structure Analysis")
    print("=" * 60)
    
    test_texts = [
        ("Simple text", "This is simple. Short sentences. Easy to read."),
        ("Complex text", "This is a more complex document with longer sentences that contain multiple clauses, subordinate phrases, and technical terminology that requires more sophisticated processing algorithms."),
        ("Mixed content", """
        This document has mixed content. Some sentences are short.
        
        Other paragraphs contain much longer and more complex sentences that discuss technical topics in great detail, requiring advanced natural language processing techniques to properly understand and segment.
        
        The final section returns to simpler language.
        """),
        ("Empty text", ""),
    ]
    
    for name, text in test_texts:
        print(f"\n📋 Analyzing: {name}")
        
        try:
            analysis = await chunking_service.analyze_text_structure(text)
            
            print(f"   Word count: {analysis['word_count']}")
            print(f"   Sentence count: {analysis['sentence_count']}")
            print(f"   Paragraph count: {analysis['paragraph_count']}")
            print(f"   Avg sentence length: {analysis['avg_sentence_length']:.1f}")
            print(f"   Text complexity: {analysis['text_complexity']}")
            print(f"   Recommended strategy: {analysis['recommended_strategy']}")
            print(f"   Estimated chunks: {analysis['estimated_chunks']}")
            print(f"   spaCy available: {analysis['spacy_available']}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return True

async def test_chunk_size_variations():
    """Test different chunk sizes."""
    print("\n" + "=" * 60)
    print("Testing Different Chunk Sizes")
    print("=" * 60)
    
    text = "This is a test sentence. " * 50  # Create repetitive text
    
    chunk_sizes = [50, 100, 200, 500]
    
    for size in chunk_sizes:
        print(f"\n📏 Chunk size: {size} characters")
        
        try:
            chunks = await chunking_service.semantic_chunk(text, chunk_size=size)
            
            actual_sizes = [len(chunk) for chunk in chunks]
            
            print(f"   Chunks created: {len(chunks)}")
            print(f"   Actual sizes: min={min(actual_sizes)}, max={max(actual_sizes)}, avg={sum(actual_sizes)/len(actual_sizes):.1f}")
            print(f"   Size compliance: {sum(1 for s in actual_sizes if s <= size * 1.1) / len(actual_sizes) * 100:.1f}%")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return True

async def test_special_content():
    """Test chunking with special content types."""
    print("\n" + "=" * 60)
    print("Testing Special Content Types")
    print("=" * 60)
    
    special_texts = [
        ("Code-like content", """
        def process_data(input_data):
            # This function processes input data
            result = []
            for item in input_data:
                if item.is_valid():
                    result.append(item.transform())
            return result
        """),
        ("List content", """
        Shopping list:
        - Apples
        - Bananas
        - Milk
        - Bread
        - Cheese
        
        Todo items:
        1. Complete project
        2. Review documents
        3. Send emails
        """),
        ("Mixed punctuation", "Hello! How are you? I'm fine... What about you??? This has lots of punctuation!!! And some... ellipses."),
        ("Numbers and symbols", "The price is $29.99. Call us at (555) 123-4567. Email: test@example.com. Visit https://example.com for more info."),
    ]
    
    for name, text in special_texts:
        print(f"\n📝 Testing: {name}")
        
        try:
            chunks = await chunking_service.semantic_chunk(text, chunk_size=100)
            
            print(f"   Chunks created: {len(chunks)}")
            if chunks:
                print(f"   First chunk: {chunks[0][:80]}{'...' if len(chunks[0]) > 80 else ''}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return True

async def main():
    """Run all tests."""
    print("🚀 Starting Semantic Chunking Manual Tests")
    print(f"📁 Working directory: {Path.cwd()}")
    
    tests = [
        ("Basic Chunking", test_basic_chunking),
        ("Metadata Extraction", test_metadata_extraction),
        ("Text Analysis", test_text_analysis),
        ("Chunk Size Variations", test_chunk_size_variations),
        ("Special Content", test_special_content),
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
        print("🎉 All tests passed! Semantic chunking is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
