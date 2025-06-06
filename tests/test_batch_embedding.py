#!/usr/bin/env python3
"""Test script for batch embedding functionality."""

import asyncio
import os
import sys
import time
from typing import List

# Add the packages to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'morag-core', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'morag-embedding', 'src'))

from morag_core.config import settings
from morag_embedding.service import GeminiEmbeddingService


async def test_batch_embedding():
    """Test batch embedding functionality."""
    
    # Check if API key is available
    if not settings.gemini_api_key:
        print("❌ GEMINI_API_KEY not found in environment")
        print("Please set your Gemini API key in the .env file or environment variables")
        return False
    
    print("🚀 Testing Gemini Batch Embedding Functionality")
    print("=" * 50)
    
    # Test texts
    test_texts = [
        "The sky is blue because of Rayleigh scattering.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
        "The Earth orbits around the Sun once every year.",
        "Quantum computing uses quantum mechanical phenomena.",
        "Photosynthesis converts light energy into chemical energy.",
        "The human brain contains approximately 86 billion neurons.",
        "Climate change is caused by greenhouse gas emissions.",
        "DNA contains the genetic instructions for life.",
        "The speed of light in vacuum is approximately 299,792,458 meters per second."
    ]
    
    print(f"📝 Test texts: {len(test_texts)} items")
    
    try:
        # Test with batch embedding enabled
        print("\n🔄 Testing with batch embedding ENABLED...")
        service_batch = GeminiEmbeddingService(
            batch_size=5,
            enable_batch_embedding=True
        )
        
        start_time = time.time()
        result_batch = await service_batch.generate_batch_embeddings(test_texts)
        batch_time = time.time() - start_time
        
        print(f"✅ Batch embedding completed in {batch_time:.2f} seconds")
        print(f"📊 Generated {len(result_batch.embeddings)} embeddings")
        print(f"🎯 Model: {result_batch.model}")
        print(f"📈 Metadata: {result_batch.metadata}")
        
        # Verify embedding dimensions
        if result_batch.embeddings:
            embedding_dim = len(result_batch.embeddings[0])
            print(f"📏 Embedding dimension: {embedding_dim}")
        
        # Test with batch embedding disabled (sequential)
        print("\n🔄 Testing with batch embedding DISABLED (sequential)...")
        service_sequential = GeminiEmbeddingService(
            batch_size=5,
            enable_batch_embedding=False
        )
        
        start_time = time.time()
        result_sequential = await service_sequential.generate_batch_embeddings(test_texts)
        sequential_time = time.time() - start_time
        
        print(f"✅ Sequential embedding completed in {sequential_time:.2f} seconds")
        print(f"📊 Generated {len(result_sequential.embeddings)} embeddings")
        print(f"🎯 Model: {result_sequential.model}")
        print(f"📈 Metadata: {result_sequential.metadata}")
        
        # Compare performance
        print("\n📈 Performance Comparison:")
        print(f"Batch embedding time: {batch_time:.2f}s")
        print(f"Sequential embedding time: {sequential_time:.2f}s")
        if sequential_time > 0:
            speedup = sequential_time / batch_time
            print(f"Speedup: {speedup:.2f}x faster with batch embedding")
        
        # Test single embedding for comparison
        print("\n🔄 Testing single embedding...")
        start_time = time.time()
        single_result = await service_batch.generate_embedding(test_texts[0])
        single_time = time.time() - start_time
        
        print(f"✅ Single embedding completed in {single_time:.2f} seconds")
        print(f"📏 Embedding dimension: {len(single_result.embedding)}")
        
        print("\n🎉 All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False


async def test_different_batch_sizes():
    """Test different batch sizes to find optimal performance."""
    
    if not settings.gemini_api_key:
        print("❌ GEMINI_API_KEY not found in environment")
        return False
    
    print("\n🔬 Testing Different Batch Sizes")
    print("=" * 40)
    
    # Test texts
    test_texts = [f"This is test text number {i+1} for batch size optimization." for i in range(20)]
    
    batch_sizes = [1, 5, 10, 15, 20]
    
    for batch_size in batch_sizes:
        try:
            print(f"\n📊 Testing batch size: {batch_size}")
            
            service = GeminiEmbeddingService(
                batch_size=batch_size,
                enable_batch_embedding=True
            )
            
            start_time = time.time()
            result = await service.generate_batch_embeddings(test_texts)
            elapsed_time = time.time() - start_time
            
            print(f"⏱️  Time: {elapsed_time:.2f}s")
            print(f"📈 Embeddings: {len(result.embeddings)}")
            print(f"🔧 Method: {result.metadata.get('method', 'unknown')}")
            
            if 'total_batches' in result.metadata:
                print(f"📦 Total batches: {result.metadata['total_batches']}")
            
        except Exception as e:
            print(f"❌ Batch size {batch_size} failed: {str(e)}")
    
    return True


if __name__ == "__main__":
    async def main():
        success1 = await test_batch_embedding()
        success2 = await test_different_batch_sizes()
        
        if success1 and success2:
            print("\n🎉 All batch embedding tests passed!")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed!")
            sys.exit(1)
    
    asyncio.run(main())
