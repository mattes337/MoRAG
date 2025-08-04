#!/usr/bin/env python3
"""Test script for LLM batch processing functionality."""

import asyncio
import os
import time
from typing import List, Dict, Any

# Set up environment
os.environ.setdefault("MORAG_ENABLE_LLM_BATCHING", "true")
os.environ.setdefault("MORAG_LLM_BATCH_SIZE", "5")
os.environ.setdefault("MORAG_LLM_BATCH_DELAY", "1.0")

try:
    from morag_reasoning.llm import LLMClient, LLMConfig
    from morag_reasoning.batch_processor import (
        batch_text_analysis, 
        batch_document_chunks,
        TextAnalysisBatchProcessor
    )
    BATCH_AVAILABLE = True
except ImportError as e:
    print(f"Batch processing not available: {e}")
    BATCH_AVAILABLE = False


async def test_basic_batch_processing():
    """Test basic batch processing functionality."""
    print("\n=== Testing Basic Batch Processing ===")
    
    if not BATCH_AVAILABLE:
        print("❌ Batch processing not available")
        return False
    
    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found - skipping LLM tests")
        return False
    
    # Create LLM client with batch configuration
    config = LLMConfig(
        provider="gemini",
        model="gemini-1.5-flash",
        api_key=api_key,
        batch_size=3,
        enable_batching=True,
        batch_delay=0.5
    )
    
    llm_client = LLMClient(config)
    
    # Test simple batch processing
    prompts = [
        "What is the capital of France?",
        "What is 2 + 2?",
        "Name a programming language.",
        "What color is the sky?",
        "What is the largest planet?"
    ]
    
    print(f"Processing {len(prompts)} prompts in batches...")
    
    start_time = time.time()
    try:
        responses = await llm_client.generate_batch(prompts)
        batch_time = time.time() - start_time
        
        print(f"✅ Batch processing completed in {batch_time:.2f}s")
        print(f"📊 Responses received: {len(responses)}")
        
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            print(f"  {i+1}. Q: {prompt}")
            print(f"     A: {response[:100]}{'...' if len(response) > 100 else ''}")
        
        return True
        
    except Exception as e:
        print(f"❌ Batch processing failed: {str(e)}")
        return False


async def test_individual_vs_batch_performance():
    """Compare individual vs batch processing performance."""
    print("\n=== Testing Individual vs Batch Performance ===")
    
    if not BATCH_AVAILABLE:
        print("❌ Batch processing not available")
        return False
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found - skipping performance test")
        return False
    
    # Create LLM clients
    batch_config = LLMConfig(
        provider="gemini",
        model="gemini-1.5-flash",
        api_key=api_key,
        batch_size=5,
        enable_batching=True,
        batch_delay=0.5
    )
    
    individual_config = LLMConfig(
        provider="gemini",
        model="gemini-1.5-flash",
        api_key=api_key,
        enable_batching=False
    )
    
    batch_client = LLMClient(batch_config)
    individual_client = LLMClient(individual_config)
    
    # Test prompts
    test_prompts = [
        "Summarize the concept of artificial intelligence in one sentence.",
        "What are the main benefits of renewable energy?",
        "Explain the water cycle briefly.",
        "What is the difference between HTTP and HTTPS?",
        "Name three programming paradigms.",
        "What is machine learning?",
        "Describe photosynthesis in simple terms.",
        "What are the primary colors?"
    ]
    
    print(f"Testing with {len(test_prompts)} prompts...")
    
    # Test individual processing
    print("\n🔄 Testing individual processing...")
    individual_start = time.time()
    try:
        individual_responses = []
        for prompt in test_prompts:
            response = await individual_client.generate_text(prompt)
            individual_responses.append(response)
        individual_time = time.time() - individual_start
        print(f"✅ Individual processing: {individual_time:.2f}s")
    except Exception as e:
        print(f"❌ Individual processing failed: {str(e)}")
        return False
    
    # Test batch processing
    print("\n🔄 Testing batch processing...")
    batch_start = time.time()
    try:
        batch_responses = await batch_client.generate_batch(test_prompts)
        batch_time = time.time() - batch_start
        print(f"✅ Batch processing: {batch_time:.2f}s")
    except Exception as e:
        print(f"❌ Batch processing failed: {str(e)}")
        return False
    
    # Compare results
    print(f"\n📊 Performance Comparison:")
    print(f"   Individual: {individual_time:.2f}s ({len(test_prompts)} API calls)")
    print(f"   Batch:      {batch_time:.2f}s (~{len(test_prompts)//batch_config.batch_size + 1} API calls)")
    
    if batch_time < individual_time:
        speedup = individual_time / batch_time
        print(f"   🚀 Batch processing is {speedup:.1f}x faster!")
    else:
        print(f"   ⚠️  Batch processing was slower (possibly due to network conditions)")
    
    # Verify response quality
    print(f"\n🔍 Response Quality Check:")
    print(f"   Individual responses: {len(individual_responses)}")
    print(f"   Batch responses:      {len(batch_responses)}")
    
    if len(individual_responses) == len(batch_responses):
        print("   ✅ Same number of responses received")
    else:
        print("   ⚠️  Different number of responses")
    
    return True


async def test_text_analysis_batch():
    """Test text analysis batch processing."""
    print("\n=== Testing Text Analysis Batch Processing ===")
    
    if not BATCH_AVAILABLE:
        print("❌ Batch processing not available")
        return False
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found - skipping text analysis test")
        return False
    
    config = LLMConfig(
        provider="gemini",
        model="gemini-1.5-flash",
        api_key=api_key,
        batch_size=3,
        enable_batching=True
    )
    
    llm_client = LLMClient(config)
    
    # Test texts
    texts = [
        "Apple Inc. is a technology company founded by Steve Jobs.",
        "The Eiffel Tower is located in Paris, France.",
        "Python is a programming language created by Guido van Rossum.",
        "The Amazon rainforest is located in South America.",
        "Microsoft was founded by Bill Gates and Paul Allen."
    ]
    
    print(f"Analyzing {len(texts)} texts for entity extraction...")
    
    try:
        results = await batch_text_analysis(
            llm_client,
            texts,
            analysis_type="entity_extraction"
        )
        
        print(f"✅ Analysis completed: {len(results)} results")
        
        for i, result in enumerate(results):
            if result.success:
                print(f"  Text {i+1}: ✅ Success")
                analysis = result.result.get("analysis", [])
                if isinstance(analysis, list):
                    print(f"    Entities found: {len(analysis)}")
                else:
                    print(f"    Analysis: {str(analysis)[:100]}...")
            else:
                print(f"  Text {i+1}: ❌ Failed - {result.error_message}")
        
        return True
        
    except Exception as e:
        print(f"❌ Text analysis batch processing failed: {str(e)}")
        return False


async def test_document_chunk_batch():
    """Test document chunk batch processing."""
    print("\n=== Testing Document Chunk Batch Processing ===")
    
    if not BATCH_AVAILABLE:
        print("❌ Batch processing not available")
        return False
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found - skipping chunk test")
        return False
    
    config = LLMConfig(
        provider="gemini",
        model="gemini-1.5-flash",
        api_key=api_key,
        batch_size=2,
        enable_batching=True
    )
    
    llm_client = LLMClient(config)
    
    # Test document chunks
    chunks = [
        {
            "id": "chunk_1",
            "text": "Apple Inc. is a multinational technology company. It was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne.",
            "document_id": "doc_1"
        },
        {
            "id": "chunk_2", 
            "text": "The iPhone is Apple's flagship smartphone product. It runs on iOS operating system.",
            "document_id": "doc_1"
        },
        {
            "id": "chunk_3",
            "text": "Tim Cook became CEO of Apple after Steve Jobs. He has led the company's expansion into new markets.",
            "document_id": "doc_1"
        }
    ]
    
    print(f"Processing {len(chunks)} document chunks...")
    
    try:
        results = await batch_document_chunks(
            llm_client,
            chunks,
            processing_type="extraction"
        )
        
        print(f"✅ Chunk processing completed: {len(results)} results")
        
        for result in results:
            if result.success:
                chunk_id = result.item.get("id", "unknown")
                entities = result.result.get("entities", [])
                relations = result.result.get("relations", [])
                print(f"  {chunk_id}: ✅ {len(entities)} entities, {len(relations)} relations")
            else:
                chunk_id = result.item.get("id", "unknown")
                print(f"  {chunk_id}: ❌ Failed - {result.error_message}")
        
        return True
        
    except Exception as e:
        print(f"❌ Document chunk batch processing failed: {str(e)}")
        return False


async def main():
    """Run all batch processing tests."""
    print("🧪 MoRAG Batch Processing Test Suite")
    print("=" * 50)
    
    tests = [
        test_basic_batch_processing,
        test_individual_vs_batch_performance,
        test_text_analysis_batch,
        test_document_chunk_batch
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {str(e)}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    passed = sum(results)
    total = len(results)
    print(f"   Passed: {passed}/{total}")
    
    if passed == total:
        print("   🎉 All tests passed!")
    else:
        print(f"   ⚠️  {total - passed} test(s) failed")
    
    print("\n💡 Tips:")
    print("   - Set GEMINI_API_KEY environment variable to run LLM tests")
    print("   - Adjust MORAG_LLM_BATCH_SIZE to control batch size")
    print("   - Set MORAG_ENABLE_LLM_BATCHING=false to disable batching")


if __name__ == "__main__":
    asyncio.run(main())
