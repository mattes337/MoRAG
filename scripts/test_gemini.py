#!/usr/bin/env python3
"""Manual testing script for Gemini integration."""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morag.services.embedding import gemini_service
from morag.utils.text_processing import prepare_text_for_embedding, prepare_text_for_summary
from morag.core.config import settings


async def test_embedding_generation():
    """Test single embedding generation."""
    print("🧪 Testing embedding generation...")
    
    try:
        text = "This is a test document about machine learning and artificial intelligence."
        prepared_text = prepare_text_for_embedding(text)
        
        result = await gemini_service.generate_embedding(prepared_text)
        
        print(f"✅ Embedding generated successfully!")
        print(f"   - Dimension: {len(result.embedding)}")
        print(f"   - Model: {result.model}")
        print(f"   - Token count: {result.token_count}")
        print(f"   - First 5 values: {result.embedding[:5]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Embedding generation failed: {str(e)}")
        return False


async def test_batch_embedding_generation():
    """Test batch embedding generation."""
    print("\n🧪 Testing batch embedding generation...")
    
    try:
        texts = [
            "First document about artificial intelligence and machine learning.",
            "Second document about data science and analytics.",
            "Third document about natural language processing."
        ]
        
        prepared_texts = [prepare_text_for_embedding(text) for text in texts]
        
        results = await gemini_service.generate_embeddings_batch(
            prepared_texts, 
            batch_size=2,
            delay_between_batches=0.5
        )
        
        print(f"✅ Batch embedding generated successfully!")
        print(f"   - Number of embeddings: {len(results)}")
        print(f"   - All dimensions: {[len(r.embedding) for r in results]}")
        print(f"   - Models: {[r.model for r in results]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Batch embedding generation failed: {str(e)}")
        return False


async def test_summary_generation():
    """Test text summarization."""
    print("\n🧪 Testing summary generation...")
    
    try:
        long_text = """
        Machine learning is a powerful subset of artificial intelligence that enables 
        computers to learn and improve from experience without being explicitly programmed. 
        It involves the development of algorithms and statistical models that computer systems 
        use to perform specific tasks effectively without using explicit instructions, 
        relying on patterns and inference instead. Machine learning algorithms build a 
        mathematical model based on sample data, known as training data, in order to make 
        predictions or decisions without being explicitly programmed to perform the task. 
        Machine learning algorithms are used in a wide variety of applications, such as 
        email filtering and computer vision, where it is difficult or infeasible to develop 
        conventional algorithms to perform the needed tasks.
        """
        
        prepared_text = prepare_text_for_summary(long_text)
        
        # Test different summary styles
        styles = ["concise", "detailed", "bullet"]
        
        for style in styles:
            result = await gemini_service.generate_summary(
                prepared_text, 
                max_length=50, 
                style=style
            )
            
            print(f"✅ {style.capitalize()} summary generated!")
            print(f"   - Length: {len(result.summary)} characters")
            print(f"   - Word count: {result.token_count}")
            print(f"   - Model: {result.model}")
            print(f"   - Summary: {result.summary[:100]}...")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Summary generation failed: {str(e)}")
        return False


async def test_health_check():
    """Test Gemini service health check."""
    print("🧪 Testing health check...")
    
    try:
        health = await gemini_service.health_check()
        
        print(f"✅ Health check completed!")
        print(f"   - Status: {health['status']}")
        print(f"   - Embedding model: {health.get('embedding_model', 'N/A')}")
        print(f"   - Generation model: {health.get('generation_model', 'N/A')}")
        
        if health['status'] == 'healthy':
            print(f"   - Embedding dimension: {health.get('embedding_dimension', 'N/A')}")
        else:
            print(f"   - Error: {health.get('error', 'Unknown error')}")
        
        return health['status'] == 'healthy'
        
    except Exception as e:
        print(f"❌ Health check failed: {str(e)}")
        return False


async def test_rate_limiting():
    """Test rate limiting behavior."""
    print("\n🧪 Testing rate limiting...")
    
    try:
        # Generate multiple embeddings quickly to test rate limiting
        texts = [f"Test document number {i} for rate limiting test." for i in range(5)]
        
        print("   - Generating 5 embeddings with rate limiting...")
        results = await gemini_service.generate_embeddings_batch(
            texts,
            batch_size=2,
            delay_between_batches=1.0
        )
        
        successful_results = [r for r in results if r.token_count > 0]
        
        print(f"✅ Rate limiting test completed!")
        print(f"   - Total requests: {len(texts)}")
        print(f"   - Successful: {len(successful_results)}")
        print(f"   - Failed: {len(results) - len(successful_results)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Rate limiting test failed: {str(e)}")
        return False


async def test_error_handling():
    """Test error handling."""
    print("\n🧪 Testing error handling...")
    
    try:
        # Test with very long text that might cause issues
        very_long_text = "word " * 50000  # Very long text
        
        try:
            result = await gemini_service.generate_embedding(very_long_text)
            print("✅ Long text handled successfully")
        except Exception as e:
            print(f"⚠️  Long text caused expected error: {type(e).__name__}")
        
        # Test with empty text
        try:
            result = await gemini_service.generate_embedding("")
            print("✅ Empty text handled successfully")
        except Exception as e:
            print(f"⚠️  Empty text caused expected error: {type(e).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {str(e)}")
        return False


def check_configuration():
    """Check Gemini configuration."""
    print("🔧 Checking Gemini configuration...")
    
    if not settings.gemini_api_key:
        print("❌ GEMINI_API_KEY not found in environment variables")
        print("   Please set your Gemini API key in the .env file:")
        print("   GEMINI_API_KEY=your_actual_api_key_here")
        return False
    
    if not settings.gemini_api_key.startswith("AI"):
        print("⚠️  Warning: Gemini API key doesn't start with 'AI'")
        print("   This might not be a valid Gemini API key format")
    
    print(f"✅ Configuration looks good!")
    print(f"   - API key: {settings.gemini_api_key[:10]}...")
    print(f"   - Embedding model: {settings.gemini_embedding_model}")
    print(f"   - Generation model: {settings.gemini_model}")
    
    return True


async def main():
    """Run all tests."""
    print("🚀 Starting Gemini integration tests...\n")
    
    # Check configuration first
    if not check_configuration():
        print("\n❌ Configuration check failed. Please fix configuration and try again.")
        return
    
    # Run all tests
    tests = [
        ("Health Check", test_health_check),
        ("Embedding Generation", test_embedding_generation),
        ("Batch Embedding Generation", test_batch_embedding_generation),
        ("Summary Generation", test_summary_generation),
        ("Rate Limiting", test_rate_limiting),
        ("Error Handling", test_error_handling),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        try:
            success = await test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {str(e)}")
            results.append((test_name, False))
    
    # Print summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print('='*50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
        if success:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Gemini integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    asyncio.run(main())
