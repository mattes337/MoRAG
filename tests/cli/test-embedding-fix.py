#!/usr/bin/env python3
"""Test script to verify GeminiEmbeddingService fixes."""

import asyncio
import sys
import os
from pathlib import Path

# Add the packages to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "packages" / "morag-core" / "src"))
sys.path.insert(0, str(project_root / "packages" / "morag-services" / "src"))

async def test_embedding_service_instantiation():
    """Test that GeminiEmbeddingService can be instantiated without errors."""
    try:
        from morag_services.embedding import GeminiEmbeddingService

        print("✅ Successfully imported GeminiEmbeddingService")

        # Try to instantiate (with dummy API key)
        service = GeminiEmbeddingService(
            api_key="dummy_key_for_testing",
            embedding_model="text-embedding-004",
            generation_model="gemini-2.0-flash-001"
        )

        print("✅ Successfully instantiated GeminiEmbeddingService")

        # Check that it has all required methods
        required_methods = [
            'initialize', 'shutdown', 'health_check',
            'generate_embedding', 'generate_embeddings',
            'get_embedding_dimension', 'get_supported_models', 'get_max_tokens'
        ]

        missing_methods = []
        for method in required_methods:
            if not hasattr(service, method):
                missing_methods.append(method)

        if missing_methods:
            print(f"❌ Missing methods: {missing_methods}")
            return False
        else:
            print("✅ All required methods are present")

        # Test method signatures and return types
        try:
            dimension = service.get_embedding_dimension()
            print(f"✅ get_embedding_dimension() works: {dimension}")

            models = service.get_supported_models()
            print(f"✅ get_supported_models() works: {models}")

            max_tokens = service.get_max_tokens()
            print(f"✅ get_max_tokens() works: {max_tokens}")

        except Exception as e:
            print(f"❌ Method call failed: {e}")
            return False

        # Test health check (without real API key)
        try:
            health = await service.health_check()
            print(f"✅ Health check method works (status: {health.get('status', 'unknown')})")
        except Exception as e:
            print(f"⚠️  Health check failed (expected without real API key): {e}")

        return True

    except Exception as e:
        error_msg = str(e)
        if "Can't instantiate abstract class GeminiEmbeddingService" in error_msg:
            print("❌ The abstract class error is still present!")
            print("   The fix may not have been applied correctly.")
            return False
        else:
            print(f"❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return False

async def main():
    """Main test function."""
    print("Testing GeminiEmbeddingService fixes...")
    print("=" * 50)

    success = await test_embedding_service_instantiation()

    print("=" * 50)
    if success:
        print("✅ All tests passed! GeminiEmbeddingService should work now.")
        print("\nThe abstract class error has been fixed!")
        print("You can now use the embedding service without instantiation errors.")
    else:
        print("❌ Tests failed. Check the errors above.")

    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
