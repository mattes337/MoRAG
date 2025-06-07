#!/usr/bin/env python3
"""Test script to verify QdrantVectorStorage fixes."""

import asyncio
import sys
import os
from pathlib import Path

# Add the packages to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "packages" / "morag-core" / "src"))
sys.path.insert(0, str(project_root / "packages" / "morag-services" / "src"))

async def test_qdrant_instantiation():
    """Test that QdrantVectorStorage can be instantiated without errors."""
    try:
        from morag_services.storage import QdrantVectorStorage
        
        print("✅ Successfully imported QdrantVectorStorage")
        
        # Try to instantiate
        storage = QdrantVectorStorage(
            host="localhost",
            port=6333,
            collection_name="test_collection"
        )
        
        print("✅ Successfully instantiated QdrantVectorStorage")
        
        # Check that it has all required methods
        required_methods = [
            'initialize', 'shutdown', 'health_check',
            'put_object', 'get_object', 'delete_object', 'list_objects', 
            'get_object_metadata', 'object_exists',
            'add_vectors', 'search_vectors', 'delete_vectors', 'update_vector_metadata'
        ]
        
        missing_methods = []
        for method in required_methods:
            if not hasattr(storage, method):
                missing_methods.append(method)
        
        if missing_methods:
            print(f"❌ Missing methods: {missing_methods}")
            return False
        else:
            print("✅ All required methods are present")
        
        # Test health check (without connecting to actual Qdrant)
        try:
            health = await storage.health_check()
            print(f"✅ Health check method works (status: {health.get('status', 'unknown')})")
        except Exception as e:
            print(f"⚠️  Health check failed (expected if Qdrant not running): {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function."""
    print("Testing QdrantVectorStorage fixes...")
    print("=" * 50)
    
    success = await test_qdrant_instantiation()
    
    print("=" * 50)
    if success:
        print("✅ All tests passed! QdrantVectorStorage should work now.")
        print("\nThe abstract class error has been fixed!")
        print("You can now run MoRAG workers without the instantiation error.")
    else:
        print("❌ Tests failed. Check the errors above.")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
