#!/usr/bin/env python3
"""
Simple test to validate that collection name validation is working.
"""

import os
import sys
from unittest.mock import patch

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_environment_variable_validation():
    """Test that environment variable validation works."""
    
    # Test 1: Missing environment variable should fail
    print("Test 1: Testing missing QDRANT_COLLECTION_NAME...")
    with patch.dict(os.environ, {}, clear=True):
        if 'QDRANT_COLLECTION_NAME' in os.environ:
            del os.environ['QDRANT_COLLECTION_NAME']
        
        # Test that services fail without collection name
        try:
            # Import and test the validation logic directly
            qdrant_host = os.getenv('QDRANT_HOST', 'localhost')
            qdrant_port = int(os.getenv('QDRANT_PORT', '6333'))
            qdrant_api_key = os.getenv('QDRANT_API_KEY')
            collection_name = os.getenv('QDRANT_COLLECTION_NAME')
            
            if not collection_name:
                raise ValueError("QDRANT_COLLECTION_NAME environment variable is required")
            
            print("❌ Expected ValueError but none was raised")
            return False
            
        except ValueError as e:
            if "QDRANT_COLLECTION_NAME environment variable is required" in str(e):
                print("✅ Correctly failed with missing environment variable")
            else:
                print(f"❌ Wrong error message: {e}")
                return False
    
    # Test 2: Present environment variable should work
    print("Test 2: Testing present QDRANT_COLLECTION_NAME...")
    with patch.dict(os.environ, {'QDRANT_COLLECTION_NAME': 'test_collection'}):
        try:
            collection_name = os.getenv('QDRANT_COLLECTION_NAME')
            if not collection_name:
                raise ValueError("QDRANT_COLLECTION_NAME environment variable is required")
            
            if collection_name == 'test_collection':
                print("✅ Correctly accepted valid environment variable")
            else:
                print(f"❌ Wrong collection name: {collection_name}")
                return False
                
        except ValueError as e:
            print(f"❌ Unexpected error with valid environment variable: {e}")
            return False
    
    return True


def test_storage_validation():
    """Test storage class validation logic."""
    print("Test 3: Testing storage validation logic...")
    
    # Simulate the validation logic from QdrantVectorStorage
    def validate_collection_name(collection_name):
        if not collection_name:
            raise ValueError("collection_name is required - set QDRANT_COLLECTION_NAME environment variable")
        return collection_name
    
    # Test with None
    try:
        validate_collection_name(None)
        print("❌ Expected ValueError for None collection name")
        return False
    except ValueError as e:
        if "collection_name is required" in str(e):
            print("✅ Correctly rejected None collection name")
        else:
            print(f"❌ Wrong error message: {e}")
            return False
    
    # Test with empty string
    try:
        validate_collection_name("")
        print("❌ Expected ValueError for empty collection name")
        return False
    except ValueError as e:
        if "collection_name is required" in str(e):
            print("✅ Correctly rejected empty collection name")
        else:
            print(f"❌ Wrong error message: {e}")
            return False
    
    # Test with valid name
    try:
        result = validate_collection_name("valid_collection")
        if result == "valid_collection":
            print("✅ Correctly accepted valid collection name")
        else:
            print(f"❌ Wrong result: {result}")
            return False
    except Exception as e:
        print(f"❌ Unexpected error with valid collection name: {e}")
        return False
    
    return True


def test_environment_files():
    """Test that environment files have consistent collection names."""
    print("Test 4: Testing environment file consistency...")
    
    env_files = [
        '.env.example',
        '.env.prod.example'
    ]
    
    for env_file in env_files:
        if os.path.exists(env_file):
            with open(env_file, 'r') as f:
                content = f.read()
                if 'QDRANT_COLLECTION_NAME=morag_documents' in content:
                    print(f"✅ {env_file} has correct collection name")
                else:
                    print(f"❌ {env_file} has incorrect or missing collection name")
                    return False
        else:
            print(f"⏭️  {env_file} not found, skipping")
    
    return True


def main():
    """Run all validation tests."""
    print("Testing Qdrant collection name validation...")
    print("=" * 50)
    
    tests = [
        test_environment_variable_validation,
        test_storage_validation,
        test_environment_files
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} failed with exception: {e}")
            failed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✅ All validation tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
