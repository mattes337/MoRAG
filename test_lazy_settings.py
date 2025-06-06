#!/usr/bin/env python3
"""
Test script to verify that settings are loaded lazily and don't fail at import time.
"""

import os
import sys
from unittest.mock import patch


def test_import_without_env_var():
    """Test that modules can be imported without QDRANT_COLLECTION_NAME set."""
    print("Testing module imports without QDRANT_COLLECTION_NAME...")
    
    # Clear environment variable
    with patch.dict(os.environ, {}, clear=True):
        if 'QDRANT_COLLECTION_NAME' in os.environ:
            del os.environ['QDRANT_COLLECTION_NAME']
        
        try:
            # Test core config import
            print("1. Testing morag_core.config import...")
            from packages.morag_core.src.morag_core.config import settings
            print("   ✅ morag_core.config imported successfully")
            
            # Test worker import
            print("2. Testing morag.worker import...")
            sys.path.insert(0, 'packages/morag/src')
            from morag.worker import celery_app
            print("   ✅ morag.worker imported successfully")
            
            # Test file handling import
            print("3. Testing morag_core.utils.file_handling import...")
            from packages.morag_core.src.morag_core.utils.file_handling import generate_temp_path
            print("   ✅ morag_core.utils.file_handling imported successfully")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Import failed: {e}")
            return False


def test_settings_access_with_env_var():
    """Test that settings work correctly when environment variable is set."""
    print("\nTesting settings access with QDRANT_COLLECTION_NAME set...")
    
    with patch.dict(os.environ, {'QDRANT_COLLECTION_NAME': 'test_collection'}):
        try:
            # Test accessing settings
            print("1. Testing settings access...")
            from packages.morag_core.src.morag_core.config import settings
            collection_name = settings.qdrant_collection_name
            if collection_name == 'test_collection':
                print(f"   ✅ Settings access successful: {collection_name}")
            else:
                print(f"   ❌ Wrong collection name: {collection_name}")
                return False
            
            # Test file handling with settings
            print("2. Testing file handling with settings...")
            from packages.morag_core.src.morag_core.utils.file_handling import generate_temp_path
            temp_path = generate_temp_path("test_", ".tmp")
            print(f"   ✅ Temp path generated: {temp_path}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Settings access failed: {e}")
            return False


def test_settings_access_without_env_var():
    """Test that settings fail appropriately when accessed without environment variable."""
    print("\nTesting settings access without QDRANT_COLLECTION_NAME...")
    
    with patch.dict(os.environ, {}, clear=True):
        if 'QDRANT_COLLECTION_NAME' in os.environ:
            del os.environ['QDRANT_COLLECTION_NAME']
        
        try:
            # Test accessing settings - this should fail
            print("1. Testing settings access (should fail)...")
            from packages.morag_core.src.morag_core.config import settings
            collection_name = settings.qdrant_collection_name
            print(f"   ❌ Should have failed but got: {collection_name}")
            return False
            
        except ValueError as e:
            if "QDRANT_COLLECTION_NAME environment variable is required" in str(e):
                print(f"   ✅ Correctly failed: {e}")
                return True
            else:
                print(f"   ❌ Wrong error: {e}")
                return False
        except Exception as e:
            print(f"   ❌ Unexpected error: {e}")
            return False


def main():
    """Run all tests."""
    print("Testing Lazy Settings Loading")
    print("=" * 50)
    
    tests = [
        ("Import without env var", test_import_without_env_var),
        ("Settings with env var", test_settings_access_with_env_var),
        ("Settings without env var", test_settings_access_without_env_var),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✅ {test_name}: PASSED")
            else:
                failed += 1
                print(f"\n❌ {test_name}: FAILED")
        except Exception as e:
            failed += 1
            print(f"\n❌ {test_name}: FAILED with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✅ All tests passed! Lazy settings loading is working correctly.")
        print("\nThis means:")
        print("• Modules can be imported without environment variables")
        print("• Settings are only validated when actually accessed")
        print("• Docker containers should start without immediate failures")
        return True
    else:
        print("❌ Some tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
