#!/usr/bin/env python3
"""
Test script to verify that the worker can be imported without QDRANT_COLLECTION_NAME.
"""

import os
import sys
from unittest.mock import patch


def test_worker_import():
    """Test that worker module can be imported without environment variable."""
    print("Testing worker import without QDRANT_COLLECTION_NAME...")
    
    # Clear environment variable
    with patch.dict(os.environ, {}, clear=True):
        if 'QDRANT_COLLECTION_NAME' in os.environ:
            del os.environ['QDRANT_COLLECTION_NAME']
        
        try:
            # This should work now with lazy loading
            print("Attempting to import morag.worker...")
            import morag.worker
            print("✅ morag.worker imported successfully!")
            
            # Test that celery app exists
            if hasattr(morag.worker, 'celery_app'):
                print("✅ celery_app found in worker module")
            else:
                print("❌ celery_app not found in worker module")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Import failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_worker_with_env_var():
    """Test that worker works correctly when environment variable is set."""
    print("\nTesting worker with QDRANT_COLLECTION_NAME set...")
    
    with patch.dict(os.environ, {'QDRANT_COLLECTION_NAME': 'test_collection'}):
        try:
            # Import worker
            import morag.worker
            print("✅ Worker imported with environment variable")
            
            # Test that we can access the celery app
            app = morag.worker.celery_app
            print(f"✅ Celery app accessible: {app.main}")
            
            return True
            
        except Exception as e:
            print(f"❌ Worker test failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Run the tests."""
    print("Testing Worker Import with Lazy Settings")
    print("=" * 50)
    
    tests = [
        test_worker_import,
        test_worker_with_env_var,
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
            print(f"Test failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✅ All tests passed! Worker can be imported without environment variables.")
        print("This should fix the Docker container startup issue.")
        return True
    else:
        print("❌ Some tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
