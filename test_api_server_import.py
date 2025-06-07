#!/usr/bin/env python3
"""
Test script to verify that the API server can be imported without triggering settings validation.
"""

import os
import sys
from unittest.mock import patch


def test_server_import_without_env_var():
    """Test that server module can be imported without QDRANT_COLLECTION_NAME."""
    print("Testing server import without QDRANT_COLLECTION_NAME...")
    
    # Clear environment variable
    with patch.dict(os.environ, {}, clear=True):
        if 'QDRANT_COLLECTION_NAME' in os.environ:
            del os.environ['QDRANT_COLLECTION_NAME']
        
        try:
            # This should work now with lazy loading
            print("Attempting to import morag.server...")
            import morag.server
            print("✅ morag.server imported successfully!")
            
            # Test that create_app function exists
            if hasattr(morag.server, 'create_app'):
                print("✅ create_app function found")
            else:
                print("❌ create_app function not found")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Import failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_create_app_without_env_var():
    """Test that create_app can be called without QDRANT_COLLECTION_NAME."""
    print("\nTesting create_app without QDRANT_COLLECTION_NAME...")
    
    # Clear environment variable
    with patch.dict(os.environ, {}, clear=True):
        if 'QDRANT_COLLECTION_NAME' in os.environ:
            del os.environ['QDRANT_COLLECTION_NAME']
        
        try:
            # Import server
            import morag.server
            
            # This should work now - create_app should not trigger validation
            print("Attempting to call create_app...")
            app = morag.server.create_app()
            print("✅ create_app called successfully!")
            
            # Test that we got a FastAPI app
            if hasattr(app, 'routes'):
                print(f"✅ FastAPI app created with {len(app.routes)} routes")
            else:
                print("❌ Invalid app object returned")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ create_app failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_api_access_with_env_var():
    """Test that API access works when environment variable is set."""
    print("\nTesting API access with QDRANT_COLLECTION_NAME set...")
    
    with patch.dict(os.environ, {'QDRANT_COLLECTION_NAME': 'test_collection'}):
        try:
            # Import and create app
            import morag.server
            app = morag.server.create_app()
            print("✅ App created with environment variable")
            
            # The actual API initialization should happen when endpoints are called
            # For now, just verify the app was created successfully
            if hasattr(app, 'routes') and len(app.routes) > 0:
                print(f"✅ App has {len(app.routes)} routes configured")
                return True
            else:
                print("❌ App has no routes")
                return False
            
        except Exception as e:
            print(f"❌ API access failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Run all tests."""
    print("Testing API Server Import with Lazy Loading")
    print("=" * 50)
    
    tests = [
        ("Server import", test_server_import_without_env_var),
        ("Create app", test_create_app_without_env_var),
        ("API access with env var", test_api_access_with_env_var),
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
        print("✅ All tests passed! API server can be imported and created without environment variables.")
        print("This should fix the Docker container startup issue for the API service.")
        return True
    else:
        print("❌ Some tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
