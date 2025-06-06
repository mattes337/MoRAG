#!/usr/bin/env python3
"""
Test script to verify Docker environment variables are working.
This script can be run inside a Docker container to check environment setup.
"""

import os
import sys


def test_environment_variables():
    """Test that required environment variables are available."""
    print("Testing Docker Environment Variables")
    print("=" * 50)
    
    # List of important environment variables
    env_vars = [
        'QDRANT_COLLECTION_NAME',
        'QDRANT_HOST',
        'QDRANT_PORT',
        'REDIS_URL',
        'GEMINI_API_KEY',
        'TEMP_DIR',
        'LOG_LEVEL'
    ]
    
    print("Environment Variables:")
    print("-" * 30)
    
    missing_vars = []
    for var in env_vars:
        value = os.getenv(var)
        if value:
            # Mask sensitive values
            if 'KEY' in var or 'PASSWORD' in var:
                display_value = f"{value[:8]}..." if len(value) > 8 else "***"
            else:
                display_value = value
            print(f"✅ {var}: {display_value}")
        else:
            print(f"❌ {var}: NOT SET")
            missing_vars.append(var)
    
    print()
    
    if missing_vars:
        print(f"❌ Missing {len(missing_vars)} environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        return False
    else:
        print("✅ All environment variables are set!")
        return True


def test_settings_import():
    """Test that settings can be imported and accessed."""
    print("\nTesting Settings Import")
    print("-" * 30)
    
    try:
        # Test import
        print("1. Importing morag_core.config...")
        from morag_core.config import settings
        print("   ✅ Import successful")
        
        # Test access to critical setting
        print("2. Accessing qdrant_collection_name...")
        collection_name = settings.qdrant_collection_name
        print(f"   ✅ Collection name: {collection_name}")
        
        # Test other settings
        print("3. Accessing other settings...")
        temp_dir = settings.temp_dir
        log_level = settings.log_level
        print(f"   ✅ Temp dir: {temp_dir}")
        print(f"   ✅ Log level: {log_level}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Settings test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_worker_import():
    """Test that worker module can be imported."""
    print("\nTesting Worker Import")
    print("-" * 30)
    
    try:
        print("1. Importing morag.worker...")
        import morag.worker
        print("   ✅ Worker import successful")
        
        print("2. Checking celery app...")
        app = morag.worker.celery_app
        print(f"   ✅ Celery app: {app.main}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Worker import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_package_imports():
    """Test that all MoRAG packages can be imported."""
    print("\nTesting Package Imports")
    print("-" * 30)
    
    packages = [
        'morag_core',
        'morag_services',
        'morag_document',
        'morag_audio',
        'morag_video',
        'morag_image',
        'morag_web',
        'morag_youtube',
        'morag_embedding',
        'morag'
    ]
    
    failed_imports = []
    
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except Exception as e:
            print(f"❌ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed to import {len(failed_imports)} packages:")
        for package in failed_imports:
            print(f"   - {package}")
        return False
    else:
        print("\n✅ All packages imported successfully!")
        return True


def main():
    """Run all tests."""
    print("Docker Environment Test Suite")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print()
    
    tests = [
        ("Environment Variables", test_environment_variables),
        ("Settings Import", test_settings_import),
        ("Worker Import", test_worker_import),
        ("Package Imports", test_package_imports),
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
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✅ All tests passed! Docker environment is working correctly.")
        print("\nThe container should be able to:")
        print("• Load environment variables from .env file")
        print("• Import all MoRAG packages without errors")
        print("• Access settings with lazy validation")
        print("• Start Celery workers successfully")
        return True
    else:
        print("❌ Some tests failed!")
        print("\nCheck the following:")
        print("• Ensure .env file is present and readable")
        print("• Verify all required environment variables are set")
        print("• Check that all MoRAG packages are installed correctly")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
