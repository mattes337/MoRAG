#!/usr/bin/env python3
"""
Docker verification test script.
This script can be run inside Docker containers to verify the environment is working correctly.
"""

import os
import sys
import json


def check_environment():
    """Check that required environment variables are available."""
    print("🔍 Checking Environment Variables")
    print("-" * 40)
    
    required_vars = ['QDRANT_COLLECTION_NAME']
    optional_vars = ['QDRANT_HOST', 'QDRANT_PORT', 'REDIS_URL', 'GEMINI_API_KEY']
    
    missing_required = []
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: NOT SET")
            missing_required.append(var)
    
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            # Mask sensitive values
            if 'KEY' in var or 'PASSWORD' in var:
                display_value = f"{value[:8]}..." if len(value) > 8 else "***"
            else:
                display_value = value
            print(f"✅ {var}: {display_value}")
        else:
            print(f"⚠️  {var}: NOT SET (optional)")
    
    return len(missing_required) == 0


def test_settings_import():
    """Test that settings can be imported and accessed."""
    print("\n🔧 Testing Settings Import")
    print("-" * 40)
    
    try:
        print("Importing morag_core.config...")
        from morag_core.config import settings
        print("✅ Settings imported successfully")
        
        print("Accessing qdrant_collection_name...")
        collection_name = settings.qdrant_collection_name
        print(f"✅ Collection name: {collection_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Settings test failed: {e}")
        return False


def test_worker_functionality():
    """Test that worker can be imported and initialized."""
    print("\n👷 Testing Worker Functionality")
    print("-" * 40)
    
    try:
        print("Importing morag.worker...")
        import morag.worker
        print("✅ Worker imported successfully")
        
        print("Checking Celery app...")
        app = morag.worker.celery_app
        print(f"✅ Celery app: {app.main}")
        
        # Test that we can get the API instance
        print("Testing get_morag_api function...")
        api = morag.worker.get_morag_api()
        print(f"✅ MoRAG API instance: {type(api).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Worker test failed: {e}")
        return False


def test_api_server():
    """Test that API server can be created."""
    print("\n🌐 Testing API Server")
    print("-" * 40)
    
    try:
        print("Importing morag.server...")
        import morag.server
        print("✅ Server imported successfully")
        
        print("Creating FastAPI app...")
        app = morag.server.create_app()
        print("✅ FastAPI app created successfully")
        
        print(f"✅ App has {len(app.routes)} routes configured")
        
        return True
        
    except Exception as e:
        print(f"❌ API server test failed: {e}")
        return False


def test_package_availability():
    """Test that all MoRAG packages are available."""
    print("\n📦 Testing Package Availability")
    print("-" * 40)
    
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
    
    available = 0
    total = len(packages)
    
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package}")
            available += 1
        except ImportError as e:
            print(f"❌ {package}: {e}")
        except Exception as e:
            print(f"⚠️  {package}: {e}")
    
    print(f"\n📊 Package availability: {available}/{total}")
    return available >= total * 0.8  # At least 80% should be available


def main():
    """Run all verification tests."""
    print("🐳 Docker Environment Verification")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Container type: {os.getenv('CONTAINER_TYPE', 'unknown')}")
    print()
    
    tests = [
        ("Environment Variables", check_environment),
        ("Settings Import", test_settings_import),
        ("Worker Functionality", test_worker_functionality),
        ("API Server", test_api_server),
        ("Package Availability", test_package_availability),
    ]
    
    results = {}
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed += 1
                print(f"\n✅ {test_name}: PASSED")
            else:
                failed += 1
                print(f"\n❌ {test_name}: FAILED")
        except Exception as e:
            results[test_name] = False
            failed += 1
            print(f"\n❌ {test_name}: FAILED with exception: {e}")
    
    print("\n" + "=" * 60)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {passed/len(tests)*100:.1f}%")
    
    print("\nDetailed Results:")
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
    
    # Determine overall status
    critical_tests = ["Environment Variables", "Settings Import"]
    critical_passed = all(results.get(test, False) for test in critical_tests)
    
    if critical_passed and passed >= len(tests) * 0.8:
        print("\n🎉 VERIFICATION SUCCESSFUL!")
        print("The Docker container is ready for production use.")
        return_code = 0
    elif critical_passed:
        print("\n⚠️  VERIFICATION PARTIALLY SUCCESSFUL")
        print("Critical functionality is working, but some optional features may be unavailable.")
        return_code = 0
    else:
        print("\n💥 VERIFICATION FAILED!")
        print("Critical issues detected. Container may not function correctly.")
        return_code = 1
    
    print("\nNext steps:")
    if return_code == 0:
        print("• Container is ready to handle requests")
        print("• All services should start successfully")
        print("• Environment variables are configured correctly")
    else:
        print("• Check environment variable configuration")
        print("• Verify .env file is properly mounted")
        print("• Check container logs for additional errors")
    
    return return_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
