#!/usr/bin/env python3
"""
Alpine Container Testing Script

This script tests the Alpine Docker container functionality to ensure:
1. MoRAG can be imported successfully
2. Basic configuration is working
3. Services can be started
4. API endpoints are accessible
"""

import os
import sys
import time
import requests
import subprocess
from pathlib import Path

# Colors for output
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'  # No Color

def log_info(message):
    print(f"{BLUE}[INFO]{NC} {message}")

def log_success(message):
    print(f"{GREEN}[SUCCESS]{NC} {message}")

def log_warning(message):
    print(f"{YELLOW}[WARNING]{NC} {message}")

def log_error(message):
    print(f"{RED}[ERROR]{NC} {message}")

def test_python_import():
    """Test if MoRAG can be imported successfully"""
    log_info("Testing MoRAG import...")
    try:
        import morag
        log_success("MoRAG import successful")
        return True
    except ImportError as e:
        log_error(f"MoRAG import failed: {e}")
        return False

def test_basic_imports():
    """Test basic MoRAG component imports"""
    log_info("Testing basic component imports...")
    
    components = [
        "morag.core.config",
        "morag.api.main",
        "morag.processors.document",
        "morag.services.embedding",
        "morag.converters.universal_converter"
    ]
    
    success_count = 0
    for component in components:
        try:
            __import__(component)
            log_success(f"✓ {component}")
            success_count += 1
        except ImportError as e:
            log_error(f"✗ {component}: {e}")
    
    log_info(f"Import test: {success_count}/{len(components)} components imported successfully")
    return success_count == len(components)

def test_environment_config():
    """Test environment configuration"""
    log_info("Testing environment configuration...")
    
    try:
        from morag.core.config import settings
        log_success("Settings loaded successfully")
        
        # Check critical settings
        if hasattr(settings, 'gemini_api_key'):
            if settings.gemini_api_key:
                log_success("✓ Gemini API key configured")
            else:
                log_warning("⚠ Gemini API key not set")
        
        if hasattr(settings, 'qdrant_host'):
            log_info(f"Qdrant host: {settings.qdrant_host}")
        
        if hasattr(settings, 'redis_url'):
            log_info(f"Redis URL: {settings.redis_url}")
        
        return True
    except Exception as e:
        log_error(f"Environment configuration test failed: {e}")
        return False

def test_redis_connection():
    """Test Redis connection"""
    log_info("Testing Redis connection...")
    
    try:
        import redis
        from morag.core.config import settings
        
        # Try to connect to Redis
        r = redis.from_url(settings.redis_url)
        r.ping()
        log_success("✓ Redis connection successful")
        return True
    except Exception as e:
        log_error(f"Redis connection failed: {e}")
        return False

def test_qdrant_connection():
    """Test Qdrant connection (external server)"""
    log_info("Testing Qdrant connection...")
    
    try:
        from morag.core.config import settings
        
        # Try to connect to Qdrant health endpoint
        qdrant_url = f"http://{settings.qdrant_host}:{settings.qdrant_port}/health"
        response = requests.get(qdrant_url, timeout=5)
        
        if response.status_code == 200:
            log_success("✓ Qdrant connection successful")
            return True
        else:
            log_error(f"Qdrant health check failed: {response.status_code}")
            return False
    except Exception as e:
        log_error(f"Qdrant connection failed: {e}")
        log_warning("Make sure your external Qdrant server is running and accessible")
        return False

def test_api_startup():
    """Test if the API can start (basic validation)"""
    log_info("Testing API startup capability...")
    
    try:
        from morag.api.main import app
        log_success("✓ FastAPI app can be imported")
        
        # Test if we can create the app instance
        if app:
            log_success("✓ FastAPI app instance created")
            return True
        else:
            log_error("FastAPI app instance is None")
            return False
    except Exception as e:
        log_error(f"API startup test failed: {e}")
        return False

def test_basic_functionality():
    """Test basic MoRAG functionality"""
    log_info("Testing basic functionality...")
    
    try:
        from morag.converters.universal_converter import UniversalConverter
        
        # Create converter instance
        converter = UniversalConverter()
        log_success("✓ UniversalConverter created")
        
        # Test text conversion
        test_text = "This is a test document for Alpine container validation."
        result = converter.convert_text(test_text, source_format="txt")
        
        if result and result.content:
            log_success("✓ Basic text conversion works")
            return True
        else:
            log_error("Basic text conversion failed")
            return False
    except Exception as e:
        log_error(f"Basic functionality test failed: {e}")
        return False

def run_all_tests():
    """Run all container tests"""
    log_info("Starting Alpine Container Test Suite...")
    print("=" * 60)
    
    tests = [
        ("Python Import", test_python_import),
        ("Basic Imports", test_basic_imports),
        ("Environment Config", test_environment_config),
        ("Redis Connection", test_redis_connection),
        ("Qdrant Connection", test_qdrant_connection),
        ("API Startup", test_api_startup),
        ("Basic Functionality", test_basic_functionality)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'-' * 40}")
        log_info(f"Running: {test_name}")
        results[test_name] = test_func()
    
    # Summary
    print(f"\n{'=' * 60}")
    log_info("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        color = GREEN if result else RED
        print(f"{color}{status:>6}{NC} | {test_name}")
        if result:
            passed += 1
    
    print(f"\n{BLUE}Results: {passed}/{total} tests passed{NC}")
    
    if passed == total:
        log_success("🎉 All tests passed! Alpine container is working correctly.")
        return True
    else:
        log_warning(f"⚠ {total - passed} test(s) failed. Check the issues above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
