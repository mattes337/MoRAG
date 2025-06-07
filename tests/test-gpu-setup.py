#!/usr/bin/env python3
"""Comprehensive GPU worker setup test script."""

import os
import sys
import subprocess
import tempfile
import requests
from pathlib import Path

def test_gpu_availability():
    """Test if GPU is available and accessible."""
    print("🔍 Testing GPU availability...")
    
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            gpu_info = result.stdout.strip()
            print(f"✅ GPU detected: {gpu_info}")
            return True
        else:
            print(f"❌ nvidia-smi failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ nvidia-smi command timed out")
        return False
    except FileNotFoundError:
        print("❌ nvidia-smi not found - NVIDIA drivers may not be installed")
        return False

def test_cuda_installation():
    """Test CUDA installation."""
    print("🔍 Testing CUDA installation...")
    
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            cuda_version = result.stdout.strip().split('\n')[-1]
            print(f"✅ CUDA installed: {cuda_version}")
            return True
        else:
            print("❌ CUDA not found or not working")
            return False
    except subprocess.TimeoutExpired:
        print("❌ nvcc command timed out")
        return False
    except FileNotFoundError:
        print("⚠️  nvcc not found - CUDA may not be installed or not in PATH")
        return False

def test_python_packages():
    """Test required Python packages."""
    print("🔍 Testing Python packages...")
    
    required_packages = [
        'celery',
        'redis',
        'structlog',
        'httpx',
        'aiofiles'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} installed")
        except ImportError:
            print(f"❌ {package} not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("   Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def test_environment_variables():
    """Test required environment variables."""
    print("🔍 Testing environment variables...")
    
    required_vars = [
        'REDIS_URL',
        'QDRANT_URL',
        'QDRANT_COLLECTION_NAME',
        'GEMINI_API_KEY'
    ]
    
    missing_vars = []
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Mask sensitive values
            if 'API_KEY' in var or 'PASSWORD' in var:
                masked_value = value[:8] + '...' if len(value) > 8 else '***'
                print(f"✅ {var}: {masked_value}")
            else:
                print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: Not set")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        return False
    
    return True

def test_file_access():
    """Test file access permissions."""
    print("🔍 Testing file access...")
    
    temp_dir = os.getenv('TEMP_DIR', '/tmp')
    
    try:
        # Test write access
        test_file = Path(temp_dir) / 'gpu_worker_test.tmp'
        test_file.write_text('test')
        test_file.unlink()
        print(f"✅ Temp directory writable: {temp_dir}")
        return True
    except Exception as e:
        print(f"❌ Cannot write to temp directory {temp_dir}: {e}")
        return False

def test_server_connectivity():
    """Test connectivity to main server."""
    print("🔍 Testing server connectivity...")
    
    redis_url = os.getenv('REDIS_URL')
    qdrant_url = os.getenv('QDRANT_URL')
    
    if not redis_url or not qdrant_url:
        print("❌ REDIS_URL or QDRANT_URL not set")
        return False
    
    # Test Redis
    try:
        result = subprocess.run(['redis-cli', '-u', redis_url, 'ping'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and 'PONG' in result.stdout:
            print("✅ Redis connection successful")
        else:
            print(f"❌ Redis connection failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Redis test failed: {e}")
        return False
    
    # Test Qdrant
    try:
        response = requests.get(f"{qdrant_url}/collections", timeout=10)
        if response.status_code == 200:
            print("✅ Qdrant connection successful")
        else:
            print(f"❌ Qdrant connection failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Qdrant test failed: {e}")
        return False
    
    return True

def main():
    """Run comprehensive GPU worker setup tests."""
    print("🧪 MoRAG GPU Worker Setup Test")
    print("=" * 40)
    
    tests = [
        ("GPU Availability", test_gpu_availability),
        ("CUDA Installation", test_cuda_installation),
        ("Python Packages", test_python_packages),
        ("Environment Variables", test_environment_variables),
        ("File Access", test_file_access),
        ("Server Connectivity", test_server_connectivity)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        if test_func():
            passed += 1
        print()
    
    # Results
    print("📊 Test Results")
    print("=" * 20)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All tests passed! GPU worker is ready to start.")
        print("\n🚀 Next steps:")
        print("1. Start the GPU worker: ./scripts/start-gpu-worker.sh")
        print("2. Test with: python tests/test-gpu-workers.py")
        sys.exit(0)
    else:
        print("❌ Some tests failed! Please fix the issues before starting GPU worker.")
        sys.exit(1)

if __name__ == "__main__":
    main()
