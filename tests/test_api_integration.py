#!/usr/bin/env python3
"""Test script for API key integration with server endpoints."""

import asyncio
import requests
import json
import os
import sys
from pathlib import Path

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_USER_ID = "test_user_123"

def test_api_key_creation():
    """Test API key creation endpoint."""
    print("1. Testing API key creation...")
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/auth/create-key",
            data={
                "user_id": TEST_USER_ID,
                "description": "Test API key for remote GPU workers",
                "expires_days": 30
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            api_key = data.get("api_key")
            print(f"✅ API key created: {api_key[:8]}...")
            return api_key
        else:
            print(f"❌ API key creation failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ API key creation error: {e}")
        return None

def test_api_key_validation(api_key):
    """Test API key validation endpoint."""
    print("\n2. Testing API key validation...")
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/auth/validate-key",
            data={"api_key": api_key}
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("valid"):
                print(f"✅ API key validation successful: {data}")
                return True
            else:
                print(f"❌ API key validation failed: {data}")
                return False
        else:
            print(f"❌ API key validation request failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ API key validation error: {e}")
        return False

def test_queue_info_anonymous():
    """Test queue info endpoint without authentication."""
    print("\n3. Testing queue info (anonymous)...")
    
    try:
        response = requests.get(f"{BASE_URL}/api/v1/auth/queue-info")
        
        if response.status_code == 200:
            data = response.json()
            if not data.get("authenticated"):
                print(f"✅ Anonymous queue info: {data}")
                return True
            else:
                print(f"❌ Expected anonymous access, got authenticated: {data}")
                return False
        else:
            print(f"❌ Queue info request failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Queue info error: {e}")
        return False

def test_queue_info_authenticated(api_key):
    """Test queue info endpoint with authentication."""
    print("\n4. Testing queue info (authenticated)...")
    
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(f"{BASE_URL}/api/v1/auth/queue-info", headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("authenticated") and data.get("user_id") == TEST_USER_ID:
                print(f"✅ Authenticated queue info: {data}")
                return True
            else:
                print(f"❌ Expected authenticated access for {TEST_USER_ID}, got: {data}")
                return False
        else:
            print(f"❌ Authenticated queue info request failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Authenticated queue info error: {e}")
        return False

def test_file_processing_with_gpu(api_key):
    """Test file processing with GPU flag and authentication."""
    print("\n5. Testing file processing with GPU flag...")
    
    try:
        # Create a simple test file
        test_content = "This is a test document for GPU processing."
        test_file_path = Path("test_gpu_file.txt")
        test_file_path.write_text(test_content)
        
        headers = {"Authorization": f"Bearer {api_key}"}
        
        with open(test_file_path, 'rb') as f:
            files = {'file': ('test_gpu_file.txt', f, 'text/plain')}
            data = {
                'content_type': 'document',
                'gpu': 'true'
            }
            
            response = requests.post(
                f"{BASE_URL}/process/file",
                headers=headers,
                files=files,
                data=data
            )
        
        # Clean up test file
        test_file_path.unlink()
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                print(f"✅ GPU file processing successful: {data.get('processing_time')}s")
                return True
            else:
                print(f"❌ GPU file processing failed: {data}")
                return False
        else:
            print(f"❌ GPU file processing request failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ GPU file processing error: {e}")
        return False

def test_url_processing_with_gpu(api_key):
    """Test URL processing with GPU flag and authentication."""
    print("\n6. Testing URL processing with GPU flag...")
    
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        
        payload = {
            "url": "https://example.com",
            "content_type": "web",
            "gpu": True
        }
        
        response = requests.post(
            f"{BASE_URL}/process/url",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                print(f"✅ GPU URL processing successful: {data.get('processing_time')}s")
                return True
            else:
                print(f"❌ GPU URL processing failed: {data}")
                return False
        else:
            print(f"❌ GPU URL processing request failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ GPU URL processing error: {e}")
        return False

def main():
    """Run all API integration tests."""
    print("🚀 Starting API Key Integration Tests\n")
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print("❌ Server health check failed. Make sure the MoRAG server is running.")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server at {BASE_URL}. Make sure it's running.")
        print(f"   Error: {e}")
        return False
    
    print("✅ Server is running and accessible\n")
    
    # Run tests
    results = []
    
    # Test 1: Create API key
    api_key = test_api_key_creation()
    results.append(api_key is not None)
    
    if not api_key:
        print("\n❌ Cannot continue tests without API key")
        return False
    
    # Test 2: Validate API key
    results.append(test_api_key_validation(api_key))
    
    # Test 3: Queue info (anonymous)
    results.append(test_queue_info_anonymous())
    
    # Test 4: Queue info (authenticated)
    results.append(test_queue_info_authenticated(api_key))
    
    # Test 5: File processing with GPU
    results.append(test_file_processing_with_gpu(api_key))
    
    # Test 6: URL processing with GPU
    results.append(test_url_processing_with_gpu(api_key))
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - API Key Integration successful!")
        return True
    else:
        print(f"\n❌ {total-passed} TESTS FAILED - Check implementation")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
