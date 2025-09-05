#!/usr/bin/env python3
"""CLI test script for temp directory validation fixes."""

import sys
import os
import tempfile
import shutil
from pathlib import Path
import httpx
import time

# Add the packages to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "morag" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "morag-core" / "src"))

def test_api_startup_validation():
    """Test that API server validates temp directory on startup."""
    print("🚀 Testing API server startup validation...")
    
    try:
        # Try to connect to the API server
        response = httpx.get("http://localhost:8000/health", timeout=5)
        
        if response.status_code == 200:
            print("[OK] API server is running and healthy")
            health_data = response.json()
            print(f"   Health status: {health_data}")
            return True
        else:
            print(f"[FAIL] API server health check failed (status: {response.status_code})")
            return False
            
    except httpx.ConnectError:
        print("[FAIL] Cannot connect to API server (not running or startup failed)")
        print("   This could indicate temp directory validation failed during startup")
        return False
    except Exception as e:
        print(f"[FAIL] API server test error: {e}")
        return False

def test_file_upload_endpoint():
    """Test file upload endpoint to verify temp directory usage."""
    print("📤 Testing file upload endpoint...")
    
    try:
        # Create a test file
        test_content = "This is a test file for temp directory validation."
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            test_file_path = f.name
        
        try:
            # Upload the file
            with open(test_file_path, 'rb') as f:
                files = {'file': ('test.txt', f, 'text/plain')}
                response = httpx.post(
                    "http://localhost:8000/process/file",
                    files=files,
                    timeout=30
                )
            
            if response.status_code == 200:
                print("[OK] File upload and processing successful")
                result = response.json()
                print(f"   Processing success: {result.get('success', False)}")
                return True
            else:
                print(f"[FAIL] File upload failed (status: {response.status_code})")
                print(f"   Response: {response.text}")
                return False
                
        finally:
            # Clean up test file
            if os.path.exists(test_file_path):
                os.unlink(test_file_path)
                
    except Exception as e:
        print(f"[FAIL] File upload test error: {e}")
        return False

def test_temp_directory_consistency():
    """Test that temp directory is consistently used across components."""
    print("[FILES] Testing temp directory consistency...")
    
    try:
        from morag.utils.file_upload import get_upload_handler
        
        # Get upload handler
        handler = get_upload_handler()
        temp_dir = handler.temp_dir
        
        print(f"   Upload handler temp dir: {temp_dir}")
        
        # Check if it's using shared volume
        if str(temp_dir).startswith('/app/temp'):
            print("[OK] Using shared Docker volume (/app/temp)")
        elif str(temp_dir).startswith('./temp'):
            print("[OK] Using local development directory (./temp)")
        elif str(temp_dir).startswith('/tmp/'):
            print("[WARN]  Using system temp directory (may cause issues in containers)")
        else:
            print(f"❓ Using unknown temp directory: {temp_dir}")
        
        # Test write permissions
        test_file = temp_dir / "consistency_test.txt"
        test_file.write_text("consistency test")
        test_file.unlink()
        print("[OK] Write permissions verified")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Temp directory consistency test error: {e}")
        return False

def test_worker_file_access():
    """Test that workers can access files uploaded via API."""
    print("👷 Testing worker file access...")
    
    try:
        # Create a test file for ingestion (background processing)
        test_content = "This is a test file for worker access validation."
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            test_file_path = f.name
        
        try:
            # Submit file for ingestion (background processing)
            with open(test_file_path, 'rb') as f:
                files = {'file': ('worker_test.txt', f, 'text/plain')}
                data = {'source_type': 'document'}
                response = httpx.post(
                    "http://localhost:8000/api/v1/ingest/file",
                    files=files,
                    data=data,
                    timeout=30
                )
            
            if response.status_code == 200:
                result = response.json()
                task_id = result.get('task_id')
                print(f"[OK] File submitted for background processing (task: {task_id})")
                
                # Check task status after a short delay
                time.sleep(2)
                status_response = httpx.get(
                    f"http://localhost:8000/api/v1/status/{task_id}",
                    timeout=10
                )
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    print(f"   Task status: {status_data.get('status', 'unknown')}")
                    
                    if status_data.get('status') == 'FAILURE':
                        error = status_data.get('error', 'Unknown error')
                        if 'File not found' in error:
                            print("[FAIL] Worker cannot access uploaded file - volume mapping issue!")
                            return False
                        else:
                            print(f"   Task failed with different error: {error}")
                    
                    print("[OK] Worker can access uploaded files")
                    return True
                else:
                    print(f"[FAIL] Cannot check task status (status: {status_response.status_code})")
                    return False
            else:
                print(f"[FAIL] File ingestion failed (status: {response.status_code})")
                print(f"   Response: {response.text}")
                return False
                
        finally:
            # Clean up test file
            if os.path.exists(test_file_path):
                os.unlink(test_file_path)
                
    except Exception as e:
        print(f"[FAIL] Worker file access test error: {e}")
        return False

def main():
    """Run all temp directory validation tests."""
    print("🧪 Testing Temp Directory Validation Fixes")
    print("=" * 50)
    
    tests = [
        ("API Startup Validation", test_api_startup_validation),
        ("Temp Directory Consistency", test_temp_directory_consistency),
        ("File Upload Endpoint", test_file_upload_endpoint),
        ("Worker File Access", test_worker_file_access),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}")
        print("-" * 30)
        
        try:
            if test_func():
                passed += 1
            else:
                print(f"[FAIL] {test_name} failed")
        except Exception as e:
            print(f"[FAIL] {test_name} failed with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("[SUCCESS] All temp directory validation tests passed!")
        return True
    else:
        print("[FAIL] Some tests failed - check temp directory configuration")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
