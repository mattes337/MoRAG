#!/usr/bin/env python3
"""Test script to validate file upload API fix."""

import requests
import json
import time
import tempfile
from pathlib import Path
import sys


def create_test_files():
    """Create test files for upload testing."""
    test_files = {}
    
    # Create a small text file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a test text file for MoRAG file upload validation.\n")
        f.write("It contains multiple lines to test text processing.\n")
        f.write("The file upload API should handle this correctly.\n")
        test_files['text'] = Path(f.name)
    
    # Create a small PDF-like file (just text with .pdf extension for testing)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdf', delete=False) as f:
        f.write("%PDF-1.4\n")
        f.write("1 0 obj\n")
        f.write("<<\n")
        f.write("/Type /Catalog\n")
        f.write("/Pages 2 0 R\n")
        f.write(">>\n")
        f.write("endobj\n")
        test_files['pdf'] = Path(f.name)
    
    return test_files


def test_health_check():
    """Test API health check."""
    print("Testing API health check...")
    try:
        response = requests.get('http://localhost:8000/health/', timeout=10)
        if response.status_code == 200:
            print("✅ Health check passed")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {str(e)}")
        return False


def test_file_upload(file_path: Path, content_type: str = None):
    """Test file upload endpoint."""
    print(f"\nTesting file upload: {file_path.name}")
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (file_path.name, f, content_type)}
            data = {}
            if content_type:
                data['content_type'] = content_type
            
            print(f"  Uploading {file_path.name} ({file_path.stat().st_size} bytes)...")
            start_time = time.time()
            
            response = requests.post(
                'http://localhost:8000/process/file',
                files=files,
                data=data,
                timeout=60
            )
            
            processing_time = time.time() - start_time
            
            print(f"  Response status: {response.status_code}")
            print(f"  Processing time: {processing_time:.2f}s")
            
            if response.status_code == 200:
                data = response.json()
                print("  ✅ File upload successful")
                print(f"  Success: {data.get('success', 'unknown')}")
                print(f"  Content length: {len(data.get('content', ''))}")
                print(f"  Metadata keys: {list(data.get('metadata', {}).keys())}")
                print(f"  Processing time: {data.get('processing_time', 'unknown')}s")
                return True
            else:
                print(f"  ❌ File upload failed")
                try:
                    error_data = response.json()
                    print(f"  Error: {error_data.get('detail', 'Unknown error')}")
                except:
                    print(f"  Error: {response.text}")
                return False
                
    except requests.exceptions.Timeout:
        print("  ❌ Request timed out")
        return False
    except Exception as e:
        print(f"  ❌ Upload failed: {str(e)}")
        return False


def test_invalid_file():
    """Test upload of invalid file type."""
    print("\nTesting invalid file type rejection...")
    
    try:
        # Create a file with invalid extension
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
            f.write("Invalid file content")
            invalid_file = Path(f.name)
        
        try:
            with open(invalid_file, 'rb') as f:
                files = {'file': (invalid_file.name, f, 'application/xyz')}
                
                response = requests.post(
                    'http://localhost:8000/process/file',
                    files=files,
                    timeout=30
                )
                
                if response.status_code == 400:
                    print("  ✅ Invalid file type correctly rejected")
                    error_data = response.json()
                    print(f"  Error message: {error_data.get('detail', 'Unknown')}")
                    return True
                else:
                    print(f"  ❌ Expected 400 status, got {response.status_code}")
                    return False
        finally:
            invalid_file.unlink()
            
    except Exception as e:
        print(f"  ❌ Test failed: {str(e)}")
        return False


def test_large_file():
    """Test upload of large file (should be rejected)."""
    print("\nTesting large file rejection...")
    
    try:
        # Create a large file (2MB)
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.txt', delete=False) as f:
            large_content = b"x" * (2 * 1024 * 1024)  # 2MB
            f.write(large_content)
            large_file = Path(f.name)
        
        try:
            with open(large_file, 'rb') as f:
                files = {'file': (large_file.name, f, 'text/plain')}
                
                response = requests.post(
                    'http://localhost:8000/process/file',
                    files=files,
                    timeout=30
                )
                
                if response.status_code == 400:
                    print("  ✅ Large file correctly rejected")
                    error_data = response.json()
                    print(f"  Error message: {error_data.get('detail', 'Unknown')}")
                    return True
                else:
                    print(f"  ❌ Expected 400 status, got {response.status_code}")
                    return False
        finally:
            large_file.unlink()
            
    except Exception as e:
        print(f"  ❌ Test failed: {str(e)}")
        return False


def test_options_parsing():
    """Test file upload with processing options."""
    print("\nTesting options parsing...")
    
    try:
        # Create a small test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test file for options parsing")
            test_file = Path(f.name)
        
        try:
            options = {
                "language": "en",
                "quality": "high",
                "test_option": True
            }
            
            with open(test_file, 'rb') as f:
                files = {'file': (test_file.name, f, 'text/plain')}
                data = {
                    'content_type': 'document',
                    'options': json.dumps(options)
                }
                
                response = requests.post(
                    'http://localhost:8000/process/file',
                    files=files,
                    data=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    print("  ✅ Options parsing successful")
                    return True
                else:
                    print(f"  ❌ Options parsing failed: {response.status_code}")
                    try:
                        error_data = response.json()
                        print(f"  Error: {error_data.get('detail', 'Unknown')}")
                    except:
                        print(f"  Error: {response.text}")
                    return False
        finally:
            test_file.unlink()
            
    except Exception as e:
        print(f"  ❌ Test failed: {str(e)}")
        return False


def main():
    """Run all file upload tests."""
    print("MoRAG File Upload API Test")
    print("=" * 50)
    
    # Test health check first
    if not test_health_check():
        print("\n❌ API is not available. Make sure the server is running.")
        sys.exit(1)
    
    # Create test files
    print("\nCreating test files...")
    test_files = create_test_files()
    
    try:
        results = []
        
        # Test valid file uploads
        results.append(test_file_upload(test_files['text'], 'text/plain'))
        results.append(test_file_upload(test_files['pdf'], 'application/pdf'))
        
        # Test error cases
        results.append(test_invalid_file())
        results.append(test_large_file())
        results.append(test_options_parsing())
        
        # Summary
        print("\n" + "=" * 50)
        print("Test Summary:")
        passed = sum(results)
        total = len(results)
        print(f"Passed: {passed}/{total}")
        
        if passed == total:
            print("✅ All tests passed! File upload API is working correctly.")
        else:
            print("❌ Some tests failed. Check the output above for details.")
            sys.exit(1)
            
    finally:
        # Clean up test files
        for file_path in test_files.values():
            try:
                file_path.unlink()
            except:
                pass


if __name__ == "__main__":
    main()
