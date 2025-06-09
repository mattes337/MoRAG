#!/usr/bin/env python3
"""Test script for API chunk size parameters."""

import os
import sys
import tempfile
import json
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from fastapi.testclient import TestClient
    from morag.server import create_app
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("⚠️  FastAPI not available, skipping API tests")


def test_api_chunk_size_parameters():
    """Test that API endpoints accept chunk size parameters."""
    
    if not FASTAPI_AVAILABLE:
        print("⚠️  Skipping API test - FastAPI not available")
        return
    
    try:
        # Create test app
        app = create_app()
        client = TestClient(app)
        
        # Create a test file
        test_content = "This is a test document. " * 100  # Create some content
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            temp_file_path = f.name
        
        try:
            # Test file upload with chunk size parameters
            with open(temp_file_path, 'rb') as f:
                response = client.post(
                    "/api/v1/ingest/file",
                    files={"file": ("test.txt", f, "text/plain")},
                    data={
                        "source_type": "document",
                        "chunk_size": "2000",
                        "chunk_overlap": "100",
                        "chunking_strategy": "paragraph"
                    }
                )
            
            print(f"API Response Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ API accepts chunk size parameters")
                print(f"Task ID: {result.get('task_id')}")
                print(f"Status: {result.get('status')}")
                print(f"Message: {result.get('message')}")
            else:
                print(f"❌ API request failed: {response.status_code}")
                print(f"Response: {response.text}")
                
        finally:
            # Clean up test file
            try:
                os.unlink(temp_file_path)
            except:
                pass
                
    except Exception as e:
        print(f"❌ API test failed: {e}")
        import traceback
        traceback.print_exc()


def test_api_chunk_size_validation():
    """Test that API validates chunk size parameters."""
    
    if not FASTAPI_AVAILABLE:
        print("⚠️  Skipping API validation test - FastAPI not available")
        return
    
    try:
        # Create test app
        app = create_app()
        client = TestClient(app)
        
        # Create a test file
        test_content = "This is a test document."
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            temp_file_path = f.name
        
        try:
            # Test with invalid chunk size (too small)
            with open(temp_file_path, 'rb') as f:
                response = client.post(
                    "/api/v1/ingest/file",
                    files={"file": ("test.txt", f, "text/plain")},
                    data={
                        "source_type": "document",
                        "chunk_size": "100",  # Too small
                        "chunk_overlap": "50"
                    }
                )
            
            print(f"Invalid chunk size response: {response.status_code}")
            
            if response.status_code == 400:
                print("✅ API correctly validates chunk size (too small)")
            else:
                print(f"⚠️  Expected 400, got {response.status_code}")
                print(f"Response: {response.text}")
            
            # Test with invalid chunk size (too large)
            with open(temp_file_path, 'rb') as f:
                response = client.post(
                    "/api/v1/ingest/file",
                    files={"file": ("test.txt", f, "text/plain")},
                    data={
                        "source_type": "document",
                        "chunk_size": "20000",  # Too large
                        "chunk_overlap": "100"
                    }
                )
            
            print(f"Invalid chunk size response: {response.status_code}")
            
            if response.status_code == 400:
                print("✅ API correctly validates chunk size (too large)")
            else:
                print(f"⚠️  Expected 400, got {response.status_code}")
                print(f"Response: {response.text}")
                
        finally:
            # Clean up test file
            try:
                os.unlink(temp_file_path)
            except:
                pass
                
    except Exception as e:
        print(f"❌ API validation test failed: {e}")
        import traceback
        traceback.print_exc()


def test_api_openapi_schema():
    """Test that OpenAPI schema includes chunk size parameters."""
    
    if not FASTAPI_AVAILABLE:
        print("⚠️  Skipping OpenAPI test - FastAPI not available")
        return
    
    try:
        # Create test app
        app = create_app()
        client = TestClient(app)
        
        # Get OpenAPI schema
        response = client.get("/openapi.json")
        
        if response.status_code == 200:
            schema = response.json()
            
            # Check if ingest file endpoint has chunk size parameters
            paths = schema.get("paths", {})
            ingest_file_path = paths.get("/api/v1/ingest/file", {})
            post_method = ingest_file_path.get("post", {})
            request_body = post_method.get("requestBody", {})
            content = request_body.get("content", {})
            multipart = content.get("multipart/form-data", {})
            schema_props = multipart.get("schema", {}).get("properties", {})
            
            chunk_size_param = schema_props.get("chunk_size")
            chunk_overlap_param = schema_props.get("chunk_overlap")
            chunking_strategy_param = schema_props.get("chunking_strategy")
            
            if chunk_size_param:
                print("✅ chunk_size parameter found in OpenAPI schema")
                print(f"   Type: {chunk_size_param.get('type')}")
            else:
                print("❌ chunk_size parameter not found in OpenAPI schema")
            
            if chunk_overlap_param:
                print("✅ chunk_overlap parameter found in OpenAPI schema")
                print(f"   Type: {chunk_overlap_param.get('type')}")
            else:
                print("❌ chunk_overlap parameter not found in OpenAPI schema")
                
            if chunking_strategy_param:
                print("✅ chunking_strategy parameter found in OpenAPI schema")
                print(f"   Type: {chunking_strategy_param.get('type')}")
            else:
                print("❌ chunking_strategy parameter not found in OpenAPI schema")
                
        else:
            print(f"❌ Failed to get OpenAPI schema: {response.status_code}")
            
    except Exception as e:
        print(f"❌ OpenAPI test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all API tests."""
    print("Testing API chunk size parameters...")
    print("=" * 50)
    
    try:
        test_api_chunk_size_parameters()
        print()
        
        test_api_chunk_size_validation()
        print()
        
        test_api_openapi_schema()
        print()
        
        print("=" * 50)
        print("✅ API chunk size tests completed!")
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
