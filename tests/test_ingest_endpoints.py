#!/usr/bin/env python3
"""Test script for the new ingest endpoints."""

import requests
import json
import time
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint."""
    print("🔍 Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print("✅ Health check passed")
        return True
    else:
        print("❌ Health check failed")
        return False

def test_ingest_url():
    """Test URL ingestion endpoint."""
    print("\n🔍 Testing URL ingestion...")
    
    data = {
        "source_type": "web",
        "url": "https://httpbin.org/json",
        "metadata": {
            "test": True,
            "category": "api_test"
        }
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/ingest/url",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"✅ URL ingestion started")
        print(f"Task ID: {result.get('task_id')}")
        print(f"Message: {result.get('message')}")
        return result.get('task_id')
    else:
        print(f"❌ URL ingestion failed: {response.text}")
        return None

def test_task_status(task_id):
    """Test task status endpoint."""
    if not task_id:
        return
        
    print(f"\n🔍 Testing task status for {task_id}...")
    
    response = requests.get(f"{BASE_URL}/api/v1/status/{task_id}")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Task status retrieved")
        print(f"Status: {result.get('status')}")
        print(f"Progress: {result.get('progress', 0):.2f}")
        print(f"Message: {result.get('message', 'N/A')}")
        return result
    else:
        print(f"❌ Task status failed: {response.text}")
        return None

def test_list_active_tasks():
    """Test list active tasks endpoint."""
    print("\n🔍 Testing list active tasks...")
    
    response = requests.get(f"{BASE_URL}/api/v1/status/")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Active tasks listed")
        print(f"Active tasks: {result.get('count', 0)}")
        return result
    else:
        print(f"❌ List active tasks failed: {response.text}")
        return None

def test_queue_stats():
    """Test queue statistics endpoint."""
    print("\n🔍 Testing queue statistics...")
    
    response = requests.get(f"{BASE_URL}/api/v1/status/stats/queues")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Queue stats retrieved")
        print(f"Pending: {result.get('pending', 0)}")
        print(f"Active: {result.get('active', 0)}")
        return result
    else:
        print(f"❌ Queue stats failed: {response.text}")
        return None

def test_batch_ingest():
    """Test batch ingestion endpoint."""
    print("\n🔍 Testing batch ingestion...")
    
    data = {
        "items": [
            {
                "source_type": "web",
                "url": "https://httpbin.org/json"
            },
            {
                "source_type": "web", 
                "url": "https://httpbin.org/uuid"
            }
        ],
        "webhook_url": None
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/ingest/batch",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Batch ingestion started")
        print(f"Batch ID: {result.get('batch_id')}")
        print(f"Task IDs: {result.get('task_ids')}")
        print(f"Total items: {result.get('total_items')}")
        return result.get('task_ids', [])
    else:
        print(f"❌ Batch ingestion failed: {response.text}")
        return []

def test_swagger_docs():
    """Test that Swagger docs are accessible."""
    print("\n🔍 Testing Swagger documentation...")
    
    response = requests.get(f"{BASE_URL}/docs")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        print("✅ Swagger docs accessible")
        return True
    else:
        print("❌ Swagger docs not accessible")
        return False

def test_openapi_schema():
    """Test OpenAPI schema endpoint."""
    print("\n🔍 Testing OpenAPI schema...")
    
    response = requests.get(f"{BASE_URL}/openapi.json")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        schema = response.json()
        print("✅ OpenAPI schema accessible")
        
        # Check for ingest endpoints
        paths = schema.get('paths', {})
        ingest_endpoints = [path for path in paths.keys() if '/api/v1/ingest/' in path]
        print(f"Ingest endpoints found: {len(ingest_endpoints)}")
        for endpoint in ingest_endpoints:
            print(f"  - {endpoint}")
        
        return True
    else:
        print("❌ OpenAPI schema not accessible")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing MoRAG Ingest Endpoints")
    print("=" * 50)
    
    # Test basic connectivity
    if not test_health():
        print("❌ Server not accessible. Make sure MoRAG server is running.")
        return
    
    # Test documentation
    test_swagger_docs()
    test_openapi_schema()
    
    # Test ingest endpoints
    task_id = test_ingest_url()
    
    # Test task management
    test_task_status(task_id)
    test_list_active_tasks()
    test_queue_stats()
    
    # Test batch processing
    batch_task_ids = test_batch_ingest()
    
    print("\n" + "=" * 50)
    print("🎉 Test completed!")
    print("\nTo view the interactive API documentation:")
    print(f"  📖 Swagger UI: {BASE_URL}/docs")
    print(f"  📖 ReDoc: {BASE_URL}/redoc")
    
    if task_id or batch_task_ids:
        print("\nTo monitor task progress:")
        if task_id:
            print(f"  🔍 Single task: {BASE_URL}/api/v1/status/{task_id}")
        for tid in batch_task_ids:
            print(f"  🔍 Batch task: {BASE_URL}/api/v1/status/{tid}")

if __name__ == "__main__":
    main()
