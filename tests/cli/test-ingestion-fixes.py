#!/usr/bin/env python3
"""Test script for ingestion fixes: auto-detection and options variable fix."""

import json
import requests
import tempfile
import time
from pathlib import Path


def print_section(title):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_result(key, value):
    """Print a key-value result."""
    print(f"  {key}: {value}")


def test_auto_detection_file():
    """Test automatic content type detection for file uploads."""
    print_section("Testing Auto-Detection for File Upload")
    
    # Create a test PDF file
    test_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n>>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000074 00000 n \n0000000120 00000 n \ntrailer\n<<\n/Size 4\n/Root 1 0 R\n>>\nstartxref\n179\n%%EOF"
    
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
        f.write(test_content)
        temp_file = Path(f.name)
    
    try:
        print("🔄 Testing file upload without source_type...")
        
        # Test without source_type (should auto-detect)
        with open(temp_file, 'rb') as f:
            files = {'file': ('test.pdf', f, 'application/pdf')}
            data = {
                'metadata': json.dumps({
                    'test': 'auto_detection',
                    'expected_type': 'document'
                })
            }
            
            response = requests.post(
                'http://localhost:8000/api/v1/ingest/file',
                files=files,
                data=data,
                timeout=30
            )
        
        if response.status_code == 200:
            result = response.json()
            print_result("✅ Auto-detection successful", f"Task ID: {result['task_id']}")
            print_result("Status", result['status'])
            print_result("Message", result['message'])
            return result['task_id']
        else:
            print_result("❌ Auto-detection failed", f"Status: {response.status_code}")
            print_result("Error", response.text)
            return None
            
    finally:
        temp_file.unlink(missing_ok=True)


def test_auto_detection_url():
    """Test automatic content type detection for URLs."""
    print_section("Testing Auto-Detection for URL Ingestion")
    
    # Test YouTube URL auto-detection
    print("🔄 Testing YouTube URL auto-detection...")
    
    data = {
        'url': 'https://youtube.com/watch?v=dQw4w9WgXcQ',
        'metadata': {
            'test': 'auto_detection',
            'expected_type': 'youtube'
        }
    }
    
    response = requests.post(
        'http://localhost:8000/api/v1/ingest/url',
        json=data,
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        print_result("✅ YouTube auto-detection successful", f"Task ID: {result['task_id']}")
        print_result("Status", result['status'])
        print_result("Message", result['message'])
        youtube_task_id = result['task_id']
    else:
        print_result("❌ YouTube auto-detection failed", f"Status: {response.status_code}")
        print_result("Error", response.text)
        youtube_task_id = None
    
    # Test web URL auto-detection
    print("\n🔄 Testing web URL auto-detection...")
    
    data = {
        'url': 'https://httpbin.org/html',
        'metadata': {
            'test': 'auto_detection',
            'expected_type': 'web'
        }
    }
    
    response = requests.post(
        'http://localhost:8000/api/v1/ingest/url',
        json=data,
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        print_result("✅ Web auto-detection successful", f"Task ID: {result['task_id']}")
        print_result("Status", result['status'])
        print_result("Message", result['message'])
        web_task_id = result['task_id']
    else:
        print_result("❌ Web auto-detection failed", f"Status: {response.status_code}")
        print_result("Error", response.text)
        web_task_id = None
    
    return youtube_task_id, web_task_id


def test_batch_auto_detection():
    """Test automatic content type detection for batch ingestion."""
    print_section("Testing Auto-Detection for Batch Ingestion")
    
    print("🔄 Testing batch ingestion with mixed auto-detection...")
    
    data = {
        'items': [
            {
                'url': 'https://httpbin.org/json',
                'metadata': {'test': 'batch_auto_detection', 'item': 1}
            },
            {
                'url': 'https://youtube.com/watch?v=dQw4w9WgXcQ',
                'metadata': {'test': 'batch_auto_detection', 'item': 2}
            }
        ],
        'webhook_url': None
    }
    
    response = requests.post(
        'http://localhost:8000/api/v1/ingest/batch',
        json=data,
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        print_result("✅ Batch auto-detection successful", f"Batch ID: {result['batch_id']}")
        print_result("Task IDs", result['task_ids'])
        print_result("Total items", result['total_items'])
        return result['task_ids']
    else:
        print_result("❌ Batch auto-detection failed", f"Status: {response.status_code}")
        print_result("Error", response.text)
        return []


def monitor_task(task_id, description="Task"):
    """Monitor a task until completion."""
    if not task_id:
        return None
    
    print(f"\n🔍 Monitoring {description} ({task_id})...")
    
    for attempt in range(30):  # 30 attempts, 2 seconds each = 1 minute max
        try:
            response = requests.get(f'http://localhost:8000/api/v1/status/{task_id}')
            
            if response.status_code == 200:
                status = response.json()
                print(f"  Status: {status['status']} (Progress: {status.get('progress', 0):.1%})")
                
                if status['status'] in ['SUCCESS', 'FAILURE']:
                    if status['status'] == 'SUCCESS':
                        print_result(f"✅ {description} completed", "Success")
                        if status.get('result'):
                            result = status['result']
                            if 'metadata' in result:
                                print_result("Detected source type", 
                                           result['metadata'].get('source_type', 'unknown'))
                    else:
                        print_result(f"❌ {description} failed", status.get('error', 'Unknown error'))
                    
                    return status
                    
            time.sleep(2)
            
        except Exception as e:
            print(f"  Error checking status: {e}")
            time.sleep(2)
    
    print_result(f"⏰ {description} timeout", "Task did not complete within 1 minute")
    return None


def main():
    """Run all ingestion fix tests."""
    print_section("MoRAG Ingestion Fixes Test Suite")
    print("Testing automatic content type detection and options variable fix")
    
    # Test file auto-detection
    file_task_id = test_auto_detection_file()
    
    # Test URL auto-detection
    youtube_task_id, web_task_id = test_auto_detection_url()
    
    # Test batch auto-detection
    batch_task_ids = test_batch_auto_detection()
    
    # Monitor all tasks
    print_section("Monitoring Task Completion")
    
    if file_task_id:
        monitor_task(file_task_id, "File auto-detection")
    
    if youtube_task_id:
        monitor_task(youtube_task_id, "YouTube auto-detection")
    
    if web_task_id:
        monitor_task(web_task_id, "Web auto-detection")
    
    for i, task_id in enumerate(batch_task_ids):
        monitor_task(task_id, f"Batch item {i+1}")
    
    print_section("Test Summary")
    print("✅ All ingestion fix tests completed!")
    print("📝 Check the logs above for any failures or issues.")
    print("🔧 The options variable error should be fixed.")
    print("🎯 Auto-detection should work for files and URLs.")


if __name__ == "__main__":
    main()
