#!/usr/bin/env python3
"""
Demo script to test webhook functionality.

This script demonstrates:
1. Starting the webhook receiver
2. Making API calls with webhook URLs
3. Receiving webhook notifications
4. Checking status history

Usage:
1. Start the webhook receiver: python webhook_receiver.py
2. Run this demo: python test_webhook_demo.py
"""

import asyncio
import httpx
import json
import time
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"
WEBHOOK_URL = "http://localhost:8001/webhook"
API_KEY = "test-api-key"

async def test_webhook_integration():
    """Test the complete webhook integration."""
    
    print("🚀 Starting Webhook Integration Test")
    print("=" * 50)
    
    # Test 1: Check if API is running
    print("\n1. Testing API connectivity...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_BASE_URL}/status/", 
                                      headers={"Authorization": f"Bearer {API_KEY}"})
            if response.status_code == 200:
                print("✅ API is running")
            else:
                print(f"❌ API returned status {response.status_code}")
                return
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("Make sure the MoRAG API is running on localhost:8000")
        return
    
    # Test 2: Check if webhook receiver is running
    print("\n2. Testing webhook receiver connectivity...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8001/")
            if response.status_code == 200:
                print("✅ Webhook receiver is running")
            else:
                print(f"❌ Webhook receiver returned status {response.status_code}")
                return
    except Exception as e:
        print(f"❌ Cannot connect to webhook receiver: {e}")
        print("Make sure to start the webhook receiver: python webhook_receiver.py")
        return
    
    # Test 3: Clear previous webhooks
    print("\n3. Clearing previous webhooks...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.delete("http://localhost:8001/webhooks")
            print(f"✅ Cleared webhooks: {response.json()['message']}")
    except Exception as e:
        print(f"⚠️  Could not clear webhooks: {e}")
    
    # Test 4: Create a test file for ingestion
    print("\n4. Creating test file...")
    test_file_path = Path("test_document.txt")
    test_content = """
    This is a test document for webhook integration testing.
    
    The document contains multiple paragraphs to test the chunking
    and embedding generation process.
    
    When this document is processed, it should trigger webhook
    notifications at various stages of the pipeline.
    """
    
    test_file_path.write_text(test_content)
    print(f"✅ Created test file: {test_file_path}")
    
    # Test 5: Submit ingestion task with webhook
    print("\n5. Submitting ingestion task with webhook...")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            with open(test_file_path, 'rb') as f:
                files = {"file": ("test_document.txt", f, "text/plain")}
                data = {
                    "source_type": "document",
                    "webhook_url": WEBHOOK_URL,
                    "metadata": json.dumps({
                        "test_run": True,
                        "description": "Webhook integration test"
                    })
                }
                
                response = await client.post(
                    f"{API_BASE_URL}/ingest/file",
                    headers={"Authorization": f"Bearer {API_KEY}"},
                    files=files,
                    data=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    task_id = result["task_id"]
                    print(f"✅ Task submitted successfully: {task_id}")
                else:
                    print(f"❌ Failed to submit task: {response.status_code}")
                    print(response.text)
                    return
                    
    except Exception as e:
        print(f"❌ Error submitting task: {e}")
        return
    
    # Test 6: Monitor task progress and webhooks
    print(f"\n6. Monitoring task progress for {task_id}...")
    
    max_wait_time = 120  # 2 minutes
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        try:
            # Check task status
            async with httpx.AsyncClient() as client:
                status_response = await client.get(
                    f"{API_BASE_URL}/status/{task_id}",
                    headers={"Authorization": f"Bearer {API_KEY}"}
                )
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    print(f"📊 Task Status: {status_data['status']} - Progress: {status_data.get('progress', 0)*100:.1f}%")
                    
                    if status_data['status'] in ['success', 'failure']:
                        print(f"🏁 Task completed with status: {status_data['status']}")
                        break
                
                # Check received webhooks
                webhook_response = await client.get("http://localhost:8001/webhooks")
                if webhook_response.status_code == 200:
                    webhook_data = webhook_response.json()
                    webhook_count = webhook_data['total_webhooks']
                    print(f"📨 Received {webhook_count} webhooks")
                    
                    if webhook_count > 0:
                        latest = webhook_data['webhooks'][-1]
                        event_type = latest['payload'].get('event_type', 'unknown')
                        print(f"   Latest: {event_type}")
                
        except Exception as e:
            print(f"⚠️  Error checking status: {e}")
        
        await asyncio.sleep(5)  # Wait 5 seconds before next check
    
    # Test 7: Check final webhook status
    print("\n7. Checking final webhook status...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8001/webhooks")
            if response.status_code == 200:
                data = response.json()
                print(f"📨 Total webhooks received: {data['total_webhooks']}")
                
                if data['total_webhooks'] > 0:
                    print("\n📋 Webhook Summary:")
                    for i, webhook in enumerate(data['webhooks'], 1):
                        payload = webhook['payload']
                        event_type = payload.get('event_type', 'unknown')
                        task_data = payload.get('data', {})
                        status = task_data.get('status', 'unknown')
                        print(f"   {i}. {event_type} - {status}")
                        
                        if 'progress' in task_data:
                            print(f"      Progress: {task_data['progress']*100:.1f}%")
                        if 'message' in task_data:
                            print(f"      Message: {task_data['message']}")
                        if 'error' in task_data:
                            print(f"      Error: {task_data['error']}")
                else:
                    print("❌ No webhooks received!")
                    
    except Exception as e:
        print(f"❌ Error checking webhooks: {e}")
    
    # Test 8: Check task history
    print(f"\n8. Checking task history for {task_id}...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{API_BASE_URL}/status/{task_id}/history",
                headers={"Authorization": f"Bearer {API_KEY}"}
            )
            
            if response.status_code == 200:
                history_data = response.json()
                print(f"📚 Task history events: {history_data['event_count']}")
                
                if history_data['event_count'] > 0:
                    print("\n📋 History Summary:")
                    for i, event in enumerate(history_data['history'][:5], 1):  # Show first 5
                        print(f"   {i}. {event['status']} - {event.get('message', 'No message')}")
                        if event.get('progress') is not None:
                            print(f"      Progress: {event['progress']*100:.1f}%")
            else:
                print(f"❌ Failed to get task history: {response.status_code}")
                
    except Exception as e:
        print(f"❌ Error checking task history: {e}")
    
    # Cleanup
    print("\n9. Cleaning up...")
    try:
        test_file_path.unlink()
        print("✅ Removed test file")
    except Exception as e:
        print(f"⚠️  Could not remove test file: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Webhook Integration Test Complete!")
    print("\nTo view detailed webhook data, visit: http://localhost:8001/webhooks")

if __name__ == "__main__":
    asyncio.run(test_webhook_integration())
