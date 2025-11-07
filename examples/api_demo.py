#!/usr/bin/env python3
"""
MoRAG Ingestion API Demo

This script demonstrates how to use the MoRAG ingestion API to process various types of content.
"""

import json
import time
from pathlib import Path

import requests

# API Configuration
API_BASE_URL = "http://localhost:8000"
API_KEY = "demo-api-key"  # Replace with your actual API key


def make_request(method, endpoint, **kwargs):
    """Make an authenticated API request."""
    headers = kwargs.pop("headers", {})
    headers["Authorization"] = f"Bearer {API_KEY}"

    url = f"{API_BASE_URL}{endpoint}"
    response = requests.request(method, url, headers=headers, **kwargs)

    print(f"{method} {endpoint} -> {response.status_code}")
    if response.status_code >= 400:
        print(f"Error: {response.text}")

    return response


def demo_health_check():
    """Test the health endpoint."""
    print("\n=== Health Check ===")
    response = requests.get(f"{API_BASE_URL}/health/")
    print(f"Health Status: {response.json()}")


def demo_file_upload():
    """Demo file upload functionality."""
    print("\n=== File Upload Demo ===")

    # Create a sample text file
    sample_content = """
    # Sample Document

    This is a sample document for testing the MoRAG ingestion API.

    ## Features
    - Document parsing
    - Text chunking
    - Embedding generation
    - Vector storage

    The system can process various file types including PDF, DOCX, and plain text.
    """

    # Save to temporary file
    temp_file = Path("temp_sample.txt")
    temp_file.write_text(sample_content)

    try:
        # Upload the file
        with open(temp_file, "rb") as f:
            files = {"file": ("sample.txt", f, "text/plain")}
            data = {
                "source_type": "document",
                "metadata": json.dumps(
                    {
                        "tags": ["demo", "sample"],
                        "priority": 1,
                        "notes": "Demo file upload",
                    }
                ),
            }

            response = make_request(
                "POST", "/api/v1/ingest/file", files=files, data=data
            )

            if response.status_code == 200:
                result = response.json()
                print(f"Upload successful! Task ID: {result['task_id']}")
                print(f"Status: {result['status']}")
                print(f"Message: {result['message']}")
                print(f"Estimated time: {result['estimated_time']} seconds")
                return result["task_id"]
            else:
                print(f"Upload failed: {response.text}")
                return None

    finally:
        # Clean up
        if temp_file.exists():
            temp_file.unlink()


def demo_url_ingestion():
    """Demo URL ingestion functionality."""
    print("\n=== URL Ingestion Demo ===")

    # Ingest a web page
    data = {
        "source_type": "web",
        "url": "https://example.com",
        "metadata": {"tags": ["web", "demo"], "category": "website"},
    }

    response = make_request("POST", "/api/v1/ingest/url", json=data)

    if response.status_code == 200:
        result = response.json()
        print(f"URL ingestion successful! Task ID: {result['task_id']}")
        print(f"Status: {result['status']}")
        print(f"Message: {result['message']}")
        return result["task_id"]
    else:
        print(f"URL ingestion failed: {response.text}")
        return None


def demo_batch_ingestion():
    """Demo batch ingestion functionality."""
    print("\n=== Batch Ingestion Demo ===")

    # Batch ingest multiple URLs
    data = {
        "items": [
            {
                "source_type": "web",
                "url": "https://example.com",
                "metadata": {"category": "example"},
            },
            {
                "source_type": "web",
                "url": "https://httpbin.org/html",
                "metadata": {"category": "test"},
            },
        ],
        "webhook_url": "https://webhook.example.com/notify",
    }

    response = make_request("POST", "/api/v1/ingest/batch", json=data)

    if response.status_code == 200:
        result = response.json()
        print(f"Batch ingestion successful! Batch ID: {result['batch_id']}")
        print(f"Task IDs: {result['task_ids']}")
        print(f"Total items: {result['total_items']}")
        return result["task_ids"]
    else:
        print(f"Batch ingestion failed: {response.text}")
        return []


def demo_task_status(task_id):
    """Demo task status checking."""
    print(f"\n=== Task Status Demo (Task: {task_id}) ===")

    response = make_request("GET", f"/api/v1/status/{task_id}")

    if response.status_code == 200:
        result = response.json()
        print(f"Task ID: {result['task_id']}")
        print(f"Status: {result['status']}")
        print(f"Progress: {result.get('progress', 'N/A')}")
        print(f"Message: {result.get('message', 'N/A')}")
        if result.get("created_at"):
            print(f"Created: {result['created_at']}")
        if result.get("estimated_time_remaining"):
            print(f"Est. time remaining: {result['estimated_time_remaining']} seconds")
    else:
        print(f"Status check failed: {response.text}")


def demo_list_active_tasks():
    """Demo listing active tasks."""
    print("\n=== Active Tasks Demo ===")

    response = make_request("GET", "/api/v1/status/")

    if response.status_code == 200:
        result = response.json()
        print(f"Active tasks count: {result['count']}")
        print(f"Active tasks: {result['active_tasks']}")
    else:
        print(f"Failed to list active tasks: {response.text}")


def demo_queue_stats():
    """Demo queue statistics."""
    print("\n=== Queue Statistics Demo ===")

    response = make_request("GET", "/api/v1/status/stats/queues")

    if response.status_code == 200:
        result = response.json()
        print("Queue Statistics:")
        for key, value in result.items():
            print(f"  {key}: {value}")
    else:
        print(f"Failed to get queue stats: {response.text}")


def main():
    """Run the complete API demo."""
    print("MoRAG Ingestion API Demo")
    print("=" * 40)

    # Check if API is running
    try:
        demo_health_check()
    except requests.exceptions.ConnectionError:
        print(
            "Error: Cannot connect to API. Make sure the MoRAG API server is running on http://localhost:8000"
        )
        return

    # Demo file upload
    file_task_id = demo_file_upload()

    # Demo URL ingestion
    url_task_id = demo_url_ingestion()

    # Demo batch ingestion
    batch_task_ids = demo_batch_ingestion()

    # Demo task status checking
    if file_task_id:
        demo_task_status(file_task_id)

    if url_task_id:
        demo_task_status(url_task_id)

    # Demo listing active tasks
    demo_list_active_tasks()

    # Demo queue statistics
    demo_queue_stats()

    print("\n=== Demo Complete ===")
    print("Note: This demo shows the API structure. Actual task processing")
    print("requires the full MoRAG system with Redis, Qdrant, and Gemini API.")


if __name__ == "__main__":
    main()
