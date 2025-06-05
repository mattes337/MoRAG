#!/usr/bin/env python3
"""Simple test of MoRAG video processing."""

import requests
import json
import time
from pathlib import Path

def test_video_processing():
    """Test video processing via REST API."""
    
    video_file = "uploads/test_video.mp4"
    
    if not Path(video_file).exists():
        print(f"Video file {video_file} not found")
        return
    
    print(f"Testing MoRAG with {video_file}")
    print("=" * 50)
    
    # Test file upload and processing
    try:
        with open(video_file, 'rb') as f:
            files = {'file': f}
            
            print("Uploading and processing video file...")
            start_time = time.time()
            
            response = requests.post(
                'http://localhost:8000/process/file',
                files=files,
                timeout=300  # 5 minutes timeout
            )
            
            processing_time = time.time() - start_time
            
            print(f"Request completed in {processing_time:.2f} seconds")
            print(f"Status code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("Processing successful!")
                print(f"Content length: {len(result.get('content', ''))}")
                print(f"Metadata keys: {list(result.get('metadata', {}).keys())}")
                
                # Show preview of content
                content = result.get('content', '')
                if content:
                    preview = content[:500]
                    print(f"\nContent preview:\n{preview}")
                    if len(content) > 500:
                        print("... (truncated)")
                        
            else:
                print(f"Processing failed: {response.text}")
                
    except requests.exceptions.Timeout:
        print("Request timed out - video processing may take longer")
    except Exception as e:
        print(f"Error: {str(e)}")

def test_health():
    """Test health endpoint."""
    try:
        response = requests.get('http://localhost:8000/health', timeout=10)
        print("Health check:")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            health = response.json()
            print(f"Overall status: {health.get('overall_status', 'unknown')}")
            print(f"Services: {len(health.get('services', {}))}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Health check failed: {str(e)}")

if __name__ == "__main__":
    print("MoRAG Simple Test")
    print("=" * 30)
    
    # Test health first
    test_health()
    print()
    
    # Test video processing
    test_video_processing()
