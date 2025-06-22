#!/usr/bin/env python3
"""CLI tool for testing the remote conversion system."""

import sys
import os
import argparse
import tempfile
import json
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from the project root
from dotenv import load_dotenv
env_path = project_root / '.env'
load_dotenv(env_path)

# Load environment variables from the project root
from dotenv import load_dotenv
env_path = project_root / '.env'
load_dotenv(env_path)
sys.path.insert(0, str(project_root / "packages" / "morag-core" / "src"))
sys.path.insert(0, str(project_root / "packages" / "morag" / "src"))

def test_remote_job_lifecycle():
    """Test the complete remote job lifecycle."""
    print("🔄 Testing Remote Job Lifecycle")
    print("=" * 40)
    
    try:
        from morag.repositories.remote_job_repository import RemoteJobRepository
        from morag.services.remote_job_service import RemoteJobService
        from morag.models.remote_job_api import CreateRemoteJobRequest, SubmitResultRequest
        
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"📁 Using temp directory: {temp_dir}")
            
            # Initialize repository and service
            repository = RemoteJobRepository(data_dir=temp_dir)
            service = RemoteJobService(repository=repository)
            
            # Step 1: Create a remote job
            print("\n1️⃣ Creating remote job...")
            request = CreateRemoteJobRequest(
                source_file_path="/tmp/test_audio.mp3",
                content_type="audio",
                task_options={
                    "webhook_url": "http://example.com/webhook",
                    "metadata": {"source": "test"}
                },
                ingestion_task_id="test-ingestion-123"
            )
            
            job = service.create_job(request)
            print(f"✅ Job created: {job.id}")
            print(f"   Status: {job.status}")
            print(f"   Content Type: {job.content_type}")
            
            # Step 2: Poll for jobs (simulate worker)
            print("\n2️⃣ Polling for jobs (worker simulation)...")
            available_jobs = service.poll_available_jobs(
                worker_id="test-worker-1",
                content_types=["audio", "video"],
                max_jobs=1
            )
            
            if len(available_jobs) != 1:
                raise Exception(f"Expected 1 job, got {len(available_jobs)}")
            
            polled_job = available_jobs[0]
            print(f"✅ Job polled: {polled_job.id}")
            print(f"   Status: {polled_job.status}")
            print(f"   Worker: {polled_job.worker_id}")
            
            # Step 3: Submit successful result
            print("\n3️⃣ Submitting successful result...")
            result_request = SubmitResultRequest(
                success=True,
                content="This is the processed audio transcript with speaker diarization.",
                metadata={
                    "duration": 125.7,
                    "speakers": ["Speaker_00", "Speaker_01"],
                    "topics": [
                        {"timestamp": 0, "topic": "Introduction"},
                        {"timestamp": 60, "topic": "Main Discussion"}
                    ]
                },
                processing_time=45.3
            )
            
            completed_job = service.submit_result(job.id, result_request)
            if not completed_job or completed_job.status != "completed":
                raise Exception(f"Job completion failed: {completed_job}")
            
            print(f"✅ Result submitted successfully")
            print(f"   Final Status: {completed_job.status}")
            print(f"   Processing Time: {completed_job.processing_duration:.1f}s")
            
            print("\n🎉 Remote job lifecycle test passed!")
            return True
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_endpoints():
    """Test the API endpoints using requests."""
    print("\n🌐 Testing API Endpoints")
    print("=" * 40)
    
    try:
        import requests
        
        # Test server health (assuming server is running on localhost:8000)
        base_url = "http://localhost:8000"
        
        print("1️⃣ Testing server health...")
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                print("✅ Server is running")
            else:
                print(f"⚠️ Server responded with status {response.status_code}")
        except requests.exceptions.RequestException:
            print("❌ Server is not running. Start the server with: python -m morag.server")
            return False
        
        # Test creating a remote job
        print("\n2️⃣ Testing remote job creation...")
        job_data = {
            "source_file_path": "/tmp/test_audio.mp3",
            "content_type": "audio",
            "task_options": {"webhook_url": "http://example.com/webhook"},
            "ingestion_task_id": "test-api-123"
        }
        
        response = requests.post(f"{base_url}/api/v1/remote-jobs/", json=job_data)
        if response.status_code == 200:
            job_response = response.json()
            job_id = job_response["job_id"]
            print(f"✅ Remote job created: {job_id}")
            
            # Test job status
            print("\n3️⃣ Testing job status...")
            status_response = requests.get(f"{base_url}/api/v1/remote-jobs/{job_id}/status")
            if status_response.status_code == 200:
                status_data = status_response.json()
                print(f"✅ Job status: {status_data['status']}")
            else:
                print(f"❌ Failed to get job status: {status_response.status_code}")
                
        else:
            print(f"❌ Failed to create remote job: {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        print("\n🎉 API endpoints test passed!")
        return True
        
    except ImportError:
        print("❌ requests library not available. Install with: pip install requests")
        return False
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Test the remote conversion system")
    parser.add_argument(
        "--test", 
        choices=["lifecycle", "api", "all"], 
        default="all",
        help="Which test to run"
    )
    
    args = parser.parse_args()
    
    print("🚀 Remote Conversion System - CLI Test Tool")
    print("=" * 50)
    
    success = True
    
    if args.test in ["lifecycle", "all"]:
        success &= test_remote_job_lifecycle()
    
    if args.test in ["api", "all"]:
        success &= test_api_endpoints()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed!")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
