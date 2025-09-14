#!/usr/bin/env python3
"""End-to-end integration test for remote conversion system."""

import sys
import os
import tempfile
import json
import time
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "packages" / "morag-core" / "src"))
sys.path.insert(0, str(project_root / "packages" / "morag" / "src"))

def test_remote_job_lifecycle():
    """Test the complete remote job lifecycle."""
    print("🔄 Testing Remote Job Lifecycle")
    print("=" * 40)
    
    try:
        # Import directly from specific modules to avoid circular imports
        import sys
        sys.path.insert(0, str(project_root / "packages" / "morag" / "src"))

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
            
            assert len(available_jobs) == 1, "Should find one available job"
            polled_job = available_jobs[0]
            print(f"✅ Job polled: {polled_job.id}")
            print(f"   Status: {polled_job.status}")
            print(f"   Worker: {polled_job.worker_id}")
            
            # Step 3: Check job status
            print("\n3️⃣ Checking job status...")
            status_job = service.get_job_status(job.id)
            assert status_job.status == "processing", f"Expected processing, got {status_job.status}"
            assert status_job.worker_id == "test-worker-1", f"Expected test-worker-1, got {status_job.worker_id}"
            print(f"✅ Status check passed")
            
            # Step 4: Submit successful result
            print("\n4️⃣ Submitting successful result...")
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
            assert completed_job is not None, "Result submission should succeed"
            assert completed_job.status == "completed", f"Expected completed, got {completed_job.status}"
            print(f"✅ Result submitted successfully")
            print(f"   Final Status: {completed_job.status}")
            print(f"   Processing Time: {completed_job.processing_duration:.1f}s")
            
            # Step 5: Verify job data persistence
            print("\n5️⃣ Verifying data persistence...")
            final_job = service.get_job_status(job.id)
            assert final_job.status == "completed"
            assert final_job.result_data["content"] == result_request.content
            assert final_job.result_data["metadata"]["duration"] == 125.7
            print(f"✅ Data persistence verified")
            
            # Step 6: Test cleanup functions
            print("\n6️⃣ Testing cleanup functions...")
            expired_count = service.cleanup_expired_jobs()
            old_count = service.cleanup_old_jobs(days_old=0)  # Clean everything
            print(f"✅ Cleanup completed (expired: {expired_count}, old: {old_count})")
            
            print("\n🎉 All tests passed! Remote job lifecycle working correctly.")
            return True
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_worker_polling_scenarios():
    """Test various worker polling scenarios."""
    print("\n🔄 Testing Worker Polling Scenarios")
    print("=" * 40)
    
    try:
        # Import directly from specific modules to avoid circular imports
        import sys
        sys.path.insert(0, str(project_root / "packages" / "morag" / "src"))

        from morag.repositories.remote_job_repository import RemoteJobRepository
        from morag.services.remote_job_service import RemoteJobService
        from morag.models.remote_job_api import CreateRemoteJobRequest
        
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = RemoteJobRepository(data_dir=temp_dir)
            service = RemoteJobService(repository=repository)
            
            # Create jobs of different types
            print("\n1️⃣ Creating jobs of different types...")
            audio_job = service.create_job(CreateRemoteJobRequest(
                source_file_path="/tmp/audio.mp3",
                content_type="audio",
                task_options={},
                ingestion_task_id="audio-task"
            ))
            
            video_job = service.create_job(CreateRemoteJobRequest(
                source_file_path="/tmp/video.mp4", 
                content_type="video",
                task_options={},
                ingestion_task_id="video-task"
            ))
            
            doc_job = service.create_job(CreateRemoteJobRequest(
                source_file_path="/tmp/doc.pdf",
                content_type="document", 
                task_options={},
                ingestion_task_id="doc-task"
            ))
            
            print(f"✅ Created 3 jobs: audio, video, document")
            
            # Test audio-only worker
            print("\n2️⃣ Testing audio-only worker...")
            audio_jobs = service.poll_available_jobs("audio-worker", ["audio"], max_jobs=5)
            assert len(audio_jobs) == 1, f"Expected 1 audio job, got {len(audio_jobs)}"
            assert audio_jobs[0].content_type == "audio"
            print(f"✅ Audio worker got audio job")
            
            # Test video-only worker
            print("\n3️⃣ Testing video-only worker...")
            video_jobs = service.poll_available_jobs("video-worker", ["video"], max_jobs=5)
            assert len(video_jobs) == 1, f"Expected 1 video job, got {len(video_jobs)}"
            assert video_jobs[0].content_type == "video"
            print(f"✅ Video worker got video job")
            
            # Test multi-type worker (should get document since audio/video are claimed)
            print("\n4️⃣ Testing multi-type worker...")
            multi_jobs = service.poll_available_jobs("multi-worker", ["audio", "video", "document"], max_jobs=5)
            assert len(multi_jobs) == 1, f"Expected 1 job, got {len(multi_jobs)}"
            assert multi_jobs[0].content_type == "document"
            print(f"✅ Multi-type worker got remaining document job")
            
            # Test worker with no matching content types
            print("\n5️⃣ Testing worker with no matching types...")
            no_jobs = service.poll_available_jobs("image-worker", ["image"], max_jobs=5)
            assert len(no_jobs) == 0, f"Expected 0 jobs, got {len(no_jobs)}"
            print(f"✅ Image worker got no jobs (as expected)")
            
            print("\n🎉 All polling scenarios passed!")
            return True
            
    except Exception as e:
        print(f"\n❌ Polling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all end-to-end tests."""
    print("🚀 Remote Conversion System - End-to-End Tests")
    print("=" * 50)
    
    tests = [
        test_remote_job_lifecycle,
        test_worker_polling_scenarios
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print("📊 End-to-End Test Summary")
    print("=" * 50)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All end-to-end tests passed!")
        print("✅ Remote conversion system is working correctly!")
        return 0
    else:
        print(f"\n💥 {failed} test(s) failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
