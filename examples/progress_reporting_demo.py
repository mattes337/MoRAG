#!/usr/bin/env python3
"""
Demo script showing how to use the MoRAG progress reporting system.

This script demonstrates:
1. Parsing progress events from log messages
2. Updating job entities with progress information
3. Handling progress from different sources (remote workers, Celery tasks, etc.)
"""

import json
import time
from datetime import datetime, timezone

# Add MoRAG core to path
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "packages" / "morag-core" / "src"))

from morag_core.jobs.progress_parser import ProgressEventParser, ProgressEvent
from morag_core.jobs.progress_handler import ProgressHandler


def demo_progress_parsing():
    """Demonstrate parsing progress events from log messages."""
    print("=== Progress Event Parsing Demo ===\n")
    
    parser = ProgressEventParser()
    
    # Sample log messages from different sources
    sample_logs = [
        '{"event": "Processing progress: Audio processing: Initializing audio processing (52%)", "logger": "remote_converter", "level": "info", "timestamp": "2025-06-11T11:26:50.780725Z"}',
        '{"event": "Processing progress: Audio processing: Extracting audio metadata (54%)", "logger": "remote_converter", "level": "info", "timestamp": "2025-06-11T11:26:50.789601Z"}',
        '{"event": "Processing progress: Audio processing: Transcribing audio content (56%)", "logger": "remote_converter", "level": "info", "timestamp": "2025-06-11T11:26:57.212718Z"}',
        'Processing... 75%',  # Plain text format
        'Progress: 85% - Finalizing transcription',  # Alternative format
        '[90%] Cleaning up temporary files',  # Bracket format
    ]
    
    print("Parsing sample log messages:")
    for i, log_line in enumerate(sample_logs, 1):
        print(f"\n{i}. Input: {log_line[:80]}...")
        
        event = parser.parse_json_log(log_line)
        if event:
            print(f"   Parsed: {event.percentage}% - {event.message}")
            print(f"   Timestamp: {event.timestamp}")
            print(f"   Logger: {event.logger_name}")
        else:
            print("   No progress information found")
    
    # Demonstrate getting latest progress
    print(f"\n--- Latest Progress ---")
    latest = parser.get_latest_progress(sample_logs)
    if latest:
        print(f"Latest progress: {latest.percentage}% - {latest.message}")
        print(f"Timestamp: {latest.timestamp}")


def demo_progress_handler():
    """Demonstrate handling progress events and updating jobs."""
    print("\n\n=== Progress Handler Demo ===\n")
    
    # Create a mock progress handler (in real usage, this would connect to database)
    class MockJobTracker:
        def __init__(self):
            self.jobs = {}
        
        def update_progress(self, job_id, percentage, status, summary, user_id):
            if job_id not in self.jobs:
                self.jobs[job_id] = {"percentage": 0, "status": "PENDING", "summary": ""}
            
            self.jobs[job_id].update({
                "percentage": percentage,
                "status": status.value if status else self.jobs[job_id]["status"],
                "summary": summary or self.jobs[job_id]["summary"]
            })
            
            print(f"Job {job_id} updated: {percentage}% - {summary}")
            return True
        
        def mark_completed(self, job_id, summary, user_id):
            if job_id in self.jobs:
                self.jobs[job_id].update({"percentage": 100, "status": "FINISHED", "summary": summary})
                print(f"Job {job_id} completed: {summary}")
            return True
        
        def mark_failed(self, job_id, error_message, user_id):
            if job_id in self.jobs:
                self.jobs[job_id].update({"percentage": 0, "status": "FAILED", "summary": error_message})
                print(f"Job {job_id} failed: {error_message}")
            return True
    
    # Create handler with mock tracker
    handler = ProgressHandler()
    handler.job_tracker = MockJobTracker()
    
    # Demo 1: Remote worker progress
    print("1. Remote Worker Progress Updates:")
    worker_id = "remote-worker-gpu-01"
    job_id = "job-audio-123"
    
    handler.register_job_mapping(worker_id, job_id)
    
    remote_progress = [
        (10, "Initializing audio processing"),
        (25, "Loading audio file"),
        (50, "Extracting features"),
        (75, "Transcribing content"),
        (90, "Post-processing"),
        (100, "Audio processing completed")
    ]
    
    for percentage, message in remote_progress:
        handler.process_remote_worker_progress(worker_id, percentage, message)
        time.sleep(0.1)  # Simulate time between updates
    
    # Demo 2: Log message processing
    print(f"\n2. Log Message Processing:")
    video_job_id = "job-video-456"
    
    log_messages = [
        '{"event": "Processing progress: Video processing: Extracting frames (20%)", "timestamp": "2025-06-11T11:26:50.780725Z"}',
        '{"event": "Processing progress: Video processing: Audio extraction (40%)", "timestamp": "2025-06-11T11:26:51.780725Z"}',
        '{"event": "Processing progress: Video processing: Transcription (80%)", "timestamp": "2025-06-11T11:26:52.780725Z"}',
    ]
    
    for log_message in log_messages:
        handler.process_log_line(log_message, job_id=video_job_id)
        time.sleep(0.1)
    
    # Demo 3: Celery task progress
    print(f"\n3. Celery Task Progress:")
    celery_task_id = "celery-task-789"
    
    celery_progress = [
        (5, "Task started"),
        (30, "File downloaded"),
        (60, "Processing document"),
        (85, "Storing results"),
        (100, "Task completed")
    ]
    
    for percentage, message in celery_progress:
        handler.process_celery_task_progress(celery_task_id, percentage, message)
        time.sleep(0.1)
    
    # Demo 4: Job completion
    print(f"\n4. Job Completion:")
    handler.handle_job_completion("job-success-001", True, "Processing completed successfully")
    handler.handle_job_completion("job-failure-002", False, "Processing failed due to timeout")
    
    # Show final job states
    print(f"\n--- Final Job States ---")
    for job_id, job_data in handler.job_tracker.jobs.items():
        print(f"{job_id}: {job_data['percentage']}% - {job_data['status']} - {job_data['summary']}")


def demo_real_time_simulation():
    """Simulate real-time progress reporting."""
    print("\n\n=== Real-time Progress Simulation ===\n")
    
    parser = ProgressEventParser()
    
    # Simulate receiving log messages in real-time
    simulated_logs = [
        ("2025-06-11T11:26:50.000Z", "Processing progress: Document analysis: Loading document (5%)"),
        ("2025-06-11T11:26:51.000Z", "Processing progress: Document analysis: Extracting text (15%)"),
        ("2025-06-11T11:26:52.000Z", "Processing progress: Document analysis: Analyzing content (35%)"),
        ("2025-06-11T11:26:53.000Z", "Processing progress: Document analysis: Generating embeddings (60%)"),
        ("2025-06-11T11:26:54.000Z", "Processing progress: Document analysis: Storing results (85%)"),
        ("2025-06-11T11:26:55.000Z", "Processing progress: Document analysis: Finalizing (100%)"),
    ]
    
    print("Simulating real-time log processing:")
    for timestamp, message in simulated_logs:
        log_line = json.dumps({
            "event": message,
            "logger": "document_processor",
            "level": "info",
            "timestamp": timestamp
        })
        
        event = parser.parse_json_log(log_line)
        if event:
            print(f"[{timestamp}] {event.percentage:3d}% - {event.message}")
        
        time.sleep(0.5)  # Simulate real-time delay
    
    print("\nReal-time simulation completed!")


def main():
    """Run all demos."""
    print("MoRAG Progress Reporting System Demo")
    print("=" * 50)
    
    try:
        demo_progress_parsing()
        demo_progress_handler()
        demo_real_time_simulation()
        
        print(f"\n{'=' * 50}")
        print("Demo completed successfully!")
        print("\nKey features demonstrated:")
        print("✓ Parse progress events from JSON and plain text logs")
        print("✓ Handle progress from remote workers, Celery tasks, and direct logs")
        print("✓ Update job entities with progress information")
        print("✓ Handle job completion and failure scenarios")
        print("✓ Real-time progress monitoring simulation")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
