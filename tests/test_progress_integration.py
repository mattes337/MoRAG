"""Integration tests for progress reporting from workers to job entities."""

import pytest
import json
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from morag_core.jobs.progress_handler import ProgressHandler
from morag_core.jobs.models import JobStatus


class TestProgressIntegration:
    """Test end-to-end progress reporting integration."""
    
    @pytest.fixture
    def mock_database(self):
        """Mock database session and job operations."""
        mock_session = Mock()
        mock_job = Mock()
        mock_job.id = "test-job-123"
        mock_job.status = JobStatus.PROCESSING
        mock_job.percentage = 0
        mock_job.summary = ""
        
        mock_session.query.return_value.filter_by.return_value.first.return_value = mock_job
        return mock_session, mock_job
    
    @pytest.fixture
    def progress_handler_with_db(self, mock_database):
        """Progress handler with mocked database."""
        mock_session, mock_job = mock_database
        
        with patch('morag_core.jobs.progress_handler.JobTracker') as mock_tracker_class:
            mock_tracker = Mock()
            mock_tracker_class.return_value = mock_tracker
            
            handler = ProgressHandler()
            handler.job_tracker = mock_tracker
            return handler, mock_tracker, mock_job
    
    def test_remote_worker_progress_flow(self, progress_handler_with_db):
        """Test complete flow from remote worker progress to job update."""
        handler, mock_tracker, mock_job = progress_handler_with_db
        
        # Simulate remote worker starting a job
        worker_id = "remote-worker-gpu-01"
        job_id = "job-audio-processing-123"
        
        # Register the job mapping
        handler.register_job_mapping(worker_id, job_id)
        
        # Simulate progress updates from remote worker
        progress_updates = [
            (10, "Initializing audio processing"),
            (25, "Loading audio file"),
            (52, "Extracting audio metadata"),
            (54, "Transcribing audio content"),
            (85, "Finalizing transcription"),
            (100, "Audio processing completed")
        ]
        
        for percentage, message in progress_updates:
            result = handler.process_remote_worker_progress(worker_id, percentage, message)
            assert result is True
        
        # Verify all progress updates were made
        assert mock_tracker.update_progress.call_count == len(progress_updates)
        
        # Check the final call
        final_call = mock_tracker.update_progress.call_args_list[-1]
        assert final_call[1]['job_id'] == job_id
        assert final_call[1]['percentage'] == 100
        assert "Audio processing completed" in final_call[1]['summary']
    
    def test_log_parsing_integration(self, progress_handler_with_db):
        """Test parsing log messages and updating job progress."""
        handler, mock_tracker, mock_job = progress_handler_with_db
        
        job_id = "job-video-processing-456"
        
        # Simulate log messages from a worker
        log_messages = [
            '{"event": "Processing progress: Video processing: Initializing (10%)", "logger": "remote_converter", "level": "info", "timestamp": "2025-06-11T11:26:50.780725Z"}',
            '{"event": "Processing progress: Video processing: Extracting frames (30%)", "logger": "remote_converter", "level": "info", "timestamp": "2025-06-11T11:26:51.780725Z"}',
            '{"event": "Processing progress: Video processing: Audio extraction (60%)", "logger": "remote_converter", "level": "info", "timestamp": "2025-06-11T11:26:52.780725Z"}',
            '{"event": "Processing progress: Video processing: Transcription (85%)", "logger": "remote_converter", "level": "info", "timestamp": "2025-06-11T11:26:53.780725Z"}',
            '{"event": "Processing progress: Video processing: Finalizing (100%)", "logger": "remote_converter", "level": "info", "timestamp": "2025-06-11T11:26:54.780725Z"}',
        ]
        
        # Process each log message
        for log_message in log_messages:
            result = handler.process_log_line(log_message, job_id=job_id)
            assert result is True
        
        # Verify progress updates were made
        assert mock_tracker.update_progress.call_count == len(log_messages)
        
        # Check progression of percentages
        call_percentages = [call[1]['percentage'] for call in mock_tracker.update_progress.call_args_list]
        assert call_percentages == [10, 30, 60, 85, 100]
    
    def test_celery_task_integration(self, progress_handler_with_db):
        """Test integration with Celery task progress reporting."""
        handler, mock_tracker, mock_job = progress_handler_with_db
        
        task_id = "celery-task-789"
        
        # Simulate Celery task progress updates
        celery_updates = [
            (5, "Task started"),
            (20, "File downloaded"),
            (40, "Processing file"),
            (70, "Storing results"),
            (95, "Cleaning up"),
            (100, "Task completed")
        ]
        
        for percentage, message in celery_updates:
            result = handler.process_celery_task_progress(task_id, percentage, message)
            assert result is True
        
        # Verify updates were made with task_id as job_id
        assert mock_tracker.update_progress.call_count == len(celery_updates)
        
        # Check that task_id was used as job_id
        for call in mock_tracker.update_progress.call_args_list:
            assert call[1]['job_id'] == task_id
    
    def test_mixed_progress_sources(self, progress_handler_with_db):
        """Test handling progress from multiple sources."""
        handler, mock_tracker, mock_job = progress_handler_with_db
        
        # Set up multiple jobs
        remote_worker_id = "remote-worker-1"
        remote_job_id = "remote-job-1"
        celery_task_id = "celery-task-1"
        
        handler.register_job_mapping(remote_worker_id, remote_job_id)
        
        # Process updates from different sources
        handler.process_remote_worker_progress(remote_worker_id, 25, "Remote processing")
        handler.process_celery_task_progress(celery_task_id, 50, "Celery processing")
        
        log_line = '{"event": "Processing progress: Direct log processing (75%)", "timestamp": "2025-06-11T11:26:50.780725Z"}'
        handler.process_log_line(log_line, job_id="direct-job-1")
        
        # Verify all updates were processed
        assert mock_tracker.update_progress.call_count == 3
        
        # Check that different job IDs were used
        job_ids = [call[1]['job_id'] for call in mock_tracker.update_progress.call_args_list]
        assert remote_job_id in job_ids
        assert celery_task_id in job_ids
        assert "direct-job-1" in job_ids
    
    def test_error_handling(self, progress_handler_with_db):
        """Test error handling in progress reporting."""
        handler, mock_tracker, mock_job = progress_handler_with_db
        
        # Test with invalid log line
        invalid_log = "This is not a valid JSON log line"
        result = handler.process_log_line(invalid_log, job_id="test-job")
        assert result is False  # Should handle gracefully
        
        # Test with log line without progress info
        no_progress_log = '{"event": "Some other event", "timestamp": "2025-06-11T11:26:50.780725Z"}'
        result = handler.process_log_line(no_progress_log, job_id="test-job")
        assert result is False  # Should return False for non-progress events
        
        # Test with missing job mapping
        result = handler.process_remote_worker_progress("unknown-worker", 50, "Test message")
        assert result is False  # Should handle missing mapping gracefully
    
    def test_job_completion_handling(self, progress_handler_with_db):
        """Test handling job completion scenarios."""
        handler, mock_tracker, mock_job = progress_handler_with_db
        
        job_id = "completion-test-job"
        
        # Test successful completion
        result = handler.handle_job_completion(job_id, True, "Job completed successfully")
        assert result is True
        mock_tracker.mark_completed.assert_called_once_with(job_id, "Job completed successfully", None)
        
        # Test failed completion
        result = handler.handle_job_completion(job_id, False, "Job failed with error")
        assert result is True
        mock_tracker.mark_failed.assert_called_once_with(job_id, "Job failed with error", None)
        
        # Test error handling shortcut
        result = handler.handle_job_error(job_id, "Critical error occurred")
        assert result is True
        # Should call mark_failed with formatted message
        expected_message = "Job failed: Critical error occurred"
        mock_tracker.mark_failed.assert_called_with(job_id, expected_message, None)


@pytest.mark.asyncio
async def test_async_progress_reporting():
    """Test asynchronous progress reporting scenarios."""
    
    # Mock an async progress callback
    progress_updates = []
    
    async def mock_progress_callback(progress: float, message: str = None):
        progress_updates.append((progress, message))
        await asyncio.sleep(0.01)  # Simulate async work
    
    # Simulate multiple concurrent progress updates
    tasks = []
    for i in range(10):
        progress = i / 10.0
        message = f"Step {i+1}"
        tasks.append(mock_progress_callback(progress, message))
    
    await asyncio.gather(*tasks)
    
    # Verify all updates were captured
    assert len(progress_updates) == 10
    assert progress_updates[0] == (0.0, "Step 1")
    assert progress_updates[-1] == (0.9, "Step 10")


if __name__ == "__main__":
    pytest.main([__file__])
