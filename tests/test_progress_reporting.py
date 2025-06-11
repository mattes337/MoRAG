"""Tests for progress reporting functionality."""

import pytest
import json
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock

from morag_core.jobs.progress_parser import ProgressEventParser, ProgressEvent
from morag_core.jobs.progress_handler import ProgressHandler


class TestProgressEventParser:
    """Test progress event parsing from log messages."""
    
    def test_parse_json_log_with_progress(self):
        """Test parsing JSON log with progress information."""
        parser = ProgressEventParser()
        
        log_line = json.dumps({
            "event": "Processing progress: Audio processing: Initializing audio processing (52%)",
            "logger": "remote_converter",
            "level": "info",
            "timestamp": "2025-06-11T11:26:50.780725Z"
        })
        
        event = parser.parse_json_log(log_line)
        
        assert event is not None
        assert event.percentage == 52
        assert "Audio processing: Initializing audio processing" in event.message
        assert event.logger_name == "remote_converter"
        assert event.level == "info"
    
    def test_parse_json_log_different_formats(self):
        """Test parsing different progress message formats."""
        parser = ProgressEventParser()
        
        test_cases = [
            ("Processing progress: Audio processing: Extracting audio metadata (54%)", 54, "Audio processing: Extracting audio metadata"),
            ("Processing progress: Audio processing: Transcribing audio content (56%)", 56, "Audio processing: Transcribing audio content"),
            ("Processing... 75%", 75, "Processing..."),
            ("Progress: 45% - Converting video", 45, "Converting video"),
            ("Stage: transcription (67%)", 67, "transcription"),
            ("Audio processing: 80% complete", 80, "Audio processing"),
            ("[75%] Processing audio content", 75, "Processing audio content"),
        ]
        
        for message, expected_percentage, expected_description in test_cases:
            log_line = json.dumps({
                "event": message,
                "logger": "test",
                "level": "info",
                "timestamp": "2025-06-11T11:26:50.780725Z"
            })
            
            event = parser.parse_json_log(log_line)
            
            assert event is not None, f"Failed to parse: {message}"
            assert event.percentage == expected_percentage, f"Wrong percentage for: {message}"
            assert expected_description in event.message, f"Wrong description for: {message}"
    
    def test_parse_plain_text_log(self):
        """Test parsing plain text log lines."""
        parser = ProgressEventParser()
        
        event = parser.parse_plain_text_log("Processing progress: Video conversion (75%)")
        
        assert event is not None
        assert event.percentage == 75
        assert "Video conversion" in event.message
    
    def test_parse_log_stream(self):
        """Test parsing multiple log lines."""
        parser = ProgressEventParser()
        
        log_lines = [
            json.dumps({"event": "Processing progress: Starting (10%)", "timestamp": "2025-06-11T11:26:50.780725Z"}),
            json.dumps({"event": "Processing progress: Middle (50%)", "timestamp": "2025-06-11T11:26:51.780725Z"}),
            json.dumps({"event": "Processing progress: Finishing (90%)", "timestamp": "2025-06-11T11:26:52.780725Z"}),
            "Not a progress line",
            json.dumps({"event": "Some other event", "timestamp": "2025-06-11T11:26:53.780725Z"}),
        ]
        
        events = parser.parse_log_stream(log_lines)
        
        assert len(events) == 3
        assert events[0].percentage == 10
        assert events[1].percentage == 50
        assert events[2].percentage == 90
    
    def test_get_latest_progress(self):
        """Test getting the latest progress from log lines."""
        parser = ProgressEventParser()
        
        log_lines = [
            json.dumps({"event": "Processing progress: Starting (10%)", "timestamp": "2025-06-11T11:26:50.780725Z"}),
            json.dumps({"event": "Processing progress: Middle (50%)", "timestamp": "2025-06-11T11:26:52.780725Z"}),
            json.dumps({"event": "Processing progress: Earlier (30%)", "timestamp": "2025-06-11T11:26:51.780725Z"}),
        ]
        
        latest = parser.get_latest_progress(log_lines)
        
        assert latest is not None
        assert latest.percentage == 50  # Should be the latest by timestamp


class TestProgressHandler:
    """Test progress event handling and job updates."""
    
    @pytest.fixture
    def mock_job_tracker(self):
        """Mock job tracker."""
        return Mock()
    
    @pytest.fixture
    def progress_handler(self, mock_job_tracker):
        """Progress handler with mocked dependencies."""
        with patch('morag_core.jobs.progress_handler.JobTracker', return_value=mock_job_tracker):
            handler = ProgressHandler()
            handler.job_tracker = mock_job_tracker
            return handler
    
    def test_register_job_mapping(self, progress_handler):
        """Test registering job mappings."""
        progress_handler.register_job_mapping("worker-1", "job-123")
        
        assert progress_handler.get_job_id_for_worker("worker-1") == "job-123"
    
    def test_process_log_line_with_job_id(self, progress_handler):
        """Test processing log line with direct job ID."""
        log_line = json.dumps({
            "event": "Processing progress: Audio processing (75%)",
            "timestamp": "2025-06-11T11:26:50.780725Z"
        })
        
        result = progress_handler.process_log_line(log_line, job_id="job-123")
        
        assert result is True
        progress_handler.job_tracker.update_progress.assert_called_once()
        call_args = progress_handler.job_tracker.update_progress.call_args
        assert call_args[1]['job_id'] == "job-123"
        assert call_args[1]['percentage'] == 75
    
    def test_process_log_line_with_worker_mapping(self, progress_handler):
        """Test processing log line with worker ID mapping."""
        progress_handler.register_job_mapping("worker-1", "job-456")
        
        log_line = json.dumps({
            "event": "Processing progress: Video processing (60%)",
            "timestamp": "2025-06-11T11:26:50.780725Z"
        })
        
        result = progress_handler.process_log_line(log_line, worker_id="worker-1")
        
        assert result is True
        progress_handler.job_tracker.update_progress.assert_called_once()
        call_args = progress_handler.job_tracker.update_progress.call_args
        assert call_args[1]['job_id'] == "job-456"
        assert call_args[1]['percentage'] == 60
    
    def test_process_remote_worker_progress(self, progress_handler):
        """Test processing progress from remote worker."""
        progress_handler.register_job_mapping("remote-worker-1", "job-789")
        
        result = progress_handler.process_remote_worker_progress(
            "remote-worker-1", 85, "Transcribing audio"
        )
        
        assert result is True
        progress_handler.job_tracker.update_progress.assert_called_once()
        call_args = progress_handler.job_tracker.update_progress.call_args
        assert call_args[1]['job_id'] == "job-789"
        assert call_args[1]['percentage'] == 85
        assert "Transcribing audio" in call_args[1]['summary']
    
    def test_process_celery_task_progress(self, progress_handler):
        """Test processing progress from Celery task."""
        result = progress_handler.process_celery_task_progress(
            "task-123", 45, "Processing document"
        )
        
        assert result is True
        progress_handler.job_tracker.update_progress.assert_called_once()
        call_args = progress_handler.job_tracker.update_progress.call_args
        assert call_args[1]['job_id'] == "task-123"  # Uses task_id as job_id
        assert call_args[1]['percentage'] == 45
    
    def test_handle_job_completion_success(self, progress_handler):
        """Test handling successful job completion."""
        result = progress_handler.handle_job_completion(
            "job-123", True, "Processing completed successfully"
        )
        
        assert result is True
        progress_handler.job_tracker.mark_completed.assert_called_once_with(
            "job-123", "Processing completed successfully", None
        )
    
    def test_handle_job_completion_failure(self, progress_handler):
        """Test handling failed job completion."""
        result = progress_handler.handle_job_completion(
            "job-123", False, "Processing failed with error"
        )
        
        assert result is True
        progress_handler.job_tracker.mark_failed.assert_called_once_with(
            "job-123", "Processing failed with error", None
        )
    
    def test_sync_job_with_logs(self, progress_handler):
        """Test syncing job with latest progress from logs."""
        log_lines = [
            json.dumps({"event": "Processing progress: Starting (10%)", "timestamp": "2025-06-11T11:26:50.780725Z"}),
            json.dumps({"event": "Processing progress: Finishing (90%)", "timestamp": "2025-06-11T11:26:52.780725Z"}),
        ]
        
        result = progress_handler.sync_job_with_logs("job-123", log_lines)
        
        assert result is True
        progress_handler.job_tracker.update_progress.assert_called_once()
        call_args = progress_handler.job_tracker.update_progress.call_args
        assert call_args[1]['job_id'] == "job-123"
        assert call_args[1]['percentage'] == 90  # Should use latest progress


if __name__ == "__main__":
    pytest.main([__file__])
