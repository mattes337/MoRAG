#!/usr/bin/env python3
"""
Integration test for the file upload race condition fix.

This test simulates the actual race condition scenario that was occurring
in the production logs and verifies that our fix resolves it.
"""

import asyncio
import tempfile
import time
import threading
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import pytest

from morag.utils.file_upload import FileUploadHandler, FileUploadConfig
from morag.ingest_tasks import ingest_file_task


class TestRaceConditionIntegration:
    """Integration test for race condition fix."""
    
    @pytest.mark.asyncio
    async def test_file_survives_until_worker_processes_it(self):
        """Test that uploaded files survive long enough for worker to process them."""
        # Create handler with very short cleanup timeout for testing
        config = FileUploadConfig(cleanup_timeout=2)  # 2 seconds
        handler = FileUploadHandler(config)
        
        try:
            # Create a test file
            test_file = handler.temp_dir / "test_document.pdf"
            test_file.write_text("Test PDF content for race condition test")
            
            # Schedule cleanup (this starts the background thread)
            handler._schedule_cleanup(test_file)
            
            # Verify file exists immediately after scheduling cleanup
            assert test_file.exists()
            
            # Simulate worker delay (worker takes time to start processing)
            # In the real scenario, this was about 17 seconds
            await asyncio.sleep(0.5)  # Simulate some delay
            
            # File should still exist during worker processing
            assert test_file.exists(), "File was cleaned up too early - race condition detected!"
            
            # Simulate worker processing the file
            file_content = test_file.read_text()
            assert file_content == "Test PDF content for race condition test"
            
            # File should still exist after processing (cleanup hasn't run yet)
            assert test_file.exists()
            
            # Wait for cleanup to complete
            time.sleep(2.2)
            
            # Now file should be cleaned up
            assert not test_file.exists(), "File was not cleaned up after timeout"
            
        finally:
            # Cleanup temp directory
            handler.cleanup_temp_dir()
    
    @pytest.mark.asyncio
    async def test_multiple_files_no_race_condition(self):
        """Test that multiple files can be processed without race conditions."""
        config = FileUploadConfig(cleanup_timeout=1)  # 1 second
        handler = FileUploadHandler(config)
        
        try:
            # Create multiple test files
            test_files = []
            for i in range(3):
                test_file = handler.temp_dir / f"test_document_{i}.pdf"
                test_file.write_text(f"Test PDF content {i}")
                test_files.append(test_file)
                
                # Schedule cleanup for each file
                handler._schedule_cleanup(test_file)
            
            # All files should exist initially
            for test_file in test_files:
                assert test_file.exists()
            
            # Simulate concurrent worker processing
            async def simulate_worker_processing(file_path):
                await asyncio.sleep(0.2)  # Simulate processing delay
                if file_path.exists():
                    content = file_path.read_text()
                    return f"Processed: {content}"
                else:
                    raise FileNotFoundError(f"File {file_path} was cleaned up prematurely")
            
            # Process all files concurrently
            tasks = [simulate_worker_processing(f) for f in test_files]
            results = await asyncio.gather(*tasks)
            
            # All files should have been processed successfully
            assert len(results) == 3
            for i, result in enumerate(results):
                assert result == f"Processed: Test PDF content {i}"
            
            # Files should still exist (cleanup hasn't run yet)
            for test_file in test_files:
                assert test_file.exists()
            
            # Wait for cleanup to complete
            time.sleep(1.2)
            
            # All files should now be cleaned up
            for test_file in test_files:
                assert not test_file.exists()
                
        finally:
            handler.cleanup_temp_dir()
    
    def test_cleanup_threads_are_daemon_threads(self):
        """Test that cleanup threads are daemon threads to prevent blocking shutdown."""
        config = FileUploadConfig(cleanup_timeout=10)  # Long timeout
        handler = FileUploadHandler(config)
        
        try:
            # Create a test file
            test_file = handler.temp_dir / "test_daemon.txt"
            test_file.write_text("Test content")
            
            # Schedule cleanup
            handler._schedule_cleanup(test_file)
            
            # Verify cleanup thread was created and is a daemon thread
            assert len(handler._cleanup_threads) == 1
            cleanup_thread = handler._cleanup_threads[0]
            assert isinstance(cleanup_thread, threading.Thread)
            assert cleanup_thread.daemon is True, "Cleanup thread should be a daemon thread"
            
            # Daemon threads should not prevent program exit
            # This is important for proper shutdown behavior
            
        finally:
            handler.cleanup_temp_dir()
    
    @patch('morag.ingest_tasks.logger')
    def test_enhanced_error_logging_on_file_not_found(self, mock_logger):
        """Test that enhanced error logging provides useful debugging information."""
        # Create a non-existent file path
        non_existent_path = "/tmp/non_existent_file_12345.pdf"
        
        # Mock the Celery task request
        mock_request = Mock()
        mock_request.id = "test-task-123"
        
        # Create a mock task instance
        mock_task = Mock()
        mock_task.request = mock_request
        mock_task.update_state = Mock()
        
        # Test the enhanced error handling
        try:
            # This should trigger the enhanced error logging
            with patch('morag.ingest_tasks.get_morag_api'):
                # Call the task function directly to test error handling
                result = ingest_file_task.apply(
                    args=[non_existent_path, "document", {}],
                    task_id="test-task-123"
                )
        except Exception:
            # Expected to fail, we're testing the error logging
            pass
        
        # Verify that error logging was called with detailed information
        # The enhanced logging should provide debugging context
        error_calls = [call for call in mock_logger.error.call_args_list 
                      if 'file_path' in str(call)]
        
        # Should have at least one detailed error log
        assert len(error_calls) > 0, "Enhanced error logging should be called"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
