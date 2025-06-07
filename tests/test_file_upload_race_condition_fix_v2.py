#!/usr/bin/env python3
"""
Test for the updated file upload race condition fix.

This test verifies that the threading-based cleanup mechanism
prevents race conditions between file upload and task processing.
"""

import asyncio
import tempfile
import time
import threading
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

from morag.utils.file_upload import FileUploadHandler, FileUploadConfig


class TestFileUploadRaceConditionFixV2:
    """Test the updated race condition fix using threading instead of asyncio."""
    
    def test_cleanup_uses_threading_not_asyncio(self):
        """Test that cleanup uses threading instead of asyncio tasks."""
        config = FileUploadConfig(cleanup_timeout=1)  # 1 second for testing
        handler = FileUploadHandler(config)
        
        # Create a temporary file
        test_file = handler.temp_dir / "test_file.txt"
        test_file.write_text("test content")
        
        # Verify file exists
        assert test_file.exists()
        
        # Schedule cleanup
        handler._schedule_cleanup(test_file)
        
        # Verify cleanup thread was created
        assert hasattr(handler, '_cleanup_threads')
        assert len(handler._cleanup_threads) == 1
        assert isinstance(handler._cleanup_threads[0], threading.Thread)
        assert handler._cleanup_threads[0].daemon is True
        
        # File should still exist immediately after scheduling
        assert test_file.exists()
        
        # Wait for cleanup to complete (1 second + small buffer)
        time.sleep(1.2)
        
        # File should now be cleaned up
        assert not test_file.exists()
    
    def test_multiple_cleanup_threads(self):
        """Test that multiple cleanup threads can be scheduled."""
        config = FileUploadConfig(cleanup_timeout=2)  # 2 seconds for testing
        handler = FileUploadHandler(config)
        
        # Create multiple temporary files
        test_files = []
        for i in range(3):
            test_file = handler.temp_dir / f"test_file_{i}.txt"
            test_file.write_text(f"test content {i}")
            test_files.append(test_file)
            
            # Schedule cleanup for each file
            handler._schedule_cleanup(test_file)
        
        # Verify all files exist initially
        for test_file in test_files:
            assert test_file.exists()
        
        # Verify multiple cleanup threads were created
        assert len(handler._cleanup_threads) == 3
        for thread in handler._cleanup_threads:
            assert isinstance(thread, threading.Thread)
            assert thread.daemon is True
        
        # Files should still exist immediately after scheduling
        for test_file in test_files:
            assert test_file.exists()
        
        # Wait for cleanup to complete
        time.sleep(2.2)
        
        # All files should now be cleaned up
        for test_file in test_files:
            assert not test_file.exists()
    
    def test_cleanup_handles_missing_file_gracefully(self):
        """Test that cleanup handles already-deleted files gracefully."""
        config = FileUploadConfig(cleanup_timeout=0.5)  # 0.5 seconds for testing
        handler = FileUploadHandler(config)
        
        # Create a temporary file
        test_file = handler.temp_dir / "test_file.txt"
        test_file.write_text("test content")
        
        # Schedule cleanup
        handler._schedule_cleanup(test_file)
        
        # Manually delete the file before cleanup runs
        test_file.unlink()
        
        # Wait for cleanup to complete - should not raise an error
        time.sleep(0.7)
        
        # Verify cleanup thread was created and completed without error
        assert len(handler._cleanup_threads) == 1
    
    def test_cleanup_survives_request_context_end(self):
        """Test that cleanup threads survive when HTTP request context ends."""
        config = FileUploadConfig(cleanup_timeout=1)
        handler = FileUploadHandler(config)
        
        # Create a temporary file
        test_file = handler.temp_dir / "test_file.txt"
        test_file.write_text("test content")
        
        # Schedule cleanup
        handler._schedule_cleanup(test_file)
        
        # Simulate end of request context by creating and ending an asyncio event loop
        async def simulate_request():
            # This simulates what happens during a FastAPI request
            await asyncio.sleep(0.1)
            return "request_complete"
        
        # Run and complete the simulated request
        result = asyncio.run(simulate_request())
        assert result == "request_complete"
        
        # File should still exist (cleanup hasn't run yet)
        assert test_file.exists()
        
        # Wait for cleanup to complete
        time.sleep(1.2)
        
        # File should now be cleaned up, proving the thread survived the event loop ending
        assert not test_file.exists()
    
    def test_handler_initialization_creates_cleanup_threads_list(self):
        """Test that handler initialization creates the cleanup threads list."""
        handler = FileUploadHandler()
        
        # Verify cleanup threads list is initialized
        assert hasattr(handler, '_cleanup_threads')
        assert isinstance(handler._cleanup_threads, list)
        assert len(handler._cleanup_threads) == 0
    
    @patch('morag.utils.file_upload.logger')
    def test_cleanup_logging(self, mock_logger):
        """Test that cleanup operations are properly logged."""
        config = FileUploadConfig(cleanup_timeout=0.5)
        handler = FileUploadHandler(config)
        
        # Create a temporary file
        test_file = handler.temp_dir / "test_file.txt"
        test_file.write_text("test content")
        
        # Schedule cleanup
        handler._schedule_cleanup(test_file)
        
        # Wait for cleanup to complete
        time.sleep(0.7)
        
        # Verify logging calls were made
        mock_logger.debug.assert_called()
        mock_logger.info.assert_called()
        
        # Check that the debug log was called with correct parameters
        debug_calls = [call for call in mock_logger.debug.call_args_list 
                      if 'Scheduled cleanup task started' in str(call)]
        assert len(debug_calls) > 0
        
        # Check that the info log was called for successful cleanup
        info_calls = [call for call in mock_logger.info.call_args_list 
                     if 'Cleaned up temporary file after timeout' in str(call)]
        assert len(info_calls) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
