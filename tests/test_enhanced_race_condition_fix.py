#!/usr/bin/env python3
"""
Test for the enhanced race condition fix with improved temp directory handling.

This test verifies that the enhanced solution addresses both the asyncio cancellation
issue and the temp directory persistence issue.
"""

import tempfile
import time
import threading
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

from morag.utils.file_upload import FileUploadHandler, FileUploadConfig, get_upload_handler


class TestEnhancedRaceConditionFix:
    """Test the enhanced race condition fix."""
    
    def test_temp_directory_location_preference(self):
        """Test that handler prefers app temp directory over system temp."""
        # Create a mock temp directory
        with tempfile.TemporaryDirectory() as mock_app_temp:
            app_temp_path = Path(mock_app_temp)
            
            # Mock the Path("./temp") to point to our mock directory
            with patch('morag.utils.file_upload.Path') as mock_path_class:
                def path_side_effect(path_str):
                    if path_str == "./temp":
                        return app_temp_path
                    else:
                        return Path(path_str)
                
                mock_path_class.side_effect = path_side_effect
                
                # Create handler
                config = FileUploadConfig()
                handler = FileUploadHandler(config)
                
                # Verify it used the app temp directory
                assert str(handler.temp_dir).startswith(str(app_temp_path))
                assert "morag_uploads_" in str(handler.temp_dir)
                
                # Cleanup
                handler.cleanup_temp_dir()
    
    def test_fallback_to_system_temp(self):
        """Test fallback to system temp when app temp is not available."""
        # Mock Path("./temp") to not exist
        with patch('morag.utils.file_upload.Path') as mock_path_class:
            def path_side_effect(path_str):
                if path_str == "./temp":
                    mock_path = Mock()
                    mock_path.exists.return_value = False
                    return mock_path
                else:
                    return Path(path_str)
            
            mock_path_class.side_effect = path_side_effect
            
            # Create handler
            config = FileUploadConfig()
            handler = FileUploadHandler(config)
            
            # Should have fallen back to system temp
            assert handler.temp_dir.exists()
            assert "morag_uploads_" in str(handler.temp_dir)
            
            # Cleanup
            handler.cleanup_temp_dir()
    
    def test_marker_file_creation(self):
        """Test that marker file is created for tracking."""
        config = FileUploadConfig()
        handler = FileUploadHandler(config)
        
        try:
            # Check that marker file exists
            marker_file = handler.temp_dir / ".morag_upload_handler_active"
            assert marker_file.exists()
            
            # Check marker file content
            content = marker_file.read_text()
            assert "Created at" in content
            assert content.replace("Created at ", "").strip().replace(".", "").isdigit()
            
        finally:
            handler.cleanup_temp_dir()
    
    def test_enhanced_logging_on_garbage_collection(self):
        """Test that garbage collection is logged for debugging."""
        with patch('morag.utils.file_upload.logger') as mock_logger:
            config = FileUploadConfig()
            handler = FileUploadHandler(config)
            temp_dir = handler.temp_dir
            
            # Manually call __del__ to test logging
            handler.__del__()
            
            # Verify warning was logged
            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args
            assert "FileUploadHandler being garbage collected" in str(call_args)
            # Check that temp_dir path is in the call (normalize path separators)
            temp_dir_str = str(temp_dir).replace('\\', '/')
            call_args_str = str(call_args).replace('\\\\', '/').replace('\\', '/')
            assert temp_dir_str in call_args_str
            
            # Cleanup
            handler.cleanup_temp_dir()
    
    def test_get_upload_handler_recreates_on_missing_temp_dir(self):
        """Test that get_upload_handler recreates handler if temp dir is missing."""
        with patch('morag.utils.file_upload.logger') as mock_logger:
            # Get initial handler
            handler1 = get_upload_handler()
            temp_dir1 = handler1.temp_dir
            
            # Manually remove the temp directory to simulate the issue
            import shutil
            shutil.rmtree(temp_dir1, ignore_errors=True)
            
            # Get handler again - should detect missing temp dir and recreate
            handler2 = get_upload_handler()
            temp_dir2 = handler2.temp_dir
            
            # Should be a different temp directory
            assert temp_dir1 != temp_dir2
            assert temp_dir2.exists()
            
            # Should have logged the recreation
            warning_calls = [call for call in mock_logger.warning.call_args_list 
                           if "temp directory missing" in str(call)]
            assert len(warning_calls) > 0
            
            # Cleanup
            handler2.cleanup_temp_dir()
    
    def test_threading_cleanup_survives_context_changes(self):
        """Test that threading-based cleanup survives various context changes."""
        config = FileUploadConfig(cleanup_timeout=1)  # 1 second for testing
        handler = FileUploadHandler(config)
        
        try:
            # Create test file
            test_file = handler.temp_dir / "test_survival.txt"
            test_file.write_text("test content")
            
            # Schedule cleanup
            handler._schedule_cleanup(test_file)
            
            # Verify cleanup thread was created
            assert len(handler._cleanup_threads) == 1
            cleanup_thread = handler._cleanup_threads[0]
            assert cleanup_thread.daemon is True
            assert cleanup_thread.is_alive()
            
            # File should exist initially
            assert test_file.exists()
            
            # Simulate various context changes that might affect asyncio tasks
            import asyncio
            import gc
            
            # Force garbage collection
            gc.collect()
            
            # Create and destroy event loops
            loop1 = asyncio.new_event_loop()
            asyncio.set_event_loop(loop1)
            loop1.close()
            
            loop2 = asyncio.new_event_loop()
            asyncio.set_event_loop(loop2)
            loop2.close()
            
            # File should still exist (cleanup hasn't run yet)
            assert test_file.exists()
            
            # Thread should still be alive
            assert cleanup_thread.is_alive()
            
            # Wait for cleanup to complete
            time.sleep(1.2)
            
            # File should now be cleaned up
            assert not test_file.exists()
            
        finally:
            handler.cleanup_temp_dir()
    
    @patch('morag.utils.file_upload.logger')
    def test_configure_upload_handler_logging(self, mock_logger):
        """Test that configure_upload_handler logs properly."""
        from morag.utils.file_upload import configure_upload_handler
        
        # Configure new handler
        config = FileUploadConfig(max_file_size=50 * 1024 * 1024)
        configure_upload_handler(config)
        
        # Verify logging was called
        info_calls = [call for call in mock_logger.info.call_args_list 
                     if "Configuring new FileUploadHandler" in str(call)]
        assert len(info_calls) > 0
        
        # Check that config details were logged
        call_str = str(info_calls[0])
        assert "new_config" in call_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
