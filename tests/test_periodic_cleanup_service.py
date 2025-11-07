#!/usr/bin/env python3
"""
Test for the periodic cleanup service.

This test verifies that the periodic cleanup service properly manages
temporary files without causing race conditions.
"""

import os
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from morag.services.cleanup_service import PeriodicCleanupService
from morag.utils.file_upload import FileUploadConfig, FileUploadHandler


class TestPeriodicCleanupService:
    """Test the periodic cleanup service."""

    def test_cleanup_service_initialization(self):
        """Test that cleanup service initializes correctly."""
        service = PeriodicCleanupService(
            cleanup_interval_hours=2, max_file_age_hours=48, max_disk_usage_mb=500
        )

        assert service.cleanup_interval_hours == 2
        assert service.max_file_age_hours == 48
        assert service.max_disk_usage_mb == 500
        assert not service.is_running

    def test_cleanup_service_start_stop(self):
        """Test starting and stopping the cleanup service."""
        service = PeriodicCleanupService(
            cleanup_interval_hours=24
        )  # Long interval for testing

        # Start service
        service.start()
        assert service.is_running

        # Stop service
        service.stop()
        assert not service.is_running

    def test_cleanup_old_files_by_age(self):
        """Test that old files are cleaned up based on age."""
        config = FileUploadConfig()
        handler = FileUploadHandler(config)

        try:
            # Create test files with different ages
            old_file = handler.temp_dir / "old_file.txt"
            new_file = handler.temp_dir / "new_file.txt"

            old_file.write_text("old content")
            new_file.write_text("new content")

            # Manually set old file's modification time to 25 hours ago
            old_time = time.time() - (25 * 3600)  # 25 hours ago
            os.utime(old_file, (old_time, old_time))

            # Run cleanup with 24 hour max age
            deleted_count = handler.cleanup_old_files(
                max_age_hours=24, max_disk_usage_mb=1000
            )

            # Old file should be deleted, new file should remain
            assert deleted_count == 1
            assert not old_file.exists()
            assert new_file.exists()

        finally:
            handler.cleanup_temp_dir()

    def test_cleanup_by_disk_usage(self):
        """Test that files are cleaned up when disk usage exceeds limit."""
        config = FileUploadConfig()
        handler = FileUploadHandler(config)

        try:
            # Create multiple test files
            files = []
            for i in range(5):
                file_path = handler.temp_dir / f"test_file_{i}.txt"
                file_path.write_text("x" * 1000)  # 1KB each

                # Set different modification times (oldest first)
                file_time = time.time() - (i * 3600)  # i hours ago
                os.utime(file_path, (file_time, file_time))
                files.append(file_path)

            # Run cleanup with very low disk usage limit (should trigger cleanup)
            deleted_count = handler.cleanup_old_files(
                max_age_hours=48, max_disk_usage_mb=0.001
            )  # 1KB limit

            # Should delete some files due to disk usage limit
            assert deleted_count > 0

            # Check that some files were deleted (exact order may vary)
            remaining_files = [f for f in files if f.exists()]
            assert len(remaining_files) < len(files)

        finally:
            handler.cleanup_temp_dir()

    def test_cleanup_preserves_recent_files_under_disk_limit(self):
        """Test that recent files are preserved when under disk usage limit."""
        config = FileUploadConfig()
        handler = FileUploadHandler(config)

        try:
            # Create a small recent file
            recent_file = handler.temp_dir / "recent_file.txt"
            recent_file.write_text("small content")

            # Run cleanup with generous limits
            deleted_count = handler.cleanup_old_files(
                max_age_hours=24, max_disk_usage_mb=1000
            )

            # No files should be deleted
            assert deleted_count == 0
            assert recent_file.exists()

        finally:
            handler.cleanup_temp_dir()

    def test_force_cleanup(self):
        """Test force cleanup functionality."""
        service = PeriodicCleanupService()

        # Mock the upload handler
        with patch(
            "morag.services.cleanup_service.get_upload_handler"
        ) as mock_get_handler:
            mock_handler = Mock()
            mock_handler.cleanup_old_files.return_value = 5
            mock_get_handler.return_value = mock_handler

            # Force cleanup
            deleted_count = service.force_cleanup()

            # Verify cleanup was called and returned correct count
            assert deleted_count == 5
            mock_handler.cleanup_old_files.assert_called_once()

    def test_cleanup_handles_missing_temp_dir(self):
        """Test that cleanup handles missing temp directory gracefully."""
        config = FileUploadConfig()
        handler = FileUploadHandler(config)

        # Remove temp directory
        import shutil

        shutil.rmtree(handler.temp_dir, ignore_errors=True)

        # Cleanup should handle missing directory gracefully
        deleted_count = handler.cleanup_old_files()
        assert deleted_count == 0

    def test_cleanup_ignores_marker_files(self):
        """Test that cleanup ignores marker files (starting with dot)."""
        config = FileUploadConfig()
        handler = FileUploadHandler(config)

        try:
            # Create marker file and regular file
            marker_file = handler.temp_dir / ".marker_file"
            regular_file = handler.temp_dir / "regular_file.txt"

            marker_file.write_text("marker content")
            regular_file.write_text("regular content")

            # Set both files to be old
            old_time = time.time() - (25 * 3600)  # 25 hours ago
            os.utime(marker_file, (old_time, old_time))
            os.utime(regular_file, (old_time, old_time))

            # Run cleanup
            deleted_count = handler.cleanup_old_files(max_age_hours=24)

            # Only regular file should be deleted, marker file should remain
            assert deleted_count == 1
            assert marker_file.exists()
            assert not regular_file.exists()

        finally:
            handler.cleanup_temp_dir()

    @patch("morag.utils.file_upload.logger")
    def test_cleanup_logging(self, mock_logger):
        """Test that cleanup operations are properly logged."""
        config = FileUploadConfig()
        handler = FileUploadHandler(config)

        try:
            # Create an old file
            old_file = handler.temp_dir / "old_file.txt"
            old_file.write_text("old content")

            # Set file to be old
            old_time = time.time() - (25 * 3600)  # 25 hours ago
            os.utime(old_file, (old_time, old_time))

            # Run cleanup
            deleted_count = handler.cleanup_old_files(max_age_hours=24)

            # Verify that at least one file was deleted
            assert deleted_count > 0

            # Verify debug logging was called
            debug_calls = [
                call
                for call in mock_logger.debug.call_args_list
                if "Cleanup scan results" in str(call)
            ]
            assert len(debug_calls) > 0

        finally:
            handler.cleanup_temp_dir()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
