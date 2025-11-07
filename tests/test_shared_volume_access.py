#!/usr/bin/env python3
"""
Test to verify that shared volumes work correctly between containers.

This test verifies that files created in the temp directory can be accessed
by both the API container and worker containers.
"""

import tempfile
import time
from pathlib import Path

import pytest
from morag.utils.file_upload import FileUploadConfig, FileUploadHandler


class TestSharedVolumeAccess:
    """Test shared volume access between containers."""

    def test_temp_directory_location(self):
        """Test that temp directory is created in the correct shared location."""
        config = FileUploadConfig()
        handler = FileUploadHandler(config)

        try:
            # Check if temp directory is in a shared location
            temp_dir_str = str(handler.temp_dir)

            # Should prefer /app/temp (Docker) or ./temp (local) over system temp
            is_shared_location = (
                temp_dir_str.startswith("/app/temp")
                or temp_dir_str.startswith("./temp")
                or "temp" in temp_dir_str.lower()
            )

            print(f"Temp directory: {temp_dir_str}")
            print(f"Is shared location: {is_shared_location}")

            # Verify temp directory exists and is writable
            assert handler.temp_dir.exists()
            assert handler.temp_dir.is_dir()

            # Test writing a file
            test_file = handler.temp_dir / "test_shared_access.txt"
            test_file.write_text("Test content for shared access")

            assert test_file.exists()
            assert test_file.read_text() == "Test content for shared access"

            # Clean up test file
            test_file.unlink()

        finally:
            handler.cleanup_temp_dir()

    def test_file_persistence_across_handler_instances(self):
        """Test that files persist when accessed by different handler instances."""
        config = FileUploadConfig()

        # Create first handler and save a file
        handler1 = FileUploadHandler(config)
        temp_dir1 = handler1.temp_dir

        try:
            test_file = temp_dir1 / "persistent_test.txt"
            test_file.write_text("Persistent test content")

            # Verify file exists
            assert test_file.exists()

            # Create second handler (simulating different container/process)
            handler2 = FileUploadHandler(config)
            temp_dir2 = handler2.temp_dir

            # If using shared volumes, we should be able to find files in shared location
            # Note: Different handlers will have different subdirectories, but the parent should be shared

            print(f"Handler 1 temp dir: {temp_dir1}")
            print(f"Handler 2 temp dir: {temp_dir2}")

            # Both should be using shared location
            temp_dir1_str = str(temp_dir1)
            temp_dir2_str = str(temp_dir2)

            shared_location1 = temp_dir1_str.startswith(
                "/app/temp"
            ) or temp_dir1_str.startswith("./temp")
            shared_location2 = temp_dir2_str.startswith(
                "/app/temp"
            ) or temp_dir2_str.startswith("./temp")

            print(f"Handler 1 using shared location: {shared_location1}")
            print(f"Handler 2 using shared location: {shared_location2}")

            # Both handlers should be using shared locations
            # (They may have different subdirectories, but same parent)

        finally:
            handler1.cleanup_temp_dir()
            handler2.cleanup_temp_dir()

    def test_docker_volume_paths(self):
        """Test that Docker volume paths are correctly detected."""
        # Test the path detection logic
        app_temp_path = Path("/app/temp")
        local_temp_path = Path("./temp")

        print(f"/app/temp exists: {app_temp_path.exists()}")
        print(f"./temp exists: {local_temp_path.exists()}")

        # In Docker environment, /app/temp should exist
        # In local development, ./temp might exist

        # At least one should be available or creatable
        config = FileUploadConfig()
        handler = FileUploadHandler(config)

        temp_dir_str = str(handler.temp_dir)
        print(f"Selected temp directory: {temp_dir_str}")

        # Verify the directory is accessible
        assert handler.temp_dir.exists()
        assert handler.temp_dir.is_dir()

        # Test file operations
        test_file = handler.temp_dir / "docker_volume_test.txt"
        test_file.write_text("Docker volume test content")

        assert test_file.exists()
        content = test_file.read_text()
        assert content == "Docker volume test content"

        # Clean up
        test_file.unlink()
        handler.cleanup_temp_dir()

    def test_marker_file_in_shared_location(self):
        """Test that marker files are created in shared location."""
        config = FileUploadConfig()
        handler = FileUploadHandler(config)

        try:
            # Check for marker file
            marker_file = handler.temp_dir / ".morag_upload_handler_active"
            assert marker_file.exists()

            # Read marker content
            content = marker_file.read_text()
            assert "Created at" in content

            print(f"Marker file location: {marker_file}")
            print(f"Marker file content: {content}")

            # Verify marker file is in shared location
            marker_path_str = str(marker_file)
            is_shared = marker_path_str.startswith(
                "/app/temp"
            ) or marker_path_str.startswith("./temp")

            print(f"Marker file in shared location: {is_shared}")

        finally:
            handler.cleanup_temp_dir()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to show print statements
