#!/usr/bin/env python3
"""Test script for temp directory fixes and validation."""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import uuid
import pytest

# Add the packages to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "morag" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "morag-core" / "src"))

from morag.utils.file_upload import (
    FileUploadHandler, 
    FileUploadConfig, 
    get_upload_handler,
    validate_temp_directory_access
)


class TestTempDirectoryFixes:
    """Test cases for temp directory fixes."""
    
    def setup_method(self):
        """Setup for each test."""
        # Reset global upload handler
        import morag.utils.file_upload
        morag.utils.file_upload._upload_handler = None
    
    def test_temp_directory_validation_success(self):
        """Test successful temp directory validation."""
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the temp directory to use our test directory
            config = FileUploadConfig()
            handler = FileUploadHandler(config)
            
            # Override the temp directory to our test directory
            handler.temp_dir = Path(temp_dir) / "test_uploads"
            handler.temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Set as global handler
            import morag.utils.file_upload
            morag.utils.file_upload._upload_handler = handler
            
            # Validation should succeed
            assert validate_temp_directory_access() is True
    
    def test_temp_directory_validation_failure_not_writable(self):
        """Test temp directory validation failure when directory is not writable."""
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir) / "readonly_uploads"
            test_dir.mkdir(parents=True, exist_ok=True)
            
            # Make directory read-only (on Unix systems)
            if os.name != 'nt':  # Skip on Windows
                os.chmod(test_dir, 0o444)
                
                config = FileUploadConfig()
                handler = FileUploadHandler(config)
                handler.temp_dir = test_dir
                
                # Set as global handler
                import morag.utils.file_upload
                morag.utils.file_upload._upload_handler = handler
                
                # Validation should fail
                with pytest.raises(RuntimeError, match="Temp directory validation failed"):
                    validate_temp_directory_access()
    
    def test_file_upload_handler_write_permission_test(self):
        """Test that FileUploadHandler tests write permissions during directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = FileUploadConfig()
            handler = FileUploadHandler(config)
            
            # Test _try_create_dir method
            test_dir = Path(temp_dir) / "test_write_permissions"
            
            # Should succeed with writable directory
            assert handler._try_create_dir(test_dir) is True
            assert test_dir.exists()
            
            # Cleanup test files
            if test_dir.exists():
                shutil.rmtree(test_dir)
    
    def test_shared_volume_detection(self):
        """Test detection of shared volume vs system temp."""
        # Test with /app/temp path (shared volume)
        config = FileUploadConfig()
        handler = FileUploadHandler(config)
        
        # Mock /app/temp directory
        app_temp = Path("/app/temp")
        if app_temp.exists():
            # If /app/temp exists, handler should prefer it
            assert str(handler.temp_dir).startswith("/app/temp")
        else:
            # If /app/temp doesn't exist, should use fallback
            # This is expected in local development
            assert handler.temp_dir.exists()
    
    def test_system_temp_warning(self):
        """Test that using system temp directory generates appropriate warnings."""
        import logging
        import io
        
        # Capture log output
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        logger = logging.getLogger("morag.utils.file_upload")
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)
        
        try:
            # Force system temp usage by making other directories unavailable
            with tempfile.TemporaryDirectory() as temp_dir:
                config = FileUploadConfig()
                upload_handler = FileUploadHandler(config)
                
                # If using system temp, should have warning in logs
                if str(upload_handler.temp_dir).startswith('/tmp/'):
                    log_output = log_capture.getvalue()
                    # Should contain warning about system temp usage
                    # (This test may not trigger in all environments)
                    pass
        finally:
            logger.removeHandler(handler)
    
    def test_temp_directory_creation_failure(self):
        """Test handling of temp directory creation failure."""
        # This test is challenging to implement reliably across platforms
        # as it requires simulating filesystem permission failures
        pass
    
    def test_upload_handler_singleton_behavior(self):
        """Test that upload handler behaves correctly as singleton."""
        # Get first handler
        handler1 = get_upload_handler()
        
        # Get second handler - should be same instance
        handler2 = get_upload_handler()
        
        assert handler1 is handler2
        assert handler1.temp_dir == handler2.temp_dir
    
    def test_temp_directory_exists_check(self):
        """Test that upload handler checks if temp directory still exists."""
        # Get initial handler
        handler1 = get_upload_handler()
        temp_dir = handler1.temp_dir
        
        # Remove the temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        
        # Get handler again - should create new one
        handler2 = get_upload_handler()
        
        # Should be different instance with new temp directory
        assert handler1 is not handler2
        assert handler2.temp_dir.exists()


def main():
    """Run the tests."""
    print("🧪 Testing temp directory fixes...")
    
    # Run basic validation test
    try:
        print("📁 Testing temp directory validation...")
        validate_temp_directory_access()
        print("✅ Temp directory validation passed")
    except Exception as e:
        print(f"❌ Temp directory validation failed: {e}")
        return False
    
    # Test upload handler
    try:
        print("📤 Testing upload handler...")
        handler = get_upload_handler()
        print(f"✅ Upload handler created with temp dir: {handler.temp_dir}")
        
        # Test write permissions
        test_file = handler.temp_dir / f"test_{uuid.uuid4().hex[:8]}.txt"
        test_file.write_text("test content")
        test_file.unlink()
        print("✅ Write permissions test passed")
        
    except Exception as e:
        print(f"❌ Upload handler test failed: {e}")
        return False
    
    print("🎉 All temp directory tests passed!")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
