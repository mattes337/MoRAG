"""File upload handling utilities for MoRAG system."""

import tempfile
import time
import uuid
import asyncio
import aiofiles
import mimetypes
from pathlib import Path
from typing import Optional, List, Dict, Any, Set
import structlog
from fastapi import UploadFile, HTTPException

logger = structlog.get_logger(__name__)


class FileUploadError(Exception):
    """Exception raised for file upload errors."""
    pass


class FileUploadConfig:
    """Configuration for file upload handling."""
    
    def __init__(
        self,
        max_file_size: int = 100 * 1024 * 1024,  # 100MB default
        allowed_extensions: Optional[Set[str]] = None,
        temp_dir_prefix: str = "morag_uploads_",
        cleanup_timeout: int = 3600,  # 1 hour
        allowed_mime_types: Optional[Set[str]] = None
    ):
        self.max_file_size = max_file_size
        self.allowed_extensions = allowed_extensions or {
            '.pdf', '.txt', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls',
            '.mp3', '.wav', '.m4a', '.flac', '.ogg',
            '.mp4', '.avi', '.mov', '.mkv', '.webm',
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp',
            '.html', '.htm', '.md', '.rtf'
        }
        self.temp_dir_prefix = temp_dir_prefix
        self.cleanup_timeout = cleanup_timeout
        self.allowed_mime_types = allowed_mime_types or {
            # Documents
            'application/pdf', 'text/plain', 'text/markdown',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/msword',
            'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            'application/vnd.ms-powerpoint',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'application/vnd.ms-excel',
            # Audio
            'audio/mpeg', 'audio/wav', 'audio/mp4', 'audio/flac', 'audio/ogg',
            # Video
            'video/mp4', 'video/avi', 'video/quicktime', 'video/x-msvideo',
            'video/x-matroska', 'video/webm',
            # Images
            'image/jpeg', 'image/png', 'image/gif', 'image/bmp',
            'image/tiff', 'image/webp',
            # Web
            'text/html', 'application/rtf'
        }


class FileUploadHandler:
    """Handles file uploads with validation, security, and cleanup."""
    
    def __init__(self, config: Optional[FileUploadConfig] = None):
        self.config = config or FileUploadConfig()

        # Try to create temp directory in shared location for Docker containers
        # This ensures all containers (API and workers) can access the same files
        temp_dir_created = False

        try:
            # First try /app/temp (Docker shared volume) - REQUIRED for container deployments
            app_temp_dir = Path("/app/temp")
            if app_temp_dir.exists() or self._try_create_dir(app_temp_dir):
                self.temp_dir = app_temp_dir / f"{self.config.temp_dir_prefix}{uuid.uuid4().hex[:8]}"
                self.temp_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Using shared Docker temp directory", temp_dir=str(self.temp_dir))
                temp_dir_created = True
            else:
                # Try ./temp directory (local development)
                local_temp_dir = Path("./temp")
                if local_temp_dir.exists() or self._try_create_dir(local_temp_dir):
                    self.temp_dir = local_temp_dir / f"{self.config.temp_dir_prefix}{uuid.uuid4().hex[:8]}"
                    self.temp_dir.mkdir(parents=True, exist_ok=True)
                    logger.info("Using local temp directory", temp_dir=str(self.temp_dir))
                    temp_dir_created = True
                else:
                    # CRITICAL: System temp directory will NOT work in container environments
                    # because workers won't have access to the same files
                    self.temp_dir = Path(tempfile.mkdtemp(prefix=self.config.temp_dir_prefix))
                    logger.error("CRITICAL: Using system temp directory - files will NOT be shared between containers!",
                               temp_dir=str(self.temp_dir),
                               warning="This will cause file access errors in worker processes")
                    temp_dir_created = True

        except Exception as e:
            logger.error("CRITICAL: Failed to create any temp directory",
                        error=str(e))
            raise RuntimeError(f"Cannot create temporary directory for file uploads: {str(e)}")

        if not temp_dir_created:
            raise RuntimeError("Failed to create any usable temporary directory")

        # No longer tracking individual cleanup threads - using periodic cleanup instead

        # Create a marker file to help track directory lifecycle
        marker_file = self.temp_dir / ".morag_upload_handler_active"
        marker_file.write_text(f"Created at {time.time()}")

        logger.info("FileUploadHandler initialized",
                   temp_dir=str(self.temp_dir),
                   max_file_size=self.config.max_file_size)

    def _try_create_dir(self, dir_path: Path) -> bool:
        """Try to create a directory and return True if successful.

        Args:
            dir_path: Path to directory to create

        Returns:
            True if directory exists or was created successfully
        """
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            # Test write permissions by creating a test file
            test_file = dir_path / f".write_test_{uuid.uuid4().hex[:8]}"
            test_file.write_text("test")
            test_file.unlink()
            return dir_path.exists() and dir_path.is_dir()
        except Exception as e:
            logger.debug("Failed to create directory or test write permissions",
                        dir_path=str(dir_path), error=str(e))
            return False
    
    async def save_upload(self, file: UploadFile) -> Path:
        """Save uploaded file to temporary location with validation.
        
        Args:
            file: FastAPI UploadFile object
            
        Returns:
            Path to saved temporary file
            
        Raises:
            FileUploadError: If file validation fails or save operation fails
        """
        try:
            # Validate file
            await self._validate_file(file)
            
            # Generate unique filename
            unique_filename = self._generate_unique_filename(file.filename)
            temp_path = self.temp_dir / unique_filename
            
            # Ensure temp directory exists
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Save file asynchronously
            logger.info("Saving uploaded file", 
                       filename=file.filename,
                       temp_path=str(temp_path),
                       content_type=file.content_type)
            
            async with aiofiles.open(temp_path, 'wb') as f:
                # Read file in chunks to handle large files efficiently
                chunk_size = 8192  # 8KB chunks
                total_size = 0
                
                while True:
                    chunk = await file.read(chunk_size)
                    if not chunk:
                        break
                    
                    total_size += len(chunk)
                    if total_size > self.config.max_file_size:
                        # Clean up partial file
                        if temp_path.exists():
                            temp_path.unlink()
                        raise FileUploadError(
                            f"File size exceeds maximum allowed size of "
                            f"{self.config.max_file_size} bytes"
                        )
                    
                    await f.write(chunk)
            
            # NOTE: No longer scheduling individual file cleanup to prevent race conditions
            # Files will be cleaned up by periodic cleanup process based on age and disk space
            
            logger.info("File uploaded successfully", 
                       filename=file.filename,
                       temp_path=str(temp_path),
                       file_size=total_size)
            
            return temp_path
            
        except Exception as e:
            logger.error("File upload failed", 
                        filename=getattr(file, 'filename', 'unknown'),
                        error=str(e))
            if isinstance(e, FileUploadError):
                raise
            raise FileUploadError(f"File upload failed: {str(e)}") from e
    
    async def _validate_file(self, file: UploadFile) -> None:
        """Validate uploaded file.
        
        Args:
            file: FastAPI UploadFile object
            
        Raises:
            FileUploadError: If validation fails
        """
        # Check filename
        if not file.filename:
            raise FileUploadError("No filename provided")
        
        # Sanitize and validate filename
        sanitized_filename = self._sanitize_filename(file.filename)
        if not sanitized_filename:
            raise FileUploadError("Invalid filename")
        
        # Check file extension
        file_ext = Path(sanitized_filename).suffix.lower()
        if file_ext not in self.config.allowed_extensions:
            raise FileUploadError(
                f"File extension '{file_ext}' not allowed. "
                f"Allowed extensions: {', '.join(sorted(self.config.allowed_extensions))}"
            )
        
        # Check MIME type if provided
        if file.content_type:
            if file.content_type not in self.config.allowed_mime_types:
                # Try to guess MIME type from filename
                guessed_type, _ = mimetypes.guess_type(sanitized_filename)
                if not guessed_type or guessed_type not in self.config.allowed_mime_types:
                    raise FileUploadError(
                        f"MIME type '{file.content_type}' not allowed"
                    )
        
        # Check file size (initial check)
        if hasattr(file, 'size') and file.size:
            if file.size > self.config.max_file_size:
                raise FileUploadError(
                    f"File size {file.size} exceeds maximum allowed size of "
                    f"{self.config.max_file_size} bytes"
                )
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to prevent path traversal and other issues.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        if not filename:
            return ""
        
        # Remove path components
        filename = Path(filename).name
        
        # Remove or replace dangerous characters
        dangerous_chars = '<>:"/\\|?*'
        for char in dangerous_chars:
            filename = filename.replace(char, '_')
        
        # Remove leading/trailing dots and spaces
        filename = filename.strip('. ')
        
        # Ensure filename is not empty and not too long
        if not filename:
            return ""
        
        if len(filename) > 255:
            # Keep extension but truncate name
            stem = Path(filename).stem[:200]
            suffix = Path(filename).suffix
            filename = f"{stem}{suffix}"
        
        return filename
    
    def _generate_unique_filename(self, original_filename: str) -> str:
        """Generate unique filename to prevent conflicts.
        
        Args:
            original_filename: Original filename
            
        Returns:
            Unique filename with UUID prefix
        """
        sanitized = self._sanitize_filename(original_filename)
        if not sanitized:
            sanitized = "uploaded_file"
        
        unique_id = str(uuid.uuid4())[:8]
        return f"{unique_id}_{sanitized}"
    
    def cleanup_old_files(self, max_age_hours: int = 24, max_disk_usage_mb: int = 10000) -> int:
        """Clean up old temporary files based on age and disk usage.

        Args:
            max_age_hours: Maximum age in hours before files are eligible for cleanup
            max_disk_usage_mb: Maximum disk usage in MB before aggressive cleanup

        Returns:
            Number of files cleaned up
        """
        import shutil

        if not self.temp_dir.exists():
            return 0

        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        deleted_count = 0

        try:
            # Get all files in temp directory with their ages
            files_with_ages = []
            total_size_mb = 0

            for file_path in self.temp_dir.glob("*"):
                if file_path.is_file() and not file_path.name.startswith('.'):
                    try:
                        file_age = current_time - file_path.stat().st_mtime
                        file_size = file_path.stat().st_size
                        files_with_ages.append((file_path, file_age, file_size))
                        total_size_mb += file_size / (1024 * 1024)
                    except Exception as e:
                        logger.warning("Failed to get file stats", file_path=str(file_path), error=str(e))

            # Sort by age (oldest first)
            files_with_ages.sort(key=lambda x: x[1], reverse=True)

            logger.debug("Cleanup scan results",
                        temp_dir=str(self.temp_dir),
                        total_files=len(files_with_ages),
                        total_size_mb=round(total_size_mb, 2),
                        max_age_hours=max_age_hours,
                        max_disk_usage_mb=max_disk_usage_mb)

            # Clean up files based on age and disk usage
            for file_path, file_age, file_size in files_with_ages:
                should_delete = False
                reason = ""

                # Always delete files older than max_age
                if file_age > max_age_seconds:
                    should_delete = True
                    reason = f"age {file_age/3600:.1f}h > {max_age_hours}h"

                # Delete oldest files if disk usage is too high
                elif total_size_mb > max_disk_usage_mb:
                    should_delete = True
                    reason = f"disk usage {total_size_mb:.1f}MB > {max_disk_usage_mb}MB"
                    total_size_mb -= file_size / (1024 * 1024)

                if should_delete:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                        logger.debug("Cleaned up temporary file",
                                   file_path=str(file_path),
                                   reason=reason,
                                   age_hours=round(file_age/3600, 1))
                    except Exception as e:
                        logger.warning("Failed to delete temporary file",
                                     file_path=str(file_path),
                                     error=str(e))

            if deleted_count > 0:
                logger.info("Temporary file cleanup completed",
                           temp_dir=str(self.temp_dir),
                           files_deleted=deleted_count,
                           remaining_files=len(files_with_ages) - deleted_count)

        except Exception as e:
            logger.error("Error during temporary file cleanup",
                        temp_dir=str(self.temp_dir),
                        error=str(e))

        return deleted_count
    
    def cleanup_temp_dir(self) -> None:
        """Clean up temporary directory and all files."""
        try:
            if self.temp_dir.exists():
                import shutil
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                logger.info("Cleaned up temporary directory", temp_dir=str(self.temp_dir))
        except Exception as e:
            logger.warning("Failed to clean up temporary directory", 
                         temp_dir=str(self.temp_dir), error=str(e))
    
    def __del__(self):
        """Cleanup on object destruction."""
        # NOTE: Removed aggressive temp directory cleanup to prevent race conditions
        # with background tasks. Individual files are cleaned up by scheduled tasks
        # or by the background tasks themselves after processing.

        # Log when handler is being garbage collected for debugging
        try:
            logger.warning("FileUploadHandler being garbage collected",
                         temp_dir=str(self.temp_dir) if hasattr(self, 'temp_dir') else 'unknown',
                         temp_dir_exists=self.temp_dir.exists() if hasattr(self, 'temp_dir') else False)
        except Exception:
            # Ignore any errors during destruction
            pass


# Global file upload handler instance
_upload_handler: Optional[FileUploadHandler] = None


def get_upload_handler() -> FileUploadHandler:
    """Get global file upload handler instance."""
    global _upload_handler
    if _upload_handler is None:
        logger.info("Creating new global FileUploadHandler instance")

        # Get configuration from MoRAG settings
        try:
            from morag_core.config import get_settings
            settings = get_settings()
            max_upload_size = settings.get_max_upload_size_bytes()

            config = FileUploadConfig(max_file_size=max_upload_size)
            _upload_handler = FileUploadHandler(config)

            logger.info("FileUploadHandler configured from settings",
                       max_upload_size_mb=max_upload_size / (1024 * 1024),
                       max_upload_size_bytes=max_upload_size)
        except Exception as e:
            logger.warning("Failed to load MoRAG settings, using default config",
                         error=str(e))
            _upload_handler = FileUploadHandler()
    else:
        # Check if the handler's temp directory still exists
        if not _upload_handler.temp_dir.exists():
            logger.warning("Global FileUploadHandler temp directory missing, creating new handler",
                         old_temp_dir=str(_upload_handler.temp_dir))

            # Recreate with same configuration approach
            try:
                from morag_core.config import get_settings
                settings = get_settings()
                max_upload_size = settings.get_max_upload_size_bytes()

                config = FileUploadConfig(max_file_size=max_upload_size)
                _upload_handler = FileUploadHandler(config)
            except Exception as e:
                logger.warning("Failed to load MoRAG settings for new handler, using default config",
                             error=str(e))
                _upload_handler = FileUploadHandler()
    return _upload_handler


def configure_upload_handler(config: FileUploadConfig) -> None:
    """Configure global file upload handler."""
    global _upload_handler

    old_temp_dir = None
    if _upload_handler is not None:
        old_temp_dir = str(_upload_handler.temp_dir)

    logger.info("Configuring new FileUploadHandler",
               old_temp_dir=old_temp_dir,
               new_config=config.__dict__)

    # NOTE: Don't cleanup existing temp directory to avoid race conditions
    # with background tasks that might still be processing files
    _upload_handler = FileUploadHandler(config)


def validate_temp_directory_access() -> bool:
    """Validate that the temporary directory is accessible and writable.

    This should be called during application startup to fail early if
    the temp directory is not properly configured.

    Returns:
        True if temp directory is accessible and writable

    Raises:
        RuntimeError: If temp directory is not accessible or writable
    """
    try:
        # Get the upload handler (this will create temp directory)
        upload_handler = get_upload_handler()
        temp_dir = upload_handler.temp_dir

        # Test directory exists
        if not temp_dir.exists():
            raise RuntimeError(f"Temp directory does not exist: {temp_dir}")

        # Test directory is writable
        test_file = temp_dir / f".startup_test_{uuid.uuid4().hex[:8]}"
        try:
            test_file.write_text("startup test")
            test_file.unlink()
        except Exception as e:
            raise RuntimeError(f"Temp directory is not writable: {temp_dir} - {str(e)}")

        # Check if we're using system temp (which is problematic in containers)
        if str(temp_dir).startswith('/tmp/'):
            logger.warning("STARTUP WARNING: Using system temp directory - this may cause issues in container environments",
                          temp_dir=str(temp_dir))

        logger.info("Temp directory validation successful",
                   temp_dir=str(temp_dir),
                   is_shared_volume=str(temp_dir).startswith('/app/temp'))
        return True

    except Exception as e:
        logger.error("STARTUP FAILURE: Temp directory validation failed", error=str(e))
        raise RuntimeError(f"Temp directory validation failed: {str(e)}")
