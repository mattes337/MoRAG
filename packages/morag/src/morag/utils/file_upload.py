"""File upload handling utilities for MoRAG system."""

import tempfile
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
        self.temp_dir = Path(tempfile.mkdtemp(prefix=self.config.temp_dir_prefix))
        self._cleanup_threads: List = []  # Track cleanup threads instead of asyncio tasks

        logger.info("FileUploadHandler initialized",
                   temp_dir=str(self.temp_dir),
                   max_file_size=self.config.max_file_size)
    
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
            
            # Schedule cleanup
            self._schedule_cleanup(temp_path)
            
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
    
    def _schedule_cleanup(self, file_path: Path) -> None:
        """Schedule cleanup of temporary file.

        Args:
            file_path: Path to file to clean up
        """
        import threading
        import time

        def cleanup_task():
            """Background thread cleanup task to avoid asyncio event loop issues."""
            logger.debug("Scheduled cleanup task started",
                        file_path=str(file_path),
                        timeout_seconds=self.config.cleanup_timeout)

            # Sleep in background thread to avoid event loop cancellation
            time.sleep(self.config.cleanup_timeout)

            try:
                if file_path.exists():
                    file_path.unlink()
                    logger.info("Cleaned up temporary file after timeout",
                               file_path=str(file_path),
                               timeout_seconds=self.config.cleanup_timeout)
                else:
                    logger.debug("Temporary file already cleaned up",
                                file_path=str(file_path))
            except Exception as e:
                logger.warning("Failed to clean up temporary file",
                             file_path=str(file_path), error=str(e))

        # Use daemon thread to avoid blocking shutdown and prevent event loop cancellation
        thread = threading.Thread(target=cleanup_task, daemon=True)
        thread.start()

        # Keep reference to prevent garbage collection
        self._cleanup_threads.append(thread)
    
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
        pass


# Global file upload handler instance
_upload_handler: Optional[FileUploadHandler] = None


def get_upload_handler() -> FileUploadHandler:
    """Get global file upload handler instance."""
    global _upload_handler
    if _upload_handler is None:
        _upload_handler = FileUploadHandler()
    return _upload_handler


def configure_upload_handler(config: FileUploadConfig) -> None:
    """Configure global file upload handler."""
    global _upload_handler
    # NOTE: Don't cleanup existing temp directory to avoid race conditions
    # with background tasks that might still be processing files
    _upload_handler = FileUploadHandler(config)
