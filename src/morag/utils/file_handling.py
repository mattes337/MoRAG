from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import tempfile
import shutil
import mimetypes
import hashlib
import structlog
from fastapi import UploadFile

from morag.core.config import settings
from morag.core.exceptions import ValidationError

logger = structlog.get_logger()

class FileHandler:
    """Handles file upload and validation."""
    
    def __init__(self):
        self.upload_dir = Path(settings.upload_dir)
        self.temp_dir = Path(settings.temp_dir)
        
        # Create directories if they don't exist
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Supported MIME types
        self.supported_mimes = {
            # Documents
            'application/pdf': 'pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
            'application/msword': 'doc',
            'text/markdown': 'md',
            'text/plain': 'txt',
            
            # Audio
            'audio/mpeg': 'mp3',
            'audio/wav': 'wav',
            'audio/x-wav': 'wav',
            'audio/mp4': 'm4a',
            'audio/x-m4a': 'm4a',
            'audio/ogg': 'ogg',
            'audio/flac': 'flac',
            
            # Video
            'video/mp4': 'mp4',
            'video/quicktime': 'mov',
            'video/x-msvideo': 'avi',
            'video/webm': 'webm',
            'video/x-matroska': 'mkv',
            
            # Images
            'image/jpeg': 'jpg',
            'image/png': 'png',
            'image/gif': 'gif',
            'image/bmp': 'bmp',
            'image/tiff': 'tiff',
            'image/webp': 'webp',
        }
    
    def validate_file(self, file: UploadFile, source_type: str) -> Tuple[str, Dict[str, Any]]:
        """Validate uploaded file and return file info."""
        
        # Check file size
        if hasattr(file, 'size') and file.size:
            max_size = self._get_max_size_for_type(source_type)
            if file.size > max_size:
                raise ValidationError(f"File too large: {file.size} bytes (max: {max_size})")
        
        # Detect MIME type
        mime_type = file.content_type
        if not mime_type:
            mime_type, _ = mimetypes.guess_type(file.filename)
        
        if mime_type not in self.supported_mimes:
            raise ValidationError(f"Unsupported file type: {mime_type}")
        
        # Validate source type matches file type
        file_extension = self.supported_mimes[mime_type]
        if not self._is_valid_for_source_type(file_extension, source_type):
            raise ValidationError(f"File type {file_extension} not valid for source type {source_type}")
        
        # Generate file info
        file_info = {
            'original_filename': file.filename,
            'mime_type': mime_type,
            'file_extension': file_extension,
            'size': getattr(file, 'size', 0)
        }
        
        return file_extension, file_info
    
    async def save_uploaded_file(
        self,
        file: UploadFile,
        source_type: str
    ) -> Tuple[Path, Dict[str, Any]]:
        """Save uploaded file and return path and metadata."""
        
        # Validate file
        file_extension, file_info = self.validate_file(file, source_type)
        
        # Generate unique filename
        file_hash = hashlib.md5(f"{file.filename}{file.size}".encode()).hexdigest()[:8]
        safe_filename = f"{file_hash}_{file.filename}"
        file_path = self.upload_dir / safe_filename
        
        try:
            # Save file
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Update file info with actual size
            file_info['size'] = len(content)
            file_info['file_path'] = str(file_path)
            file_info['file_hash'] = file_hash
            
            logger.info(
                "File uploaded successfully",
                filename=safe_filename,
                size=file_info['size'],
                mime_type=file_info['mime_type']
            )
            
            return file_path, file_info
            
        except Exception as e:
            # Clean up on error
            if file_path.exists():
                file_path.unlink()
            logger.error("Failed to save uploaded file", error=str(e))
            raise ValidationError(f"Failed to save file: {str(e)}")
    
    def _get_max_size_for_type(self, source_type: str) -> int:
        """Get maximum file size for source type."""
        from ..core.config import settings

        max_sizes = {
            'document': settings.max_document_size,
            'audio': settings.max_audio_size,
            'video': settings.max_video_size,
            'image': settings.max_image_size,
        }
        return max_sizes.get(source_type, settings.max_document_size)
    
    def _is_valid_for_source_type(self, file_extension: str, source_type: str) -> bool:
        """Check if file extension is valid for source type."""
        valid_extensions = {
            'document': ['pdf', 'docx', 'doc', 'md', 'txt'],
            'audio': ['mp3', 'wav', 'm4a', 'ogg', 'flac'],
            'video': ['mp4', 'mov', 'avi', 'webm', 'mkv'],
            'image': ['jpg', 'png', 'gif', 'bmp', 'tiff', 'webp'],
        }
        return file_extension in valid_extensions.get(source_type, [])
    
    def cleanup_file(self, file_path: Path) -> None:
        """Clean up uploaded file."""
        try:
            if file_path.exists():
                file_path.unlink()
                logger.info("File cleaned up", file_path=str(file_path))
        except Exception as e:
            logger.warning("Failed to cleanup file", file_path=str(file_path), error=str(e))

# Global instance
file_handler = FileHandler()
