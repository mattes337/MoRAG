"""File handling utilities for MoRAG."""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import mimetypes
import uuid
import structlog

from ..exceptions import ValidationError, StorageError
from ..config import settings

logger = structlog.get_logger(__name__)


def ensure_directory(directory: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't.
    
    Args:
        directory: Directory path
        
    Returns:
        Path object for directory
        
    Raises:
        StorageError: If directory cannot be created
    """
    directory = Path(directory)
    try:
        os.makedirs(directory, exist_ok=True)
        return directory
    except Exception as e:
        raise StorageError(f"Failed to create directory {directory}: {str(e)}")


def get_file_info(file_path: Union[str, Path]) -> Dict[str, Union[str, int]]:
    """Get file information.
    
    Args:
        file_path: Path to file
        
    Returns:
        Dictionary with file information
        
    Raises:
        ValidationError: If file does not exist
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise ValidationError(f"File not found: {file_path}")
    
    # Get file stats
    stats = file_path.stat()
    
    # Get MIME type
    mime_type, _ = mimetypes.guess_type(file_path)
    
    return {
        "file_name": file_path.name,
        "file_path": str(file_path),
        "file_size": stats.st_size,
        "mime_type": mime_type or "application/octet-stream",
        "extension": file_path.suffix.lower().lstrip('.'),
        "created_at": stats.st_ctime,
        "modified_at": stats.st_mtime,
    }


def generate_temp_path(prefix: str = "", suffix: str = "", directory: Optional[Union[str, Path]] = None) -> Path:
    """Generate temporary file path.
    
    Args:
        prefix: Prefix for filename
        suffix: Suffix for filename (e.g., file extension)
        directory: Directory for temporary file (uses settings.temp_dir if None)
        
    Returns:
        Path object for temporary file
    """
    # Use specified directory or default temp directory
    temp_dir = Path(directory) if directory else Path(settings.temp_dir)
    ensure_directory(temp_dir)
    
    # Generate unique filename
    filename = f"{prefix}{uuid.uuid4().hex}{suffix}"
    return temp_dir / filename


def safe_delete(file_path: Union[str, Path]) -> bool:
    """Safely delete file or directory.
    
    Args:
        file_path: Path to file or directory
        
    Returns:
        True if deletion was successful, False otherwise
    """
    file_path = Path(file_path)
    
    try:
        if file_path.is_file():
            file_path.unlink()
        elif file_path.is_dir():
            shutil.rmtree(file_path)
        return True
    except Exception as e:
        logger.warning(f"Failed to delete {file_path}: {str(e)}")
        return False


def detect_format(file_path: Union[str, Path]) -> str:
    """Detect format from file extension.
    
    Args:
        file_path: Path to file
        
    Returns:
        Format type string
    """
    file_path = Path(file_path)
    extension = file_path.suffix.lower().lstrip('.')
    
    # Map common extensions to format types
    format_map = {
        # Documents
        'pdf': 'pdf',
        'txt': 'text',
        'md': 'markdown',
        'html': 'html',
        'htm': 'html',
        'xml': 'xml',
        'json': 'json',
        'csv': 'csv',
        
        # Office
        'doc': 'word',
        'docx': 'word',
        'xls': 'excel',
        'xlsx': 'excel',
        'ppt': 'powerpoint',
        'pptx': 'powerpoint',
        
        # Audio
        'mp3': 'audio',
        'wav': 'audio',
        'ogg': 'audio',
        'flac': 'audio',
        'm4a': 'audio',
        
        # Video
        'mp4': 'video',
        'avi': 'video',
        'mov': 'video',
        'mkv': 'video',
        'webm': 'video',
        
        # Images
        'jpg': 'image',
        'jpeg': 'image',
        'png': 'image',
        'gif': 'image',
        'bmp': 'image',
        'webp': 'image',
    }
    
    return format_map.get(extension, 'unknown')