"""File handling utilities for MoRAG."""

import os
import shutil
import hashlib
import tempfile
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import mimetypes
import uuid
import structlog

from ..exceptions import ValidationError, StorageError

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
    if directory:
        temp_dir = Path(directory)
    else:
        # Import settings here to avoid module-level import issues
        from ..config import settings
        temp_dir = Path(settings.temp_dir)

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


def get_file_hash(file_path: Union[str, Path], algorithm: str = "sha256") -> str:
    """Get file hash.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm (md5, sha1, sha256, etc.)

    Returns:
        File hash as hex string

    Raises:
        ValidationError: If file does not exist
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise ValidationError(f"File not found: {file_path}")

    hash_obj = hashlib.new(algorithm)

    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    except Exception as e:
        raise StorageError(f"Failed to calculate hash for {file_path}: {str(e)}")


def get_file_size(file_path: Union[str, Path]) -> int:
    """Get file size in bytes.

    Args:
        file_path: Path to file

    Returns:
        File size in bytes

    Raises:
        ValidationError: If file does not exist
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise ValidationError(f"File not found: {file_path}")

    return file_path.stat().st_size


def is_file_readable(file_path: Union[str, Path]) -> bool:
    """Check if file is readable.

    Args:
        file_path: Path to file

    Returns:
        True if file is readable, False otherwise
    """
    file_path = Path(file_path)

    try:
        return file_path.exists() and file_path.is_file() and os.access(file_path, os.R_OK)
    except Exception:
        return False


def parse_size_string(size_str: str) -> int:
    """Parse size string like '100MB', '5GB' to bytes.

    Args:
        size_str: Size string (e.g., '100MB', '5GB', '1024')

    Returns:
        Size in bytes

    Raises:
        ValueError: If size string format is invalid
    """
    if not size_str:
        raise ValueError("Size string cannot be empty")

    size_str = size_str.strip().upper()

    # If it's just a number, return as-is (assume bytes)
    if size_str.isdigit():
        return int(size_str)

    # Define size multipliers
    multipliers = {
        'B': 1,
        'KB': 1024,
        'MB': 1024 * 1024,
        'GB': 1024 * 1024 * 1024,
        'TB': 1024 * 1024 * 1024 * 1024,
    }

    # Extract number and unit
    import re
    match = re.match(r'^(\d+(?:\.\d+)?)\s*([KMGT]?B)$', size_str)
    if not match:
        raise ValueError(f"Invalid size format: {size_str}. Expected format like '100MB', '5GB', etc.")

    number_str, unit = match.groups()
    number = float(number_str)

    if unit not in multipliers:
        raise ValueError(f"Unknown size unit: {unit}. Supported units: {list(multipliers.keys())}")

    return int(number * multipliers[unit])


def cleanup_temp_files(pattern: str = "morag_*", max_age_hours: int = 24) -> int:
    """Clean up temporary files.

    Args:
        pattern: File pattern to match
        max_age_hours: Maximum age in hours before deletion

    Returns:
        Number of files deleted
    """
    import time

    temp_dir = Path(tempfile.gettempdir())
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    deleted_count = 0

    try:
        for file_path in temp_dir.glob(pattern):
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    if safe_delete(file_path):
                        deleted_count += 1
                        logger.debug(f"Deleted temp file: {file_path}")
    except Exception as e:
        logger.warning(f"Error during temp file cleanup: {str(e)}")

    return deleted_count
