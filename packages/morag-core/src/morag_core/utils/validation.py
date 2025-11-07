"""Validation utilities for MoRAG."""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
from urllib.parse import urlparse

from ..exceptions import ValidationError
from ..config import settings


def validate_file_size(file_path: Union[str, Path], max_size: Optional[int] = None) -> bool:
    """Validate file size.

    Args:
        file_path: Path to file
        max_size: Maximum file size in bytes (uses settings.max_file_size if None)

    Returns:
        True if file size is valid

    Raises:
        ValidationError: If file size exceeds maximum
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise ValidationError(f"File not found: {file_path}")

    file_size = file_path.stat().st_size
    max_allowed = max_size or settings.max_file_size

    if file_size > max_allowed:
        raise ValidationError(
            f"File size ({file_size} bytes) exceeds maximum allowed size ({max_allowed} bytes)"
        )

    return True


def validate_file_type(file_path: Union[str, Path], allowed_extensions: Set[str]) -> bool:
    """Validate file type by extension.

    Args:
        file_path: Path to file
        allowed_extensions: Set of allowed file extensions (without dot)

    Returns:
        True if file type is valid

    Raises:
        ValidationError: If file type is not allowed
    """
    file_path = Path(file_path)
    extension = file_path.suffix.lower().lstrip('.')

    if not extension or extension not in allowed_extensions:
        raise ValidationError(
            f"File type '{extension}' is not allowed. Allowed types: {', '.join(allowed_extensions)}"
        )

    return True


def validate_url(url: str) -> bool:
    """Validate URL format.

    Args:
        url: URL to validate

    Returns:
        True if URL is valid

    Raises:
        ValidationError: If URL is invalid
    """
    try:
        result = urlparse(url)
        # Check if scheme and netloc are present
        if not all([result.scheme, result.netloc]):
            raise ValidationError(f"Invalid URL format: {url}")

        # Check if scheme is http or https
        if result.scheme not in ["http", "https"]:
            raise ValidationError(f"URL must use http or https scheme: {url}")

        return True
    except Exception as e:
        raise ValidationError(f"Invalid URL: {url}. Error: {str(e)}")


def validate_webhook_url(url: str) -> bool:
    """Validate webhook URL format.

    Args:
        url: Webhook URL to validate

    Returns:
        True if webhook URL is valid

    Raises:
        ValidationError: If webhook URL is invalid
    """
    # First validate as a regular URL
    validate_url(url)

    # Additional webhook-specific validation could be added here
    # For example, checking for specific domains or patterns

    return True


def validate_api_key(api_key: str, min_length: int = 8) -> bool:
    """Validate API key format.

    Args:
        api_key: API key to validate
        min_length: Minimum length for API key

    Returns:
        True if API key is valid

    Raises:
        ValidationError: If API key is invalid
    """
    if not api_key or len(api_key) < min_length:
        raise ValidationError(f"API key must be at least {min_length} characters long")

    # Check if API key contains only alphanumeric characters and common separators
    if not re.match(r'^[a-zA-Z0-9_\-\.]+$', api_key):
        raise ValidationError("API key contains invalid characters")

    return True


def validate_file_path(file_path: Union[str, Path]) -> bool:
    """Validate file path.

    Args:
        file_path: Path to validate

    Returns:
        True if file path is valid

    Raises:
        ValidationError: If file path is invalid
    """
    file_path = Path(file_path)

    # Check if path exists
    if not file_path.exists():
        raise ValidationError(f"File not found: {file_path}")

    # Check if it's a file (not a directory)
    if not file_path.is_file():
        raise ValidationError(f"Path is not a file: {file_path}")

    # Check if file is readable
    if not os.access(file_path, os.R_OK):
        raise ValidationError(f"File is not readable: {file_path}")

    return True


def validate_email(email: str) -> bool:
    """Validate email format.

    Args:
        email: Email to validate

    Returns:
        True if email is valid

    Raises:
        ValidationError: If email is invalid
    """
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

    if not re.match(email_pattern, email):
        raise ValidationError(f"Invalid email format: {email}")

    return True


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """Sanitize filename for safe filesystem usage.

    Args:
        filename: Original filename
        max_length: Maximum length for filename

    Returns:
        Sanitized filename
    """
    # Remove or replace invalid characters
    invalid_chars = r'[<>:"/\\|?*\x00-\x1f]'
    sanitized = re.sub(invalid_chars, '_', filename)

    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')

    # Ensure filename is not empty
    if not sanitized:
        sanitized = "unnamed_file"

    # Truncate if too long
    if len(sanitized) > max_length:
        name, ext = os.path.splitext(sanitized)
        max_name_length = max_length - len(ext)
        sanitized = name[:max_name_length] + ext

    return sanitized


def sanitize_filepath(filepath: Union[str, Path], base_dir: Optional[Union[str, Path]] = None) -> Path:
    """Sanitize file path to prevent traversal and injection attacks.

    Args:
        filepath: Path to sanitize
        base_dir: Base directory to validate against (defaults to current working directory)

    Returns:
        Sanitized and resolved Path object

    Raises:
        ValidationError: If path traversal or unsafe characters detected
    """
    if not filepath:
        raise ValidationError("Empty file path provided")

    path = Path(filepath)

    # Convert to string to check for dangerous patterns
    path_str = str(path)

    # Check for null bytes (can cause issues in some systems)
    if '\x00' in path_str:
        raise ValidationError(f"Null byte detected in path: {filepath}")

    # Check for dangerous command injection patterns
    dangerous_patterns = [
        r'[;&|`$()]',  # Shell metacharacters
        r'\$\(',       # Command substitution
        r'`.*`',       # Backtick command substitution
        r'>\s*/',      # Output redirection to system paths
        r'<\s*/',      # Input redirection from system paths
        r'\|\s*\w+',   # Pipe to commands
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, path_str):
            raise ValidationError(f"Dangerous characters or patterns detected in path: {filepath}")

    # Check for path traversal patterns
    traversal_patterns = [
        r'\.\./',      # Directory traversal
        r'\.\.\\'      # Windows directory traversal
    ]

    for pattern in traversal_patterns:
        if re.search(pattern, path_str):
            raise ValidationError(f"Path traversal pattern detected: {filepath}")

    # Resolve the path and validate it stays within base directory
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError) as e:
        raise ValidationError(f"Failed to resolve path {filepath}: {str(e)}")

    # Set base directory for validation
    if base_dir is None:
        base_dir = Path.cwd().resolve()
    else:
        base_dir = Path(base_dir).resolve()

    # Ensure the resolved path is within the base directory
    try:
        resolved.relative_to(base_dir)
    except ValueError:
        raise ValidationError(f"Path traversal detected - path outside base directory: {filepath}")

    # Additional filename validation
    filename = resolved.name
    if filename:
        # Check for reserved Windows filenames
        reserved_names = {
            'CON', 'PRN', 'AUX', 'NUL',
            'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
            'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
        }

        # Check base filename without extension
        base_filename = filename.split('.')[0].upper()
        if base_filename in reserved_names:
            raise ValidationError(f"Reserved filename detected: {filename}")

        # Check for filenames that are only dots or whitespace
        if filename.strip('. \t\n\r') == '':
            raise ValidationError(f"Invalid filename (only dots/whitespace): {filename}")

        # Check for filenames that start with multiple dots (potential traversal or hidden files)
        if filename.startswith('..'):
            raise ValidationError(f"Filename cannot start with double dots: {filename}")

    return resolved
