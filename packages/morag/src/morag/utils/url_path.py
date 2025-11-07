"""
URL path utilities to handle URLs without corrupting them through pathlib.
"""

from pathlib import Path
from typing import Union
import os
import structlog

logger = structlog.get_logger(__name__)


class URLPath:
    """
    A path-like object that preserves URL format without corruption.

    This class prevents URLs from being mangled when passed through pathlib.Path(),
    which normalizes paths and can corrupt URLs by removing slashes.
    """

    def __init__(self, url_str: str):
        """Initialize with URL string."""
        self.url_str = url_str

    def __str__(self) -> str:
        """Return the original URL string."""
        return self.url_str

    def __fspath__(self) -> str:
        """Return the URL string for os.fspath() compatibility."""
        return self.url_str

    def __repr__(self) -> str:
        """Return a representation of the URLPath."""
        return f"URLPath('{self.url_str}')"

    @property
    def name(self) -> str:
        """Return a filename-like name from the URL."""
        # Extract the last part of the URL path, or use a default
        parts = self.url_str.rstrip('/').split('/')
        if len(parts) > 2:  # More than just protocol://domain
            name = parts[-1]
            if name and not name.startswith('?'):
                return name

        # Fallback to domain name or generic name
        if 'youtube.com' in self.url_str or 'youtu.be' in self.url_str:
            return 'youtube_video'
        elif any(domain in self.url_str for domain in ['.com', '.org', '.net', '.edu']):
            return 'web_content'
        else:
            return 'url_content'

    def exists(self) -> bool:
        """URLs are assumed to exist for processing purposes."""
        return True

    @property
    def suffix(self) -> str:
        """Return empty suffix for URLs."""
        return ''

    @property
    def stem(self) -> str:
        """Return the name without suffix."""
        return self.name

    def stat(self):
        """Raise AttributeError for stat() calls on URLs."""
        raise AttributeError(
            f"'URLPath' object has no attribute 'stat' - URLs don't have file system statistics. "
            f"URL: {self.url_str}"
        )

    def is_file(self) -> bool:
        """URLs are not files."""
        return False

    def is_dir(self) -> bool:
        """URLs are not directories."""
        return False

    def read_text(self, *args, **kwargs):
        """URLs cannot be read as text files."""
        raise AttributeError(
            f"'URLPath' object cannot be read as a text file. "
            f"Use appropriate URL processing instead. URL: {self.url_str}"
        )

    def read_bytes(self, *args, **kwargs):
        """URLs cannot be read as binary files."""
        raise AttributeError(
            f"'URLPath' object cannot be read as a binary file. "
            f"Use appropriate URL processing instead. URL: {self.url_str}"
        )


def create_path_from_string(path_str: str) -> Union[Path, URLPath]:
    """
    Create appropriate path object from string.

    Args:
        path_str: File path or URL string

    Returns:
        Path object for local files, URLPath for URLs
    """
    logger.debug("Creating path from string", path_str=path_str)
    if path_str.startswith(('http://', 'https://')):
        logger.info("Creating URLPath for URL", url=path_str)
        return URLPath(path_str)
    else:
        logger.debug("Creating Path for local file", path=path_str)
        return Path(path_str)


def is_url(path_like) -> bool:
    """
    Check if a path-like object represents a URL.

    Args:
        path_like: Path, URLPath, or string to check

    Returns:
        True if it's a URL, False otherwise
    """
    if isinstance(path_like, URLPath):
        return True

    path_str = str(path_like)
    return path_str.startswith(('http://', 'https://'))


def get_url_string(path_like) -> str:
    """
    Get the URL string from a path-like object.

    Args:
        path_like: URLPath or string representing a URL

    Returns:
        The URL string

    Raises:
        ValueError: If the path_like object is not a URL
    """
    if isinstance(path_like, URLPath):
        return path_like.url_str

    path_str = str(path_like)
    if path_str.startswith(('http://', 'https://')):
        return path_str

    raise ValueError(f"Not a URL: {path_like}")
