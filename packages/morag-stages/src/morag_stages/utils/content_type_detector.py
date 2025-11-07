"""Content type detection utility."""

import re
from pathlib import Path
from typing import Union
import structlog

logger = structlog.get_logger(__name__)

# Import services - these are optional for content type detection
try:
    from morag_services import ContentType
    SERVICES_AVAILABLE = True
except ImportError:
    SERVICES_AVAILABLE = False
    class ContentType:  # type: ignore
        pass


class ContentTypeDetector:
    """Utility class for detecting content types from files and URLs."""

    def __init__(self):
        """Initialize content type detector."""
        # Define file extension mappings
        self.video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
        self.audio_extensions = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma'}
        self.document_extensions = {'.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx'}
        self.text_extensions = {'.txt', '.md', '.rst', '.html', '.xml', '.json', '.csv'}

        # YouTube URL patterns
        self.youtube_patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
            r'youtube\.com/v/([a-zA-Z0-9_-]{11})',
            r'youtube\.com/watch\?.*v=([a-zA-Z0-9_-]{11})'
        ]

    def detect_content_type(self, file_path: Union[str, Path]) -> Union[str, object]:
        """Detect content type from file path or URL.

        Args:
            file_path: File path or URL to analyze

        Returns:
            Content type (ContentType enum if services available, string otherwise)
        """
        file_str = str(file_path)
        logger.debug("Detecting content type", file_path=file_str)

        # Check if it's a URL
        if file_str.startswith(('http://', 'https://')):
            return self._detect_url_content_type(file_str)

        # Check file extension
        return self._detect_file_content_type(Path(file_path))

    def _detect_url_content_type(self, url: str) -> Union[str, object]:
        """Detect content type for URLs.

        Args:
            url: URL to analyze

        Returns:
            Content type for the URL
        """
        logger.debug("Analyzing URL", url=url)

        # Check for YouTube URLs
        is_youtube = any(
            re.search(pattern, url) for pattern in self.youtube_patterns
        ) or 'youtube.com' in url or 'youtu.be' in url

        if is_youtube:
            logger.debug("Detected YouTube URL", url=url)
            if SERVICES_AVAILABLE:
                return ContentType.YOUTUBE
            else:
                return "YOUTUBE"

        # Default to web content for other URLs
        logger.debug("Detected web URL", url=url)
        if SERVICES_AVAILABLE:
            return ContentType.WEB
        else:
            return "WEB"

    def _detect_file_content_type(self, file_path: Path) -> Union[str, object]:
        """Detect content type for files.

        Args:
            file_path: File path to analyze

        Returns:
            Content type for the file
        """
        file_suffix = file_path.suffix.lower()
        file_str = str(file_path)

        logger.debug("Checking file extension", file_path=file_str, suffix=file_suffix)

        # Video files
        if file_suffix in self.video_extensions:
            logger.debug("Detected video file", file_path=file_str, suffix=file_suffix)
            if SERVICES_AVAILABLE:
                return ContentType.VIDEO
            else:
                return "VIDEO"

        # Audio files
        if file_suffix in self.audio_extensions:
            logger.debug("Detected audio file", file_path=file_str, suffix=file_suffix)
            if SERVICES_AVAILABLE:
                return ContentType.AUDIO
            else:
                return "AUDIO"

        # Document files
        if file_suffix in self.document_extensions:
            logger.debug("Detected document file", file_path=file_str, suffix=file_suffix)
            if SERVICES_AVAILABLE:
                return ContentType.DOCUMENT
            else:
                return "DOCUMENT"

        # Text files
        if file_suffix in self.text_extensions:
            logger.debug("Detected text file", file_path=file_str, suffix=file_suffix)
            if SERVICES_AVAILABLE:
                return ContentType.TEXT
            else:
                return "TEXT"

        # Default to text for unknown types
        logger.debug("Using default text type for unknown extension",
                    file_path=file_str, suffix=file_suffix)
        if SERVICES_AVAILABLE:
            return ContentType.TEXT
        else:
            return "TEXT"

    def is_content_type(self, content_type: Union[str, object], expected_type: str) -> bool:
        """Check if content type matches expected type.

        Args:
            content_type: Content type (ContentType enum or string)
            expected_type: Expected type as string (e.g., "VIDEO", "AUDIO")

        Returns:
            True if content type matches expected type
        """
        if SERVICES_AVAILABLE and hasattr(ContentType, expected_type):
            return content_type == getattr(ContentType, expected_type)
        else:
            return str(content_type).upper() == expected_type.upper()


# Global detector instance
_detector = ContentTypeDetector()


def detect_content_type(file_path: Union[str, Path]) -> Union[str, object]:
    """Detect content type from file path or URL.

    Args:
        file_path: File path or URL to analyze

    Returns:
        Content type (ContentType enum if services available, string otherwise)
    """
    return _detector.detect_content_type(file_path)


def is_content_type(content_type: Union[str, object], expected_type: str) -> bool:
    """Check if content type matches expected type.

    Args:
        content_type: Content type (ContentType enum or string)
        expected_type: Expected type as string (e.g., "VIDEO", "AUDIO")

    Returns:
        True if content type matches expected type
    """
    return _detector.is_content_type(content_type, expected_type)
