"""Main API interface for MoRAG system."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import structlog
from morag.orchestrator import MoRAGOrchestrator
from morag_core.models import ProcessingResult
from morag_services import ContentType, ServiceConfig

logger = structlog.get_logger(__name__)


class MoRAGAPI:
    """Main API interface for the MoRAG system."""

    def __init__(self, config: Optional[ServiceConfig] = None):
        """Initialize the MoRAG API.

        Args:
            config: Service configuration
        """
        self.orchestrator = MoRAGOrchestrator(config)
        logger.info("MoRAG API initialized")

    async def process_url(
        self,
        url: str,
        content_type: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> ProcessingResult:
        """Process content from a URL.

        Args:
            url: URL to process
            content_type: Type of content (auto-detected if not provided)
            options: Processing options

        Returns:
            Processing result
        """
        # Auto-detect content type if not provided
        if content_type is None:
            content_type = self._detect_content_type(url)

        # Validate and convert to ContentType enum
        try:
            content_type_enum = ContentType(content_type)
        except ValueError:
            # If content type is not valid, try to map it or default to unknown
            content_type = self._normalize_content_type(content_type)
            content_type_enum = ContentType(content_type)

        return await self.orchestrator.process_content(url, content_type_enum, options)

    async def process_file(
        self,
        file_path: Union[str, Path],
        content_type: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> ProcessingResult:
        """Process content from a file.

        Args:
            file_path: Path to file
            content_type: Type of content (auto-detected if not provided)
            options: Processing options

        Returns:
            Processing result
        """
        file_path = Path(file_path)

        # Auto-detect content type if not provided
        if content_type is None:
            content_type = self._detect_content_type_from_file(file_path)

        # Validate and convert to ContentType enum
        try:
            content_type_enum = ContentType(content_type)
        except ValueError:
            # If content type is not valid, try to map it or default to unknown
            content_type = self._normalize_content_type(content_type)
            content_type_enum = ContentType(content_type)

        return await self.orchestrator.process_content(
            file_path, content_type_enum, options
        )

    async def process_web_page(
        self, url: str, options: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """Process a web page.

        Args:
            url: Web page URL
            options: Processing options

        Returns:
            Processing result
        """
        return await self.orchestrator.process_content(url, ContentType.WEB, options)

    async def process_youtube_video(
        self, url: str, options: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """Process a YouTube video.

        Args:
            url: YouTube video URL
            options: Processing options

        Returns:
            Processing result
        """
        return await self.orchestrator.process_content(
            url, ContentType.YOUTUBE, options
        )

    async def process_document(
        self, file_path: Union[str, Path], options: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """Process a document file.

        Args:
            file_path: Path to document
            options: Processing options

        Returns:
            Processing result
        """
        return await self.orchestrator.process_content(
            file_path, ContentType.DOCUMENT, options
        )

    async def process_audio(
        self, file_path: Union[str, Path], options: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """Process an audio file.

        Args:
            file_path: Path to audio file
            options: Processing options

        Returns:
            Processing result
        """
        return await self.orchestrator.process_content(
            file_path, ContentType.AUDIO, options
        )

    async def process_video(
        self, file_path: Union[str, Path], options: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """Process a video file.

        Args:
            file_path: Path to video file
            options: Processing options

        Returns:
            Processing result
        """
        return await self.orchestrator.process_content(
            file_path, ContentType.VIDEO, options
        )

    async def process_image(
        self, file_path: Union[str, Path], options: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """Process an image file.

        Args:
            file_path: Path to image file
            options: Processing options

        Returns:
            Processing result
        """
        return await self.orchestrator.process_content(
            file_path, ContentType.IMAGE, options
        )

    async def process_batch(
        self, items: List[Dict[str, Any]], options: Optional[Dict[str, Any]] = None
    ) -> List[ProcessingResult]:
        """Process multiple items in batch.

        Args:
            items: List of items to process
            options: Global processing options

        Returns:
            List of processing results
        """
        return await self.orchestrator.process_batch(items, options)

    async def search(
        self, query: str, limit: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar content.

        Args:
            query: Search query
            limit: Maximum number of results
            filters: Optional filters

        Returns:
            List of similar content items
        """
        return await self.orchestrator.search_similar(query, limit, filters)

    async def health_check(self) -> Dict[str, Any]:
        """Get health status of all components.

        Returns:
            Health status information
        """
        return await self.orchestrator.get_health_status()

    def _detect_content_type(self, url: str) -> str:
        """Auto-detect content type from URL.

        Args:
            url: URL to analyze

        Returns:
            Detected content type
        """
        url_lower = url.lower()

        # YouTube detection
        if "youtube.com" in url_lower or "youtu.be" in url_lower:
            return "youtube"

        # Default to web for URLs
        if url_lower.startswith(("http://", "https://")):
            return "web"

        # If it's not a URL, treat as file path
        return self._detect_content_type_from_file(Path(url))

    def _detect_content_type_from_file(self, file_path: Path) -> str:
        """Auto-detect content type from file extension.

        Args:
            file_path: File path to analyze

        Returns:
            Detected content type
        """
        suffix = file_path.suffix.lower()

        # Document types
        if suffix in [
            ".pdf",
            ".doc",
            ".docx",
            ".txt",
            ".md",
            ".rtf",
            ".pptx",
            ".ppt",
            ".xlsx",
            ".xls",
            ".csv",
            ".json",
            ".xml",
        ]:
            return "document"

        # Audio types
        if suffix in [".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"]:
            return "audio"

        # Video types
        if suffix in [".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm"]:
            return "video"

        # Image types
        if suffix in [
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".webp",
            ".tiff",
            ".svg",
        ]:
            return "image"

        # Web types
        if suffix in [".html", ".htm"]:
            return "web"

        # Default to document
        return "document"

    def _normalize_content_type(self, content_type: str) -> str:
        """Normalize content type to valid ContentType enum value.

        Args:
            content_type: Raw content type string

        Returns:
            Normalized content type string that matches ContentType enum
        """
        content_type_lower = content_type.lower()

        # Map common file extensions or formats to content types
        format_mapping = {
            "pdf": "document",
            "doc": "document",
            "docx": "document",
            "txt": "document",
            "md": "document",
            "rtf": "document",
            "pptx": "document",
            "ppt": "document",
            "xlsx": "document",
            "xls": "document",
            "csv": "document",
            "json": "document",
            "xml": "document",
            "html": "web",
            "htm": "web",
            "mp3": "audio",
            "wav": "audio",
            "flac": "audio",
            "m4a": "audio",
            "ogg": "audio",
            "aac": "audio",
            "mp4": "video",
            "avi": "video",
            "mkv": "video",
            "mov": "video",
            "wmv": "video",
            "flv": "video",
            "webm": "video",
            "jpg": "image",
            "jpeg": "image",
            "png": "image",
            "gif": "image",
            "bmp": "image",
            "webp": "image",
            "tiff": "image",
            "svg": "image",
        }

        # Try to map the content type
        if content_type_lower in format_mapping:
            return format_mapping[content_type_lower]

        # If it's already a valid content type, return as is
        valid_types = [
            "document",
            "audio",
            "video",
            "image",
            "web",
            "youtube",
            "text",
            "unknown",
        ]
        if content_type_lower in valid_types:
            return content_type_lower

        # Default to unknown for unrecognized types
        return "unknown"

    async def cleanup(self) -> None:
        """Clean up resources."""
        await self.orchestrator.cleanup()
        logger.info("MoRAG API cleaned up")
