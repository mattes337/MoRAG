"""Main services integration module for MoRAG system.

This module implements the Facade pattern for backward compatibility.
The actual functionality is delegated to single-purpose components:
- MoRAGServiceCoordinator: Service coordination and orchestration
- ContentProcessors: Content processing implementations
- ServiceUtilities: Content type detection utilities

This facade is deprecated - use MoRAGServiceCoordinator directly.
"""

import warnings
from typing import Any, Dict, List, Optional, Union

from morag_core.utils.logging import get_logger

from .content_processors import ContentProcessors

# Import single-purpose components
from .service_coordinator import (
    ContentType,
    MoRAGServiceCoordinator,
    ProcessingResult,
    ServiceConfig,
)
from .service_utilities import ServiceUtilities

logger = get_logger(__name__)

# Issue deprecation warning for the facade
warnings.warn(
    "MoRAGServices is deprecated. Use MoRAGServiceCoordinator directly for new code.",
    DeprecationWarning,
    stacklevel=2,
)


class ContentTypeDetector:
    """Single-purpose content type detector."""

    def __init__(self):
        self._utilities = ServiceUtilities()

    def detect(self, path_or_url: str) -> str:
        """Detect content type based on path or URL."""
        return self._utilities.detect_content_type(path_or_url)


class MoRAGServices:
    """Facade for backward compatibility only.

    This is a pure facade that delegates to single-purpose components.
    No business logic should be implemented here - only delegation.
    """

    def __init__(self, config=None, graph_config=None, data_output_dir=None):
        """Initialize MoRAG services facade."""
        logger.warning(
            "MoRAGServices is deprecated. Use MoRAGServiceCoordinator directly."
        )

        # Delegate to single-purpose components - no inheritance
        self._coordinator = MoRAGServiceCoordinator(
            config, graph_config, data_output_dir
        )
        self._processors = ContentProcessors(self._coordinator)
        self._detector = ContentTypeDetector()
        self._utilities = ServiceUtilities()

    # Pure delegation - no logic here

    def _is_document(self, path_or_url: str) -> bool:
        """Check if path/URL is a document."""
        return self._utilities.is_document(path_or_url)

    def _is_audio(self, path_or_url: str) -> bool:
        """Check if path/URL is audio."""
        return self._utilities.is_audio(path_or_url)

    def _is_video(self, path_or_url: str) -> bool:
        """Check if path/URL is video."""
        return self._utilities.is_video(path_or_url)

    def _is_image(self, path_or_url: str) -> bool:
        """Check if path/URL is an image."""
        return self._utilities.is_image(path_or_url)

    def _is_web(self, path_or_url: str) -> bool:
        """Check if this is a web URL."""
        return self._utilities.is_web(path_or_url)

    def _is_youtube(self, path_or_url: str) -> bool:
        """Check if this is a YouTube URL."""
        return self._utilities.is_youtube(path_or_url)

    def detect_content_type(self, path_or_url: str) -> str:
        """Detect content type based on path or URL."""
        return self._detector.detect(path_or_url)

    # Pure delegation - no logic here

    async def process_content(
        self,
        path_or_url: str,
        content_type: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> ProcessingResult:
        """Process content based on detected or specified type."""
        return await self._processors.process_content(
            path_or_url, content_type, options
        )

    async def process_document(
        self, document_path: str, options: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """Process document content."""
        return await self._processors.process_document(document_path, options)

    async def process_audio(
        self, audio_path: str, options: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """Process audio content."""
        return await self._processors.process_audio(audio_path, options)

    async def process_video(
        self, video_path: str, options: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """Process video content."""
        return await self._processors.process_video(video_path, options)

    async def process_image(
        self, image_path: str, options: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """Process image content."""
        return await self._processors.process_image(image_path, options)

    async def process_url(
        self, url: str, options: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """Process web URL content."""
        return await self._processors.process_url(url, options)

    async def process_youtube(
        self, url: str, options: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """Process YouTube content."""
        return await self._processors.process_youtube(url, options)

    async def process_batch(self, items: List[str]) -> Dict[str, ProcessingResult]:
        """Process multiple items concurrently."""
        return await self._processors.process_batch(items)

    # Delegate coordinator methods for backward compatibility
    def get_service(self, service_type: str):
        """Get a specific service instance."""
        return self._coordinator.get_service(service_type)

    @property
    def config(self):
        """Get coordinator configuration."""
        return self._coordinator.config


# Backward compatibility exports
__all__ = [
    "MoRAGServices",  # Deprecated facade - use MoRAGServiceCoordinator instead
    "MoRAGServiceCoordinator",  # Recommended for new code
    "ContentProcessors",  # Single-purpose content processing
    "ContentTypeDetector",  # Single-purpose content type detection
    "ServiceUtilities",  # Single-purpose utilities
    "ServiceConfig",
    "ContentType",
    "ProcessingResult",
]
