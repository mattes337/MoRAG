"""Main services integration module for MoRAG system.

REFACTORED: This module has been split for better maintainability.
The main functionality is now distributed across:
- service_coordinator.py: Main service coordination
- content_processors.py: Content processing implementations  
- service_utilities.py: Utility functions

This file provides backward compatibility.
"""

# Re-export the split components for backward compatibility
from .service_coordinator import MoRAGServiceCoordinator, ServiceConfig, ContentType, ProcessingResult
from .content_processors import ContentProcessors
from .service_utilities import ServiceUtilities

import structlog
from typing import Dict, List, Any, Optional, Union

logger = structlog.get_logger(__name__)


class MoRAGServices(MoRAGServiceCoordinator):
    """Backward compatible main MoRAG services class.
    
    This class extends MoRAGServiceCoordinator with the original MoRAGServices interface
    for backward compatibility while using the refactored components internally.
    """

    def __init__(self, config=None, graph_config=None, data_output_dir=None):
        """Initialize MoRAG services with backward compatible interface."""
        super().__init__(config, graph_config, data_output_dir)
        
        # Initialize processors
        self.content_processors = ContentProcessors(self)
        self.utilities = ServiceUtilities()

    # Content type detection methods (delegated to utilities)
    def _is_document(self, path_or_url: str) -> bool:
        """Check if path/URL is a document."""
        return self.utilities.is_document(path_or_url)
    
    def _is_audio(self, path_or_url: str) -> bool:
        """Check if path/URL is audio."""
        return self.utilities.is_audio(path_or_url)
    
    def _is_video(self, path_or_url: str) -> bool:
        """Check if path/URL is video."""
        return self.utilities.is_video(path_or_url)
    
    def _is_image(self, path_or_url: str) -> bool:
        """Check if path/URL is an image."""
        return self.utilities.is_image(path_or_url)
    
    def _is_web(self, path_or_url: str) -> bool:
        """Check if this is a web URL."""
        return self.utilities.is_web(path_or_url)
    
    def _is_youtube(self, path_or_url: str) -> bool:
        """Check if this is a YouTube URL."""
        return self.utilities.is_youtube(path_or_url)

    def detect_content_type(self, path_or_url: str) -> str:
        """Detect content type based on path or URL."""
        return self.utilities.detect_content_type(path_or_url)

    # Content processing methods (delegated to content processors)
    async def process_content(self, path_or_url: str, content_type: Optional[str] = None, options: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """Process content based on detected or specified type."""
        return await self.content_processors.process_content(path_or_url, content_type, options)

    async def process_document(self, document_path: str, options: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """Process document content."""
        return await self.content_processors.process_document(document_path, options)

    async def process_audio(self, audio_path: str, options: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """Process audio content."""
        return await self.content_processors.process_audio(audio_path, options)

    async def process_video(self, video_path: str, options: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """Process video content."""
        return await self.content_processors.process_video(video_path, options)

    async def process_image(self, image_path: str, options: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """Process image content."""
        return await self.content_processors.process_image(image_path, options)

    async def process_url(self, url: str, options: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """Process web URL content."""
        return await self.content_processors.process_url(url, options)

    async def process_youtube(self, url: str, options: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """Process YouTube content."""
        return await self.content_processors.process_youtube(url, options)

    async def process_batch(self, items: List[str]) -> Dict[str, ProcessingResult]:
        """Process multiple items concurrently."""
        return await self.content_processors.process_batch(items)


# Backward compatibility exports
__all__ = [
    "MoRAGServices", 
    "MoRAGServiceCoordinator", 
    "ContentProcessors", 
    "ServiceUtilities",
    "ServiceConfig", 
    "ContentType", 
    "ProcessingResult"
]