"""Main orchestrator for MoRAG system."""

import asyncio
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import structlog

from morag_core.models import Document, DocumentChunk, ProcessingResult
from morag_services import MoRAGServices, ServiceConfig, ContentType
from morag_web import WebProcessor, WebConverter
from morag_youtube import YouTubeProcessor

logger = structlog.get_logger(__name__)


class MoRAGOrchestrator:
    """Main orchestrator that coordinates all MoRAG components."""

    def __init__(self, config: Optional[ServiceConfig] = None):
        """Initialize the orchestrator.

        Args:
            config: Service configuration
        """
        self.config = config or ServiceConfig()
        self.services = MoRAGServices(self.config)

        # Initialize specialized processors
        self.web_processor = WebProcessor()
        self.web_converter = WebConverter()
        self.youtube_processor = YouTubeProcessor()

        logger.info("MoRAG Orchestrator initialized")

    async def process_content(
        self,
        content: Union[str, Path, Dict[str, Any]],
        content_type: ContentType,
        options: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """Process content of any supported type.

        Args:
            content: Content to process (URL, file path, or data)
            content_type: Type of content
            options: Processing options

        Returns:
            Processing result with extracted content and metadata
        """
        logger.info("Starting content processing",
                   content_type=content_type.value,
                   options=options)

        try:
            # Route to appropriate processor based on content type
            if content_type == ContentType.WEB:
                return await self._process_web_content(content, options)
            elif content_type == ContentType.YOUTUBE:
                return await self._process_youtube_content(content, options)
            elif content_type == ContentType.DOCUMENT:
                return await self._process_document_content(content, options)
            elif content_type == ContentType.AUDIO:
                return await self._process_audio_content(content, options)
            elif content_type == ContentType.VIDEO:
                return await self._process_video_content(content, options)
            elif content_type == ContentType.IMAGE:
                return await self._process_image_content(content, options)
            else:
                raise ValueError(f"Unsupported content type: {content_type}")

        except Exception as e:
            logger.error("Content processing failed",
                        content_type=content_type.value,
                        error=str(e))
            raise

    async def _process_web_content(
        self,
        content: Union[str, Dict[str, Any]],
        options: Optional[Dict[str, Any]]
    ) -> ProcessingResult:
        """Process web content using services."""
        return await self.services.process_url(content, options)

    async def _process_youtube_content(
        self,
        content: Union[str, Dict[str, Any]],
        options: Optional[Dict[str, Any]]
    ) -> ProcessingResult:
        """Process YouTube content using services."""
        return await self.services.process_youtube(content, options)

    async def _process_document_content(
        self,
        content: Union[str, Path, Dict[str, Any]],
        options: Optional[Dict[str, Any]]
    ) -> ProcessingResult:
        """Process document content using services."""
        return await self.services.process_content(content, ContentType.DOCUMENT, options)

    async def _process_audio_content(
        self,
        content: Union[str, Path, Dict[str, Any]],
        options: Optional[Dict[str, Any]]
    ) -> ProcessingResult:
        """Process audio content using services."""
        return await self.services.process_content(content, ContentType.AUDIO, options)

    async def _process_video_content(
        self,
        content: Union[str, Path, Dict[str, Any]],
        options: Optional[Dict[str, Any]]
    ) -> ProcessingResult:
        """Process video content using services."""
        return await self.services.process_content(content, ContentType.VIDEO, options)

    async def _process_image_content(
        self,
        content: Union[str, Path, Dict[str, Any]],
        options: Optional[Dict[str, Any]]
    ) -> ProcessingResult:
        """Process image content using services."""
        return await self.services.process_content(content, ContentType.IMAGE, options)

    async def process_batch(
        self,
        items: List[Dict[str, Any]],
        options: Optional[Dict[str, Any]] = None
    ) -> List[ProcessingResult]:
        """Process multiple items in batch.

        Args:
            items: List of items to process, each with 'content' and 'content_type'
            options: Global processing options

        Returns:
            List of processing results
        """
        logger.info("Starting batch processing", item_count=len(items))

        results = []
        for i, item in enumerate(items):
            try:
                content = item['content']
                content_type = ContentType(item['content_type'])
                item_options = {**(options or {}), **(item.get('options', {}))}

                result = await self.process_content(content, content_type, item_options)
                results.append(result)

                logger.debug("Processed batch item",
                           item_index=i,
                           content_type=content_type.value,
                           success=result.success)

            except Exception as e:
                logger.error("Failed to process batch item",
                           item_index=i,
                           error=str(e))
                results.append(ProcessingResult(
                    content="",
                    metadata={"error": str(e)},
                    processing_time=0.0,
                    success=False,
                    error_message=str(e)
                ))

        logger.info("Batch processing completed",
                   total_items=len(items),
                   successful_items=len([r for r in results if r.success]))

        return results

    async def search_similar(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar content using vector similarity.

        Args:
            query: Search query
            limit: Maximum number of results
            filters: Optional filters

        Returns:
            List of similar content items
        """
        return await self.services.search_similar(query, limit, filters)

    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all components.

        Returns:
            Health status information
        """
        return await self.services.get_health_status()

    async def cleanup(self) -> None:
        """Clean up resources."""
        self.services.cleanup()
        logger.info("MoRAG Orchestrator cleaned up")
