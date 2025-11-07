"""Content processing implementations for different content types."""

import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import structlog
from morag_core.exceptions import ProcessingError, UnsupportedFormatError

from .service_coordinator import ContentType, MoRAGServiceCoordinator, ProcessingResult

logger = structlog.get_logger(__name__)


class ContentProcessors:
    """Content processing implementations for MoRAG services."""

    def __init__(self, coordinator: MoRAGServiceCoordinator):
        """Initialize with service coordinator."""
        self.coordinator = coordinator

    async def process_content(
        self,
        path_or_url: str,
        content_type: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> ProcessingResult:
        """Process content based on detected or specified type.

        Args:
            path_or_url: Path to file or URL
            content_type: Optional content type override
            options: Processing options

        Returns:
            Processing result
        """
        start_time = time.time()

        try:
            # Detect content type if not provided
            if content_type is None:
                content_type = self.coordinator.detect_content_type(path_or_url)

            logger.info("Processing content", path=path_or_url, type=content_type)

            # Route to appropriate processor
            if content_type == ContentType.DOCUMENT:
                result = await self.process_document(path_or_url, options)
            elif content_type == ContentType.AUDIO:
                result = await self.process_audio(path_or_url, options)
            elif content_type == ContentType.VIDEO:
                result = await self.process_video(path_or_url, options)
            elif content_type == ContentType.IMAGE:
                result = await self.process_image(path_or_url, options)
            elif content_type == ContentType.WEB:
                result = await self.process_url(path_or_url, options)
            elif content_type == ContentType.YOUTUBE:
                result = await self.process_youtube(path_or_url, options)
            else:
                raise UnsupportedFormatError(
                    f"Unsupported content type: {content_type}"
                )

            # Add processing time
            result.processing_time = time.time() - start_time

            logger.info(
                "Content processing completed",
                path=path_or_url,
                type=content_type,
                success=result.success,
                processing_time=result.processing_time,
            )

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                "Content processing failed",
                path=path_or_url,
                error=str(e),
                processing_time=processing_time,
            )

            return ProcessingResult(
                content_type=content_type or ContentType.UNKNOWN,
                success=False,
                error=str(e),
                processing_time=processing_time,
            )

    async def process_document(
        self, document_path: str, options: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """Process document content."""
        if not self.coordinator.document_service:
            raise ProcessingError("Document service not initialized")

        try:
            result = await self.coordinator.document_service.process(
                document_path, options or {}
            )

            # Process with graph processor if available
            graph_result = None
            if self.coordinator.graph_processor and result.success:
                try:
                    graph_result = await self.coordinator.graph_processor.process_text(
                        result.content,
                        source_path=document_path,
                        content_type="document",
                        metadata=result.metadata,
                    )
                except Exception as e:
                    logger.warning(
                        "Graph processing failed for document",
                        path=document_path,
                        error=str(e),
                    )

            return ProcessingResult(
                content_type=ContentType.DOCUMENT,
                success=result.success,
                text_content=result.content,
                metadata=result.metadata,
                error=result.error if not result.success else None,
                graph_data=graph_result,
            )

        except Exception as e:
            logger.error("Document processing failed", path=document_path, error=str(e))
            return ProcessingResult(
                content_type=ContentType.DOCUMENT, success=False, error=str(e)
            )

    async def process_audio(
        self, audio_path: str, options: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """Process audio content."""
        if not self.coordinator.audio_service:
            raise ProcessingError("Audio service not initialized")

        try:
            result = await self.coordinator.audio_service.process(
                audio_path, options or {}
            )

            # Process with graph processor if available
            graph_result = None
            if self.coordinator.graph_processor and result.success:
                try:
                    graph_result = await self.coordinator.graph_processor.process_text(
                        result.content,
                        source_path=audio_path,
                        content_type="audio",
                        metadata=result.metadata,
                    )
                except Exception as e:
                    logger.warning(
                        "Graph processing failed for audio",
                        path=audio_path,
                        error=str(e),
                    )

            return ProcessingResult(
                content_type=ContentType.AUDIO,
                success=result.success,
                text_content=result.content,
                metadata=result.metadata,
                error=result.error if not result.success else None,
                graph_data=graph_result,
            )

        except Exception as e:
            logger.error("Audio processing failed", path=audio_path, error=str(e))
            return ProcessingResult(
                content_type=ContentType.AUDIO, success=False, error=str(e)
            )

    async def process_video(
        self, video_path: str, options: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """Process video content."""
        if not self.coordinator.video_service:
            raise ProcessingError("Video service not initialized")

        try:
            result = await self.coordinator.video_service.process(
                video_path, options or {}
            )

            # Process with graph processor if available
            graph_result = None
            if self.coordinator.graph_processor and result.success:
                try:
                    graph_result = await self.coordinator.graph_processor.process_text(
                        result.content,
                        source_path=video_path,
                        content_type="video",
                        metadata=result.metadata,
                    )
                except Exception as e:
                    logger.warning(
                        "Graph processing failed for video",
                        path=video_path,
                        error=str(e),
                    )

            return ProcessingResult(
                content_type=ContentType.VIDEO,
                success=result.success,
                text_content=result.content,
                metadata=result.metadata,
                error=result.error if not result.success else None,
                graph_data=graph_result,
            )

        except Exception as e:
            logger.error("Video processing failed", path=video_path, error=str(e))
            return ProcessingResult(
                content_type=ContentType.VIDEO, success=False, error=str(e)
            )

    async def process_image(
        self, image_path: str, options: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """Process image content."""
        if not self.coordinator.image_service:
            raise ProcessingError("Image service not initialized")

        try:
            result = await self.coordinator.image_service.process(
                image_path, options or {}
            )

            # Process with graph processor if available
            graph_result = None
            if self.coordinator.graph_processor and result.success:
                try:
                    graph_result = await self.coordinator.graph_processor.process_text(
                        result.content,
                        source_path=image_path,
                        content_type="image",
                        metadata=result.metadata,
                    )
                except Exception as e:
                    logger.warning(
                        "Graph processing failed for image",
                        path=image_path,
                        error=str(e),
                    )

            return ProcessingResult(
                content_type=ContentType.IMAGE,
                success=result.success,
                text_content=result.content,
                metadata=result.metadata,
                error=result.error if not result.success else None,
                graph_data=graph_result,
            )

        except Exception as e:
            logger.error("Image processing failed", path=image_path, error=str(e))
            return ProcessingResult(
                content_type=ContentType.IMAGE, success=False, error=str(e)
            )

    async def process_url(
        self, url: str, options: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """Process web URL content."""
        if not self.coordinator.web_service:
            raise ProcessingError("Web service not initialized")

        try:
            result = await self.coordinator.web_service.process(url, options or {})

            # Process with graph processor if available
            graph_result = None
            if self.coordinator.graph_processor and result.success:
                try:
                    graph_result = await self.coordinator.graph_processor.process_text(
                        result.content,
                        source_path=url,
                        content_type="web",
                        metadata=result.metadata,
                    )
                except Exception as e:
                    logger.warning(
                        "Graph processing failed for web content", url=url, error=str(e)
                    )

            return ProcessingResult(
                content_type=ContentType.WEB,
                success=result.success,
                text_content=result.content,
                metadata=result.metadata,
                error=result.error if not result.success else None,
                graph_data=graph_result,
            )

        except Exception as e:
            logger.error("Web processing failed", url=url, error=str(e))
            return ProcessingResult(
                content_type=ContentType.WEB, success=False, error=str(e)
            )

    async def process_youtube(
        self, url: str, options: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """Process YouTube content."""
        if not self.coordinator.youtube_service:
            raise ProcessingError("YouTube service not initialized")

        try:
            result = await self.coordinator.youtube_service.process(url, options or {})

            # Process with graph processor if available
            graph_result = None
            if self.coordinator.graph_processor and result.success:
                try:
                    graph_result = await self.coordinator.graph_processor.process_text(
                        result.content,
                        source_path=url,
                        content_type="youtube",
                        metadata=result.metadata,
                    )
                except Exception as e:
                    logger.warning(
                        "Graph processing failed for YouTube content",
                        url=url,
                        error=str(e),
                    )

            return ProcessingResult(
                content_type=ContentType.YOUTUBE,
                success=result.success,
                text_content=result.content,
                metadata=result.metadata,
                error=result.error if not result.success else None,
                graph_data=graph_result,
            )

        except Exception as e:
            logger.error("YouTube processing failed", url=url, error=str(e))
            return ProcessingResult(
                content_type=ContentType.YOUTUBE, success=False, error=str(e)
            )

    async def process_batch(self, items: List[str]) -> Dict[str, ProcessingResult]:
        """Process multiple items concurrently."""

        async def process_with_semaphore(item: str) -> Tuple[str, ProcessingResult]:
            async with self.coordinator._semaphore:
                result = await self.process_content(item)
                return item, result

        # Process all items concurrently with semaphore control
        tasks = [process_with_semaphore(item) for item in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert results to dictionary
        result_dict = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error("Batch processing item failed", error=str(result))
                continue

            item, processing_result = result
            result_dict[item] = processing_result

        logger.info(
            "Batch processing completed",
            total_items=len(items),
            successful_items=len(result_dict),
        )

        return result_dict


__all__ = ["ContentProcessors"]
