"""Main service coordinator for MoRAG system."""

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from pathlib import Path
from enum import Enum
import os
import structlog
from pydantic import BaseModel

from morag_core.interfaces.service import BaseService
from morag_core.models import ProcessingConfig
from morag_core.exceptions import ProcessingError, UnsupportedFormatError

# Import specialized services
from morag_document.service import DocumentService
from morag_audio.service import AudioService
from morag_video.service import VideoService
from morag_image.service import ImageService
from morag_embedding import GeminiEmbeddingService
from morag_web.service import WebService
from morag_youtube.service import YouTubeService

# Import graph processing
from .graph_processor import GraphProcessor, GraphProcessingConfig, GraphProcessingResult

logger = structlog.get_logger(__name__)

@dataclass
class ServiceConfig:
    """Configuration for MoRAG services."""
    document_config: Optional[ProcessingConfig] = None
    audio_config: Optional[ProcessingConfig] = None
    video_config: Optional[ProcessingConfig] = None
    image_config: Optional[ProcessingConfig] = None
    embedding_config: Optional[ProcessingConfig] = None
    web_config: Optional[ProcessingConfig] = None
    youtube_config: Optional[ProcessingConfig] = None
    max_concurrent_tasks: int = 5

class ContentType(str, Enum):
    """Content type enum."""
    DOCUMENT = "document"
    AUDIO = "audio"
    VIDEO = "video"
    IMAGE = "image"
    WEB = "web"
    YOUTUBE = "youtube"
    UNKNOWN = "unknown"

class ProcessingResult(BaseModel):
    """Processing result from MoRAG services."""
    content_type: str
    success: bool
    text_content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None
    graph_data: Optional[GraphProcessingResult] = None


class MoRAGServiceCoordinator:
    """Main coordinator for all MoRAG processing services."""

    def __init__(self, config: Optional[ServiceConfig] = None, graph_config: Optional[GraphProcessingConfig] = None, data_output_dir: Optional[str] = None):
        """Initialize MoRAG services coordinator.
        
        Args:
            config: Service configuration
            graph_config: Graph processing configuration
            data_output_dir: Output directory for data files
        """
        self.config = config or ServiceConfig()
        self.graph_config = graph_config
        self.data_output_dir = data_output_dir
        
        # Initialize services
        self.document_service: Optional[DocumentService] = None
        self.audio_service: Optional[AudioService] = None
        self.video_service: Optional[VideoService] = None
        self.image_service: Optional[ImageService] = None
        self.embedding_service: Optional[GeminiEmbeddingService] = None
        self.web_service: Optional[WebService] = None
        self.youtube_service: Optional[YouTubeService] = None
        
        # Graph processing
        self.graph_processor: Optional[GraphProcessor] = None
        
        # Search services
        self.search_services: Set[BaseService] = set()
        
        # Concurrency control
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)
        
        logger.info("MoRAG services coordinator initialized", max_concurrent_tasks=self.config.max_concurrent_tasks)

    async def initialize(self):
        """Initialize all services."""
        try:
            logger.info("Initializing MoRAG services...")
            
            # Initialize document service
            if self.config.document_config:
                self.document_service = DocumentService(config=self.config.document_config)
                await self.document_service.initialize()
                logger.debug("Document service initialized")
            
            # Initialize audio service
            if self.config.audio_config:
                self.audio_service = AudioService(config=self.config.audio_config)
                await self.audio_service.initialize()
                logger.debug("Audio service initialized")
            
            # Initialize video service
            if self.config.video_config:
                self.video_service = VideoService(config=self.config.video_config)
                await self.video_service.initialize()
                logger.debug("Video service initialized")
            
            # Initialize image service
            if self.config.image_config:
                self.image_service = ImageService(config=self.config.image_config)
                await self.image_service.initialize()
                logger.debug("Image service initialized")
            
            # Initialize embedding service
            self.embedding_service = GeminiEmbeddingService()
            await self.embedding_service.initialize()
            logger.debug("Embedding service initialized")
            
            # Initialize web service
            if self.config.web_config:
                self.web_service = WebService(config=self.config.web_config)
                await self.web_service.initialize()
                logger.debug("Web service initialized")
            
            # Initialize YouTube service
            if self.config.youtube_config:
                self.youtube_service = YouTubeService(config=self.config.youtube_config)
                await self.youtube_service.initialize()
                logger.debug("YouTube service initialized")
            
            # Initialize graph processor if config provided
            if self.graph_config:
                self.graph_processor = GraphProcessor(
                    config=self.graph_config,
                    data_output_dir=self.data_output_dir
                )
                await self.graph_processor.initialize()
                logger.debug("Graph processor initialized")
            
            # Initialize search services
            self._initialize_search_services()
            
            logger.info("All MoRAG services initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize MoRAG services", error=str(e))
            raise ProcessingError(f"Service initialization failed: {str(e)}")

    def _initialize_search_services(self):
        """Initialize search services for content discovery."""
        # Add services that support search functionality
        search_capable_services = [
            self.document_service,
            self.audio_service,
            self.video_service,
            self.image_service,
            self.web_service,
            self.youtube_service
        ]
        
        for service in search_capable_services:
            if service and hasattr(service, 'search'):
                self.search_services.add(service)
        
        logger.debug(f"Initialized {len(self.search_services)} search services")

    async def shutdown(self):
        """Shutdown all services gracefully."""
        logger.info("Shutting down MoRAG services...")
        
        services_to_shutdown = [
            self.document_service,
            self.audio_service,
            self.video_service,
            self.image_service,
            self.embedding_service,
            self.web_service,
            self.youtube_service,
            self.graph_processor
        ]
        
        for service in services_to_shutdown:
            if service and hasattr(service, 'shutdown'):
                try:
                    await service.shutdown()
                except Exception as e:
                    logger.warning("Error shutting down service", service=type(service).__name__, error=str(e))
        
        logger.info("MoRAG services shutdown complete")

    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all services."""
        health_status = {
            "coordinator": "healthy",
            "services": {},
            "graph_processor": None,
            "timestamp": structlog.processors.TimeStamper()(None, None, {"event": "health_check"})["timestamp"]
        }
        
        # Check individual services
        service_map = {
            "document": self.document_service,
            "audio": self.audio_service,
            "video": self.video_service,
            "image": self.image_service,
            "embedding": self.embedding_service,
            "web": self.web_service,
            "youtube": self.youtube_service
        }
        
        for name, service in service_map.items():
            if service:
                try:
                    if hasattr(service, 'get_health_status'):
                        health_status["services"][name] = await service.get_health_status()
                    else:
                        health_status["services"][name] = "healthy"
                except Exception as e:
                    health_status["services"][name] = f"error: {str(e)}"
            else:
                health_status["services"][name] = "not_initialized"
        
        # Check graph processor
        if self.graph_processor:
            try:
                health_status["graph_processor"] = await self.graph_processor.get_health_status()
            except Exception as e:
                health_status["graph_processor"] = f"error: {str(e)}"
        
        return health_status

    async def generate_embeddings(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate embeddings using the embedding service.
        
        Args:
            text: Text or list of texts to embed
            
        Returns:
            Embedding(s) as float list(s)
        """
        if not self.embedding_service:
            raise ProcessingError("Embedding service not initialized")
        
        try:
            if isinstance(text, str):
                return await self.embedding_service.generate_embedding(text)
            else:
                return await self.embedding_service.generate_batch(text)
        except Exception as e:
            logger.error("Failed to generate embeddings", error=str(e))
            raise ProcessingError(f"Embedding generation failed: {str(e)}")


__all__ = ["MoRAGServiceCoordinator", "ServiceConfig", "ContentType", "ProcessingResult"]