"""Main services integration module for MoRAG system.

This module provides the MoRAGServices class that integrates all specialized
processing packages into a cohesive API.
"""

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
from morag_embedding import EmbeddingService
from morag_web.service import WebService
from morag_youtube.service import YouTubeService

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
    TEXT = "text"
    UNKNOWN = "unknown"

class ProcessingResult(BaseModel):
    """Unified processing result for all content types."""
    content_type: str
    content_path: Optional[str] = None
    content_url: Optional[str] = None
    text_content: Optional[str] = None
    metadata: Dict[str, Any] = {}
    extracted_files: List[str] = []
    processing_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    raw_result: Optional[Any] = None

class MoRAGServices:
    """Unified service layer for MoRAG system.
    
    This class integrates all specialized processing services into a cohesive API,
    making it easy to work with multiple content types through a single interface.
    """
    
    def __init__(self, config: Optional[ServiceConfig] = None):
        """Initialize MoRAG services.
        
        Args:
            config: Configuration for services
        """
        self.config = config or ServiceConfig()
        
        # Initialize specialized services
        self.document_service = DocumentService()
        self.audio_service = AudioService()
        self.video_service = VideoService()
        self.image_service = ImageService()
        self.embedding_service = EmbeddingService()
        self.web_service = WebService()
        self.youtube_service = YouTubeService()

        # Initialize AI services for search functionality
        self._vector_storage = None
        self._gemini_embedding_service = None
        self._initialize_search_services()
        
        # Create semaphore for concurrent processing
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)

        logger.info("MoRAG Services initialized")

    def _initialize_search_services(self):
        """Initialize search-related services (embedding and vector storage)."""
        try:
            import os
            from .storage import QdrantVectorStorage
            from .embedding import GeminiEmbeddingService

            # Initialize vector storage
            qdrant_host = os.getenv('QDRANT_HOST', 'localhost')
            qdrant_port = int(os.getenv('QDRANT_PORT', '6333'))
            qdrant_api_key = os.getenv('QDRANT_API_KEY')
            collection_name = os.getenv('QDRANT_COLLECTION_NAME')
            if not collection_name:
                raise ValueError("QDRANT_COLLECTION_NAME environment variable is required")

            self._vector_storage = QdrantVectorStorage(
                host=qdrant_host,
                port=qdrant_port,
                api_key=qdrant_api_key,
                collection_name=collection_name
            )

            # Initialize embedding service
            gemini_api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
            if gemini_api_key:
                self._gemini_embedding_service = GeminiEmbeddingService(
                    api_key=gemini_api_key,
                    embedding_model="text-embedding-004",
                    generation_model="gemini-2.0-flash-001"
                )
                logger.info("Search services initialized successfully")
            else:
                logger.warning("Gemini API key not found - search functionality will be limited")

        except Exception as e:
            logger.error("Failed to initialize search services", error=str(e))
            self._vector_storage = None
            self._gemini_embedding_service = None
        
        # Register content type detectors
        self._content_type_detectors = {
            ContentType.DOCUMENT: self._is_document,
            ContentType.AUDIO: self._is_audio,
            ContentType.VIDEO: self._is_video,
            ContentType.IMAGE: self._is_image,
            ContentType.WEB: self._is_web,
            ContentType.YOUTUBE: self._is_youtube,
        }
    
    def _is_document(self, path_or_url: str) -> bool:
        """Check if content is a document."""
        if path_or_url.startswith("http"):
            return False
        
        document_extensions = {
            ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
            ".txt", ".rtf", ".md", ".csv", ".json", ".xml", ".html", ".htm"
        }
        return Path(path_or_url).suffix.lower() in document_extensions
    
    def _is_audio(self, path_or_url: str) -> bool:
        """Check if content is audio."""
        if path_or_url.startswith("http"):
            return False
        
        audio_extensions = {
            ".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a", ".wma", ".opus"
        }
        return Path(path_or_url).suffix.lower() in audio_extensions
    
    def _is_video(self, path_or_url: str) -> bool:
        """Check if content is video."""
        if path_or_url.startswith("http"):
            return False
        
        video_extensions = {
            ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"
        }
        return Path(path_or_url).suffix.lower() in video_extensions
    
    def _is_image(self, path_or_url: str) -> bool:
        """Check if content is an image."""
        if path_or_url.startswith("http"):
            return False
        
        image_extensions = {
            ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".svg"
        }
        return Path(path_or_url).suffix.lower() in image_extensions
    
    def _is_web(self, path_or_url: str) -> bool:
        """Check if content is a web URL."""
        return path_or_url.startswith("http") and not self._is_youtube(path_or_url)
    
    def _is_youtube(self, path_or_url: str) -> bool:
        """Check if content is a YouTube URL."""
        youtube_domains = ["youtube.com", "youtu.be", "youtube-nocookie.com"]
        return path_or_url.startswith("http") and any(domain in path_or_url for domain in youtube_domains)
    
    def detect_content_type(self, path_or_url: str) -> str:
        """Detect content type from path or URL.
        
        Args:
            path_or_url: File path or URL
            
        Returns:
            Content type string
        """
        for content_type, detector in self._content_type_detectors.items():
            if detector(path_or_url):
                return content_type
        
        return ContentType.UNKNOWN
    
    async def process_content(self, path_or_url: str, content_type: Optional[str] = None, options: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """Process content based on its type.

        Args:
            path_or_url: File path or URL
            content_type: Content type (auto-detected if not provided)
            options: Processing options

        Returns:
            ProcessingResult with extracted information
        """
        # Detect content type if not provided
        if content_type is None:
            content_type = self.detect_content_type(path_or_url)
        
        # Convert Path objects to strings
        if isinstance(path_or_url, Path):
            path_or_url = str(path_or_url)

        # Process based on content type
        try:
            if content_type == ContentType.DOCUMENT:
                return await self.process_document(path_or_url, options)
            elif content_type == ContentType.AUDIO:
                return await self.process_audio(path_or_url, options)
            elif content_type == ContentType.VIDEO:
                return await self.process_video(path_or_url, options)
            elif content_type == ContentType.IMAGE:
                return await self.process_image(path_or_url, options)
            elif content_type == ContentType.WEB:
                return await self.process_url(path_or_url, options)
            elif content_type == ContentType.YOUTUBE:
                return await self.process_youtube(path_or_url, options)
            else:
                raise UnsupportedFormatError(f"Unsupported content type: {content_type}")
        except Exception as e:
            logger.exception(f"Error processing content", path_or_url=path_or_url, content_type=content_type)
            return ProcessingResult(
                content_type=content_type,
                content_path=path_or_url if not path_or_url.startswith("http") else None,
                content_url=path_or_url if path_or_url.startswith("http") else None,
                text_content="",
                success=False,
                error_message=str(e)
            )
    
    async def process_document(self, document_path: str, options: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """Process a document file.

        Args:
            document_path: Path to document file
            options: Processing options

        Returns:
            ProcessingResult with extracted text and metadata
        """
        try:
            # Get both formats: markdown for Qdrant, JSON for API response
            markdown_result = await self.document_service.process_document(
                Path(document_path),
                **(options or {})
            )

            json_result = await self.document_service.process_document_to_json(
                Path(document_path),
                **(options or {})
            )

            # Use markdown content for text_content (Qdrant storage)
            text_content = ""
            if markdown_result.document:
                if markdown_result.document.chunks:
                    # Create markdown from chunks
                    markdown_parts = []
                    for chunk in markdown_result.document.chunks:
                        if chunk.section:
                            markdown_parts.append(f"## {chunk.section}\n\n{chunk.content}")
                        else:
                            markdown_parts.append(chunk.content)
                    text_content = "\n\n".join(markdown_parts)
                else:
                    # Use raw text if no chunks
                    text_content = markdown_result.document.raw_text or ""

            return ProcessingResult(
                content_type=ContentType.DOCUMENT,
                content_path=document_path,
                text_content=text_content,  # Markdown for Qdrant
                metadata=json_result.get("metadata", {}),
                extracted_files=[],  # Document service doesn't return extracted files in this format
                processing_time=json_result.get("metadata", {}).get("processing_time", 0.0),
                success=True,
                error_message=json_result.get("error"),
                raw_result=json_result  # JSON for API response
            )
        except Exception as e:
            logger.exception(f"Error processing document", document_path=document_path)
            return ProcessingResult(
                content_type=ContentType.DOCUMENT,
                content_path=str(document_path),  # Convert Path to string
                text_content="",
                success=False,
                error_message=str(e)
            )
    
    async def process_audio(self, audio_path: str, options: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """Process an audio file.

        Args:
            audio_path: Path to audio file
            options: Processing options

        Returns:
            ProcessingResult with transcription and metadata
        """
        try:
            # Get both markdown (for Qdrant) and JSON (for API response)
            markdown_result = await self.audio_service.process_file(
                Path(audio_path),
                save_output=False,  # Don't save files, just return content
                output_format="markdown"  # Use markdown format for Qdrant storage
            )

            json_result = await self.audio_service.process_file(
                Path(audio_path),
                save_output=False,  # Don't save files, just return content
                output_format="json"  # Use JSON format for API response
            )

            # Use markdown content for text_content (Qdrant storage)
            text_content = ""
            if "content" in markdown_result:
                text_content = markdown_result["content"]
            elif "markdown" in markdown_result:
                text_content = markdown_result["markdown"]
            elif "transcript" in markdown_result:
                text_content = markdown_result["transcript"]

            return ProcessingResult(
                content_type=ContentType.AUDIO,
                content_path=audio_path,
                text_content=text_content,  # Markdown for Qdrant
                metadata=json_result.get("metadata", {}),
                processing_time=json_result.get("processing_time", 0.0),
                success=json_result.get("success", False),
                error_message=json_result.get("error"),
                raw_result=json_result  # JSON for API response
            )
        except Exception as e:
            logger.exception(f"Error processing audio", audio_path=audio_path)
            return ProcessingResult(
                content_type=ContentType.AUDIO,
                content_path=audio_path,
                text_content="",
                success=False,
                error_message=str(e)
            )
    
    async def process_video(self, video_path: str, options: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """Process a video file.

        Args:
            video_path: Path to video file
            options: Processing options

        Returns:
            ProcessingResult with extracted information
        """
        try:
            # Check if thumbnails should be generated (default to False - opt-in)
            include_thumbnails = options and options.get('include_thumbnails', False)

            # Configure video processor based on options
            if not include_thumbnails:
                # Temporarily disable thumbnail generation
                original_config = self.video_service.config.generate_thumbnails
                self.video_service.config.generate_thumbnails = False

            # Get both markdown (for Qdrant) and JSON (for API response)
            markdown_result = await self.video_service.process_file(
                Path(video_path),
                save_output=False,  # Don't save files, just return content
                output_format="markdown"  # Use markdown format for Qdrant storage
            )

            json_result = await self.video_service.process_file(
                Path(video_path),
                save_output=False,  # Don't save files, just return content
                output_format="json"  # Use JSON format for API response
            )

            # Restore original config
            if not include_thumbnails:
                self.video_service.config.generate_thumbnails = original_config

            # Use markdown content for text_content (Qdrant storage)
            text_content = ""
            if "content" in markdown_result:
                text_content = markdown_result["content"]
            elif "markdown" in markdown_result:
                text_content = markdown_result["markdown"]

            return ProcessingResult(
                content_type=ContentType.VIDEO,
                content_path=video_path,
                text_content=text_content,  # Markdown for Qdrant
                metadata=json_result.get("metadata", {}),
                extracted_files=json_result.get("thumbnails", []) + json_result.get("keyframes", []),
                processing_time=json_result.get("processing_time", 0.0),
                success=json_result.get("success", False),
                error_message=json_result.get("error"),
                raw_result=json_result  # JSON for API response
            )
        except Exception as e:
            logger.exception(f"Error processing video", video_path=video_path)
            return ProcessingResult(
                content_type=ContentType.VIDEO,
                content_path=video_path,
                text_content="",
                success=False,
                error_message=str(e)
            )
    
    async def process_image(self, image_path: str, options: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """Process an image file.

        Args:
            image_path: Path to image file
            options: Processing options

        Returns:
            ProcessingResult with extracted text and metadata
        """
        try:
            # Use process_image method from ImageProcessor directly
            from morag_image.processor import ImageProcessor, ImageConfig

            # Create processor with API key from config
            api_key = getattr(self.config, 'google_api_key', None)
            processor = ImageProcessor(api_key=api_key)

            # Create image config
            image_config = ImageConfig(
                generate_caption=True,
                extract_text=True,
                extract_metadata=True,
                ocr_engine="tesseract"
            )

            result = await processor.process_image(Path(image_path), image_config)

            return ProcessingResult(
                content_type=ContentType.IMAGE,
                content_path=image_path,
                text_content=result.extracted_text or "",
                metadata={
                    "caption": result.caption,
                    "metadata": result.metadata.__dict__ if result.metadata else {},
                    "confidence_scores": result.confidence_scores
                },
                processing_time=result.processing_time,
                success=True,
                error_message=None,
                raw_result=result
            )
        except Exception as e:
            logger.exception(f"Error processing image", image_path=image_path)
            return ProcessingResult(
                content_type=ContentType.IMAGE,
                content_path=image_path,
                text_content="",
                success=False,
                error_message=str(e)
            )
    
    async def process_url(self, url: str, options: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """Process a web URL.

        Args:
            url: Web URL to process
            options: Processing options

        Returns:
            ProcessingResult with extracted content and metadata
        """
        try:
            result = await self.web_service.process_url(
                url,
                config_options=self.config.web_config.to_dict() if hasattr(self.config.web_config, 'to_dict') else None
            )
            
            return ProcessingResult(
                content_type=ContentType.WEB,
                content_url=url,
                text_content=result.content,
                metadata=result.metadata,
                processing_time=result.processing_time,
                success=result.success,
                error_message=result.error_message,
                raw_result=result
            )
        except Exception as e:
            logger.exception(f"Error processing URL", url=url)
            return ProcessingResult(
                content_type=ContentType.WEB,
                content_url=url,
                text_content="",
                success=False,
                error_message=str(e)
            )
    
    async def process_youtube(self, url: str, options: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """Process a YouTube URL.

        Args:
            url: YouTube URL to process
            options: Processing options

        Returns:
            ProcessingResult with extracted content and metadata
        """
        try:
            result = await self.youtube_service.process_video(
                url,
                config=self.config.youtube_config
            )
            
            # Convert YouTube-specific result to unified format
            extracted_files = []
            if result.video_path:
                extracted_files.append(str(result.video_path))
            if result.audio_path:
                extracted_files.append(str(result.audio_path))
            extracted_files.extend([str(p) for p in result.subtitle_paths])
            extracted_files.extend([str(p) for p in result.thumbnail_paths])
            
            # Convert metadata to dictionary
            metadata = {}
            if result.metadata:
                metadata = {
                    'id': result.metadata.id,
                    'title': result.metadata.title,
                    'description': result.metadata.description,
                    'uploader': result.metadata.uploader,
                    'upload_date': result.metadata.upload_date,
                    'duration': result.metadata.duration,
                    'view_count': result.metadata.view_count,
                    'like_count': result.metadata.like_count,
                    'comment_count': result.metadata.comment_count,
                    'tags': result.metadata.tags,
                    'categories': result.metadata.categories,
                    'thumbnail_url': result.metadata.thumbnail_url,
                    'webpage_url': result.metadata.webpage_url,
                    'channel_id': result.metadata.channel_id,
                    'channel_url': result.metadata.channel_url,
                }
            
            return ProcessingResult(
                content_type=ContentType.YOUTUBE,
                content_url=url,
                text_content=None,  # YouTube doesn't directly provide text content
                metadata=metadata,
                extracted_files=extracted_files,
                processing_time=result.processing_time,
                success=result.success,
                error_message=result.error_message,
                raw_result=result
            )
        except Exception as e:
            logger.exception(f"Error processing YouTube URL", url=url)
            return ProcessingResult(
                content_type=ContentType.YOUTUBE,
                content_url=url,
                text_content="",
                success=False,
                error_message=str(e)
            )
    
    async def process_batch(self, items: List[str]) -> Dict[str, ProcessingResult]:
        """Process multiple content items concurrently.
        
        Args:
            items: List of file paths or URLs
            
        Returns:
            Dictionary mapping items to their processing results
        """
        async def process_with_semaphore(item: str) -> Tuple[str, ProcessingResult]:
            async with self.semaphore:
                result = await self.process_content(item)
                return item, result
        
        # Create tasks for all items
        tasks = [process_with_semaphore(item) for item in items]
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert results to dictionary
        result_dict = {}
        for item, result in results:
            if isinstance(result, Exception):
                # Handle exceptions
                result_dict[item] = ProcessingResult(
                    content_type=ContentType.UNKNOWN,
                    content_path=item if not item.startswith("http") else None,
                    content_url=item if item.startswith("http") else None,
                    text_content="",
                    success=False,
                    error_message=str(result)
                )
            else:
                result_dict[item] = result
        
        return result_dict
    
    async def generate_embeddings(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for text.
        
        Args:
            text: Text or list of texts to embed
            
        Returns:
            Embeddings as list of floats or list of list of floats
        """
        return await self.embedding_service.generate_embeddings(
            text,
            config=self.config.embedding_config
        )
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all services.

        Returns:
            Health status information for all services
        """
        health_status = {
            "overall_status": "healthy",
            "services": {}
        }

        # Check each service
        services_to_check = [
            ("document", self.document_service),
            ("audio", self.audio_service),
            ("video", self.video_service),
            ("image", self.image_service),
            ("embedding", self.embedding_service),
            ("web", self.web_service),
            ("youtube", self.youtube_service),
        ]

        unhealthy_services = []

        for service_name, service in services_to_check:
            try:
                service_health = await service.health_check()
                health_status["services"][service_name] = service_health

                # Check if service is unhealthy
                if service_health.get("status") != "healthy":
                    unhealthy_services.append(service_name)

            except Exception as e:
                logger.error(f"Health check failed for {service_name} service", error=str(e))
                health_status["services"][service_name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                unhealthy_services.append(service_name)

        # Set overall status
        if unhealthy_services:
            health_status["overall_status"] = "degraded" if len(unhealthy_services) < len(services_to_check) else "unhealthy"
            health_status["unhealthy_services"] = unhealthy_services

        return health_status

    async def search_similar(self, query: str, limit: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar content using vector similarity.

        Args:
            query: Search query
            limit: Maximum number of results
            filters: Optional filters

        Returns:
            List of similar content items
        """
        try:
            # Check if search services are available
            if not self._vector_storage or not self._gemini_embedding_service:
                logger.warning("Search services not available - initializing embedding service or vector storage failed")
                return []

            # Initialize services if not already done
            if not self._vector_storage._initialized:
                await self._vector_storage.initialize()

            if not await self._gemini_embedding_service.initialize():
                logger.error("Failed to initialize Gemini embedding service")
                return []

            # Generate embedding for the query
            logger.info("Generating embedding for search query", query=query[:100])
            query_embedding = await self._gemini_embedding_service.generate_embedding(
                query,
                task_type="retrieval_query"
            )

            # Search for similar vectors
            logger.info("Searching for similar vectors", limit=limit)
            results = await self._vector_storage.search_similar(
                query_vector=query_embedding,
                limit=limit,
                score_threshold=0.5,  # Minimum similarity threshold
                filters=filters
            )

            # Format results for API response
            formatted_results = []
            for result in results:
                formatted_result = {
                    "id": result.get("id"),
                    "score": result.get("score", 0.0),
                    "text": result.get("metadata", {}).get("text", ""),
                    "metadata": result.get("metadata", {}),
                    "content_type": result.get("metadata", {}).get("content_type"),
                    "source": result.get("metadata", {}).get("source")
                }
                formatted_results.append(formatted_result)

            logger.info("Search completed successfully",
                       query=query[:50],
                       results_count=len(formatted_results))
            return formatted_results

        except Exception as e:
            logger.error("Search failed", query=query, error=str(e))
            # Return empty list instead of raising exception to maintain API stability
            return []

    def cleanup(self):
        """Clean up resources used by services."""
        # Implement cleanup for each service if needed
        pass