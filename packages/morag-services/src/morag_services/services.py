"""Main services integration module for MoRAG system.

This module provides the MoRAGServices class that integrates all specialized
processing packages into a cohesive API.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from pathlib import Path
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

class ContentType:
    """Content type constants."""
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
        
        # Create semaphore for concurrent processing
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)
        
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
    
    async def process_content(self, path_or_url: str, content_type: Optional[str] = None) -> ProcessingResult:
        """Process content based on its type.
        
        Args:
            path_or_url: File path or URL
            content_type: Content type (auto-detected if not provided)
            
        Returns:
            ProcessingResult with extracted information
        """
        # Detect content type if not provided
        if content_type is None:
            content_type = self.detect_content_type(path_or_url)
        
        # Process based on content type
        try:
            if content_type == ContentType.DOCUMENT:
                return await self.process_document(path_or_url)
            elif content_type == ContentType.AUDIO:
                return await self.process_audio(path_or_url)
            elif content_type == ContentType.VIDEO:
                return await self.process_video(path_or_url)
            elif content_type == ContentType.IMAGE:
                return await self.process_image(path_or_url)
            elif content_type == ContentType.WEB:
                return await self.process_url(path_or_url)
            elif content_type == ContentType.YOUTUBE:
                return await self.process_youtube(path_or_url)
            else:
                raise UnsupportedFormatError(f"Unsupported content type: {content_type}")
        except Exception as e:
            logger.exception(f"Error processing content", path_or_url=path_or_url, content_type=content_type)
            return ProcessingResult(
                content_type=content_type,
                content_path=path_or_url if not path_or_url.startswith("http") else None,
                content_url=path_or_url if path_or_url.startswith("http") else None,
                success=False,
                error_message=str(e)
            )
    
    async def process_document(self, document_path: str) -> ProcessingResult:
        """Process a document file.
        
        Args:
            document_path: Path to document file
            
        Returns:
            ProcessingResult with extracted text and metadata
        """
        try:
            result = await self.document_service.process_document(
                Path(document_path),
                config=self.config.document_config
            )
            
            return ProcessingResult(
                content_type=ContentType.DOCUMENT,
                content_path=document_path,
                text_content=result.text,
                metadata=result.metadata,
                extracted_files=[str(p) for p in result.extracted_files],
                processing_time=result.processing_time,
                success=result.success,
                error_message=result.error_message,
                raw_result=result
            )
        except Exception as e:
            logger.exception(f"Error processing document", document_path=document_path)
            return ProcessingResult(
                content_type=ContentType.DOCUMENT,
                content_path=document_path,
                success=False,
                error_message=str(e)
            )
    
    async def process_audio(self, audio_path: str) -> ProcessingResult:
        """Process an audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            ProcessingResult with transcription and metadata
        """
        try:
            result = await self.audio_service.process_audio(
                Path(audio_path),
                config=self.config.audio_config
            )
            
            return ProcessingResult(
                content_type=ContentType.AUDIO,
                content_path=audio_path,
                text_content=result.transcription,
                metadata=result.metadata,
                processing_time=result.processing_time,
                success=result.success,
                error_message=result.error_message,
                raw_result=result
            )
        except Exception as e:
            logger.exception(f"Error processing audio", audio_path=audio_path)
            return ProcessingResult(
                content_type=ContentType.AUDIO,
                content_path=audio_path,
                success=False,
                error_message=str(e)
            )
    
    async def process_video(self, video_path: str) -> ProcessingResult:
        """Process a video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            ProcessingResult with extracted information
        """
        try:
            result = await self.video_service.process_video(
                Path(video_path),
                config=self.config.video_config
            )
            
            return ProcessingResult(
                content_type=ContentType.VIDEO,
                content_path=video_path,
                text_content=result.transcription,
                metadata=result.metadata,
                extracted_files=[str(p) for p in result.extracted_files],
                processing_time=result.processing_time,
                success=result.success,
                error_message=result.error_message,
                raw_result=result
            )
        except Exception as e:
            logger.exception(f"Error processing video", video_path=video_path)
            return ProcessingResult(
                content_type=ContentType.VIDEO,
                content_path=video_path,
                success=False,
                error_message=str(e)
            )
    
    async def process_image(self, image_path: str) -> ProcessingResult:
        """Process an image file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            ProcessingResult with extracted text and metadata
        """
        try:
            result = await self.image_service.process_image(
                Path(image_path),
                config=self.config.image_config
            )
            
            return ProcessingResult(
                content_type=ContentType.IMAGE,
                content_path=image_path,
                text_content=result.text,
                metadata=result.metadata,
                processing_time=result.processing_time,
                success=result.success,
                error_message=result.error_message,
                raw_result=result
            )
        except Exception as e:
            logger.exception(f"Error processing image", image_path=image_path)
            return ProcessingResult(
                content_type=ContentType.IMAGE,
                content_path=image_path,
                success=False,
                error_message=str(e)
            )
    
    async def process_url(self, url: str) -> ProcessingResult:
        """Process a web URL.
        
        Args:
            url: Web URL to process
            
        Returns:
            ProcessingResult with extracted content and metadata
        """
        try:
            result = await self.web_service.process_url(
                url,
                config=self.config.web_config
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
                success=False,
                error_message=str(e)
            )
    
    async def process_youtube(self, url: str) -> ProcessingResult:
        """Process a YouTube URL.
        
        Args:
            url: YouTube URL to process
            
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
    
    def cleanup(self):
        """Clean up resources used by services."""
        # Implement cleanup for each service if needed
        pass