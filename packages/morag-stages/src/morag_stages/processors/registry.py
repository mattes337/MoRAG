"""Processor registry for dynamic loading of file-type processors."""

from typing import Dict, Optional, Type
import structlog

from .interface import StageProcessor

logger = structlog.get_logger(__name__)


class ProcessorRegistry:
    """Registry for managing stage processors."""
    
    def __init__(self):
        self._processors: Dict[str, StageProcessor] = {}
        self._processor_classes: Dict[str, Type[StageProcessor]] = {}
        self._initialize_processors()
    
    def _initialize_processors(self):
        """Initialize available processors from file-type packages."""
        # Register YouTube processor
        try:
            from .youtube_processor import YouTubeStageProcessor
            self._processor_classes["YOUTUBE"] = YouTubeStageProcessor
            logger.debug("Registered YouTube processor")
        except ImportError as e:
            logger.warning("YouTube processor not available", error=str(e))
        
        # Register video processor
        try:
            from .video_processor import VideoStageProcessor
            self._processor_classes["VIDEO"] = VideoStageProcessor
            logger.debug("Registered video processor")
        except ImportError as e:
            logger.warning("Video processor not available", error=str(e))
        
        # Register audio processor
        try:
            from .audio_processor import AudioStageProcessor
            self._processor_classes["AUDIO"] = AudioStageProcessor
            logger.debug("Registered audio processor")
        except ImportError as e:
            logger.warning("Audio processor not available", error=str(e))
        
        # Register web processor
        try:
            from .web_processor import WebStageProcessor
            self._processor_classes["WEB"] = WebStageProcessor
            logger.debug("Registered web processor")
        except ImportError as e:
            logger.warning("Web processor not available", error=str(e))
        
        # Register document processor
        try:
            from .document_processor import DocumentStageProcessor
            self._processor_classes["DOCUMENT"] = DocumentStageProcessor
            logger.debug("Registered document processor")
        except ImportError as e:
            logger.warning("Document processor not available", error=str(e))
    
    def get_processor(self, content_type: str) -> Optional[StageProcessor]:
        """Get processor for content type.
        
        Args:
            content_type: Content type (e.g., "VIDEO", "AUDIO", "YOUTUBE")
            
        Returns:
            Processor instance or None if not available
        """
        content_type = content_type.upper()
        
        # Return cached instance if available
        if content_type in self._processors:
            return self._processors[content_type]
        
        # Create new instance if class is available
        if content_type in self._processor_classes:
            try:
                processor = self._processor_classes[content_type]()
                self._processors[content_type] = processor
                logger.debug("Created processor instance", content_type=content_type)
                return processor
            except Exception as e:
                logger.error("Failed to create processor", content_type=content_type, error=str(e))
                return None
        
        logger.debug("No processor available", content_type=content_type)
        return None
    
    def supports_content_type(self, content_type: str) -> bool:
        """Check if content type is supported.
        
        Args:
            content_type: Content type to check
            
        Returns:
            True if supported, False otherwise
        """
        processor = self.get_processor(content_type)
        return processor is not None and processor.supports_content_type(content_type)
    
    def list_supported_types(self) -> list[str]:
        """List all supported content types.
        
        Returns:
            List of supported content type strings
        """
        return list(self._processor_classes.keys())


# Global registry instance
_registry = ProcessorRegistry()


def get_registry() -> ProcessorRegistry:
    """Get the global processor registry.
    
    Returns:
        ProcessorRegistry instance
    """
    return _registry
