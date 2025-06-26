"""Factory for creating chunkers with different configurations."""

from typing import Optional, Dict, Any
import structlog

from .config import ChunkingConfig, ChunkingStrategy
from .semantic_chunker import SemanticChunker

logger = structlog.get_logger(__name__)


class ChunkerFactory:
    """Factory for creating chunkers with different configurations."""
    
    @staticmethod
    def create_chunker(
        config: Optional[ChunkingConfig] = None,
        content_type: Optional[str] = None,
        **kwargs
    ) -> SemanticChunker:
        """Create a chunker with the specified configuration.
        
        Args:
            config: Chunking configuration
            content_type: Content type hint for automatic configuration
            **kwargs: Additional configuration parameters
            
        Returns:
            Configured SemanticChunker instance
        """
        # If no config provided, create based on content type
        if config is None:
            if content_type == "document":
                config = ChunkingConfig.for_documents(**kwargs)
            elif content_type == "audio":
                config = ChunkingConfig.for_audio_transcripts(**kwargs)
            elif content_type == "video":
                config = ChunkingConfig.for_video_transcripts(**kwargs)
            elif content_type == "web":
                config = ChunkingConfig.for_web_content(**kwargs)
            else:
                config = ChunkingConfig(**kwargs)
        
        logger.info(
            "Creating chunker",
            strategy=config.strategy,
            max_chunk_size=config.max_chunk_size,
            content_type=config.content_type
        )
        
        return SemanticChunker(config)
    
    @staticmethod
    def create_document_chunker(
        max_chunk_size: int = 4000,
        strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC,
        **kwargs
    ) -> SemanticChunker:
        """Create a chunker optimized for document processing."""
        config = ChunkingConfig.for_documents(
            max_chunk_size=max_chunk_size,
            strategy=strategy,
            **kwargs
        )
        return SemanticChunker(config)
    
    @staticmethod
    def create_audio_chunker(
        max_chunk_size: int = 3000,
        strategy: ChunkingStrategy = ChunkingStrategy.TOPIC_BASED,
        **kwargs
    ) -> SemanticChunker:
        """Create a chunker optimized for audio transcript processing."""
        config = ChunkingConfig.for_audio_transcripts(
            max_chunk_size=max_chunk_size,
            strategy=strategy,
            **kwargs
        )
        return SemanticChunker(config)
    
    @staticmethod
    def create_video_chunker(
        max_chunk_size: int = 3500,
        strategy: ChunkingStrategy = ChunkingStrategy.TOPIC_BASED,
        **kwargs
    ) -> SemanticChunker:
        """Create a chunker optimized for video transcript processing."""
        config = ChunkingConfig.for_video_transcripts(
            max_chunk_size=max_chunk_size,
            strategy=strategy,
            **kwargs
        )
        return SemanticChunker(config)
    
    @staticmethod
    def create_web_chunker(
        max_chunk_size: int = 3000,
        strategy: ChunkingStrategy = ChunkingStrategy.HYBRID,
        **kwargs
    ) -> SemanticChunker:
        """Create a chunker optimized for web content processing."""
        config = ChunkingConfig.for_web_content(
            max_chunk_size=max_chunk_size,
            strategy=strategy,
            **kwargs
        )
        return SemanticChunker(config)


def create_chunker(
    content_type: Optional[str] = None,
    strategy: Optional[ChunkingStrategy] = None,
    max_chunk_size: Optional[int] = None,
    **kwargs
) -> SemanticChunker:
    """Convenience function to create a chunker.
    
    Args:
        content_type: Content type hint ("document", "audio", "video", "web")
        strategy: Chunking strategy to use
        max_chunk_size: Maximum chunk size
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured SemanticChunker instance
    """
    # Build configuration parameters
    config_kwargs = {}
    if strategy is not None:
        config_kwargs["strategy"] = strategy
    if max_chunk_size is not None:
        config_kwargs["max_chunk_size"] = max_chunk_size
    config_kwargs.update(kwargs)
    
    return ChunkerFactory.create_chunker(
        content_type=content_type,
        **config_kwargs
    )
