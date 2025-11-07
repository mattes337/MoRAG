"""Universal chunking configuration for MoRAG."""

from enum import Enum
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, Field


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""

    SEMANTIC = "semantic"
    SIZE_BASED = "size_based"
    HYBRID = "hybrid"
    TOPIC_BASED = "topic_based"
    SENTENCE_BASED = "sentence_based"
    PARAGRAPH_BASED = "paragraph_based"


class ChunkingConfig(BaseModel):
    """Universal configuration for text chunking across all MoRAG components."""

    # Core chunking parameters
    strategy: ChunkingStrategy = Field(
        default=ChunkingStrategy.SEMANTIC, description="Chunking strategy to use"
    )
    max_chunk_size: int = Field(
        default=4000, ge=100, le=32000, description="Maximum characters per chunk"
    )
    min_chunk_size: int = Field(
        default=500, ge=50, le=8000, description="Minimum characters per chunk"
    )
    overlap_size: int = Field(
        default=200, ge=0, le=2000, description="Character overlap between chunks"
    )

    # Semantic chunking parameters
    min_confidence: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for semantic boundaries",
    )
    use_ai_analysis: bool = Field(
        default=True, description="Whether to use AI for semantic analysis"
    )

    # Content-specific parameters
    respect_sentence_boundaries: bool = Field(
        default=True, description="Avoid splitting sentences when possible"
    )
    respect_paragraph_boundaries: bool = Field(
        default=True, description="Prefer paragraph boundaries for splits"
    )
    preserve_code_blocks: bool = Field(
        default=True, description="Keep code blocks intact when possible"
    )
    preserve_tables: bool = Field(
        default=True, description="Keep tables intact when possible"
    )

    # Language and encoding
    language: Optional[str] = Field(
        default=None, description="Content language for language-specific processing"
    )
    encoding: str = Field(default="utf-8", description="Text encoding")

    # Performance parameters
    max_concurrent_chunks: int = Field(
        default=5, ge=1, le=20, description="Maximum concurrent chunk processing"
    )
    timeout_seconds: int = Field(
        default=30, ge=5, le=300, description="Timeout for chunk processing"
    )

    # Content type specific settings
    content_type: Optional[str] = Field(
        default=None, description="Content type hint (document, audio, video, web)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for chunking"
    )

    @classmethod
    def for_documents(
        cls,
        max_chunk_size: int = 4000,
        strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC,
        **kwargs
    ) -> "ChunkingConfig":
        """Create configuration optimized for document processing."""
        # Set defaults only if not provided in kwargs
        defaults = {
            "strategy": strategy,
            "max_chunk_size": max_chunk_size,
            "min_chunk_size": max(500, max_chunk_size // 8),
            "overlap_size": max(200, max_chunk_size // 20),
            "respect_paragraph_boundaries": True,
            "preserve_code_blocks": True,
            "preserve_tables": True,
            "content_type": "document",
        }

        # Override defaults with kwargs
        for key, value in kwargs.items():
            defaults[key] = value

        return cls(**defaults)

    @classmethod
    def for_audio_transcripts(
        cls,
        max_chunk_size: int = 3000,
        strategy: ChunkingStrategy = ChunkingStrategy.TOPIC_BASED,
        **kwargs
    ) -> "ChunkingConfig":
        """Create configuration optimized for audio transcript processing."""
        return cls(
            strategy=strategy,
            max_chunk_size=max_chunk_size,
            min_chunk_size=max(300, max_chunk_size // 10),
            overlap_size=max(150, max_chunk_size // 20),
            respect_sentence_boundaries=True,
            respect_paragraph_boundaries=False,  # Transcripts may not have clear paragraphs
            preserve_code_blocks=False,
            preserve_tables=False,
            content_type="audio",
            **kwargs
        )

    @classmethod
    def for_video_transcripts(
        cls,
        max_chunk_size: int = 3500,
        strategy: ChunkingStrategy = ChunkingStrategy.TOPIC_BASED,
        **kwargs
    ) -> "ChunkingConfig":
        """Create configuration optimized for video transcript processing."""
        return cls(
            strategy=strategy,
            max_chunk_size=max_chunk_size,
            min_chunk_size=max(400, max_chunk_size // 9),
            overlap_size=max(175, max_chunk_size // 20),
            respect_sentence_boundaries=True,
            respect_paragraph_boundaries=False,
            preserve_code_blocks=False,
            preserve_tables=False,
            content_type="video",
            **kwargs
        )

    @classmethod
    def for_web_content(
        cls,
        max_chunk_size: int = 3000,
        strategy: ChunkingStrategy = ChunkingStrategy.HYBRID,
        **kwargs
    ) -> "ChunkingConfig":
        """Create configuration optimized for web content processing."""
        return cls(
            strategy=strategy,
            max_chunk_size=max_chunk_size,
            min_chunk_size=max(300, max_chunk_size // 10),
            overlap_size=max(150, max_chunk_size // 20),
            respect_paragraph_boundaries=True,
            preserve_code_blocks=True,
            preserve_tables=True,
            content_type="web",
            **kwargs
        )

    @classmethod
    def for_code(
        cls,
        max_chunk_size: int = 2000,
        strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC,
        **kwargs
    ) -> "ChunkingConfig":
        """Create configuration optimized for code processing."""
        return cls(
            strategy=strategy,
            max_chunk_size=max_chunk_size,
            min_chunk_size=max(200, max_chunk_size // 10),
            overlap_size=max(100, max_chunk_size // 20),
            respect_sentence_boundaries=False,
            respect_paragraph_boundaries=False,
            preserve_code_blocks=True,
            preserve_tables=False,
            content_type="code",
            **kwargs
        )

    def validate_config(self) -> None:
        """Validate the chunking configuration."""
        if self.min_chunk_size >= self.max_chunk_size:
            raise ValueError("min_chunk_size must be less than max_chunk_size")

        if self.overlap_size >= self.min_chunk_size:
            raise ValueError("overlap_size must be less than min_chunk_size")

        if self.max_concurrent_chunks < 1:
            raise ValueError("max_concurrent_chunks must be at least 1")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkingConfig":
        """Create configuration from dictionary."""
        return cls(**data)
