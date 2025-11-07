"""Base interfaces for embedding services."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class EmbeddingProvider(str, Enum):
    """Embedding provider enum."""

    OPENAI = "openai"
    GEMINI = "gemini"
    HUGGINGFACE = "huggingface"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    COHERE = "cohere"
    ANTHROPIC = "anthropic"


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""

    provider: EmbeddingProvider
    model_name: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    batch_size: int = 100
    max_tokens: int = 8192
    normalize: bool = True
    timeout: float = 30.0
    retry_attempts: int = 3
    custom_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingRequest:
    """Request for embedding generation."""

    texts: List[str]
    model: Optional[str] = None
    normalize: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""

    embeddings: List[List[float]]
    model: str
    usage: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


class BaseEmbeddingService(ABC):
    """Base class for embedding services."""

    def __init__(self, config: EmbeddingConfig):
        """Initialize embedding service.

        Args:
            config: Embedding configuration
        """
        self.config = config

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the embedding service.

        Returns:
            True if initialization was successful, False otherwise
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the embedding service and release resources."""
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check service health.

        Returns:
            Dictionary with health status information
        """
        pass

    @abstractmethod
    async def generate_embeddings(
        self, texts: List[str], model: Optional[str] = None, **kwargs
    ) -> EmbeddingResult:
        """Generate embeddings for texts.

        Args:
            texts: List of texts to embed
            model: Optional model override
            **kwargs: Additional options

        Returns:
            Embedding result
        """
        pass

    @abstractmethod
    async def generate_embedding(
        self, text: str, model: Optional[str] = None, **kwargs
    ) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed
            model: Optional model override
            **kwargs: Additional options

        Returns:
            Embedding vector
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self, model: Optional[str] = None) -> int:
        """Get embedding dimension for model.

        Args:
            model: Optional model override

        Returns:
            Embedding dimension
        """
        pass

    @abstractmethod
    def get_supported_models(self) -> List[str]:
        """Get list of supported models.

        Returns:
            List of model names
        """
        pass

    @abstractmethod
    def get_max_tokens(self, model: Optional[str] = None) -> int:
        """Get maximum tokens for model.

        Args:
            model: Optional model override

        Returns:
            Maximum tokens
        """
        pass

    def validate_texts(self, texts: List[str], model: Optional[str] = None) -> bool:
        """Validate texts for embedding.

        Args:
            texts: Texts to validate
            model: Optional model override

        Returns:
            True if texts are valid

        Raises:
            ValueError: If texts are invalid
        """
        if not texts:
            raise ValueError("No texts provided")

        max_tokens = self.get_max_tokens(model)

        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(f"Empty text at index {i}")

            # Simple token estimation (rough)
            token_count = len(text.split())
            if token_count > max_tokens:
                raise ValueError(
                    f"Text at index {i} exceeds max tokens: {token_count} > {max_tokens}"
                )

        return True

    def chunk_texts(
        self, texts: List[str], max_chunk_size: Optional[int] = None
    ) -> List[List[str]]:
        """Chunk texts into batches.

        Args:
            texts: Texts to chunk
            max_chunk_size: Maximum chunk size (uses config batch_size if None)

        Returns:
            List of text chunks
        """
        chunk_size = max_chunk_size or self.config.batch_size
        chunks = []

        for i in range(0, len(texts), chunk_size):
            chunks.append(texts[i : i + chunk_size])

        return chunks
