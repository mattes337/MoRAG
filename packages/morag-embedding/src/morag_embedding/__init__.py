"""MoRAG Embedding Service."""

from .service import GeminiEmbeddingService

# Alias for backward compatibility
EmbeddingService = GeminiEmbeddingService

__version__ = "0.1.0"

__all__ = [
    "GeminiEmbeddingService",
    "EmbeddingService",
]