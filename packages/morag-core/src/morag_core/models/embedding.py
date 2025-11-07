"""Embedding models for MoRAG."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import uuid


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    text: str
    embedding: List[float]
    model: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "id": self.id,
            "text": self.text,
            "embedding": self.embedding,
            "model": self.model,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class BatchEmbeddingResult:
    """Result of batch embedding generation."""
    texts: List[str]
    embeddings: List[List[float]]
    model: str
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "texts": self.texts,
            "embeddings": self.embeddings,
            "model": self.model,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    def get_individual_results(self) -> List[EmbeddingResult]:
        """Get individual embedding results.

        Returns:
            List of individual embedding results
        """
        results = []
        for i, (text, embedding) in enumerate(zip(self.texts, self.embeddings)):
            result = EmbeddingResult(
                text=text,
                embedding=embedding,
                model=self.model,
                created_at=self.created_at,
                metadata={
                    **self.metadata,
                    "batch_index": i,
                }
            )
            results.append(result)
        return results


@dataclass
class SummaryResult:
    """Result of text summarization."""
    original_text: str
    summary: str
    model: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "id": self.id,
            "original_text": self.original_text,
            "summary": self.summary,
            "model": self.model,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }
