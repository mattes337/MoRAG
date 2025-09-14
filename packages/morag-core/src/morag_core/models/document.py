"""Document models for MoRAG."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import uuid


class DocumentType(str, Enum):
    """Document type enum."""
    TEXT = "text"
    PDF = "pdf"
    WORD = "word"
    EXCEL = "excel"
    POWERPOINT = "powerpoint"
    HTML = "html"
    MARKDOWN = "markdown"
    JSON = "json"
    XML = "xml"
    CSV = "csv"
    AUDIO = "audio"
    VIDEO = "video"
    IMAGE = "image"
    ARCHIVE = "archive"
    EBOOK = "ebook"
    URL = "url"
    UNKNOWN = "unknown"


@dataclass
class DocumentMetadata:
    """Document metadata."""
    source_type: DocumentType
    source_name: str
    source_path: Optional[str] = None
    source_url: Optional[str] = None
    mime_type: Optional[str] = None
    file_size: Optional[int] = None
    checksum: Optional[str] = None
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    author: Optional[str] = None
    title: Optional[str] = None
    language: Optional[str] = None
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    custom: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "source_type": self.source_type.value,
            "source_name": self.source_name,
            "source_path": self.source_path,
            "source_url": self.source_url,
            "mime_type": self.mime_type,
            "file_size": self.file_size,
            "checksum": self.checksum,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "modified_at": self.modified_at.isoformat() if self.modified_at else None,
            "author": self.author,
            "title": self.title,
            "language": self.language,
            "page_count": self.page_count,
            "word_count": self.word_count,
            "custom": self.custom,
        }


@dataclass
class DocumentChunk:
    """Document chunk."""
    document_id: str
    content: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    chunk_index: int = 0
    page_number: Optional[int] = None
    section: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "id": self.id,
            "document_id": self.document_id,
            "content": self.content,
            "metadata": self.metadata,
            "chunk_index": self.chunk_index,
            "page_number": self.page_number,
            "section": self.section,
        }


@dataclass
class Document:
    """Document model."""
    metadata: DocumentMetadata
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    chunks: List[DocumentChunk] = field(default_factory=list)
    raw_text: Optional[str] = None
    processed_at: datetime = field(default_factory=datetime.now)
    
    def add_chunk(self, content: str, **kwargs) -> DocumentChunk:
        """Add a chunk to the document.
        
        Args:
            content: Chunk content
            **kwargs: Additional chunk metadata
            
        Returns:
            Created document chunk
        """
        chunk = DocumentChunk(
            document_id=self.id,
            content=content,
            chunk_index=len(self.chunks),
            **kwargs
        )
        self.chunks.append(chunk)
        return chunk
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "id": self.id,
            "metadata": {
                "source_type": self.metadata.source_type.value,
                "source_name": self.metadata.source_name,
                "source_path": self.metadata.source_path,
                "source_url": self.metadata.source_url,
                "mime_type": self.metadata.mime_type,
                "file_size": self.metadata.file_size,
                "checksum": self.metadata.checksum,
                "created_at": self.metadata.created_at.isoformat() if self.metadata.created_at else None,
                "modified_at": self.metadata.modified_at.isoformat() if self.metadata.modified_at else None,
                "author": self.metadata.author,
                "title": self.metadata.title,
                "language": self.metadata.language,
                "page_count": self.metadata.page_count,
                "word_count": self.metadata.word_count,
                "custom": self.metadata.custom,
            },
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "processed_at": self.processed_at.isoformat(),
        }