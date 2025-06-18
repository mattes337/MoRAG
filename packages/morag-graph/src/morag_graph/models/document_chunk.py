"""DocumentChunk model for knowledge graph.

This module defines the DocumentChunk model which represents chunks or sections
of documents in the knowledge graph. DocumentChunks contain the actual text content
and are linked to both their parent Document and the entities mentioned within them.
"""

import json
import uuid
from typing import Dict, List, Optional, Any, ClassVar

from pydantic import BaseModel, Field, field_validator

from .types import EntityId


class DocumentChunk(BaseModel):
    """DocumentChunk model representing a chunk of text from a document.
    
    A document chunk represents a section or chunk of text from a source document.
    It contains the actual text content and metadata about its position within
    the document. Chunks are linked to their parent document and to the entities
    mentioned within them.
    
    Attributes:
        id: Unique identifier for the document chunk
        document_id: ID of the parent document
        chunk_index: Index/position of this chunk within the document
        text: The actual text content of this chunk
        start_position: Starting character position in the original document
        end_position: Ending character position in the original document
        chunk_type: Type of chunk (e.g., 'paragraph', 'section', 'page')
        metadata: Additional metadata about the chunk
    """
    
    id: EntityId = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: EntityId
    chunk_index: int
    text: str
    start_position: Optional[int] = None
    end_position: Optional[int] = None
    chunk_type: str = "section"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Class variables for Neo4J integration
    _neo4j_label: ClassVar[str] = "DocumentChunk"
    
    @field_validator('chunk_index')
    @classmethod
    def validate_chunk_index(cls, v: int) -> int:
        """Validate that chunk_index is non-negative."""
        if v < 0:
            raise ValueError(f"Chunk index must be non-negative, got {v}")
        return v
    
    @field_validator('text')
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Validate that text is not empty."""
        if not v or not v.strip():
            raise ValueError("Text content cannot be empty")
        return v
    
    @field_validator('document_id')
    @classmethod
    def validate_document_id(cls, v: EntityId) -> EntityId:
        """Validate that document_id is not empty."""
        if not v or not v.strip():
            raise ValueError("Document ID cannot be empty")
        return v
    
    def __hash__(self) -> int:
        """Make DocumentChunk hashable based on its ID."""
        return hash(self.id)
    
    def __eq__(self, other) -> bool:
        """Compare document chunks based on their ID."""
        if not isinstance(other, DocumentChunk):
            return False
        return self.id == other.id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document chunk to dictionary for JSON serialization."""
        return self.model_dump()
    
    def to_neo4j_node(self) -> Dict[str, Any]:
        """Convert document chunk to Neo4J node properties."""
        properties = self.model_dump()
        
        # Serialize metadata to JSON string for Neo4J storage
        if 'metadata' in properties:
            properties['metadata'] = json.dumps(properties['metadata'])
            
        # Add label for Neo4J
        properties['_labels'] = [self._neo4j_label]
        
        return properties
    
    @classmethod
    def from_neo4j_node(cls, node: Dict[str, Any]) -> 'DocumentChunk':
        """Create document chunk from Neo4J node properties."""
        # Make a copy to avoid modifying the original
        node = node.copy()
        
        # Deserialize metadata from JSON string
        if 'metadata' in node and isinstance(node['metadata'], str):
            node['metadata'] = json.loads(node['metadata'])
        
        # Remove Neo4J specific properties
        if '_labels' in node:
            node.pop('_labels')
            
        return cls(**node)