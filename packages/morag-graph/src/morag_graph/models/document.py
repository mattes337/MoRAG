"""Document model for knowledge graph.

This module defines the Document model which represents source documents
in the knowledge graph. Documents contain metadata about the source files
and are linked to DocumentChunk nodes that contain the actual text content.
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, ClassVar

from pydantic import BaseModel, Field, field_validator, model_validator

from .types import EntityId
from ..utils.id_generation import UnifiedIDGenerator, IDValidator


class Document(BaseModel):
    """Document model representing a source document in the knowledge graph.

    A document represents a source file or data source that has been ingested
    into the knowledge graph. It contains metadata about the source and is
    linked to DocumentChunk nodes that contain the actual text content.

    Attributes:
        id: Unique identifier for the document
        name: Display name of the document (typically the filename)
        source_file: Path or identifier of the source file
        file_name: Name of the source file
        file_size: Size of the source file in bytes
        checksum: SHA256 checksum of the source file content
        mime_type: MIME type of the source file
        ingestion_timestamp: When the document was ingested
        last_modified: When the source file was last modified
        model: Model used for extraction (e.g., 'gemini-1.5-flash')
        metadata: Additional metadata about the document
    """

    id: EntityId = Field(default="")
    name: str  # Display name for the document (required)
    source_file: str
    file_name: Optional[str] = None
    file_size: Optional[int] = None
    checksum: Optional[str] = None
    mime_type: Optional[str] = None
    ingestion_timestamp: datetime = Field(default_factory=datetime.now)
    last_modified: Optional[datetime] = None
    model: Optional[str] = None
    summary: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Class variables for Neo4J integration
    _neo4j_label: ClassVar[str] = "Document"
    
    def __init__(self, **data):
        """Initialize document with unified ID generation."""
        # Generate unified ID if not provided
        if 'id' not in data or not data['id']:
            data['id'] = UnifiedIDGenerator.generate_document_id(
                source_file=data.get('source_file', ''),
                checksum=data.get('checksum')
            )
        super().__init__(**data)
    
    @field_validator('id')
    @classmethod
    def validate_id_format(cls, v):
        """Validate document ID format."""
        if v and not IDValidator.validate_document_id(v):
            raise ValueError(f"Invalid document ID format: {v}")
        return v
    
    def get_unified_id(self) -> str:
        """Get unified document ID."""
        return self.id
    
    def is_unified_format(self) -> bool:
        """Check if using unified ID format."""
        return IDValidator.validate_document_id(self.id)
    
    def __hash__(self) -> int:
        """Make Document hashable based on its ID."""
        return hash(self.id)
    
    def __eq__(self, other) -> bool:
        """Compare documents based on their ID."""
        if not isinstance(other, Document):
            return False
        return self.id == other.id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary for JSON serialization."""
        return self.model_dump()
    
    def to_neo4j_node(self) -> Dict[str, Any]:
        """Convert document to Neo4J node properties."""
        properties = self.model_dump()
        
        # Convert datetime objects to ISO strings for Neo4J
        if 'ingestion_timestamp' in properties and properties['ingestion_timestamp']:
            properties['ingestion_timestamp'] = properties['ingestion_timestamp'].isoformat()
        if 'last_modified' in properties and properties['last_modified']:
            properties['last_modified'] = properties['last_modified'].isoformat()
            
        # Serialize metadata to JSON string for Neo4J storage
        if 'metadata' in properties:
            properties['metadata'] = json.dumps(properties['metadata'])
            
        # Add label for Neo4J
        properties['_labels'] = [self._neo4j_label]
        
        return properties
    
    @classmethod
    def from_neo4j_node(cls, node: Dict[str, Any]) -> 'Document':
        """Create document from Neo4J node properties."""
        # Make a copy to avoid modifying the original
        node = node.copy()
        
        # Deserialize metadata from JSON string
        if 'metadata' in node and isinstance(node['metadata'], str):
            node['metadata'] = json.loads(node['metadata'])
        
        # Convert ISO strings back to datetime objects
        if 'ingestion_timestamp' in node and isinstance(node['ingestion_timestamp'], str):
            node['ingestion_timestamp'] = datetime.fromisoformat(node['ingestion_timestamp'])
        if 'last_modified' in node and isinstance(node['last_modified'], str):
            node['last_modified'] = datetime.fromisoformat(node['last_modified'])
        
        # Remove Neo4J specific properties
        if '_labels' in node:
            node.pop('_labels')
            
        return cls(**node)