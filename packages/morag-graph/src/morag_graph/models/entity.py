"""Entity model for graph-augmented RAG."""

import json
import uuid
import hashlib
from typing import Dict, List, Optional, Any, Union, ClassVar, Set

from pydantic import BaseModel, Field, field_validator

from .types import EntityId, EntityAttributes
from ..utils.id_generation import UnifiedIDGenerator, IDValidator


class Entity(BaseModel):
    """Entity model representing a node in the knowledge graph.
    
    An entity can be a person, organization, location, concept, etc.
    Each entity has a unique ID, a name, a type, and optional attributes.
    Entities are global and can be referenced from multiple documents through
    DocumentChunk -> MENTIONS -> Entity relationships.
    
    Attributes:
        id: Unique identifier for the entity
        name: Human-readable name of the entity
        type: Type of the entity (e.g., PERSON, ORGANIZATION)
        attributes: Additional attributes of the entity
        confidence: Confidence score of the entity extraction (0.0 to 1.0)
    """
    
    id: EntityId = Field(default="")
    name: str
    type: str = "CUSTOM"
    description: str = Field(default="", description="Description of the entity")
    attributes: EntityAttributes = Field(default_factory=dict)
    source_doc_id: Optional[str] = None
    confidence: float = 1.0
    
    # Cross-system integration fields
    mentioned_in_chunks: Set[str] = Field(default_factory=set, description="Set of chunk IDs where this entity is mentioned")
    qdrant_vector_ids: Set[str] = Field(default_factory=set, description="Set of Qdrant vector IDs associated with this entity")
    
    # Class variables for Neo4J integration
    _neo4j_label: ClassVar[str] = "Entity"
    
    def __init__(self, **data):
        """Initialize entity with unified deterministic ID based on name, type, and source document."""
        if 'id' not in data or not data['id']:
            # Generate unified deterministic ID based on name, type, and source document
            name = data.get('name', '')
            entity_type = data.get('type', "CUSTOM")
            source_doc_id = data.get('source_doc_id', '')
            if hasattr(entity_type, 'value'):
                entity_type = entity_type.value
            data['id'] = UnifiedIDGenerator.generate_entity_id(name, entity_type, source_doc_id)
        super().__init__(**data)
    
    @field_validator('id')
    @classmethod
    def validate_id_format(cls, v):
        """Validate entity ID format."""
        if v and not IDValidator.validate_entity_id(v):
            raise ValueError(f"Invalid entity ID format: {v}")
        return v
    
    def __setattr__(self, name, value):
        """Override setattr to regenerate ID when source_doc_id changes."""
        if name == 'source_doc_id' and hasattr(self, 'source_doc_id') and self.source_doc_id != value:
            # Set the new value first
            super().__setattr__(name, value)
            # Regenerate ID with new source_doc_id using unified generator
            entity_type = self.type
            if hasattr(entity_type, 'value'):
                entity_type = entity_type.value
            super().__setattr__('id', UnifiedIDGenerator.generate_entity_id(self.name, entity_type, value or ''))
        else:
            super().__setattr__(name, value)
    
    def get_unified_id(self) -> str:
        """Get unified entity ID."""
        return self.id
    
    def is_unified_format(self) -> bool:
        """Check if using unified ID format."""
        return IDValidator.validate_entity_id(self.id)
    
    def add_chunk_reference(self, chunk_id: str) -> None:
        """Add a reference to a chunk where this entity is mentioned.
        
        Args:
            chunk_id: Unified chunk ID
        """
        if not IDValidator.validate_chunk_id(chunk_id):
            raise ValueError(f"Invalid chunk ID format: {chunk_id}")
        self.mentioned_in_chunks.add(chunk_id)
    
    def remove_chunk_reference(self, chunk_id: str) -> None:
        """Remove a reference to a chunk.
        
        Args:
            chunk_id: Unified chunk ID
        """
        self.mentioned_in_chunks.discard(chunk_id)
    
    def add_qdrant_vector_id(self, vector_id: str) -> None:
        """Add a Qdrant vector ID associated with this entity.
        
        Args:
            vector_id: Qdrant vector ID
        """
        self.qdrant_vector_ids.add(vector_id)
    
    def remove_qdrant_vector_id(self, vector_id: str) -> None:
        """Remove a Qdrant vector ID.
        
        Args:
            vector_id: Qdrant vector ID
        """
        self.qdrant_vector_ids.discard(vector_id)
    
    def get_document_ids_from_chunks(self) -> Set[str]:
        """Get unique document IDs from all referenced chunks.
        
        Returns:
            Set of document IDs
        """
        document_ids = set()
        for chunk_id in self.mentioned_in_chunks:
            try:
                doc_id = UnifiedIDGenerator.extract_document_id_from_chunk(chunk_id)
                document_ids.add(doc_id)
            except ValueError:
                # Skip invalid chunk IDs
                continue
        return document_ids
    
    def get_chunk_count(self) -> int:
        """Get the number of chunks this entity is mentioned in.
        
        Returns:
            Number of chunks
        """
        return len(self.mentioned_in_chunks)
    
    def get_qdrant_vector_count(self) -> int:
        """Get the number of Qdrant vectors associated with this entity.
        
        Returns:
            Number of Qdrant vectors
        """
        return len(self.qdrant_vector_ids)
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Validate that confidence is between 0.0 and 1.0."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {v}")
        return v
    
    @field_validator('type')
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Validate entity type is a non-empty string."""
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Entity type must be a non-empty string")
        return v.strip()
    
    def __hash__(self) -> int:
        """Make Entity hashable based on its ID."""
        return hash(self.id)
    
    def __eq__(self, other) -> bool:
        """Compare entities based on their ID."""
        if not isinstance(other, Entity):
            return False
        return self.id == other.id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary for JSON serialization."""
        data = self.model_dump()

        # Convert type to string for JSON serialization
        if hasattr(data['type'], 'value'):
            # Handle enum types - get just the value without the class prefix
            data['type'] = data['type'].value
        else:
            # Handle string types
            data['type'] = str(data['type'])

        # Convert sets to lists for JSON serialization
        if 'mentioned_in_chunks' in data and isinstance(data['mentioned_in_chunks'], set):
            data['mentioned_in_chunks'] = list(data['mentioned_in_chunks'])

        if 'qdrant_vector_ids' in data and isinstance(data['qdrant_vector_ids'], set):
            data['qdrant_vector_ids'] = list(data['qdrant_vector_ids'])

        return data
    
    def to_neo4j_node(self) -> Dict[str, Any]:
        """Convert entity to Neo4J node properties."""
        # Get the clean type value - always a string now
        type_value = str(self.type)
        # Clean up any enum string representations that might still exist
        if type_value.startswith('EntityType.'):
            type_value = type_value.replace('EntityType.', '')

        # Use model_dump but manually handle enum serialization
        properties = self.model_dump()

        # Force the type to be the clean value (override any enum serialization)
        properties['type'] = type_value

        # Serialize attributes to JSON string for Neo4J storage
        if 'attributes' in properties:
            properties['attributes'] = json.dumps(properties['attributes'])

        # Convert sets to lists for Neo4J storage
        if 'mentioned_in_chunks' in properties:
            properties['mentioned_in_chunks'] = list(properties['mentioned_in_chunks'])

        if 'qdrant_vector_ids' in properties:
            properties['qdrant_vector_ids'] = list(properties['qdrant_vector_ids'])

        # Add label for Neo4J (sanitize type for valid Neo4j label)
        # Use only the clean type value without any prefixes
        type_label = type_value.replace('.', '_').replace(' ', '_').replace('-', '_')

        properties['_labels'] = [self._neo4j_label, type_label]

        return properties
    
    @classmethod
    def from_neo4j_node(cls, node: Dict[str, Any]) -> 'Entity':
        """Create entity from Neo4J node properties."""
        # Make a copy to avoid modifying the original
        node = node.copy()
        
        # Deserialize attributes from JSON string
        if 'attributes' in node and isinstance(node['attributes'], str):
            node['attributes'] = json.loads(node['attributes'])
        
        # Convert lists back to sets
        if 'mentioned_in_chunks' in node and isinstance(node['mentioned_in_chunks'], list):
            node['mentioned_in_chunks'] = set(node['mentioned_in_chunks'])
        
        if 'qdrant_vector_ids' in node and isinstance(node['qdrant_vector_ids'], list):
            node['qdrant_vector_ids'] = set(node['qdrant_vector_ids'])
        
        # Remove Neo4J specific properties
        if '_labels' in node:
            node.pop('_labels')
            
        return cls(**node)