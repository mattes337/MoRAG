"""Entity model for graph-augmented RAG."""

import json
import uuid
import hashlib
from typing import Dict, List, Optional, Any, Union, ClassVar

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
    

    
    # Class variables for Neo4J integration
    _neo4j_label: ClassVar[str] = "Entity"
    
    def __init__(self, **data):
        """Initialize entity with unified deterministic ID based on name, type, and source document."""
        if 'id' not in data or not data['id']:
            # Generate unified deterministic ID based on name, type, and source document
            name = data.get('name', '')
            entity_type = str(data.get('type', "CUSTOM"))
            source_doc_id = data.get('source_doc_id', '')
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
            entity_type = str(self.type)
            super().__setattr__('id', UnifiedIDGenerator.generate_entity_id(self.name, entity_type, value or ''))
        else:
            super().__setattr__(name, value)
    
    def get_unified_id(self) -> str:
        """Get unified entity ID."""
        return self.id
    
    def is_unified_format(self) -> bool:
        """Check if using unified ID format."""
        return IDValidator.validate_entity_id(self.id)

    def _normalize_label(self, type_value: str) -> str:
        """Normalize entity type to valid Neo4j label format.

        Simple normalization that only ensures Neo4j compatibility:
        - Uppercase
        - Valid Neo4j label format (alphanumeric and underscore only)

        The LLM is responsible for generating appropriate entity types.

        Args:
            type_value: The entity type to normalize

        Returns:
            Normalized uppercase label suitable for Neo4j
        """
        if not type_value:
            return "UNKNOWN"

        # Convert to uppercase and clean whitespace
        normalized = type_value.upper().strip()

        # Sanitize for valid Neo4j label (no dots, spaces, special chars)
        normalized = normalized.replace('.', '_').replace(' ', '_').replace('-', '_')
        normalized = normalized.replace('(', '').replace(')', '').replace('/', '_')
        normalized = normalized.replace('&', '_AND_').replace('+', '_PLUS_')

        # Remove any double underscores
        while '__' in normalized:
            normalized = normalized.replace('__', '_')

        # Remove leading/trailing underscores
        normalized = normalized.strip('_')

        # Ensure the label is valid (starts with letter, contains only alphanumeric and underscore)
        if not normalized or not normalized[0].isalpha():
            normalized = f"TYPE_{normalized}" if normalized else "UNKNOWN"

        return normalized
    

    
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

        # Ensure type is string for JSON serialization
        data['type'] = str(data['type'])

        return data
    
    def to_neo4j_node(self) -> Dict[str, Any]:
        """Convert entity to Neo4J node properties.

        Excludes unnecessary fields to simplify graph structure:
        - attributes: Used for auto creation, not needed in graph
        - source_doc_id: Document reference, entities connect via chunks
        - mentioned_in_chunks: Handled via relationships, not node properties
        - description: Redundant with entity type and name
        """
        # Get the type value as string
        type_value = str(self.type)

        # Use model_dump for properties
        properties = self.model_dump()

        # Ensure type is clean string value
        properties['type'] = type_value

        # Remove unnecessary fields to simplify graph structure
        properties.pop('attributes', None)  # Used for auto creation, not needed in graph
        properties.pop('source_doc_id', None)  # Document reference, entities connect via chunks
        properties.pop('description', None)  # Redundant with entity type and name

        # Add label for Neo4J - use only the LLM-determined type as label
        # Normalize to uppercase singular canonical form
        type_label = self._normalize_label(type_value)

        properties['_labels'] = [type_label]

        return properties
    
    @classmethod
    def from_neo4j_node(cls, node: Dict[str, Any]) -> 'Entity':
        """Create entity from Neo4J node properties."""
        # Make a copy to avoid modifying the original
        node = node.copy()
        
        # Deserialize attributes from JSON string
        if 'attributes' in node and isinstance(node['attributes'], str):
            node['attributes'] = json.loads(node['attributes'])
        
        # Remove Neo4J specific properties
        if '_labels' in node:
            node.pop('_labels')
            
        return cls(**node)