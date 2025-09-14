"""Relation model for graph-augmented RAG."""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, ClassVar

from pydantic import BaseModel, Field, field_validator

from .types import RelationId, RelationAttributes, EntityId
from ..utils.id_generation import UnifiedIDGenerator, IDValidator


class Relation(BaseModel):
    """Relation model representing an edge in the knowledge graph.
    
    A relation connects two entities and describes their relationship.
    Each relation has a unique ID, source and target entities, a type, and optional attributes.
    
    Attributes:
        id: Unique identifier for the relation
        source_entity_id: ID of the source entity
        target_entity_id: ID of the target entity
        type: Type of the relation (e.g., WORKS_FOR, LOCATED_IN)
        attributes: Additional attributes of the relation
        source_text: Original text from which the relation was extracted
        source_doc_id: ID of the document from which the relation was extracted
        confidence: Confidence score of the relation extraction (0.0 to 1.0)
        weight: Weight of the relation for graph algorithms (default: 1.0)
    """
    
    id: RelationId = Field(default="")
    source_entity_id: EntityId
    target_entity_id: EntityId
    type: str = Field(default="CUSTOM", description="Relation type")
    description: str = Field(default="", description="Description of the relation")
    context: str = Field(default="", description="Context where the relation was found")
    attributes: RelationAttributes = Field(default_factory=dict)
    source_doc_id: Optional[str] = None
    confidence: float = 1.0
    weight: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: Optional[datetime] = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(default_factory=datetime.now, description="Last update timestamp")
    
    # Class variables for Neo4J integration
    _neo4j_type: ClassVar[str] = "RELATION"
    
    def __init__(self, **data):
        """Initialize relation with unified ID generation."""
        # Generate unified relation ID if not provided
        if 'id' not in data or not data['id']:
            relation_type = str(data.get('type', "CUSTOM"))
            data['id'] = UnifiedIDGenerator.generate_relation_id(
                source_entity_id=data['source_entity_id'],
                target_entity_id=data['target_entity_id'],
                relation_type=relation_type
            )
        super().__init__(**data)
    
    @field_validator('id')
    @classmethod
    def validate_id_format(cls, v):
        """Validate relation ID format."""
        if v and not IDValidator.validate_relation_id(v):
            raise ValueError(f"Invalid relation ID format: {v}")
        return v

    @field_validator('type')
    @classmethod
    def validate_type_format(cls, v):
        """Validate relation type format."""
        if not v or not v.strip():
            raise ValueError("Relation type must be a non-empty string")
        return v.strip()
    
    def get_unified_id(self) -> str:
        """Get unified relation ID."""
        return self.id
    
    def is_unified_format(self) -> bool:
        """Check if using unified ID format."""
        return IDValidator.validate_relation_id(self.id)
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Validate that confidence is between 0.0 and 1.0."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {v}")
        return v
    
    @field_validator('weight')
    @classmethod
    def validate_weight(cls, v: float) -> float:
        """Validate that weight is positive."""
        if v <= 0.0:
            raise ValueError(f"Weight must be positive, got {v}")
        return v
    
    @field_validator('type')
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Validate relation type is a non-empty string."""
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Relation type must be a non-empty string")
        return v.strip()
    
    @field_validator('source_entity_id', 'target_entity_id')
    @classmethod
    def validate_entity_ids(cls, v: EntityId) -> EntityId:
        """Validate that entity IDs are not empty."""
        if not v or not v.strip():
            raise ValueError("Entity ID cannot be empty")
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert relation to dictionary for JSON serialization."""
        data = self.model_dump()

        # Ensure type is string for JSON serialization
        data['type'] = str(data['type'])

        return data
    
    def to_neo4j_relationship(self) -> Dict[str, Any]:
        """Convert relation to Neo4J relationship properties.

        Excludes unnecessary fields to reduce memory consumption:
        - context: Full chunk text (entities already refer to chunks)
        - source_doc_id: Document reference (entities already have chunk references)
        - description: Redundant with relation type
        """
        # Get the type value as string
        type_value = str(self.type)

        # Use model_dump for properties
        properties = self.model_dump()

        # Ensure type is clean string value
        properties['type'] = type_value

        # Serialize attributes to JSON string for Neo4J storage
        if 'attributes' in properties:
            properties['attributes'] = json.dumps(properties['attributes'])

        # Remove entity IDs as they are handled by Neo4J relationship structure
        properties.pop('source_entity_id', None)
        properties.pop('target_entity_id', None)

        # Remove unnecessary fields to reduce memory consumption
        properties.pop('context', None)  # Full chunk text - entities already refer to chunks
        properties.pop('source_doc_id', None)  # Document reference - entities have chunk references
        properties.pop('description', None)  # Redundant with relation type

        return properties
    
    @classmethod
    def from_neo4j_relationship(
        cls, 
        relationship: Any, 
        source_entity_id: EntityId, 
        target_entity_id: EntityId
    ) -> 'Relation':
        """Create relation from Neo4J relationship properties."""
        # Convert Neo4j relationship object to dictionary
        try:
            if hasattr(relationship, 'items'):
                # Neo4j relationship object - convert to dict
                relationship_dict = dict(relationship.items())
            elif isinstance(relationship, dict):
                # Already a dictionary - make a copy
                relationship_dict = relationship.copy()
            else:
                # Handle other types by converting to dict
                relationship_dict = dict(relationship)
        except (TypeError, ValueError) as e:
            # If conversion fails, create minimal relationship dict
            relationship_dict = {}
            
        # Ensure we have required fields with defaults
        if 'id' not in relationship_dict:
            relationship_dict['id'] = f"{source_entity_id}_{target_entity_id}_{hash(str(relationship))}"
        if 'type' not in relationship_dict:
            relationship_dict['type'] = 'RELATED_TO'
        if 'confidence' not in relationship_dict:
            relationship_dict['confidence'] = 1.0
        if 'source_text' not in relationship_dict:
            relationship_dict['source_text'] = ''
        
        # Deserialize attributes from JSON string
        if 'attributes' in relationship_dict and isinstance(relationship_dict['attributes'], str):
            try:
                relationship_dict['attributes'] = json.loads(relationship_dict['attributes'])
            except json.JSONDecodeError:
                relationship_dict['attributes'] = {}
        elif 'attributes' not in relationship_dict:
            relationship_dict['attributes'] = {}
        
        # Add entity IDs back
        relationship_dict['source_entity_id'] = source_entity_id
        relationship_dict['target_entity_id'] = target_entity_id
        
        return cls(**relationship_dict)
    
    def get_neo4j_type(self) -> str:
        """Get the Neo4J relationship type."""
        # Get the type value as string
        type_value = str(self.type)

        # Normalize relation type: uppercase, singular form, and sanitize
        normalized_type = self._normalize_relation_type(type_value)
        return normalized_type

    def _normalize_relation_type(self, relation_type: str) -> str:
        """Normalize relation type to uppercase singular form.

        Args:
            relation_type: Raw relation type string

        Returns:
            Normalized relation type (uppercase, singular, Neo4j-compatible)
        """
        if not relation_type:
            return "RELATES"

        # Convert to uppercase and clean whitespace
        normalized = relation_type.upper().strip()

        # Sanitize for valid Neo4j relationship type (no dots, spaces, special chars)
        normalized = normalized.replace('.', '_').replace(' ', '_').replace('-', '_')
        normalized = normalized.replace('(', '').replace(')', '').replace('/', '_')
        normalized = normalized.replace('&', '_AND_').replace('+', '_PLUS_')

        # Remove any double underscores
        while '__' in normalized:
            normalized = normalized.replace('__', '_')

        # Remove leading/trailing underscores
        normalized = normalized.strip('_')

        # Ensure the type is valid (starts with letter, contains only alphanumeric and underscore)
        if not normalized or not normalized[0].isalpha():
            normalized = f"REL_{normalized}" if normalized else "RELATES"

        return normalized