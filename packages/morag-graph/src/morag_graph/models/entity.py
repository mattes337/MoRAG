"""Entity model for graph-augmented RAG."""

import json
import uuid
import hashlib
from typing import Dict, List, Optional, Any, Union, ClassVar

from pydantic import BaseModel, Field, field_validator

from .types import EntityType, EntityId, EntityAttributes


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
    type: Union[EntityType, str] = EntityType.CUSTOM
    attributes: EntityAttributes = Field(default_factory=dict)
    source_doc_id: Optional[str] = None
    confidence: float = 1.0
    
    # Class variables for Neo4J integration
    _neo4j_label: ClassVar[str] = "Entity"
    
    def __init__(self, **data):
        """Initialize entity with deterministic ID based on name and type."""
        if 'id' not in data or not data['id']:
            # Generate deterministic ID based on name and type
            name = data.get('name', '')
            entity_type = data.get('type', EntityType.CUSTOM)
            if isinstance(entity_type, EntityType):
                entity_type = entity_type.value
            data['id'] = self._generate_deterministic_id(name, entity_type)
        super().__init__(**data)
    
    @staticmethod
    def _generate_deterministic_id(name: str, entity_type: str) -> str:
        """Generate a deterministic ID based on entity name and type.
        
        This ensures that entities with the same name and type always get
        the same ID, preventing duplicate nodes in the graph.
        """
        # Normalize inputs for consistent hashing
        normalized_name = name.strip().lower()
        normalized_type = str(entity_type).strip().lower()
        
        # Create a deterministic hash
        content = f"{normalized_name}:{normalized_type}"
        hash_object = hashlib.sha256(content.encode('utf-8'))
        return hash_object.hexdigest()[:32]  # Use first 32 characters for readability
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Validate that confidence is between 0.0 and 1.0."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {v}")
        return v
    
    @field_validator('type')
    @classmethod
    def validate_type(cls, v: Union[EntityType, str]) -> Union[EntityType, str]:
        """Convert string type to EntityType enum if possible."""
        if isinstance(v, str) and v in [e.value for e in EntityType]:
            return EntityType(v)
        return v
    
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
        return self.model_dump()
    
    def to_neo4j_node(self) -> Dict[str, Any]:
        """Convert entity to Neo4J node properties."""
        properties = self.model_dump()
        
        # Convert type to string for Neo4J
        if isinstance(properties['type'], EntityType):
            properties['type'] = properties['type'].value
            
        # Serialize attributes to JSON string for Neo4J storage
        if 'attributes' in properties:
            properties['attributes'] = json.dumps(properties['attributes'])
            
        # Add label for Neo4J
        properties['_labels'] = [self._neo4j_label, properties['type']]
        
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