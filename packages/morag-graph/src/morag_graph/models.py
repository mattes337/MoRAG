"""Graph models for entities and relations."""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from datetime import datetime


@dataclass
class Entity:
    """Represents an entity in the knowledge graph."""
    
    id: str
    name: str
    type: str
    properties: Dict[str, Any]
    confidence: float = 1.0
    source_document: Optional[str] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class Relation:
    """Represents a relation between entities."""
    
    id: str
    source_id: str
    target_id: str
    type: str
    properties: Dict[str, Any]
    confidence: float = 1.0
    source_document: Optional[str] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class GraphNode:
    """Generic graph node."""
    
    id: str
    label: str
    properties: Dict[str, Any]
    
    
@dataclass
class GraphEdge:
    """Generic graph edge."""
    
    source: str
    target: str
    type: str
    properties: Dict[str, Any]


class GraphSchema:
    """Schema definition for the knowledge graph."""
    
    def __init__(self):
        self.entity_types: List[str] = []
        self.relation_types: List[str] = []
        self.constraints: Dict[str, Any] = {}
    
    def add_entity_type(self, entity_type: str):
        """Add an entity type to the schema."""
        if entity_type not in self.entity_types:
            self.entity_types.append(entity_type)
    
    def add_relation_type(self, relation_type: str):
        """Add a relation type to the schema."""
        if relation_type not in self.relation_types:
            self.relation_types.append(relation_type)
    
    def validate_entity(self, entity: Entity) -> bool:
        """Validate an entity against the schema."""
        return entity.type in self.entity_types if self.entity_types else True
    
    def validate_relation(self, relation: Relation) -> bool:
        """Validate a relation against the schema."""
        return relation.type in self.relation_types if self.relation_types else True
