"""Fact model for structured knowledge extraction."""

import hashlib
from datetime import datetime
from typing import Optional, List, Dict, Any, ClassVar
from pydantic import BaseModel, Field

from .types import FactId


class Fact(BaseModel):
    """Structured fact extracted from document content.
    
    A fact represents actionable, specific knowledge that can be used to answer
    questions. Unlike generic entities, facts contain structured information
    with clear subject-object relationships and optional approach/solution details.
    
    Attributes:
        id: Unique fact identifier
        subject: What the fact is about (main entity or concept)
        object: What is being described, studied, or acted upon
        approach: How something is done or achieved (optional)
        solution: What solves a problem or achieves a goal (optional)
        remarks: Additional context, limitations, or qualifications (optional)
        source_chunk_id: Source document chunk ID for provenance
        source_document_id: Source document ID for provenance
        extraction_confidence: Confidence in the extraction (0.0 to 1.0)
        fact_type: Type of fact (research, process, definition, etc.)
        domain: Domain or topic area (optional)
        keywords: Key terms for indexing and search
        created_at: Extraction timestamp
        language: Language of the fact content
    """
    
    id: FactId = Field(default="", description="Unique fact identifier")
    subject: str = Field(..., description="What the fact is about")
    object: str = Field(..., description="What is being described or acted upon")
    approach: Optional[str] = Field(default=None, description="How something is done/achieved")
    solution: Optional[str] = Field(default=None, description="What solves a problem/achieves goal")
    remarks: Optional[str] = Field(default=None, description="Additional context/qualifications")
    
    # Provenance
    source_chunk_id: str = Field(..., description="Source document chunk ID")
    source_document_id: str = Field(..., description="Source document ID")
    extraction_confidence: float = Field(ge=0.0, le=1.0, description="Confidence in extraction")
    
    # Classification
    fact_type: str = Field(..., description="Type of fact (research, process, definition, etc.)")
    domain: Optional[str] = Field(default=None, description="Domain/topic area")
    keywords: List[str] = Field(default_factory=list, description="Key terms for indexing")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Extraction timestamp")
    language: str = Field(default="en", description="Language of the fact")
    
    # Class variables for Neo4j integration
    _neo4j_label: ClassVar[str] = "Fact"
    
    def __init__(self, **data):
        """Initialize fact with auto-generated ID if not provided."""
        if 'id' not in data or not data['id']:
            # Generate deterministic ID based on content
            content_for_hash = f"{data.get('subject', '')}{data.get('object', '')}{data.get('source_chunk_id', '')}"
            content_hash = hashlib.md5(content_for_hash.encode()).hexdigest()[:12]
            data['id'] = f"fact_{content_hash}"
        super().__init__(**data)
    
    def get_neo4j_properties(self) -> Dict[str, Any]:
        """Get properties for Neo4j storage.
        
        Returns:
            Dictionary of properties suitable for Neo4j node creation
        """
        return {
            "id": self.id,
            "subject": self.subject,
            "object": self.object,
            "approach": self.approach,
            "solution": self.solution,
            "remarks": self.remarks,
            "fact_type": self.fact_type,
            "domain": self.domain,
            "confidence": self.extraction_confidence,
            "language": self.language,
            "created_at": self.created_at.isoformat(),
            "keywords": ",".join(self.keywords) if self.keywords else "",
            "source_chunk_id": self.source_chunk_id,
            "source_document_id": self.source_document_id
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert fact to dictionary representation.
        
        Returns:
            Dictionary representation of the fact
        """
        return {
            "id": self.id,
            "subject": self.subject,
            "object": self.object,
            "approach": self.approach,
            "solution": self.solution,
            "remarks": self.remarks,
            "source_chunk_id": self.source_chunk_id,
            "source_document_id": self.source_document_id,
            "extraction_confidence": self.extraction_confidence,
            "fact_type": self.fact_type,
            "domain": self.domain,
            "keywords": self.keywords,
            "created_at": self.created_at.isoformat(),
            "language": self.language
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Fact':
        """Create fact from dictionary representation.
        
        Args:
            data: Dictionary containing fact data
            
        Returns:
            Fact instance
        """
        # Handle datetime conversion
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        return cls(**data)
    
    def get_display_text(self) -> str:
        """Get human-readable display text for the fact.
        
        Returns:
            Formatted text representation of the fact
        """
        parts = [f"Subject: {self.subject}", f"Object: {self.object}"]
        
        if self.approach:
            parts.append(f"Approach: {self.approach}")
        if self.solution:
            parts.append(f"Solution: {self.solution}")
        if self.remarks:
            parts.append(f"Remarks: {self.remarks}")
            
        return " | ".join(parts)
    
    def is_complete(self) -> bool:
        """Check if fact has minimum required information.
        
        Returns:
            True if fact has subject, object, and at least one of approach/solution
        """
        return bool(
            self.subject and 
            self.object and 
            (self.approach or self.solution)
        )
    
    def get_search_text(self) -> str:
        """Get text for full-text search indexing.
        
        Returns:
            Combined text of all fact components for search
        """
        components = [self.subject, self.object]
        
        if self.approach:
            components.append(self.approach)
        if self.solution:
            components.append(self.solution)
        if self.remarks:
            components.append(self.remarks)
        if self.keywords:
            components.extend(self.keywords)
            
        return " ".join(components)


class FactRelation(BaseModel):
    """Relationship between two facts.
    
    Represents semantic relationships between facts such as support,
    contradiction, elaboration, or sequence.
    """
    
    id: str = Field(..., description="Unique relation identifier")
    source_fact_id: FactId = Field(..., description="Source fact ID")
    target_fact_id: FactId = Field(..., description="Target fact ID")
    relation_type: str = Field(..., description="Type of relationship")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the relationship")
    context: Optional[str] = Field(default=None, description="Context explaining the relationship")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    
    # Class variables for Neo4j integration
    _neo4j_label: ClassVar[str] = "FACT_RELATION"
    
    def __init__(self, **data):
        """Initialize relation with auto-generated ID if not provided."""
        if 'id' not in data or not data['id']:
            # Generate deterministic ID based on content
            content_for_hash = f"{data.get('source_fact_id', '')}{data.get('target_fact_id', '')}{data.get('relation_type', '')}"
            content_hash = hashlib.md5(content_for_hash.encode()).hexdigest()[:12]
            data['id'] = f"fact_rel_{content_hash}"
        super().__init__(**data)
    
    def get_neo4j_properties(self) -> Dict[str, Any]:
        """Get properties for Neo4j relationship storage.
        
        Returns:
            Dictionary of properties suitable for Neo4j relationship creation
        """
        return {
            "id": self.id,
            "relation_type": self.relation_type,
            "confidence": self.confidence,
            "context": self.context,
            "created_at": self.created_at.isoformat()
        }


# Fact type constants
class FactType:
    """Constants for fact types."""
    
    RESEARCH = "research"
    PROCESS = "process"
    DEFINITION = "definition"
    CAUSAL = "causal"
    COMPARATIVE = "comparative"
    TEMPORAL = "temporal"
    STATISTICAL = "statistical"
    METHODOLOGICAL = "methodological"
    
    @classmethod
    def all_types(cls) -> List[str]:
        """Get all available fact types."""
        return [
            cls.RESEARCH, cls.PROCESS, cls.DEFINITION, cls.CAUSAL,
            cls.COMPARATIVE, cls.TEMPORAL, cls.STATISTICAL, cls.METHODOLOGICAL
        ]


# Fact relation type constants
class FactRelationType:
    """Constants for fact relationship types."""
    
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    ELABORATES = "ELABORATES"
    SEQUENCE = "SEQUENCE"
    COMPARISON = "COMPARISON"
    CAUSATION = "CAUSATION"
    TEMPORAL_ORDER = "TEMPORAL_ORDER"
    
    @classmethod
    def all_types(cls) -> List[str]:
        """Get all available fact relation types."""
        return [
            cls.SUPPORTS, cls.CONTRADICTS, cls.ELABORATES, cls.SEQUENCE,
            cls.COMPARISON, cls.CAUSATION, cls.TEMPORAL_ORDER
        ]
