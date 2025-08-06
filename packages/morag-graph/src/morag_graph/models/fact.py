"""Fact model for structured knowledge extraction."""

import hashlib
from datetime import datetime
from pathlib import Path
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
    condition: Optional[str] = Field(default=None, description="Question/precondition/situation when this fact applies")
    remarks: Optional[str] = Field(default=None, description="Additional context/qualifications")
    
    # Provenance
    source_chunk_id: str = Field(..., description="Source document chunk ID")
    source_document_id: str = Field(..., description="Source document ID")
    extraction_confidence: float = Field(ge=0.0, le=1.0, description="Confidence in extraction")

    # Detailed source metadata for reconstruction and citation
    source_file_path: Optional[str] = Field(default=None, description="Original file path")
    source_file_name: Optional[str] = Field(default=None, description="Original file name")
    page_number: Optional[int] = Field(default=None, description="Page number in document")
    chapter_title: Optional[str] = Field(default=None, description="Chapter or section title")
    chapter_index: Optional[int] = Field(default=None, description="Chapter or section index")
    paragraph_index: Optional[int] = Field(default=None, description="Paragraph index within chunk")
    timestamp_start: Optional[float] = Field(default=None, description="Start timestamp for audio/video (seconds)")
    timestamp_end: Optional[float] = Field(default=None, description="End timestamp for audio/video (seconds)")
    topic_header: Optional[str] = Field(default=None, description="Topic header for audio/video content")
    speaker_label: Optional[str] = Field(default=None, description="Speaker label for audio/video content")
    source_text_excerpt: Optional[str] = Field(default=None, description="Exact text excerpt from source")

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
            "condition": self.condition,
            "remarks": self.remarks,
            "fact_type": self.fact_type,
            "domain": self.domain,
            "confidence": self.extraction_confidence,
            "language": self.language,
            "created_at": self.created_at.isoformat(),
            "keywords": ",".join(self.keywords) if self.keywords else "",
            "source_chunk_id": self.source_chunk_id,
            "source_document_id": self.source_document_id,
            # Detailed source metadata
            "source_file_path": self.source_file_path,
            "source_file_name": self.source_file_name,
            "page_number": self.page_number,
            "chapter_title": self.chapter_title,
            "chapter_index": self.chapter_index,
            "paragraph_index": self.paragraph_index,
            "timestamp_start": self.timestamp_start,
            "timestamp_end": self.timestamp_end,
            "topic_header": self.topic_header,
            "speaker_label": self.speaker_label,
            "source_text_excerpt": self.source_text_excerpt
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
            "language": self.language,
            # Detailed source metadata
            "source_file_path": self.source_file_path,
            "source_file_name": self.source_file_name,
            "page_number": self.page_number,
            "chapter_title": self.chapter_title,
            "chapter_index": self.chapter_index,
            "paragraph_index": self.paragraph_index,
            "timestamp_start": self.timestamp_start,
            "timestamp_end": self.timestamp_end,
            "topic_header": self.topic_header,
            "speaker_label": self.speaker_label,
            "source_text_excerpt": self.source_text_excerpt
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

    def get_citation(self) -> str:
        """Generate a citation string for this fact.

        Returns:
            Formatted citation string with source attribution
        """
        citation_parts = []

        # Add file name if available
        if self.source_file_name:
            citation_parts.append(self.source_file_name)
        elif self.source_file_path:
            citation_parts.append(Path(self.source_file_path).name)

        # Add page number if available
        if self.page_number:
            citation_parts.append(f"page {self.page_number}")

        # Add chapter if available
        if self.chapter_title:
            citation_parts.append(f"chapter '{self.chapter_title}'")
        elif self.chapter_index is not None:
            citation_parts.append(f"chapter {self.chapter_index}")

        # Add timestamp for audio/video
        if self.timestamp_start is not None:
            if self.timestamp_end is not None:
                citation_parts.append(f"timestamp {self.timestamp_start:.1f}-{self.timestamp_end:.1f}s")
            else:
                citation_parts.append(f"timestamp {self.timestamp_start:.1f}s")

        # Add topic header for audio/video
        if self.topic_header:
            citation_parts.append(f"topic '{self.topic_header}'")

        # Add speaker if available
        if self.speaker_label:
            citation_parts.append(f"speaker {self.speaker_label}")

        # Add chunk reference as fallback
        if not citation_parts:
            citation_parts.append(f"chunk {self.source_chunk_id}")

        return " | ".join(citation_parts)

    def get_machine_readable_source(self) -> str:
        """Generate machine-readable source format for API responses.

        Returns:
            Machine-readable source string in format [filename:chunk_index:topic]
        """
        parts = []

        # File name
        if self.source_file_name:
            parts.append(self.source_file_name)
        elif self.source_file_path:
            parts.append(Path(self.source_file_path).name)
        else:
            parts.append(self.source_document_id)

        # Chunk or page reference
        if self.page_number:
            parts.append(f"page_{self.page_number}")
        elif self.chapter_index is not None:
            parts.append(f"chapter_{self.chapter_index}")
        else:
            # Extract chunk index from chunk_id if possible
            chunk_parts = self.source_chunk_id.split('_')
            if len(chunk_parts) > 1 and chunk_parts[-1].isdigit():
                parts.append(f"chunk_{chunk_parts[-1]}")
            else:
                parts.append("chunk_0")

        # Topic or timestamp
        if self.topic_header:
            parts.append(self.topic_header.replace(':', '_').replace(' ', '_'))
        elif self.timestamp_start is not None:
            parts.append(f"t_{int(self.timestamp_start)}")
        else:
            parts.append("content")

        return "[" + ":".join(parts) + "]"


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
        try:
            # Validate that data is a dictionary
            if not isinstance(data, dict):
                raise ValueError(f"FactRelation data must be a dictionary, got {type(data)}: {data}")

            if 'id' not in data or not data['id']:
                # Generate deterministic ID based on content
                source_id = data.get('source_fact_id', '')
                target_id = data.get('target_fact_id', '')
                relation_type = data.get('relation_type', '')

                content_for_hash = f"{source_id}{target_id}{relation_type}"
                content_hash = hashlib.md5(content_for_hash.encode()).hexdigest()[:12]
                data['id'] = f"fact_rel_{content_hash}"

            super().__init__(**data)

        except Exception as e:
            # Add detailed error information for debugging
            import structlog
            logger = structlog.get_logger(__name__)
            logger.error(
                "FactRelation initialization failed",
                data_type=type(data),
                data_content=str(data)[:200] if data else "None",
                error=str(e),
                error_type=type(e).__name__
            )
            raise
    
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
