"""Data models for multi-hop reasoning."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field


class ReasoningStrategy(str, Enum):
    """Available reasoning strategies."""
    FORWARD_CHAINING = "forward_chaining"
    BACKWARD_CHAINING = "backward_chaining"
    BIDIRECTIONAL = "bidirectional"


class ContextGapType(str, Enum):
    """Types of context gaps."""
    MISSING_ENTITY = "missing_entity"
    MISSING_RELATION = "missing_relation"
    INSUFFICIENT_DETAIL = "insufficient_detail"


class MultiHopQuery(BaseModel):
    """Multi-hop reasoning query."""
    query: str = Field(..., description="The query to answer")
    start_entities: List[str] = Field(..., description="Starting entities for reasoning")
    target_entities: Optional[List[str]] = Field(None, description="Optional target entities")
    strategy: ReasoningStrategy = Field(ReasoningStrategy.FORWARD_CHAINING, description="Reasoning strategy")
    max_depth: int = Field(4, ge=1, le=10, description="Maximum reasoning depth")
    max_paths: int = Field(50, ge=1, le=100, description="Maximum paths to discover")


class PathScore(BaseModel):
    """Path relevance score."""
    path_id: str = Field(..., description="Unique path identifier")
    relevance_score: float = Field(..., ge=0.0, le=10.0, description="Relevance score (0-10)")
    confidence: float = Field(..., ge=0.0, le=10.0, description="Confidence in score (0-10)")
    reasoning: str = Field(..., description="Reasoning for the score")


class ContextGap(BaseModel):
    """Context gap information."""
    gap_type: ContextGapType = Field(..., description="Type of gap")
    description: str = Field(..., description="Description of the gap")
    entities_needed: List[str] = Field(default_factory=list, description="Entities needed to fill gap")
    relations_needed: List[str] = Field(default_factory=list, description="Relations needed to fill gap")
    priority: float = Field(1.0, ge=0.0, le=1.0, description="Priority for filling this gap")


class ContextAnalysis(BaseModel):
    """Analysis of context sufficiency."""
    is_sufficient: bool = Field(..., description="Whether context is sufficient")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in sufficiency assessment")
    gaps: List[ContextGap] = Field(default_factory=list, description="Identified context gaps")
    reasoning: str = Field(..., description="Reasoning for the assessment")
    suggested_queries: List[str] = Field(default_factory=list, description="Suggested queries to fill gaps")


class ReasoningResult(BaseModel):
    """Result of multi-hop reasoning."""
    query: str = Field(..., description="Original query")
    paths_found: int = Field(..., description="Number of reasoning paths found")
    selected_paths: int = Field(..., description="Number of paths selected")
    iterations_used: int = Field(..., description="Number of refinement iterations used")
    final_confidence: float = Field(..., ge=0.0, le=1.0, description="Final confidence in result")
    reasoning_time_ms: float = Field(..., description="Total reasoning time in milliseconds")
    context_sufficient: bool = Field(..., description="Whether final context was sufficient")


class ReasoningConfig(BaseModel):
    """Configuration for multi-hop reasoning."""
    max_iterations: int = Field(5, ge=1, le=10, description="Maximum refinement iterations")
    sufficiency_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Confidence threshold for stopping")
    max_paths_to_select: int = Field(10, ge=1, le=50, description="Maximum paths to select")
    llm_temperature: float = Field(0.1, ge=0.0, le=2.0, description="LLM temperature for reasoning")
    enable_fallback: bool = Field(True, description="Enable fallback mechanisms")


class ReasoningMetrics(BaseModel):
    """Metrics for reasoning performance."""
    total_queries: int = Field(0, description="Total queries processed")
    successful_queries: int = Field(0, description="Successfully answered queries")
    average_paths_found: float = Field(0.0, description="Average paths found per query")
    average_iterations: float = Field(0.0, description="Average iterations per query")
    average_response_time_ms: float = Field(0.0, description="Average response time in milliseconds")
    success_rate: float = Field(0.0, ge=0.0, le=1.0, description="Success rate (0-1)")


@dataclass
class ReasoningContext:
    """Internal context for reasoning operations."""
    entities: Dict[str, Any] = field(default_factory=dict)
    relations: List[Dict[str, Any]] = field(default_factory=list)
    documents: List[Dict[str, Any]] = field(default_factory=list)
    paths: List[Any] = field(default_factory=list)  # GraphPath objects
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "entities": self.entities,
            "relations": self.relations,
            "documents": self.documents,
            "paths": [path.to_dict() if hasattr(path, 'to_dict') else str(path) for path in self.paths],
            "metadata": self.metadata
        }

    def entity_count(self) -> int:
        """Get number of entities in context."""
        return len(self.entities)

    def relation_count(self) -> int:
        """Get number of relations in context."""
        return len(self.relations)

    def document_count(self) -> int:
        """Get number of documents in context."""
        return len(self.documents)

    def path_count(self) -> int:
        """Get number of paths in context."""
        return len(self.paths)
