"""Enhanced query models for graph-augmented RAG."""

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class QueryType(str, Enum):
    """Types of queries supported by the enhanced API."""
    SIMPLE = "simple"
    ENTITY_FOCUSED = "entity_focused"
    RELATION_FOCUSED = "relation_focused"
    MULTI_HOP = "multi_hop"
    ANALYTICAL = "analytical"


class ExpansionStrategy(str, Enum):
    """Strategies for expanding query context using graph traversal."""
    DIRECT_NEIGHBORS = "direct_neighbors"
    BREADTH_FIRST = "breadth_first"
    SHORTEST_PATH = "shortest_path"
    ADAPTIVE = "adaptive"
    NONE = "none"


class FusionStrategy(str, Enum):
    """Strategies for fusing vector and graph retrieval results."""
    WEIGHTED = "weighted"
    RRF = "reciprocal_rank_fusion"
    ADAPTIVE = "adaptive"
    VECTOR_ONLY = "vector_only"
    GRAPH_ONLY = "graph_only"


class EnhancedQueryRequest(BaseModel):
    """Enhanced query request with graph-augmented capabilities."""
    query: str = Field(..., description="The user's query text")
    query_type: QueryType = Field(default=QueryType.SIMPLE, description="Type of query processing")
    max_results: int = Field(default=10, ge=1, le=100, description="Maximum number of results")
    
    # Graph-specific parameters
    expansion_strategy: ExpansionStrategy = Field(default=ExpansionStrategy.ADAPTIVE, description="Context expansion strategy")
    expansion_depth: int = Field(default=2, ge=1, le=5, description="Maximum expansion depth")
    fusion_strategy: FusionStrategy = Field(default=FusionStrategy.ADAPTIVE, description="Result fusion strategy")
    
    # Filtering and constraints
    entity_types: Optional[List[str]] = Field(default=None, description="Filter by entity types")
    relation_types: Optional[List[str]] = Field(default=None, description="Filter by relation types")
    time_range: Optional[Dict[str, datetime]] = Field(default=None, description="Time range filter")
    
    # Advanced options
    include_graph_context: bool = Field(default=True, description="Include graph context in response")
    include_reasoning_path: bool = Field(default=False, description="Include reasoning path")
    enable_multi_hop: bool = Field(default=True, description="Enable multi-hop reasoning")
    
    # Quality and performance
    min_relevance_score: float = Field(default=0.1, ge=0.0, le=1.0, description="Minimum relevance threshold")
    timeout_seconds: int = Field(default=30, ge=1, le=300, description="Query timeout")


class EntityInfo(BaseModel):
    """Information about an entity in the graph context."""
    id: str
    name: str
    type: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    relevance_score: float = Field(ge=0.0, le=1.0)
    source_documents: List[str] = Field(default_factory=list)


class RelationInfo(BaseModel):
    """Information about a relation in the graph context."""
    id: str
    source_entity: str
    target_entity: str
    relation_type: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    source_documents: List[str] = Field(default_factory=list)


class GraphContext(BaseModel):
    """Graph context information for enhanced query results."""
    entities: Dict[str, EntityInfo] = Field(default_factory=dict)
    relations: List[RelationInfo] = Field(default_factory=list)
    expansion_path: List[str] = Field(default_factory=list)
    reasoning_steps: Optional[List[str]] = Field(default=None)


class EnhancedResult(BaseModel):
    """Enhanced result with graph context information."""
    id: str
    content: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    source_type: str  # "vector", "graph", "hybrid"
    document_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Graph-specific fields
    connected_entities: List[str] = Field(default_factory=list)
    relation_context: List[RelationInfo] = Field(default_factory=list)
    reasoning_path: Optional[List[str]] = Field(default=None)


class EnhancedQueryResponse(BaseModel):
    """Enhanced query response with comprehensive graph context."""
    query_id: str
    query: str
    results: List[EnhancedResult]
    graph_context: GraphContext
    
    # Metadata
    total_results: int
    processing_time_ms: float
    fusion_strategy_used: FusionStrategy
    expansion_strategy_used: ExpansionStrategy
    
    # Quality metrics
    confidence_score: float = Field(ge=0.0, le=1.0)
    completeness_score: float = Field(ge=0.0, le=1.0)
    
    # Debug information (optional)
    debug_info: Optional[Dict[str, Any]] = Field(default=None)


class EntityQueryRequest(BaseModel):
    """Request for querying specific entities and their relationships."""
    entity_id: Optional[str] = Field(default=None, description="Specific entity ID")
    entity_name: Optional[str] = Field(default=None, description="Entity name to search")
    entity_type: Optional[str] = Field(default=None, description="Entity type filter")
    
    include_relations: bool = Field(default=True, description="Include entity relations")
    relation_depth: int = Field(default=1, ge=1, le=3, description="Relation traversal depth")
    max_relations: int = Field(default=50, ge=1, le=200, description="Maximum relations to return")


class GraphTraversalRequest(BaseModel):
    """Request for graph traversal between entities."""
    start_entity: str = Field(..., description="Starting entity ID")
    end_entity: Optional[str] = Field(default=None, description="Target entity ID")
    
    traversal_type: str = Field(default="shortest_path", description="Type of traversal")
    max_depth: int = Field(default=3, ge=1, le=6, description="Maximum traversal depth")
    max_paths: int = Field(default=10, ge=1, le=50, description="Maximum paths to return")
    
    relation_filters: Optional[List[str]] = Field(default=None, description="Allowed relation types")
    entity_filters: Optional[List[str]] = Field(default=None, description="Allowed entity types")


class GraphPath(BaseModel):
    """A path through the graph between entities."""
    entities: List[str]
    relations: List[str]
    total_weight: float
    confidence: float = Field(ge=0.0, le=1.0)


class GraphTraversalResponse(BaseModel):
    """Response for graph traversal requests."""
    start_entity: str
    end_entity: Optional[str]
    paths: List[GraphPath]
    total_paths_found: int
    processing_time_ms: float
