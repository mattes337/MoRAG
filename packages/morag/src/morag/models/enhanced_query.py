"""Enhanced query models for graph-augmented RAG."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


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
    query_type: QueryType = Field(
        default=QueryType.SIMPLE, description="Type of query processing"
    )
    max_results: int = Field(
        default=10, ge=1, le=100, description="Maximum number of results"
    )

    # Graph-specific parameters
    expansion_strategy: ExpansionStrategy = Field(
        default=ExpansionStrategy.ADAPTIVE, description="Context expansion strategy"
    )
    expansion_depth: int = Field(
        default=2, ge=1, le=5, description="Maximum expansion depth"
    )
    fusion_strategy: FusionStrategy = Field(
        default=FusionStrategy.ADAPTIVE, description="Result fusion strategy"
    )

    # Filtering and constraints
    entity_types: Optional[List[str]] = Field(
        default=None, description="Filter by entity types"
    )
    relation_types: Optional[List[str]] = Field(
        default=None, description="Filter by relation types"
    )
    time_range: Optional[Dict[str, datetime]] = Field(
        default=None, description="Time range filter"
    )

    # Advanced options
    include_graph_context: bool = Field(
        default=True, description="Include graph context in response"
    )
    include_reasoning_path: bool = Field(
        default=False, description="Include reasoning path"
    )
    enable_multi_hop: bool = Field(
        default=True, description="Enable multi-hop reasoning"
    )

    # Quality and performance
    min_relevance_score: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Minimum relevance threshold"
    )
    timeout_seconds: int = Field(default=30, ge=1, le=300, description="Query timeout")

    # Database server configuration
    database_servers: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional array of database server configurations. If not provided, uses environment defaults.",
    )

    # Fact-based retrieval options
    use_fact_retrieval: bool = Field(
        default=True,
        description="Use fact-based retrieval instead of traditional hybrid retrieval",
    )
    max_depth: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum traversal depth for fact extraction",
    )
    max_facts_per_node: int = Field(
        default=1000,
        ge=1,
        le=10000,
        description="Maximum facts to extract per node (set high for exhaustive retrieval)",
    )
    min_fact_score: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Minimum score threshold for facts"
    )
    max_total_facts: int = Field(
        default=10000,
        ge=1,
        le=100000,
        description="Maximum total facts to collect (set high for exhaustive retrieval)",
    )
    facts_only: bool = Field(
        default=False, description="Return only facts without final answer synthesis"
    )
    skip_fact_evaluation: bool = Field(
        default=False, description="Skip fact evaluation for faster processing"
    )
    decay_rate: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Rate at which fact scores decay per depth level",
    )
    language: str = Field(
        default="en", description="Language for fact extraction and processing"
    )


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


class FactInfo(BaseModel):
    """Information about an extracted fact."""

    fact_text: str = Field(..., description="The extracted fact text")
    source_node_id: str = Field(..., description="Source node ID in the graph")
    source_property: str = Field(..., description="Source property name")
    source_qdrant_chunk_id: Optional[str] = Field(
        None, description="Source Qdrant chunk ID"
    )
    source_metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Source metadata"
    )
    extracted_from_depth: int = Field(
        ..., description="Depth at which this fact was extracted"
    )
    score: float = Field(..., ge=0.0, le=1.0, description="Fact relevance score")
    final_decayed_score: float = Field(
        ..., ge=0.0, le=1.0, description="Final score after decay"
    )
    source_description: str = Field(
        ..., description="Human-readable source description"
    )


class TraversalStepInfo(BaseModel):
    """Information about a traversal step."""

    node_id: str = Field(..., description="Node ID")
    node_name: str = Field(..., description="Node name")
    depth: int = Field(..., description="Traversal depth")
    facts_extracted: int = Field(..., description="Number of facts extracted")
    next_nodes_decision: str = Field(..., description="Decision about next nodes")
    reasoning: str = Field(..., description="Reasoning for the decision")


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

    # Fact-based retrieval results (when use_fact_retrieval=True)
    facts: Optional[List[FactInfo]] = Field(
        default=None, description="Extracted facts from graph traversal"
    )
    final_answer: Optional[str] = Field(
        default=None, description="Final synthesized answer"
    )
    initial_entities: Optional[List[str]] = Field(
        default=None, description="Initial entities identified"
    )
    total_nodes_explored: Optional[int] = Field(
        default=None, description="Total nodes explored"
    )
    max_depth_reached: Optional[int] = Field(
        default=None, description="Maximum depth reached"
    )
    traversal_steps: Optional[List[TraversalStepInfo]] = Field(
        default=None, description="Traversal steps"
    )
    total_raw_facts: Optional[int] = Field(
        default=None, description="Total raw facts extracted"
    )
    total_scored_facts: Optional[int] = Field(
        default=None, description="Total scored facts"
    )
    gta_llm_calls: Optional[int] = Field(
        default=None, description="GraphTraversalAgent LLM calls"
    )
    fca_llm_calls: Optional[int] = Field(
        default=None, description="FactCriticAgent LLM calls"
    )
    final_llm_calls: Optional[int] = Field(
        default=None, description="Final synthesis LLM calls"
    )

    # Debug information (optional)
    debug_info: Optional[Dict[str, Any]] = Field(default=None)


class EntityQueryRequest(BaseModel):
    """Request for querying specific entities and their relationships."""

    entity_id: Optional[str] = Field(default=None, description="Specific entity ID")
    entity_name: Optional[str] = Field(
        default=None, description="Entity name to search"
    )
    entity_type: Optional[str] = Field(default=None, description="Entity type filter")

    include_relations: bool = Field(
        default=True, description="Include entity relations"
    )
    relation_depth: int = Field(
        default=1, ge=1, le=3, description="Relation traversal depth"
    )
    max_relations: int = Field(
        default=50, ge=1, le=200, description="Maximum relations to return"
    )

    # Database server configuration
    database_servers: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional array of database server configurations. If not provided, uses environment defaults.",
    )


class GraphTraversalRequest(BaseModel):
    """Request for graph traversal between entities."""

    start_entity: str = Field(..., description="Starting entity ID")
    end_entity: Optional[str] = Field(default=None, description="Target entity ID")

    traversal_type: str = Field(
        default="shortest_path", description="Type of traversal"
    )
    max_depth: int = Field(default=3, ge=1, le=6, description="Maximum traversal depth")
    max_paths: int = Field(
        default=10, ge=1, le=50, description="Maximum paths to return"
    )

    relation_filters: Optional[List[str]] = Field(
        default=None, description="Allowed relation types"
    )
    entity_filters: Optional[List[str]] = Field(
        default=None, description="Allowed entity types"
    )

    # Database server configuration
    database_servers: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional array of database server configurations. If not provided, uses environment defaults.",
    )


class GraphAnalyticsRequest(BaseModel):
    """Request for graph analytics and statistics."""

    metric_type: str = Field(
        default="overview",
        description="Type of analytics metric (overview, centrality, communities)",
    )

    # Database server configuration
    database_servers: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional array of database server configurations. If not provided, uses environment defaults.",
    )


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
