"""Models for intelligent entity-based retrieval with recursive path following."""

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

# Import existing database configuration models
from morag_graph.models.database_config import DatabaseServerConfig, DatabaseType


class PathDecision(str, Enum):
    """LLM decisions for path following."""
    FOLLOW = "follow"
    SKIP = "skip"
    STOP = "stop"


class SourceInfo(BaseModel):
    """Source information for key facts."""
    document_id: str = Field(..., description="Document identifier")
    chunk_id: str = Field(..., description="Chunk identifier")
    document_name: str = Field(..., description="Human-readable document name")
    chunk_text: str = Field(..., description="Original chunk text")
    relevance_score: float = Field(ge=0.0, le=1.0, description="Relevance score")
    page_number: Optional[int] = Field(None, description="Page number if applicable")
    section: Optional[str] = Field(None, description="Document section")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class KeyFact(BaseModel):
    """A key fact extracted from chunks with source information."""
    fact: str = Field(..., description="The extracted key fact")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the fact")
    relevance_to_query: float = Field(ge=0.0, le=1.0, description="Relevance to user query")
    fact_type: str = Field(..., description="Type of fact (e.g., 'definition', 'relationship', 'statistic')")
    sources: List[SourceInfo] = Field(..., description="Source information for this fact")
    supporting_entities: List[str] = Field(default_factory=list, description="Entities that support this fact")


class EntityPath(BaseModel):
    """A path through the graph starting from an entity."""
    entity_id: str = Field(..., description="Starting entity ID")
    entity_name: str = Field(..., description="Starting entity name")
    path_entities: List[str] = Field(..., description="Entities in the path")
    path_relations: List[str] = Field(..., description="Relations in the path")
    depth: int = Field(..., description="Path depth")
    relevance_score: float = Field(ge=0.0, le=1.0, description="Path relevance score")
    llm_decision: PathDecision = Field(..., description="LLM decision for this path")
    decision_reasoning: str = Field(..., description="LLM reasoning for the decision")


class RetrievalIteration(BaseModel):
    """Information about a single iteration of recursive retrieval."""
    iteration: int = Field(..., description="Iteration number")
    entities_explored: List[str] = Field(..., description="Entities explored in this iteration")
    paths_found: List[EntityPath] = Field(..., description="Paths discovered")
    paths_followed: List[EntityPath] = Field(..., description="Paths the LLM decided to follow")
    chunks_retrieved: int = Field(..., description="Number of chunks retrieved")
    llm_stop_reason: Optional[str] = Field(None, description="Reason LLM decided to stop")


class IntelligentRetrievalRequest(BaseModel):
    """Request for intelligent entity-based retrieval."""
    query: str = Field(..., description="User query/prompt")
    max_iterations: int = Field(default=5, ge=1, le=10, description="Maximum recursive iterations")
    max_entities_per_iteration: int = Field(default=10, ge=1, le=50, description="Max entities to explore per iteration")
    max_paths_per_entity: int = Field(default=5, ge=1, le=20, description="Max paths to consider per entity")
    max_depth: int = Field(default=3, ge=1, le=5, description="Maximum path depth")
    min_relevance_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum relevance threshold")
    include_debug_info: bool = Field(default=False, description="Include debug information in response")
    
    # Database configuration
    neo4j_database: Optional[str] = Field(None, description="Neo4j database name")
    qdrant_collection: Optional[str] = Field(None, description="Qdrant collection name")
    language: Optional[str] = Field(None, description="Language for processing")

    # Custom database server configurations (optional - overrides server defaults)
    neo4j_server: Optional[DatabaseServerConfig] = Field(None, description="Custom Neo4j server configuration")
    qdrant_server: Optional[DatabaseServerConfig] = Field(None, description="Custom Qdrant server configuration")

    # Retry configuration for handling model overload (503 errors)
    max_retries: int = Field(default=8, ge=1, le=20, description="Maximum retry attempts for overload errors")
    retry_base_delay: float = Field(default=2.0, ge=0.1, le=10.0, description="Base delay for exponential backoff (seconds)")
    retry_max_delay: float = Field(default=120.0, ge=1.0, le=300.0, description="Maximum delay between retries (seconds)")
    retry_jitter: bool = Field(default=True, description="Add random jitter to retry delays")


class IntelligentRetrievalResponse(BaseModel):
    """Response from intelligent entity-based retrieval."""
    query_id: str = Field(..., description="Unique query identifier")
    query: str = Field(..., description="Original user query")
    
    # Main results
    key_facts: List[KeyFact] = Field(..., description="Extracted key facts relevant to the query")
    
    # Process information
    total_iterations: int = Field(..., description="Total iterations performed")
    iterations: List[RetrievalIteration] = Field(..., description="Details of each iteration")
    
    # Entity and path information
    initial_entities: List[str] = Field(..., description="Initially identified entities")
    total_entities_explored: int = Field(..., description="Total unique entities explored")
    total_chunks_retrieved: int = Field(..., description="Total chunks retrieved")
    
    # Quality metrics
    confidence_score: float = Field(ge=0.0, le=1.0, description="Overall confidence in results")
    completeness_score: float = Field(ge=0.0, le=1.0, description="Estimated completeness of information")
    
    # Metadata
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    llm_calls_made: int = Field(..., description="Number of LLM calls made")
    
    # Debug information (optional)
    debug_info: Optional[Dict[str, Any]] = Field(None, description="Debug information if requested")


class FactExtractionRequest(BaseModel):
    """Request for extracting key facts from chunks."""
    query: str = Field(..., description="Original user query")
    chunks: List[Dict[str, Any]] = Field(..., description="Retrieved chunks with metadata")
    max_facts: int = Field(default=20, ge=1, le=100, description="Maximum facts to extract")
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum confidence threshold")


class PathFollowingRequest(BaseModel):
    """Request for LLM-based path following decision."""
    query: str = Field(..., description="Original user query")
    current_entities: List[str] = Field(..., description="Currently explored entities")
    available_paths: List[EntityPath] = Field(..., description="Available paths to consider")
    iteration: int = Field(..., description="Current iteration number")
    max_iterations: int = Field(..., description="Maximum allowed iterations")
