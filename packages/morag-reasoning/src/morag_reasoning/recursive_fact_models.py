"""Data models for recursive fact retrieval system."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class SourceMetadata(BaseModel):
    """Detailed source metadata for facts."""
    document_name: Optional[str] = Field(None, description="Name of the source document")
    chunk_index: Optional[int] = Field(None, description="Index of the chunk within the document")
    page_number: Optional[int] = Field(None, description="Page number if applicable")
    section: Optional[str] = Field(None, description="Document section")
    timestamp: Optional[str] = Field(None, description="Timestamp for audio/video content")
    additional_metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional source metadata")


class RawFact(BaseModel):
    """A raw fact extracted by the GraphTraversalAgent."""
    fact_text: str = Field(..., description="A comprehensive, detailed piece of information with full context")
    source_node_id: str = Field(..., description="ID of the node where this fact was found")
    source_property: Optional[str] = Field(None, description="Optional property name if fact from property")
    source_qdrant_chunk_id: Optional[str] = Field(None, description="Optional ID of Qdrant chunk if fact from chunk content")
    source_metadata: SourceMetadata = Field(default_factory=SourceMetadata, description="Detailed source metadata")
    extracted_from_depth: int = Field(..., description="Integer representing the traversal depth when this fact was extracted (0 for initial entities)")


class ScoredFact(BaseModel):
    """A fact that has been scored by the FactCriticAgent."""
    fact_text: str = Field(..., description="A comprehensive, detailed piece of information with full context")
    source_node_id: str = Field(..., description="ID of the node where this fact was found")
    source_property: Optional[str] = Field(None, description="Optional property name if fact from property")
    source_qdrant_chunk_id: Optional[str] = Field(None, description="Optional ID of Qdrant chunk if fact from chunk content")
    source_metadata: SourceMetadata = Field(default_factory=SourceMetadata, description="Detailed source metadata")
    extracted_from_depth: int = Field(..., description="Integer representing the traversal depth when this fact was extracted")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score assigned by FCA")
    source_description: str = Field(..., description="A user-friendly description of the source")


class FinalFact(BaseModel):
    """A fact with final decayed score after relevance decay."""
    fact_text: str = Field(..., description="A comprehensive, detailed piece of information with full context")
    source_node_id: str = Field(..., description="ID of the node where this fact was found")
    source_property: Optional[str] = Field(None, description="Optional property name if fact from property")
    source_qdrant_chunk_id: Optional[str] = Field(None, description="Optional ID of Qdrant chunk if fact from chunk content")
    source_metadata: SourceMetadata = Field(default_factory=SourceMetadata, description="Detailed source metadata")
    extracted_from_depth: int = Field(..., description="Integer representing the traversal depth when this fact was extracted")
    score: float = Field(..., ge=0.0, le=1.0, description="Original relevance score assigned by FCA")
    source_description: str = Field(..., description="A user-friendly description of the source")
    final_decayed_score: float = Field(..., ge=0.0, le=1.0, description="Score after applying depth-based decay")


class RecursiveFactRetrievalRequest(BaseModel):
    """Request for recursive fact retrieval."""
    user_query: str = Field(..., description="The original query from the user")
    max_depth: int = Field(default=3, ge=1, le=10, description="The maximum traversal depth for the GraphTraversalAgent")
    decay_rate: float = Field(default=0.2, ge=0.0, le=1.0, description="The rate at which fact scores decay per depth level")
    max_facts_per_node: int = Field(default=1000, ge=1, le=10000, description="Maximum facts to extract per node (set high for exhaustive retrieval)")
    min_fact_score: float = Field(default=0.1, ge=0.0, le=1.0, description="Minimum score threshold for facts")
    max_total_facts: int = Field(default=10000, ge=1, le=100000, description="Maximum total facts to collect (set high for exhaustive retrieval)")
    facts_only: bool = Field(default=False, description="If true, return only facts without final answer synthesis")
    skip_fact_evaluation: bool = Field(default=False, description="If true, skip fact evaluation and return all raw facts without scoring")

    # Database configuration
    neo4j_database: Optional[str] = Field(None, description="Neo4j database name (optional)")
    qdrant_collection: Optional[str] = Field(None, description="Qdrant collection name (optional)")
    language: Optional[str] = Field(None, description="Language for processing (optional)")

    # Database server configuration
    database_servers: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional array of database server configurations. If not provided, uses environment defaults."
    )


class TraversalStep(BaseModel):
    """Information about a single traversal step."""
    node_id: str = Field(..., description="Node ID being explored")
    node_name: str = Field(..., description="Human-readable node name")
    depth: int = Field(..., description="Depth at which this node was explored")
    facts_extracted: int = Field(..., description="Number of facts extracted from this node")
    next_nodes_decision: str = Field(..., description="LLM decision about next nodes to explore")
    reasoning: str = Field(..., description="LLM reasoning for the decision")


class RecursiveFactRetrievalResponse(BaseModel):
    """Response from recursive fact retrieval."""
    query_id: str = Field(..., description="Unique identifier for this query")
    user_query: str = Field(..., description="Original user query")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    
    # Traversal information
    initial_entities: List[str] = Field(..., description="Initial entities identified from query")
    total_nodes_explored: int = Field(..., description="Total number of nodes explored")
    max_depth_reached: int = Field(..., description="Maximum depth reached during traversal")
    traversal_steps: List[TraversalStep] = Field(..., description="Detailed traversal steps")
    
    # Fact information
    total_raw_facts: int = Field(..., description="Total raw facts extracted")
    total_scored_facts: int = Field(..., description="Total facts after scoring")
    final_facts: List[FinalFact] = Field(..., description="Final facts after decay and filtering")
    
    # LLM usage
    gta_llm_calls: int = Field(..., description="Number of GraphTraversalAgent LLM calls")
    fca_llm_calls: int = Field(..., description="Number of FactCriticAgent LLM calls")
    final_llm_calls: int = Field(..., description="Number of final synthesis LLM calls")
    
    # Final answer
    final_answer: Optional[str] = Field(None, description="Final synthesized answer from the stronger LLM (null if facts_only=true)")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence in the answer")
    
    # Debug information
    debug_info: Optional[Dict[str, Any]] = Field(None, description="Optional debug information")


class GTAResponse(BaseModel):
    """Response from GraphTraversalAgent."""
    extracted_facts: List[RawFact] = Field(..., description="Raw facts extracted from the current node")
    next_nodes_to_explore: str = Field(..., description="Decision about next nodes to explore")
    reasoning: str = Field(..., description="Reasoning for the decision")


class FCAResponse(BaseModel):
    """Response from FactCriticAgent."""
    scored_fact: ScoredFact = Field(..., description="The fact with assigned score and source description")
