"""Structured response models for MoRAG AI agents."""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class ConfidenceLevel(str, Enum):
    """Confidence levels for AI outputs."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class Entity(BaseModel):
    """Represents an extracted entity."""

    name: str = Field(description="The name or text of the entity")
    type: str = Field(description="The type of the entity (LLM-determined)")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score for the entity")
    start_pos: Optional[int] = Field(default=None, description="Start position in the text")
    end_pos: Optional[int] = Field(default=None, description="End position in the text")
    context: Optional[str] = Field(default=None, description="Surrounding context")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class Relation(BaseModel):
    """Represents a relation between entities."""

    source_entity: str = Field(description="The source entity name")
    target_entity: str = Field(description="The target entity name")
    relation_type: str = Field(description="The type of relation (LLM-determined)")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score for the relation")
    context: Optional[str] = Field(default=None, description="Context where the relation was found")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class EntityExtractionResult(BaseModel):
    """Result of entity extraction."""
    
    entities: List[Entity] = Field(description="List of extracted entities")
    confidence: ConfidenceLevel = Field(description="Overall confidence level")
    processing_time: Optional[float] = Field(default=None, description="Processing time in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class RelationExtractionResult(BaseModel):
    """Result of relation extraction."""
    
    relations: List[Relation] = Field(description="List of extracted relations")
    confidence: ConfidenceLevel = Field(description="Overall confidence level")
    processing_time: Optional[float] = Field(default=None, description="Processing time in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SummaryResult(BaseModel):
    """Result of text summarization."""
    
    summary: str = Field(description="The generated summary")
    key_points: List[str] = Field(description="Key points from the text")
    confidence: ConfidenceLevel = Field(description="Confidence in the summary quality")
    word_count: int = Field(description="Word count of the summary")
    compression_ratio: float = Field(description="Ratio of summary length to original length")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class TopicBoundary(BaseModel):
    """Represents a topic boundary in text."""
    
    position: int = Field(description="Character position of the boundary")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the boundary")
    topic_before: Optional[str] = Field(default=None, description="Topic before the boundary")
    topic_after: Optional[str] = Field(default=None, description="Topic after the boundary")
    reason: Optional[str] = Field(default=None, description="Reason for the boundary")


class SemanticChunkingResult(BaseModel):
    """Result of semantic chunking."""
    
    boundaries: List[TopicBoundary] = Field(description="List of topic boundaries")
    chunks: List[str] = Field(description="List of semantic chunks")
    chunk_topics: List[str] = Field(description="Topic for each chunk")
    confidence: ConfidenceLevel = Field(description="Overall confidence level")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class QueryAnalysisResult(BaseModel):
    """Result of query analysis."""
    
    intent: str = Field(description="The detected intent of the query")
    entities: List[str] = Field(description="Entities mentioned in the query")
    keywords: List[str] = Field(description="Important keywords from the query")
    query_type: str = Field(description="Type of query (factual, analytical, etc.)")
    complexity: str = Field(description="Complexity level (simple, medium, complex)")
    confidence: ConfidenceLevel = Field(description="Confidence in the analysis")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ContentAnalysisResult(BaseModel):
    """Result of content analysis."""
    
    content_type: str = Field(description="Type of content")
    language: str = Field(description="Detected language")
    sentiment: str = Field(description="Overall sentiment")
    topics: List[str] = Field(description="Main topics in the content")
    quality_score: float = Field(ge=0.0, le=1.0, description="Content quality score")
    readability_score: float = Field(ge=0.0, le=1.0, description="Readability score")
    confidence: ConfidenceLevel = Field(description="Confidence in the analysis")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class TranscriptAnalysisResult(BaseModel):
    """Result of transcript analysis."""
    
    speakers: List[str] = Field(description="Identified speakers")
    topics: List[str] = Field(description="Main topics discussed")
    key_moments: List[Dict[str, Any]] = Field(description="Key moments with timestamps")
    sentiment_timeline: List[Dict[str, Any]] = Field(description="Sentiment changes over time")
    summary: str = Field(description="Summary of the transcript")
    confidence: ConfidenceLevel = Field(description="Confidence in the analysis")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class BaseAgentResult(BaseModel):
    """Base class for all agent results."""
    
    success: bool = Field(description="Whether the operation was successful")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the result was generated")
    agent_version: Optional[str] = Field(default=None, description="Version of the agent that generated this result")
    model_used: Optional[str] = Field(default=None, description="Model used for generation")
    processing_time: Optional[float] = Field(default=None, description="Processing time in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
