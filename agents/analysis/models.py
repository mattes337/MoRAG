"""Data models for analysis agents."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class QueryIntent(str, Enum):
    """Query intent categories."""
    SEARCH = "search"
    QUESTION = "question"
    COMPARISON = "comparison"
    ANALYSIS = "analysis"
    PROCEDURE = "procedure"
    RECOMMENDATION = "recommendation"
    CLARIFICATION = "clarification"
    SUMMARY = "summary"
    CREATION = "creation"
    TROUBLESHOOTING = "troubleshooting"


class QueryType(str, Enum):
    """Query type categories."""
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    PROCEDURAL = "procedural"
    COMPARATIVE = "comparative"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    CAUSAL = "causal"
    HYPOTHETICAL = "hypothetical"


class ComplexityLevel(str, Enum):
    """Complexity levels."""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


class ConfidenceLevel(str, Enum):
    """Confidence levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class SentimentPolarity(str, Enum):
    """Sentiment polarity."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class QueryAnalysisResult(BaseModel):
    """Result from query analysis."""
    
    intent: QueryIntent = Field(..., description="Primary intent of the query")
    entities: List[str] = Field(default_factory=list, description="Entities mentioned in query")
    keywords: List[str] = Field(default_factory=list, description="Important keywords")
    query_type: QueryType = Field(..., description="Type of query")
    complexity: ComplexityLevel = Field(..., description="Complexity level")
    confidence: ConfidenceLevel = Field(..., description="Confidence in analysis")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ContentAnalysisResult(BaseModel):
    """Result from content analysis."""
    
    main_topics: List[str] = Field(..., description="Main topics identified")
    key_concepts: List[str] = Field(..., description="Key concepts")
    content_type: str = Field(..., description="Type of content")
    complexity: ComplexityLevel = Field(..., description="Content complexity")
    readability_score: Optional[float] = Field(None, description="Readability score")
    word_count: int = Field(..., description="Word count")
    confidence: ConfidenceLevel = Field(..., description="Confidence in analysis")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SentimentAnalysisResult(BaseModel):
    """Result from sentiment analysis."""
    
    polarity: SentimentPolarity = Field(..., description="Overall sentiment polarity")
    intensity: float = Field(..., ge=0.0, le=1.0, description="Sentiment intensity")
    emotions: Dict[str, float] = Field(default_factory=dict, description="Emotion scores")
    confidence: ConfidenceLevel = Field(..., description="Confidence in analysis")
    aspects: List[Dict[str, Any]] = Field(default_factory=list, description="Aspect-based sentiment")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class TopicAnalysisResult(BaseModel):
    """Result from topic analysis."""
    
    primary_topic: str = Field(..., description="Primary topic")
    secondary_topics: List[str] = Field(default_factory=list, description="Secondary topics")
    topic_distribution: Dict[str, float] = Field(..., description="Topic probability distribution")
    coherence_score: float = Field(..., ge=0.0, le=1.0, description="Topic coherence score")
    confidence: ConfidenceLevel = Field(..., description="Confidence in analysis")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
