"""Data models for generation agents."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class ConfidenceLevel(str, Enum):
    """Confidence levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class SummaryType(str, Enum):
    """Types of summaries."""
    EXTRACTIVE = "extractive"
    ABSTRACTIVE = "abstractive"
    HYBRID = "hybrid"


class SummarizationResult(BaseModel):
    """Result from summarization."""
    
    summary: str = Field(..., description="Generated summary")
    key_points: List[str] = Field(..., description="Key points extracted")
    summary_type: SummaryType = Field(..., description="Type of summary")
    compression_ratio: float = Field(..., description="Compression ratio")
    confidence: ConfidenceLevel = Field(..., description="Confidence in summary")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ResponseGenerationResult(BaseModel):
    """Result from response generation."""
    
    response: str = Field(..., description="Generated response")
    sources: List[str] = Field(default_factory=list, description="Sources used")
    confidence: ConfidenceLevel = Field(..., description="Confidence in response")
    citations: List[Dict[str, Any]] = Field(default_factory=list, description="Citations")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ExplanationResult(BaseModel):
    """Result from explanation generation."""
    
    explanation: str = Field(..., description="Generated explanation")
    reasoning_steps: List[str] = Field(..., description="Reasoning steps")
    examples: List[str] = Field(default_factory=list, description="Supporting examples")
    confidence: ConfidenceLevel = Field(..., description="Confidence in explanation")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SynthesisResult(BaseModel):
    """Result from synthesis."""
    
    synthesis: str = Field(..., description="Synthesized content")
    sources_integrated: int = Field(..., description="Number of sources integrated")
    coherence_score: float = Field(..., description="Coherence score")
    confidence: ConfidenceLevel = Field(..., description="Confidence in synthesis")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
