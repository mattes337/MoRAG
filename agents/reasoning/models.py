"""Data models for reasoning agents."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class ReasoningStrategy(str, Enum):
    """Reasoning strategies."""
    FORWARD = "forward"
    BACKWARD = "backward"
    BIDIRECTIONAL = "bidirectional"
    BREADTH_FIRST = "breadth_first"
    DEPTH_FIRST = "depth_first"


class ConfidenceLevel(str, Enum):
    """Confidence levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class PathSelectionResult(BaseModel):
    """Result from path selection."""
    
    selected_paths: List[Dict[str, Any]] = Field(..., description="Selected reasoning paths")
    total_paths_considered: int = Field(..., description="Total paths considered")
    selection_criteria: Dict[str, Any] = Field(..., description="Criteria used for selection")
    confidence: ConfidenceLevel = Field(..., description="Confidence in selection")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ReasoningResult(BaseModel):
    """Result from reasoning process."""
    
    conclusion: str = Field(..., description="Main conclusion")
    reasoning_steps: List[str] = Field(..., description="Steps in reasoning process")
    evidence: List[str] = Field(..., description="Supporting evidence")
    confidence: ConfidenceLevel = Field(..., description="Confidence in reasoning")
    alternative_conclusions: List[str] = Field(default_factory=list, description="Alternative conclusions")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class DecisionResult(BaseModel):
    """Result from decision making."""
    
    decision: str = Field(..., description="Final decision")
    rationale: str = Field(..., description="Rationale for decision")
    alternatives: List[Dict[str, Any]] = Field(..., description="Alternative options considered")
    confidence: ConfidenceLevel = Field(..., description="Confidence in decision")
    risk_assessment: Dict[str, Any] = Field(default_factory=dict, description="Risk assessment")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ContextAnalysisResult(BaseModel):
    """Result from context analysis."""
    
    context_summary: str = Field(..., description="Summary of context")
    key_factors: List[str] = Field(..., description="Key contextual factors")
    relevance_scores: Dict[str, float] = Field(..., description="Relevance scores for different aspects")
    context_gaps: List[str] = Field(default_factory=list, description="Identified gaps in context")
    confidence: ConfidenceLevel = Field(..., description="Confidence in analysis")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
