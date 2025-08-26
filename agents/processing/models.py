"""Data models for processing agents."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class ConfidenceLevel(str, Enum):
    """Confidence levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ChunkingResult(BaseModel):
    """Result from chunking."""
    
    chunks: List[str] = Field(..., description="Text chunks")
    chunk_count: int = Field(..., description="Number of chunks")
    avg_chunk_size: float = Field(..., description="Average chunk size")
    chunking_strategy: str = Field(..., description="Strategy used")
    confidence: ConfidenceLevel = Field(..., description="Confidence in chunking")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ClassificationResult(BaseModel):
    """Result from classification."""
    
    category: str = Field(..., description="Predicted category")
    confidence_score: float = Field(..., description="Confidence score")
    all_categories: Dict[str, float] = Field(..., description="All category scores")
    confidence: ConfidenceLevel = Field(..., description="Overall confidence")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ValidationResult(BaseModel):
    """Result from validation."""
    
    is_valid: bool = Field(..., description="Whether input is valid")
    validation_errors: List[str] = Field(default_factory=list, description="Validation errors")
    quality_score: float = Field(..., description="Quality score")
    confidence: ConfidenceLevel = Field(..., description="Confidence in validation")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class FilteringResult(BaseModel):
    """Result from filtering."""
    
    filtered_items: List[Any] = Field(..., description="Items that passed filtering")
    rejected_items: List[Any] = Field(default_factory=list, description="Items that were rejected")
    filter_criteria: Dict[str, Any] = Field(..., description="Criteria used for filtering")
    confidence: ConfidenceLevel = Field(..., description="Confidence in filtering")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
