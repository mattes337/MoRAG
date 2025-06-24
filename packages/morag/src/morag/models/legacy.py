"""Legacy API models for backward compatibility."""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class LegacyQueryRequest(BaseModel):
    """Legacy query request model for backward compatibility."""
    query: str = Field(..., description="The user's query text")
    max_results: Optional[int] = Field(default=10, description="Maximum number of results")
    min_score: Optional[float] = Field(default=0.1, description="Minimum relevance score")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Optional filters")


class LegacyResult(BaseModel):
    """Legacy result format."""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LegacyQueryResponse(BaseModel):
    """Legacy query response model."""
    query: str
    results: List[LegacyResult]
    total_results: int
    processing_time_ms: float
