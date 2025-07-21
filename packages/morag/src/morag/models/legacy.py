"""Legacy API models for backward compatibility."""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class LegacyResult(BaseModel):
    """Legacy result format for v1 API compatibility."""
    id: str = Field(..., description="Unique result identifier")
    content: str = Field(..., description="Result content")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class LegacyQueryRequest(BaseModel):
    """Legacy query request format for v1 API compatibility."""
    query: str = Field(..., description="The user's query text")
    max_results: Optional[int] = Field(default=10, ge=1, le=100, description="Maximum number of results")
    min_score: Optional[float] = Field(default=0.1, ge=0.0, le=1.0, description="Minimum relevance score")
    
    # Legacy field names for backward compatibility
    limit: Optional[int] = Field(default=None, ge=1, le=100, description="Alias for max_results")
    
    def __init__(self, **data):
        # Handle legacy field mapping
        if 'limit' in data and 'max_results' not in data:
            data['max_results'] = data['limit']
        super().__init__(**data)


class LegacyQueryResponse(BaseModel):
    """Legacy query response format for v1 API compatibility."""
    query: str = Field(..., description="The original query")
    results: List[LegacyResult] = Field(..., description="Query results")
    total_results: int = Field(..., description="Total number of results")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    
    # Legacy field names for backward compatibility
    total: Optional[int] = Field(default=None, description="Alias for total_results")
    
    def __init__(self, **data):
        # Ensure legacy field compatibility
        if 'total_results' in data and 'total' not in data:
            data['total'] = data['total_results']
        super().__init__(**data)


class LegacyHealthResponse(BaseModel):
    """Legacy health check response with deprecation notice."""
    status: str = Field(default="healthy", description="Service status")
    deprecated: bool = Field(default=True, description="Indicates this is a deprecated endpoint")
    migration_info: Dict[str, Any] = Field(
        default_factory=lambda: {
            "new_endpoint": "/api/v2/health",
            "deprecation_date": "2024-01-01",
            "end_of_support": "2024-12-01",
            "migration_guide": "/api/v1/migration-guide"
        },
        description="Migration information"
    )
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


class MigrationGuideResponse(BaseModel):
    """Migration guide response for v1 to v2 API transition."""
    message: str = Field(
        default="API v1 is deprecated. Please migrate to v2 for enhanced features.",
        description="Migration message"
    )
    migration_guide: Dict[str, Any] = Field(
        default_factory=lambda: {
            "old_endpoint": "/api/v1/query",
            "new_endpoint": "/api/v2/query",
            "key_changes": [
                "Enhanced request model with more configuration options",
                "Graph context included in responses",
                "Support for different query types and strategies",
                "Improved result ranking and fusion"
            ],
            "breaking_changes": [
                "Response format includes additional fields",
                "Some field names have changed",
                "New optional fields in request model"
            ],
            "migration_steps": [
                "Update endpoint URL from /api/v1/query to /api/v2/query",
                "Update request model to EnhancedQueryRequest",
                "Handle new response fields in EnhancedQueryResponse",
                "Test with existing queries",
                "Gradually adopt new features"
            ],
            "examples": {
                "v1_request": {
                    "query": "What is machine learning?",
                    "max_results": 10,
                    "min_score": 0.1
                },
                "v2_request": {
                    "query": "What is machine learning?",
                    "query_type": "simple",
                    "max_results": 10,
                    "min_relevance_score": 0.1
                }
            }
        },
        description="Detailed migration guide"
    )
    timeline: Dict[str, str] = Field(
        default_factory=lambda: {
            "deprecation_date": "2024-01-01",
            "end_of_support": "2024-12-01",
            "removal_date": "TBD (will be announced 6 months in advance)"
        },
        description="Migration timeline"
    )
    support: Dict[str, str] = Field(
        default_factory=lambda: {
            "documentation": "/docs",
            "migration_guide": "/docs/migration-guide",
            "examples": "/docs/examples"
        },
        description="Support resources"
    )
