"""Legacy API endpoints for backward compatibility."""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Response
from typing import Dict, Any
import structlog
import warnings
import uuid
from datetime import datetime

from morag.models.legacy import (
    LegacyQueryRequest, LegacyQueryResponse, LegacyResult,
    LegacyHealthResponse, MigrationGuideResponse
)
from morag.models.enhanced_query import (
    EnhancedQueryRequest, QueryType, ExpansionStrategy, FusionStrategy
)
from morag.dependencies import get_hybrid_retrieval_coordinator
from morag.endpoints.enhanced_query import enhanced_query

router = APIRouter(prefix="/api/v1", tags=["legacy"])
logger = structlog.get_logger(__name__)


def add_deprecation_headers(response: Response) -> None:
    """Add deprecation headers to legacy API responses."""
    response.headers["X-API-Deprecation-Date"] = "2024-01-01"
    response.headers["X-API-End-Of-Support"] = "2024-12-01"
    response.headers["X-API-Migration-Guide"] = "/api/v1/migration-guide"
    response.headers["X-API-Latest-Version"] = "v2"
    response.headers["Warning"] = '299 - "This API version is deprecated. Please use /api/v2/ endpoints."'


@router.post("/query", response_model=LegacyQueryResponse)
async def legacy_query(
    request: LegacyQueryRequest,
    response: Response,
    hybrid_system = Depends(get_hybrid_retrieval_coordinator),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Legacy query endpoint for backward compatibility."""
    # Add deprecation headers
    add_deprecation_headers(response)
    
    # Issue deprecation warning
    warnings.warn(
        "The /api/v1/query endpoint is deprecated. Please use /api/v2/query for enhanced features.",
        DeprecationWarning,
        stacklevel=2
    )
    
    logger.info(f"Legacy query received: {request.query[:100]}...")
    
    try:
        # Convert legacy request to enhanced request
        enhanced_request = EnhancedQueryRequest(
            query=request.query,
            query_type=QueryType.SIMPLE,
            max_results=request.max_results or 10,
            expansion_strategy=ExpansionStrategy.ADAPTIVE,
            expansion_depth=2,
            fusion_strategy=FusionStrategy.ADAPTIVE,
            include_graph_context=False,  # Legacy doesn't include graph context
            include_reasoning_path=False,
            enable_multi_hop=False,  # Keep it simple for legacy
            min_relevance_score=request.min_score or 0.1,
            timeout_seconds=30
        )
        
        # Use enhanced query endpoint
        enhanced_response = await enhanced_query(enhanced_request, hybrid_system, background_tasks)
        
        # Convert enhanced response to legacy format
        legacy_results = []
        for result in enhanced_response.results:
            legacy_results.append(LegacyResult(
                id=result.id,
                content=result.content,
                score=result.relevance_score,
                metadata={
                    "document_id": result.document_id,
                    "source_type": result.source_type,
                    **result.metadata
                }
            ))
        
        return LegacyQueryResponse(
            query=request.query,
            results=legacy_results,
            total_results=len(legacy_results),
            processing_time_ms=enhanced_response.processing_time_ms
        )
        
    except Exception as e:
        logger.error(f"Error in legacy query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=LegacyHealthResponse)
async def legacy_health(response: Response):
    """Legacy health check with deprecation notice."""
    add_deprecation_headers(response)
    
    logger.info("Legacy health check requested")
    
    return LegacyHealthResponse()


@router.get("/migration-guide", response_model=MigrationGuideResponse)
async def migration_guide(response: Response):
    """Provide migration guidance for legacy API users."""
    add_deprecation_headers(response)
    
    logger.info("Migration guide requested")
    
    return MigrationGuideResponse()


@router.get("/status")
async def legacy_status(response: Response):
    """Legacy status endpoint with feature comparison."""
    add_deprecation_headers(response)
    
    return {
        "api_version": "v1",
        "status": "deprecated",
        "features": {
            "basic_query": True,
            "graph_context": False,
            "multi_hop_reasoning": False,
            "entity_extraction": False,
            "streaming": False
        },
        "v2_features": {
            "enhanced_query": True,
            "graph_context": True,
            "multi_hop_reasoning": True,
            "entity_extraction": True,
            "streaming": True,
            "graph_traversal": True,
            "analytics": True
        },
        "migration": {
            "guide": "/api/v1/migration-guide",
            "examples": "/docs/examples",
            "support": "/docs"
        }
    }
