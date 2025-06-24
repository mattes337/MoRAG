"""Legacy API endpoints for backward compatibility."""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional
import structlog
import warnings
from datetime import date

from morag.models.legacy import LegacyQueryRequest, LegacyQueryResponse, LegacyResult
from morag.models.enhanced_query import EnhancedQueryRequest, QueryType, ExpansionStrategy
from morag.dependencies import get_hybrid_retrieval_coordinator
from morag.endpoints.enhanced_query import enhanced_query

router = APIRouter(prefix="/api/v1", tags=["legacy"])
logger = structlog.get_logger(__name__)


@router.post("/query", response_model=LegacyQueryResponse)
async def legacy_query(
    request: LegacyQueryRequest,
    hybrid_system = Depends(get_hybrid_retrieval_coordinator)
):
    """Legacy query endpoint for backward compatibility."""
    # Issue deprecation warning
    warnings.warn(
        "The /api/v1/query endpoint is deprecated. Please use /api/v2/query for enhanced features.",
        DeprecationWarning,
        stacklevel=2
    )
    
    logger.info("Legacy query received", 
               query=request.query[:100],
               max_results=request.max_results)
    
    try:
        # Convert legacy request to enhanced request
        enhanced_request = EnhancedQueryRequest(
            query=request.query,
            query_type=QueryType.SIMPLE,
            max_results=request.max_results or 10,
            expansion_strategy=ExpansionStrategy.ADAPTIVE,
            expansion_depth=2,
            include_graph_context=False,  # Legacy doesn't include graph context
            include_reasoning_path=False,
            enable_multi_hop=False,  # Keep it simple for legacy
            min_relevance_score=request.min_score or 0.1
        )
        
        # Use enhanced query endpoint
        enhanced_response = await enhanced_query(enhanced_request, hybrid_system)
        
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
        logger.error("Error in legacy query", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def legacy_health_check():
    """Legacy health check endpoint."""
    return {
        "status": "healthy", 
        "version": "1.0", 
        "deprecated": True,
        "migration_info": {
            "new_version": "2.0",
            "migration_endpoint": "/api/v1/migration-guide"
        }
    }


@router.get("/migration-guide")
async def migration_guide():
    """Provide migration guidance for legacy API users."""
    return {
        "message": "API v1 is deprecated. Please migrate to v2.",
        "migration_guide": {
            "old_endpoint": "/api/v1/query",
            "new_endpoint": "/api/v2/query",
            "key_changes": [
                "Enhanced request model with more configuration options",
                "Graph context included in responses",
                "Support for different query types and strategies",
                "Improved result ranking and fusion",
                "Streaming query support",
                "Entity and graph traversal endpoints"
            ],
            "breaking_changes": [
                "Response format includes additional fields",
                "Some field names have changed",
                "New optional fields in request model"
            ],
            "migration_steps": [
                "1. Update client to use /api/v2/query endpoint",
                "2. Update request model to EnhancedQueryRequest",
                "3. Handle new response fields in EnhancedQueryResponse",
                "4. Test with your existing queries",
                "5. Gradually adopt new features like graph context"
            ],
            "compatibility": {
                "legacy_support_until": "2024-12-01",
                "deprecation_warnings": True,
                "feature_parity": "Basic query functionality maintained"
            },
            "new_features": {
                "graph_context": "Access to knowledge graph entities and relations",
                "multi_hop_reasoning": "Advanced reasoning across multiple entities",
                "streaming": "Real-time result streaming",
                "entity_queries": "Direct entity and relationship queries",
                "graph_traversal": "Path finding and graph exploration",
                "analytics": "Graph statistics and insights"
            },
            "examples": {
                "legacy_request": {
                    "query": "What is machine learning?",
                    "max_results": 5,
                    "min_score": 0.1
                },
                "enhanced_request": {
                    "query": "What is machine learning?",
                    "query_type": "simple",
                    "max_results": 5,
                    "expansion_strategy": "adaptive",
                    "include_graph_context": True,
                    "min_relevance_score": 0.1
                }
            }
        }
    }


@router.get("/status")
async def legacy_status():
    """Legacy API status endpoint."""
    return {
        "api_version": "1.0",
        "status": "deprecated",
        "deprecation_date": "2024-01-01",
        "end_of_support": "2024-12-01",
        "replacement": {
            "version": "2.0",
            "endpoint": "/api/v2",
            "documentation": "/docs"
        },
        "features": {
            "basic_query": "supported",
            "graph_context": "not_available",
            "streaming": "not_available",
            "entity_queries": "not_available",
            "graph_traversal": "not_available"
        }
    }
