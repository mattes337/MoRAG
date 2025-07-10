"""Intelligent entity-based retrieval endpoints."""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
import structlog

from morag_reasoning import (
    IntelligentRetrievalRequest, IntelligentRetrievalResponse,
    IntelligentRetrievalService
)
from morag.dependencies import get_llm_client
from morag.database_factory import (
    get_default_neo4j_storage, get_default_qdrant_storage,
    get_connected_default_neo4j_storage, get_connected_default_qdrant_storage,
    get_neo4j_storages, get_qdrant_storages, DatabaseConnectionFactory
)
from morag_graph import DatabaseType

router = APIRouter(prefix="/api/v2", tags=["intelligent-retrieval"])
logger = structlog.get_logger(__name__)


async def get_intelligent_retrieval_service(
    request: IntelligentRetrievalRequest
) -> IntelligentRetrievalService:
    """Get intelligent retrieval service with specified databases.

    Args:
        request: Intelligent retrieval request with database configurations

    Returns:
        Configured intelligent retrieval service
    """
    # Get LLM client
    llm_client = get_llm_client()

    # Get Neo4j storage
    if request.neo4j_server:
        # Use custom Neo4j server configuration
        if request.neo4j_server.type != DatabaseType.NEO4J:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid database type for Neo4j server: {request.neo4j_server.type}"
            )
        try:
            neo4j_storage = await DatabaseConnectionFactory.create_neo4j_storage(request.neo4j_server)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to create Neo4j connection: {str(e)}"
            )
    elif request.neo4j_database:
        # Use pre-configured database by name
        neo4j_storages = get_neo4j_storages()
        neo4j_storage = neo4j_storages.get(request.neo4j_database)
        if not neo4j_storage:
            raise HTTPException(
                status_code=400,
                detail=f"Neo4j database '{request.neo4j_database}' not found"
            )
    else:
        # Use default configuration
        neo4j_storage = await get_connected_default_neo4j_storage()

    # Get Qdrant storage
    if request.qdrant_server:
        # Use custom Qdrant server configuration
        if request.qdrant_server.type != DatabaseType.QDRANT:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid database type for Qdrant server: {request.qdrant_server.type}"
            )
        try:
            qdrant_storage = await DatabaseConnectionFactory.create_qdrant_storage(request.qdrant_server)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to create Qdrant connection: {str(e)}"
            )
    elif request.qdrant_collection:
        # Use pre-configured collection by name
        qdrant_storages = get_qdrant_storages()
        qdrant_storage = qdrant_storages.get(request.qdrant_collection)
        if not qdrant_storage:
            raise HTTPException(
                status_code=400,
                detail=f"Qdrant collection '{request.qdrant_collection}' not found"
            )
    else:
        # Use default configuration
        qdrant_storage = await get_connected_default_qdrant_storage()

    if not neo4j_storage:
        raise HTTPException(
            status_code=503,
            detail="Neo4j storage not available"
        )

    if not qdrant_storage:
        raise HTTPException(
            status_code=503,
            detail="Qdrant storage not available"
        )

    return IntelligentRetrievalService(
        llm_client=llm_client,
        neo4j_storage=neo4j_storage,
        qdrant_storage=qdrant_storage
    )


@router.post("/intelligent-query", response_model=IntelligentRetrievalResponse)
async def intelligent_retrieval(
    request: IntelligentRetrievalRequest
):
    """Perform intelligent entity-based retrieval with recursive path following.
    
    This endpoint:
    1. Identifies entities from the user prompt
    2. Retrieves entity nodes/chunks from Neo4j/Qdrant
    3. Uses LLM to decide which paths to follow recursively
    4. Extracts key facts from retrieved chunks with source information
    5. Returns structured JSON with facts and sources
    """
    try:
        logger.info(
            "Starting intelligent retrieval",
            query=request.query,
            max_iterations=request.max_iterations
        )
        
        # Get service with specified databases
        service = await get_intelligent_retrieval_service(request)
        
        # Perform intelligent retrieval
        response = await service.retrieve_intelligently(request)
        
        logger.info(
            "Intelligent retrieval completed",
            query_id=response.query_id,
            key_facts=len(response.key_facts),
            processing_time_ms=response.processing_time_ms
        )
        
        return response
        
    except Exception as e:
        logger.error(
            "Intelligent retrieval failed",
            error=str(e),
            error_type=type(e).__name__,
            query=request.query
        )
        raise HTTPException(
            status_code=500,
            detail=f"Intelligent retrieval failed: {str(e)}"
        )


@router.get("/intelligent-query/health")
async def health_check():
    """Health check for intelligent retrieval endpoint."""
    try:
        # Check if required services are available
        llm_client = get_llm_client()
        neo4j_storage = get_default_neo4j_storage()
        qdrant_storage = get_default_qdrant_storage()
        
        status = {
            "status": "healthy",
            "services": {
                "llm_client": llm_client is not None,
                "neo4j_storage": neo4j_storage is not None,
                "qdrant_storage": qdrant_storage is not None
            }
        }
        
        # Check if all required services are available
        all_healthy = all(status["services"].values())
        if not all_healthy:
            status["status"] = "degraded"
        
        return status
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@router.get("/intelligent-query/info")
async def get_endpoint_info():
    """Get information about the intelligent retrieval endpoint."""
    return {
        "name": "Intelligent Entity-Based Retrieval",
        "description": "Performs intelligent retrieval using entity identification, recursive graph traversal, and fact extraction",
        "version": "1.0.0",
        "features": [
            "Entity identification from user queries",
            "Recursive graph path following with LLM decisions",
            "Key fact extraction with source tracking",
            "Configurable iteration limits and thresholds",
            "Support for multiple Neo4j databases and Qdrant collections",
            "Custom database server connections with full configuration support"
        ],
        "parameters": {
            "query": "User query/prompt (required)",
            "max_iterations": "Maximum recursive iterations (default: 5)",
            "max_entities_per_iteration": "Max entities to explore per iteration (default: 10)",
            "max_paths_per_entity": "Max paths to consider per entity (default: 5)",
            "max_depth": "Maximum path depth (default: 3)",
            "min_relevance_threshold": "Minimum relevance threshold (default: 0.3)",
            "include_debug_info": "Include debug information (default: false)",
            "neo4j_database": "Neo4j database name (optional)",
            "qdrant_collection": "Qdrant collection name (optional)",
            "language": "Language for processing (optional)",
            "neo4j_server": "Custom Neo4j server configuration (optional)",
            "qdrant_server": "Custom Qdrant server configuration (optional)"
        },
        "response_format": {
            "query_id": "Unique query identifier",
            "key_facts": "List of extracted key facts with sources",
            "iterations": "Details of each recursive iteration",
            "metrics": "Confidence and completeness scores",
            "processing_info": "Processing time and LLM calls made"
        }
    }
