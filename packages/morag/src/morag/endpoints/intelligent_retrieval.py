"""Intelligent entity-based retrieval endpoints."""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
import structlog
import asyncio
import random
import httpx

from morag_reasoning import (
    IntelligentRetrievalRequest, IntelligentRetrievalResponse,
    IntelligentRetrievalService, RecursiveFactRetrievalRequest,
    RecursiveFactRetrievalResponse, RecursiveFactRetrievalService
)
from morag.dependencies import get_llm_client, get_recursive_fact_retrieval_service
from morag.database_factory import (
    get_default_neo4j_storage, get_default_qdrant_storage,
    get_connected_default_neo4j_storage, get_connected_default_qdrant_storage,
    get_neo4j_storages, get_qdrant_storages, DatabaseConnectionFactory
)
from morag_graph import DatabaseType

router = APIRouter(prefix="/api/v2", tags=["intelligent-retrieval"])
logger = structlog.get_logger(__name__)


async def retry_on_overload(
    func,
    max_retries: int = 8,
    base_delay: float = 2.0,
    max_delay: float = 120.0,
    exponential_base: float = 2.0,
    jitter: bool = True
):
    """Retry function with exponential backoff for model overload errors.

    Args:
        func: Async function to retry
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        jitter: Whether to add random jitter to delays

    Returns:
        Result of the function call

    Raises:
        The last exception if all retries fail
    """
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            return await func()

        except Exception as e:
            last_error = e
            error_str = str(e).lower()

            # Check if this is a retryable error (503, overload, rate limit, etc.)
            is_retryable = (
                "503" in error_str or
                "overload" in error_str or
                "rate limit" in error_str or
                "quota" in error_str or
                "too many requests" in error_str or
                "service unavailable" in error_str or
                "temporarily unavailable" in error_str or
                "server error" in error_str or
                isinstance(e, httpx.HTTPStatusError) and e.response.status_code in [503, 429, 500, 502, 504]
            )

            if not is_retryable or attempt >= max_retries:
                logger.error(
                    "Function failed with non-retryable error or max retries exceeded",
                    attempt=attempt,
                    max_retries=max_retries,
                    error=str(e),
                    is_retryable=is_retryable
                )
                break

            # Calculate delay with exponential backoff
            delay = min(
                base_delay * (exponential_base ** (attempt - 1)),
                max_delay
            )

            # Add jitter to prevent thundering herd
            if jitter:
                delay *= (0.5 + random.random() * 0.5)  # Add 0-50% jitter

            logger.warning(
                "Function failed with retryable error, retrying with exponential backoff",
                attempt=attempt,
                max_retries=max_retries,
                delay=delay,
                error=str(e)
            )

            await asyncio.sleep(delay)

    # If we get here, all retries failed
    logger.error(f"Function failed after {max_retries} attempts")
    raise last_error


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

    Note: This endpoint maintains backward compatibility. For new applications,
    consider using /intelligent-query/facts which uses the improved fact-based system.
    """
    try:
        logger.info(
            "Starting intelligent retrieval (legacy)",
            query=request.query,
            max_iterations=request.max_iterations
        )

        # Get service with specified databases
        service = await get_intelligent_retrieval_service(request)

        # Perform intelligent retrieval with retry logic for model overload
        async def perform_retrieval():
            return await service.retrieve_intelligently(request)

        response = await retry_on_overload(
            perform_retrieval,
            max_retries=request.max_retries,
            base_delay=request.retry_base_delay,
            max_delay=request.retry_max_delay,
            jitter=request.retry_jitter
        )

        logger.info(
            "Intelligent retrieval completed (legacy)",
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


@router.post("/intelligent-query/facts", response_model=RecursiveFactRetrievalResponse)
async def intelligent_fact_retrieval(
    request: RecursiveFactRetrievalRequest
):
    """Perform intelligent fact-based retrieval using the new graph structure.

    This endpoint uses the RecursiveFactRetrievalService to:
    1. Extract entities from the user query
    2. Perform recursive graph traversal with fact extraction
    3. Score and filter facts based on relevance
    4. Return structured facts with source attribution
    """
    try:
        logger.info(
            "Starting intelligent fact retrieval",
            query=request.user_query,
            max_depth=request.max_depth
        )

        # Get fact retrieval service
        service = await get_recursive_fact_retrieval_service()
        if not service:
            raise HTTPException(
                status_code=503,
                detail="Recursive fact retrieval service not available"
            )

        # Perform fact retrieval with retry logic for model overload
        async def perform_retrieval():
            return await service.retrieve_facts_recursively(request)

        response = await retry_on_overload(
            perform_retrieval,
            max_retries=8,
            base_delay=2.0,
            max_delay=120.0
        )

        logger.info(
            "Intelligent fact retrieval completed",
            query_id=response.query_id,
            total_facts=len(response.final_facts),
            processing_time_ms=response.processing_time_ms
        )

        return response

    except Exception as e:
        logger.error(
            "Intelligent fact retrieval failed",
            error=str(e),
            error_type=type(e).__name__,
            query=request.user_query
        )
        raise HTTPException(
            status_code=500,
            detail=f"Intelligent fact retrieval failed: {str(e)}"
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
            "qdrant_server": "Custom Qdrant server configuration (optional)",
            "max_retries": "Maximum retry attempts for overload errors (default: 8)",
            "retry_base_delay": "Base delay for exponential backoff in seconds (default: 2.0)",
            "retry_max_delay": "Maximum delay between retries in seconds (default: 120.0)",
            "retry_jitter": "Add random jitter to retry delays (default: true)"
        },
        "response_format": {
            "query_id": "Unique query identifier",
            "key_facts": "List of extracted key facts with sources",
            "iterations": "Details of each recursive iteration",
            "metrics": "Confidence and completeness scores",
            "processing_info": "Processing time and LLM calls made"
        }
    }
