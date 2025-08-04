"""Recursive fact retrieval endpoints."""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
import structlog
import asyncio
import random
import httpx

from morag_reasoning import (
    RecursiveFactRetrievalRequest, RecursiveFactRetrievalResponse,
    RecursiveFactRetrievalService, LLMClient, LLMConfig
)
from morag.dependencies import get_llm_client
from morag.database_factory import (
    get_default_neo4j_storage, get_default_qdrant_storage,
    get_connected_default_neo4j_storage, get_connected_default_qdrant_storage,
    get_neo4j_storages, get_qdrant_storages, DatabaseConnectionFactory
)
from morag_graph import DatabaseType

router = APIRouter(prefix="/api/v2", tags=["recursive-fact-retrieval"])
logger = structlog.get_logger(__name__)


async def get_recursive_fact_retrieval_service(
    request: RecursiveFactRetrievalRequest
) -> RecursiveFactRetrievalService:
    """Get recursive fact retrieval service with specified databases."""
    
    # Get LLM client
    llm_client = get_llm_client()
    
    # Create stronger LLM client for final synthesis (could use a different model)
    stronger_llm_config = LLMConfig(
        api_key=llm_client.config.api_key,
        model=llm_client.config.model  # Could be configured to use a stronger model
    )
    stronger_llm_client = LLMClient(stronger_llm_config)
    
    # Handle database connections
    if request.database_servers:
        # Use custom database servers
        neo4j_storage = None
        qdrant_storage = None
        
        for server_config in request.database_servers:
            if server_config.get("type") == DatabaseType.NEO4J.value:
                from morag_graph import DatabaseServerConfig
                config = DatabaseServerConfig(**server_config)
                neo4j_storage = await DatabaseConnectionFactory.create_neo4j_storage(config)
            elif server_config.get("type") == DatabaseType.QDRANT.value:
                from morag_graph import DatabaseServerConfig
                config = DatabaseServerConfig(**server_config)
                qdrant_storage = await DatabaseConnectionFactory.create_qdrant_storage(config)
        
        if not neo4j_storage or not qdrant_storage:
            raise HTTPException(
                status_code=400,
                detail="Both Neo4j and Qdrant database configurations are required"
            )
    else:
        # Use default connections with optional database/collection override
        neo4j_storage = await get_connected_default_neo4j_storage(
            database=request.neo4j_database
        )
        qdrant_storage = await get_connected_default_qdrant_storage(
            collection=request.qdrant_collection
        )
    
    # Initialize embedding service for enhanced retrieval
    embedding_service = None
    try:
        from morag_services.embedding import GeminiEmbeddingService
        import os
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key:
            embedding_service = GeminiEmbeddingService(api_key=api_key)
            logger.info("Embedding service initialized for enhanced retrieval")
        else:
            logger.warning("GEMINI_API_KEY not found - enhanced retrieval disabled")
    except Exception as e:
        logger.warning("Failed to initialize embedding service", error=str(e))

    return RecursiveFactRetrievalService(
        llm_client=llm_client,
        neo4j_storage=neo4j_storage,
        qdrant_storage=qdrant_storage,
        embedding_service=embedding_service,
        stronger_llm_client=stronger_llm_client
    )


@router.post("/recursive-fact-retrieval", response_model=RecursiveFactRetrievalResponse)
async def recursive_fact_retrieval(
    request: RecursiveFactRetrievalRequest
):
    """
    Perform recursive fact retrieval using graph-based RAG system.
    
    This endpoint implements the complete recursive fact retrieval system as specified
    in RECURSIVE_FACT_RETRIEVAL.md, including:
    
    1. **GraphTraversalAgent (GTA)**: Navigates the Neo4j graph, identifies relevant 
       nodes and relationships, and extracts raw facts
    2. **FactCriticAgent (FCA)**: Evaluates the relevance and quality of raw facts, 
       assigns scores and generates user-friendly source descriptions
    3. **Orchestration Logic**: Manages the overall process including initial query 
       processing, iterative graph traversal, fact collection, scoring, relevance 
       decay, and final answer generation
    
    The system performs intelligent graph traversal, extracts facts at each node,
    evaluates their relevance, applies depth-based decay, and synthesizes a final
    answer using a stronger LLM.
    
    **Key Features:**
    - Intelligent entity identification from user queries
    - Breadth-first graph traversal with LLM-guided decisions
    - Fact extraction from node properties and associated content
    - Relevance scoring and depth-based decay
    - Final answer synthesis with confidence scoring
    - Configurable depth limits and scoring thresholds
    - Support for custom database configurations
    """
    try:
        logger.info(
            "Starting recursive fact retrieval",
            query=request.user_query,
            max_depth=request.max_depth,
            decay_rate=request.decay_rate
        )
        
        # Get service with specified databases
        service = await get_recursive_fact_retrieval_service(request)

        # Perform recursive fact retrieval with retry logic for model overload
        async def perform_retrieval():
            return await service.retrieve_facts_recursively(request)

        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                response = await perform_retrieval()
                
                logger.info(
                    "Recursive fact retrieval completed",
                    query_id=response.query_id,
                    total_facts=len(response.final_facts),
                    confidence=response.confidence_score,
                    processing_time_ms=response.processing_time_ms
                )
                
                return response
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 503 and "overloaded" in str(e).lower():
                    if attempt < max_retries - 1:
                        # Exponential backoff with jitter
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        logger.warning(
                            "Model overloaded, retrying",
                            attempt=attempt + 1,
                            delay=delay,
                            error=str(e)
                        )
                        await asyncio.sleep(delay)
                        continue
                    else:
                        logger.error("Max retries exceeded for model overload", error=str(e))
                        raise HTTPException(
                            status_code=503,
                            detail="The AI model is currently overloaded. Please try again later."
                        )
                else:
                    raise
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        "Retrieval failed, retrying",
                        attempt=attempt + 1,
                        delay=delay,
                        error=str(e)
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Recursive fact retrieval failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Recursive fact retrieval failed: {str(e)}"
        )


@router.get("/recursive-fact-retrieval/info")
async def get_recursive_fact_retrieval_info():
    """Get information about the recursive fact retrieval endpoint."""
    return {
        "name": "Recursive Fact Retrieval",
        "description": "Performs recursive fact retrieval using graph-based RAG system with GraphTraversalAgent, FactCriticAgent, and orchestration logic",
        "version": "1.0.0",
        "features": [
            "Intelligent entity identification from user queries",
            "Graph traversal with LLM-guided decisions",
            "Fact extraction from node properties and content",
            "Relevance scoring and quality evaluation",
            "Depth-based relevance decay",
            "Final answer synthesis with stronger LLM",
            "Configurable depth limits and thresholds",
            "Support for custom database configurations",
            "Comprehensive traversal tracking and debugging"
        ],
        "parameters": {
            "user_query": "The original query from the user (required)",
            "max_depth": "Maximum traversal depth (default: 3, range: 1-10)",
            "decay_rate": "Rate of score decay per depth level (default: 0.2, range: 0.0-1.0)",
            "max_facts_per_node": "Maximum facts to extract per node (default: 5, range: 1-20)",
            "min_fact_score": "Minimum score threshold for facts (default: 0.1, range: 0.0-1.0)",
            "max_total_facts": "Maximum total facts to collect (default: 50, range: 1-200)",
            "facts_only": "Return only facts without final answer synthesis (default: false)",
            "skip_fact_evaluation": "Skip fact evaluation and return all raw facts without scoring (default: false)",
            "neo4j_database": "Neo4j database name (optional)",
            "qdrant_collection": "Qdrant collection name (optional)",
            "language": "Language for processing (optional)",
            "database_servers": "Custom database server configurations (optional)"
        },
        "response_fields": {
            "query_id": "Unique identifier for the query",
            "user_query": "Original user query",
            "processing_time_ms": "Total processing time in milliseconds",
            "initial_entities": "Initial entities identified from query",
            "total_nodes_explored": "Total number of nodes explored",
            "max_depth_reached": "Maximum depth reached during traversal",
            "traversal_steps": "Detailed traversal steps with decisions",
            "total_raw_facts": "Total raw facts extracted",
            "total_scored_facts": "Total facts after scoring",
            "final_facts": "Final facts after decay and filtering",
            "gta_llm_calls": "Number of GraphTraversalAgent LLM calls",
            "fca_llm_calls": "Number of FactCriticAgent LLM calls", 
            "final_llm_calls": "Number of final synthesis LLM calls",
            "final_answer": "Final synthesized answer (null if facts_only=true)",
            "confidence_score": "Overall confidence in the answer"
        },
        "algorithm": {
            "step_1": "Extract initial entities from user query using EntityIdentificationService",
            "step_2": "Map entities to graph nodes in Neo4j",
            "step_3": "Perform breadth-first graph traversal using GraphTraversalAgent",
            "step_4": "Extract facts from each node (properties, content, relationships)",
            "step_5": "Evaluate fact relevance using FactCriticAgent (skipped if skip_fact_evaluation=true)",
            "step_6": "Apply depth-based relevance decay (skipped if skip_fact_evaluation=true)",
            "step_7": "Filter facts by minimum score and limit total count",
            "step_8": "Synthesize final answer using stronger LLM (skipped if facts_only=true)"
        }
    }


@router.get("/recursive-fact-retrieval/health")
async def check_recursive_fact_retrieval_health():
    """Check the health of recursive fact retrieval components."""
    try:
        # Check if we can create the service with default settings
        test_request = RecursiveFactRetrievalRequest(user_query="test")
        service = await get_recursive_fact_retrieval_service(test_request)
        
        health = {
            "status": "healthy",
            "components": {}
        }
        
        # Check Neo4j connection
        try:
            await service.neo4j_storage.health_check()
            health["components"]["neo4j"] = True
        except Exception as e:
            health["components"]["neo4j"] = False
            health["neo4j_error"] = str(e)
        
        # Check Qdrant connection
        try:
            await service.qdrant_storage.health_check()
            health["components"]["qdrant"] = True
        except Exception as e:
            health["components"]["qdrant"] = False
            health["qdrant_error"] = str(e)
        
        # Check LLM clients
        health["components"]["llm_client"] = service.llm_client is not None
        health["components"]["stronger_llm_client"] = service.stronger_llm_client is not None
        
        # Overall status
        if not all(health["components"].values()):
            health["status"] = "degraded"
        
        return health
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "error": str(e)
        }
