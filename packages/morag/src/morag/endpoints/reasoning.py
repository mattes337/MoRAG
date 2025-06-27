"""Multi-hop reasoning API endpoints."""

import time
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
import structlog

from morag.dependencies import (
    get_reasoning_path_finder, get_iterative_retriever, get_llm_client,
    REASONING_AVAILABLE
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/reasoning", tags=["Multi-Hop Reasoning"])


class MultiHopQuery(BaseModel):
    """Multi-hop reasoning query request."""
    query: str = Field(..., description="The query to answer using multi-hop reasoning")
    start_entities: List[str] = Field(..., description="Starting entities for reasoning")
    target_entities: Optional[List[str]] = Field(None, description="Optional target entities")
    strategy: str = Field("forward_chaining", description="Reasoning strategy (forward_chaining, backward_chaining, bidirectional)")
    max_depth: int = Field(4, ge=1, le=10, description="Maximum reasoning depth")
    max_paths: int = Field(50, ge=1, le=100, description="Maximum paths to discover")
    max_iterations: int = Field(5, ge=1, le=10, description="Maximum refinement iterations")


class PathInfo(BaseModel):
    """Information about a reasoning path."""
    path_id: str = Field(..., description="Unique path identifier")
    entities: List[str] = Field(..., description="Entities in the path")
    relations: List[str] = Field(..., description="Relations in the path")
    relevance_score: float = Field(..., description="Relevance score for the path")
    confidence: float = Field(..., description="Confidence in the path")
    reasoning: str = Field(..., description="Reasoning for path selection")


class ContextInfo(BaseModel):
    """Information about the reasoning context."""
    entity_count: int = Field(..., description="Number of entities in context")
    relation_count: int = Field(..., description="Number of relations in context")
    document_count: int = Field(..., description="Number of documents in context")
    path_count: int = Field(..., description="Number of paths in context")


class ReasoningResult(BaseModel):
    """Result of multi-hop reasoning."""
    query: str = Field(..., description="Original query")
    strategy: str = Field(..., description="Reasoning strategy used")
    paths_found: int = Field(..., description="Number of reasoning paths found")
    selected_paths: List[PathInfo] = Field(..., description="Selected reasoning paths")
    context_info: ContextInfo = Field(..., description="Final context information")
    iterations_used: int = Field(..., description="Number of refinement iterations used")
    final_confidence: float = Field(..., description="Final confidence in result")
    reasoning_time_ms: float = Field(..., description="Total reasoning time in milliseconds")
    context_sufficient: bool = Field(..., description="Whether final context was sufficient")
    reasoning_summary: str = Field(..., description="Summary of the reasoning process")


class ReasoningStatus(BaseModel):
    """Status of reasoning capabilities."""
    available: bool = Field(..., description="Whether reasoning is available")
    components: Dict[str, bool] = Field(..., description="Status of individual components")
    configuration: Dict[str, Any] = Field(..., description="Current configuration")


@router.get("/status", response_model=ReasoningStatus)
async def get_reasoning_status():
    """Get the status of multi-hop reasoning capabilities."""
    try:
        # Check component availability
        path_finder = get_reasoning_path_finder()
        iterative_retriever = get_iterative_retriever()
        llm_client = get_llm_client()
        
        components = {
            "reasoning_available": REASONING_AVAILABLE,
            "path_finder": path_finder is not None,
            "iterative_retriever": iterative_retriever is not None,
            "llm_client": llm_client is not None,
        }
        
        configuration = {
            "max_paths_default": 50,
            "max_iterations_default": 5,
            "sufficiency_threshold_default": 0.8,
            "supported_strategies": ["forward_chaining", "backward_chaining", "bidirectional"]
        }
        
        return ReasoningStatus(
            available=all(components.values()),
            components=components,
            configuration=configuration
        )
        
    except Exception as e:
        logger.error("Error checking reasoning status", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error checking reasoning status: {str(e)}")


@router.post("/query", response_model=ReasoningResult)
async def multi_hop_reasoning(
    request: MultiHopQuery,
    path_finder = Depends(get_reasoning_path_finder),
    iterative_retriever = Depends(get_iterative_retriever)
):
    """Perform multi-hop reasoning to answer a complex query."""
    if not REASONING_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Multi-hop reasoning not available. Please check reasoning components."
        )
    
    if path_finder is None or iterative_retriever is None:
        raise HTTPException(
            status_code=503,
            detail="Reasoning components not properly initialized"
        )
    
    start_time = time.time()
    
    try:
        logger.info("Starting multi-hop reasoning", 
                   query=request.query, 
                   strategy=request.strategy,
                   start_entities=request.start_entities)
        
        # Step 1: Find reasoning paths
        reasoning_paths = await path_finder.find_reasoning_paths(
            query=request.query,
            start_entities=request.start_entities,
            target_entities=request.target_entities,
            strategy=request.strategy,
            max_paths=request.max_paths
        )
        
        logger.info("Found reasoning paths", count=len(reasoning_paths))
        
        # Step 2: Create initial context from paths
        from morag_reasoning import RetrievalContext
        
        initial_context = RetrievalContext(
            paths=[path.path for path in reasoning_paths[:10]],  # Use top 10 paths
            entities={entity: {"name": entity, "type": "UNKNOWN"} for entity in request.start_entities},
            metadata={"query": request.query, "strategy": request.strategy}
        )
        
        # Step 3: Refine context iteratively
        refined_context = await iterative_retriever.refine_context(
            request.query, initial_context
        )
        
        # Step 4: Prepare response
        end_time = time.time()
        reasoning_time_ms = (end_time - start_time) * 1000
        
        # Convert paths to response format
        selected_paths = []
        for i, path_score in enumerate(reasoning_paths[:10]):  # Return top 10 paths
            path_info = PathInfo(
                path_id=f"path_{i+1}",
                entities=[entity.name for entity in path_score.path.entities],
                relations=[relation.type for relation in path_score.path.relations],
                relevance_score=path_score.relevance_score,
                confidence=path_score.confidence,
                reasoning=path_score.reasoning
            )
            selected_paths.append(path_info)
        
        # Get final analysis
        final_analysis = refined_context.metadata.get('final_analysis')
        final_confidence = final_analysis.confidence if final_analysis else 0.5
        context_sufficient = final_analysis.is_sufficient if final_analysis else False
        iterations_used = refined_context.metadata.get('iterations_used', 0)
        
        # Create context info
        context_info = ContextInfo(
            entity_count=len(refined_context.entities),
            relation_count=len(refined_context.relations),
            document_count=len(refined_context.documents),
            path_count=len(refined_context.paths)
        )
        
        # Generate reasoning summary
        reasoning_summary = _generate_reasoning_summary(
            request, reasoning_paths, refined_context, final_analysis
        )
        
        result = ReasoningResult(
            query=request.query,
            strategy=request.strategy,
            paths_found=len(reasoning_paths),
            selected_paths=selected_paths,
            context_info=context_info,
            iterations_used=iterations_used,
            final_confidence=final_confidence,
            reasoning_time_ms=reasoning_time_ms,
            context_sufficient=context_sufficient,
            reasoning_summary=reasoning_summary
        )
        
        logger.info("Multi-hop reasoning completed",
                   paths_found=len(reasoning_paths),
                   iterations_used=iterations_used,
                   final_confidence=final_confidence,
                   reasoning_time_ms=reasoning_time_ms)
        
        return result
        
    except Exception as e:
        logger.error("Error in multi-hop reasoning", error=str(e))
        raise HTTPException(status_code=500, detail=f"Reasoning failed: {str(e)}")


def _generate_reasoning_summary(request, reasoning_paths, refined_context, final_analysis) -> str:
    """Generate a summary of the reasoning process."""
    summary_parts = [
        f"Performed {request.strategy} reasoning for query: '{request.query}'",
        f"Found {len(reasoning_paths)} reasoning paths from {len(request.start_entities)} starting entities",
        f"Final context contains {len(refined_context.entities)} entities, {len(refined_context.relations)} relations, and {len(refined_context.documents)} documents"
    ]
    
    if final_analysis:
        if final_analysis.is_sufficient:
            summary_parts.append(f"Context deemed sufficient with {final_analysis.confidence:.2f} confidence")
        else:
            summary_parts.append(f"Context insufficient (confidence: {final_analysis.confidence:.2f})")
            if final_analysis.gaps:
                gap_types = [gap.gap_type for gap in final_analysis.gaps]
                summary_parts.append(f"Identified gaps: {', '.join(set(gap_types))}")
    
    return ". ".join(summary_parts) + "."


@router.get("/strategies")
async def get_reasoning_strategies():
    """Get available reasoning strategies and their descriptions."""
    strategies = {
        "forward_chaining": {
            "name": "Forward Chaining",
            "description": "Start from query entities and explore forward through relationships",
            "max_depth": 4,
            "bidirectional": False,
            "best_for": "Exploring consequences and effects of starting entities"
        },
        "backward_chaining": {
            "name": "Backward Chaining", 
            "description": "Start from potential answers and work backward to query entities",
            "max_depth": 3,
            "bidirectional": False,
            "best_for": "Finding causes and origins leading to target entities"
        },
        "bidirectional": {
            "name": "Bidirectional Search",
            "description": "Search from both ends and meet in the middle",
            "max_depth": 5,
            "bidirectional": True,
            "best_for": "Finding connections between specific start and target entities"
        }
    }
    
    return {"strategies": strategies}
