"""Enhanced query endpoints for graph-augmented RAG."""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import Dict, Any, Optional
import asyncio
import json
import uuid
from datetime import datetime
import structlog

from morag.models.enhanced_query import (
    EnhancedQueryRequest, EnhancedQueryResponse, EntityQueryRequest,
    GraphTraversalRequest, GraphTraversalResponse
)
from morag.dependencies import get_hybrid_retrieval_coordinator, get_graph_engine
from morag.utils.response_builder import EnhancedResponseBuilder
from morag.utils.query_validator import QueryValidator

router = APIRouter(prefix="/api/v2", tags=["enhanced-query"])
logger = structlog.get_logger(__name__)


@router.post("/query", response_model=EnhancedQueryResponse)
async def enhanced_query(
    request: EnhancedQueryRequest,
    hybrid_system = Depends(get_hybrid_retrieval_coordinator),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Enhanced query endpoint with graph-augmented retrieval."""
    query_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    try:
        # Validate request
        validator = QueryValidator()
        validation_result = await validator.validate_query_request(request)
        if not validation_result.is_valid:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid query: {validation_result.error_message}"
            )
        
        # Log warnings if any
        if validation_result.warnings:
            logger.warning("Query validation warnings", 
                         query_id=query_id, 
                         warnings=validation_result.warnings)
        
        # Process query
        logger.info("Processing enhanced query", 
                   query_id=query_id, 
                   query=request.query[:100],
                   query_type=request.query_type)
        
        # Execute retrieval with timeout
        try:
            retrieval_result = await asyncio.wait_for(
                hybrid_system.retrieve(
                    query=request.query,
                    max_results=request.max_results
                ),
                timeout=request.timeout_seconds
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=408,
                detail=f"Query timeout after {request.timeout_seconds} seconds"
            )
        
        # Build response
        response_builder = EnhancedResponseBuilder()
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        response = await response_builder.build_response(
            query_id=query_id,
            request=request,
            retrieval_result=retrieval_result,
            processing_time=processing_time
        )
        
        # Log query for analytics (background task)
        background_tasks.add_task(
            log_query_analytics,
            query_id=query_id,
            request=request,
            response=response,
            processing_time=response.processing_time_ms
        )
        
        logger.info("Enhanced query completed", 
                   query_id=query_id,
                   result_count=len(response.results),
                   processing_time_ms=response.processing_time_ms)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error processing enhanced query", 
                    query_id=query_id, 
                    error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.post("/query/stream")
async def enhanced_query_stream(
    request: EnhancedQueryRequest,
    hybrid_system = Depends(get_hybrid_retrieval_coordinator)
):
    """Streaming version of enhanced query for real-time results."""
    query_id = str(uuid.uuid4())
    
    async def generate_stream():
        try:
            logger.info("Starting streaming query", query_id=query_id)
            
            # For now, we'll simulate streaming by processing normally
            # and yielding partial results. In a full implementation,
            # the hybrid system would support streaming.
            
            # Send initial status
            yield f"data: {{\"status\": \"processing\", \"query_id\": \"{query_id}\"}}\n\n"
            
            # Process query
            retrieval_result = await hybrid_system.retrieve(
                query=request.query,
                max_results=request.max_results
            )
            
            # Build response
            response_builder = EnhancedResponseBuilder()
            response = await response_builder.build_response(
                query_id=query_id,
                request=request,
                retrieval_result=retrieval_result,
                processing_time=0.0  # Will be calculated properly in full implementation
            )
            
            # Stream results
            for i, result in enumerate(response.results):
                partial_response = {
                    "type": "result",
                    "index": i,
                    "result": result.dict(),
                    "query_id": query_id
                }
                yield f"data: {json.dumps(partial_response)}\n\n"
                
                # Small delay to simulate streaming
                await asyncio.sleep(0.1)
            
            # Send final response
            final_response = {
                "type": "complete",
                "query_id": query_id,
                "total_results": len(response.results),
                "graph_context": response.graph_context.dict(),
                "processing_time_ms": response.processing_time_ms
            }
            yield f"data: {json.dumps(final_response)}\n\n"
            
        except Exception as e:
            error_data = {
                "type": "error",
                "error": str(e), 
                "query_id": query_id
            }
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


@router.post("/entity/query", response_model=Dict[str, Any])
async def entity_query(
    request: EntityQueryRequest,
    graph_engine = Depends(get_graph_engine)
):
    """Query specific entities and their relationships."""
    try:
        logger.info("Processing entity query", 
                   entity_id=request.entity_id,
                   entity_name=request.entity_name)
        
        # Find entity
        entity = None
        if request.entity_id:
            entity = await graph_engine.get_entity(request.entity_id)
        elif request.entity_name:
            entities = await graph_engine.find_entities_by_name(
                request.entity_name,
                entity_type=request.entity_type
            )
            entity = entities[0] if entities else None
        
        if not entity:
            raise HTTPException(status_code=404, detail="Entity not found")
        
        # Get relations if requested
        relations = []
        if request.include_relations:
            relations = await graph_engine.get_entity_relations(
                entity.id,
                depth=request.relation_depth,
                max_relations=request.max_relations
            )
        
        # Convert to dict format
        entity_dict = entity.dict() if hasattr(entity, 'dict') else {
            'id': getattr(entity, 'id', ''),
            'name': getattr(entity, 'name', ''),
            'type': getattr(entity, 'type', ''),
            'properties': getattr(entity, 'properties', {})
        }
        
        relations_dict = []
        for rel in relations:
            if hasattr(rel, 'dict'):
                relations_dict.append(rel.dict())
            else:
                relations_dict.append({
                    'id': getattr(rel, 'id', ''),
                    'name': getattr(rel, 'name', ''),
                    'type': getattr(rel, 'type', ''),
                    'properties': getattr(rel, 'properties', {})
                })
        
        return {
            "entity": entity_dict,
            "relations": relations_dict,
            "relation_count": len(relations)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in entity query", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/graph/traverse", response_model=GraphTraversalResponse)
async def graph_traversal(
    request: GraphTraversalRequest,
    graph_engine = Depends(get_graph_engine)
):
    """Perform graph traversal between entities."""
    start_time = datetime.now()
    
    try:
        logger.info("Processing graph traversal", 
                   start_entity=request.start_entity,
                   end_entity=request.end_entity,
                   traversal_type=request.traversal_type)
        
        # Validate entities exist
        start_entity = await graph_engine.get_entity(request.start_entity)
        if not start_entity:
            raise HTTPException(status_code=404, detail="Start entity not found")
        
        end_entity = None
        if request.end_entity:
            end_entity = await graph_engine.get_entity(request.end_entity)
            if not end_entity:
                raise HTTPException(status_code=404, detail="End entity not found")
        
        # Perform traversal
        paths = []
        if request.traversal_type == "shortest_path" and end_entity:
            paths = await graph_engine.find_shortest_paths(
                request.start_entity,
                request.end_entity,
                max_paths=request.max_paths,
                relation_filters=request.relation_filters
            )
        elif request.traversal_type == "explore":
            paths = await graph_engine.explore_from_entity(
                request.start_entity,
                max_depth=request.max_depth,
                max_paths=request.max_paths,
                entity_filters=request.entity_filters,
                relation_filters=request.relation_filters
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported traversal type: {request.traversal_type}"
            )
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Convert paths to response format
        response_paths = []
        for path in paths:
            response_paths.append({
                "entities": getattr(path, 'entities', []),
                "relations": getattr(path, 'relations', []),
                "total_weight": getattr(path, 'total_weight', 1.0),
                "confidence": getattr(path, 'confidence', 0.8)
            })
        
        return GraphTraversalResponse(
            start_entity=request.start_entity,
            end_entity=request.end_entity,
            paths=response_paths,
            total_paths_found=len(paths),
            processing_time_ms=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in graph traversal", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/analytics")
async def graph_analytics(
    metric_type: str = "overview",
    graph_engine = Depends(get_graph_engine)
):
    """Get graph analytics and statistics."""
    try:
        logger.info("Processing graph analytics request", metric_type=metric_type)
        
        if metric_type == "overview":
            try:
                stats = await graph_engine.get_graph_statistics()
                return {
                    "entity_count": getattr(stats, 'entity_count', 0),
                    "relation_count": getattr(stats, 'relation_count', 0),
                    "avg_degree": getattr(stats, 'avg_degree', 0.0),
                    "connected_components": getattr(stats, 'connected_components', 0),
                    "density": getattr(stats, 'density', 0.0)
                }
            except Exception as e:
                logger.warning("Graph statistics not available", error=str(e))
                return {
                    "entity_count": 0,
                    "relation_count": 0,
                    "avg_degree": 0.0,
                    "connected_components": 0,
                    "density": 0.0,
                    "note": "Statistics not available"
                }
        
        elif metric_type == "centrality":
            try:
                centrality = await graph_engine.calculate_centrality_measures()
                return {
                    "top_entities_by_degree": getattr(centrality, 'degree_centrality', [])[:10],
                    "top_entities_by_betweenness": getattr(centrality, 'betweenness_centrality', [])[:10],
                    "top_entities_by_pagerank": getattr(centrality, 'pagerank', [])[:10]
                }
            except Exception as e:
                logger.warning("Centrality measures not available", error=str(e))
                return {
                    "top_entities_by_degree": [],
                    "top_entities_by_betweenness": [],
                    "top_entities_by_pagerank": [],
                    "note": "Centrality measures not available"
                }
        
        elif metric_type == "communities":
            try:
                communities = await graph_engine.detect_communities()
                return {
                    "community_count": len(communities),
                    "largest_community_size": max(len(getattr(c, 'entities', [])) for c in communities) if communities else 0,
                    "modularity_score": getattr(communities[0], 'modularity', 0.0) if communities else 0.0
                }
            except Exception as e:
                logger.warning("Community detection not available", error=str(e))
                return {
                    "community_count": 0,
                    "largest_community_size": 0,
                    "modularity_score": 0.0,
                    "note": "Community detection not available"
                }
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown metric type: {metric_type}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting graph analytics", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


async def log_query_analytics(
    query_id: str,
    request: EnhancedQueryRequest,
    response: EnhancedQueryResponse,
    processing_time: float
):
    """Log query for analytics (background task)."""
    try:
        # Log to analytics system
        analytics_data = {
            "query_id": query_id,
            "query": request.query,
            "query_type": request.query_type,
            "result_count": len(response.results),
            "processing_time_ms": processing_time,
            "confidence_score": response.confidence_score,
            "fusion_strategy": response.fusion_strategy_used,
            "timestamp": datetime.now().isoformat()
        }
        
        # For now, just log to structured logger
        # In production, this would send to analytics service
        logger.info("Query analytics", **analytics_data)
        
    except Exception as e:
        logger.error("Error logging analytics", error=str(e))
