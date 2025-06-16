# Task 3.1: API Integration

**Phase**: 3 - API Integration  
**Priority**: High  
**Estimated Time**: 4-5 days total  
**Dependencies**: Task 2.3 (Hybrid Retrieval System)

## Overview

This task implements enhanced API endpoints that expose graph-augmented retrieval capabilities while maintaining backward compatibility with existing clients. It provides new endpoints for advanced querying, entity exploration, and graph traversal, along with comprehensive request/response models.

## Subtasks

### 3.2.1: Enhanced Query Endpoints
**Estimated Time**: 2-3 days  
**Priority**: High

#### Implementation Steps

1. **Enhanced Query Models**
   ```python
   # src/morag_api/models/enhanced_query.py
   from typing import Dict, Any, List, Optional, Union
   from pydantic import BaseModel, Field
   from datetime import datetime
   from enum import Enum
   
   class QueryType(str, Enum):
       SIMPLE = "simple"
       ENTITY_FOCUSED = "entity_focused"
       RELATION_FOCUSED = "relation_focused"
       MULTI_HOP = "multi_hop"
       ANALYTICAL = "analytical"
   
   class ExpansionStrategy(str, Enum):
       DIRECT_NEIGHBORS = "direct_neighbors"
       BREADTH_FIRST = "breadth_first"
       SHORTEST_PATH = "shortest_path"
       ADAPTIVE = "adaptive"
       NONE = "none"
   
   class FusionStrategy(str, Enum):
       WEIGHTED = "weighted"
       RRF = "reciprocal_rank_fusion"
       ADAPTIVE = "adaptive"
       VECTOR_ONLY = "vector_only"
       GRAPH_ONLY = "graph_only"
   
   class EnhancedQueryRequest(BaseModel):
       query: str = Field(..., description="The user's query text")
       query_type: QueryType = Field(default=QueryType.SIMPLE, description="Type of query processing")
       max_results: int = Field(default=10, ge=1, le=100, description="Maximum number of results")
       
       # Graph-specific parameters
       expansion_strategy: ExpansionStrategy = Field(default=ExpansionStrategy.ADAPTIVE, description="Context expansion strategy")
       expansion_depth: int = Field(default=2, ge=1, le=5, description="Maximum expansion depth")
       fusion_strategy: FusionStrategy = Field(default=FusionStrategy.ADAPTIVE, description="Result fusion strategy")
       
       # Filtering and constraints
       entity_types: Optional[List[str]] = Field(default=None, description="Filter by entity types")
       relation_types: Optional[List[str]] = Field(default=None, description="Filter by relation types")
       time_range: Optional[Dict[str, datetime]] = Field(default=None, description="Time range filter")
       
       # Advanced options
       include_graph_context: bool = Field(default=True, description="Include graph context in response")
       include_reasoning_path: bool = Field(default=False, description="Include reasoning path")
       enable_multi_hop: bool = Field(default=True, description="Enable multi-hop reasoning")
       
       # Quality and performance
       min_relevance_score: float = Field(default=0.1, ge=0.0, le=1.0, description="Minimum relevance threshold")
       timeout_seconds: int = Field(default=30, ge=1, le=300, description="Query timeout")
   
   class EntityInfo(BaseModel):
       id: str
       name: str
       type: str
       properties: Dict[str, Any] = Field(default_factory=dict)
       relevance_score: float = Field(ge=0.0, le=1.0)
       source_documents: List[str] = Field(default_factory=list)
   
   class RelationInfo(BaseModel):
       id: str
       source_entity: str
       target_entity: str
       relation_type: str
       properties: Dict[str, Any] = Field(default_factory=dict)
       confidence: float = Field(ge=0.0, le=1.0)
       source_documents: List[str] = Field(default_factory=list)
   
   class GraphContext(BaseModel):
       entities: Dict[str, EntityInfo] = Field(default_factory=dict)
       relations: List[RelationInfo] = Field(default_factory=list)
       expansion_path: List[str] = Field(default_factory=list)
       reasoning_steps: Optional[List[str]] = Field(default=None)
   
   class EnhancedResult(BaseModel):
       id: str
       content: str
       relevance_score: float = Field(ge=0.0, le=1.0)
       source_type: str  # "vector", "graph", "hybrid"
       document_id: str
       metadata: Dict[str, Any] = Field(default_factory=dict)
       
       # Graph-specific fields
       connected_entities: List[str] = Field(default_factory=list)
       relation_context: List[RelationInfo] = Field(default_factory=list)
       reasoning_path: Optional[List[str]] = Field(default=None)
   
   class EnhancedQueryResponse(BaseModel):
       query_id: str
       query: str
       results: List[EnhancedResult]
       graph_context: GraphContext
       
       # Metadata
       total_results: int
       processing_time_ms: float
       fusion_strategy_used: FusionStrategy
       expansion_strategy_used: ExpansionStrategy
       
       # Quality metrics
       confidence_score: float = Field(ge=0.0, le=1.0)
       completeness_score: float = Field(ge=0.0, le=1.0)
       
       # Debug information (optional)
       debug_info: Optional[Dict[str, Any]] = Field(default=None)
   
   class EntityQueryRequest(BaseModel):
       entity_id: Optional[str] = Field(default=None, description="Specific entity ID")
       entity_name: Optional[str] = Field(default=None, description="Entity name to search")
       entity_type: Optional[str] = Field(default=None, description="Entity type filter")
       
       include_relations: bool = Field(default=True, description="Include entity relations")
       relation_depth: int = Field(default=1, ge=1, le=3, description="Relation traversal depth")
       max_relations: int = Field(default=50, ge=1, le=200, description="Maximum relations to return")
   
   class GraphTraversalRequest(BaseModel):
       start_entity: str = Field(..., description="Starting entity ID")
       end_entity: Optional[str] = Field(default=None, description="Target entity ID")
       
       traversal_type: str = Field(default="shortest_path", description="Type of traversal")
       max_depth: int = Field(default=3, ge=1, le=6, description="Maximum traversal depth")
       max_paths: int = Field(default=10, ge=1, le=50, description="Maximum paths to return")
       
       relation_filters: Optional[List[str]] = Field(default=None, description="Allowed relation types")
       entity_filters: Optional[List[str]] = Field(default=None, description="Allowed entity types")
   
   class GraphPath(BaseModel):
       entities: List[str]
       relations: List[str]
       total_weight: float
       confidence: float = Field(ge=0.0, le=1.0)
   
   class GraphTraversalResponse(BaseModel):
       start_entity: str
       end_entity: Optional[str]
       paths: List[GraphPath]
       total_paths_found: int
       processing_time_ms: float
   ```

2. **Enhanced API Endpoints**
   ```python
   # src/morag_api/endpoints/enhanced_query.py
   from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
   from fastapi.responses import StreamingResponse
   from typing import Dict, Any, Optional
   import asyncio
   import json
   import uuid
   from datetime import datetime
   import logging
   
   from ..models.enhanced_query import (
       EnhancedQueryRequest, EnhancedQueryResponse, EntityQueryRequest,
       GraphTraversalRequest, GraphTraversalResponse
   )
   from ..dependencies import get_hybrid_retrieval_system, get_graph_engine
   from ..utils.response_builder import EnhancedResponseBuilder
   from ..utils.query_validator import QueryValidator
   
   router = APIRouter(prefix="/api/v2", tags=["enhanced-query"])
   logger = logging.getLogger(__name__)
   
   @router.post("/query", response_model=EnhancedQueryResponse)
   async def enhanced_query(
       request: EnhancedQueryRequest,
       hybrid_system = Depends(get_hybrid_retrieval_system),
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
           
           # Process query
           logger.info(f"Processing enhanced query {query_id}: {request.query[:100]}...")
           
           # Configure retrieval based on request parameters
           retrieval_config = {
               "expansion_strategy": request.expansion_strategy,
               "expansion_depth": request.expansion_depth,
               "fusion_strategy": request.fusion_strategy,
               "max_results": request.max_results,
               "min_relevance_score": request.min_relevance_score,
               "entity_types": request.entity_types,
               "relation_types": request.relation_types,
               "enable_multi_hop": request.enable_multi_hop
           }
           
           # Execute retrieval
           retrieval_result = await hybrid_system.retrieve(
               query=request.query,
               query_type=request.query_type,
               config=retrieval_config
           )
           
           # Build response
           response_builder = EnhancedResponseBuilder()
           response = await response_builder.build_response(
               query_id=query_id,
               request=request,
               retrieval_result=retrieval_result,
               processing_time=(datetime.now() - start_time).total_seconds() * 1000
           )
           
           # Log query for analytics (background task)
           background_tasks.add_task(
               log_query_analytics,
               query_id=query_id,
               request=request,
               response=response,
               processing_time=response.processing_time_ms
           )
           
           return response
           
       except asyncio.TimeoutError:
           raise HTTPException(
               status_code=408,
               detail=f"Query timeout after {request.timeout_seconds} seconds"
           )
       except Exception as e:
           logger.error(f"Error processing query {query_id}: {str(e)}")
           raise HTTPException(
               status_code=500,
               detail=f"Internal server error: {str(e)}"
           )
   
   @router.post("/query/stream")
   async def enhanced_query_stream(
       request: EnhancedQueryRequest,
       hybrid_system = Depends(get_hybrid_retrieval_system)
   ):
       """Streaming version of enhanced query for real-time results."""
       query_id = str(uuid.uuid4())
       
       async def generate_stream():
           try:
               # Stream results as they become available
               async for partial_result in hybrid_system.retrieve_stream(
                   query=request.query,
                   query_type=request.query_type,
                   config={
                       "expansion_strategy": request.expansion_strategy,
                       "expansion_depth": request.expansion_depth,
                       "fusion_strategy": request.fusion_strategy,
                       "max_results": request.max_results
                   }
               ):
                   yield f"data: {json.dumps(partial_result.dict())}\n\n"
               
               # Send completion signal
               yield f"data: {{\"status\": \"complete\", \"query_id\": \"{query_id}\"}}\n\n"
               
           except Exception as e:
               error_data = {"error": str(e), "query_id": query_id}
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
           
           return {
               "entity": entity.dict(),
               "relations": [r.dict() for r in relations],
               "relation_count": len(relations)
           }
           
       except Exception as e:
           logger.error(f"Error in entity query: {str(e)}")
           raise HTTPException(status_code=500, detail=str(e))
   
   @router.post("/graph/traverse", response_model=GraphTraversalResponse)
   async def graph_traversal(
       request: GraphTraversalRequest,
       graph_engine = Depends(get_graph_engine)
   ):
       """Perform graph traversal between entities."""
       start_time = datetime.now()
       
       try:
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
           if request.traversal_type == "shortest_path" and end_entity:
               paths = await graph_engine.find_shortest_paths(
                   start_entity.id,
                   end_entity.id,
                   max_paths=request.max_paths,
                   relation_filters=request.relation_filters
               )
           elif request.traversal_type == "explore":
               paths = await graph_engine.explore_from_entity(
                   start_entity.id,
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
           
           return GraphTraversalResponse(
               start_entity=request.start_entity,
               end_entity=request.end_entity,
               paths=paths,
               total_paths_found=len(paths),
               processing_time_ms=processing_time
           )
           
       except Exception as e:
           logger.error(f"Error in graph traversal: {str(e)}")
           raise HTTPException(status_code=500, detail=str(e))
   
   @router.get("/graph/analytics")
   async def graph_analytics(
       metric_type: str = "overview",
       graph_engine = Depends(get_graph_engine)
   ):
       """Get graph analytics and statistics."""
       try:
           if metric_type == "overview":
               stats = await graph_engine.get_graph_statistics()
               return {
                   "entity_count": stats.entity_count,
                   "relation_count": stats.relation_count,
                   "avg_degree": stats.avg_degree,
                   "connected_components": stats.connected_components,
                   "density": stats.density
               }
           elif metric_type == "centrality":
               centrality = await graph_engine.calculate_centrality_measures()
               return {
                   "top_entities_by_degree": centrality.degree_centrality[:10],
                   "top_entities_by_betweenness": centrality.betweenness_centrality[:10],
                   "top_entities_by_pagerank": centrality.pagerank[:10]
               }
           elif metric_type == "communities":
               communities = await graph_engine.detect_communities()
               return {
                   "community_count": len(communities),
                   "largest_community_size": max(len(c.entities) for c in communities),
                   "modularity_score": communities[0].modularity if communities else 0
               }
           else:
               raise HTTPException(
                   status_code=400,
                   detail=f"Unknown metric type: {metric_type}"
               )
               
       except Exception as e:
           logger.error(f"Error getting graph analytics: {str(e)}")
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
           
           # Send to analytics service (implement based on your setup)
           logger.info(f"Query analytics: {json.dumps(analytics_data)}")
           
       except Exception as e:
           logger.error(f"Error logging analytics: {str(e)}")
   ```

#### Deliverables
- Enhanced query request/response models
- New API endpoints for graph-augmented queries
- Entity-specific query endpoints
- Graph traversal endpoints
- Streaming query support
- Analytics endpoints

### 3.2.2: Backward Compatibility
**Estimated Time**: 2 days  
**Priority**: High

#### Implementation Steps

1. **Legacy Endpoint Wrapper**
   ```python
   # src/morag_api/endpoints/legacy.py
   from fastapi import APIRouter, HTTPException, Depends
   from typing import Dict, Any, List, Optional
   import logging
   import warnings
   
   from ..models.legacy import LegacyQueryRequest, LegacyQueryResponse
   from ..models.enhanced_query import EnhancedQueryRequest, QueryType, ExpansionStrategy
   from ..dependencies import get_hybrid_retrieval_system
   from .enhanced_query import enhanced_query
   
   router = APIRouter(prefix="/api/v1", tags=["legacy"])
   logger = logging.getLogger(__name__)
   
   @router.post("/query", response_model=LegacyQueryResponse)
   async def legacy_query(
       request: LegacyQueryRequest,
       hybrid_system = Depends(get_hybrid_retrieval_system)
   ):
       """Legacy query endpoint for backward compatibility."""
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
               legacy_results.append({
                   "id": result.id,
                   "content": result.content,
                   "score": result.relevance_score,
                   "metadata": {
                       "document_id": result.document_id,
                       "source_type": result.source_type,
                       **result.metadata
                   }
               })
           
           return LegacyQueryResponse(
               query=request.query,
               results=legacy_results,
               total_results=len(legacy_results),
               processing_time_ms=enhanced_response.processing_time_ms
           )
           
       except Exception as e:
           logger.error(f"Error in legacy query: {str(e)}")
           raise HTTPException(status_code=500, detail=str(e))
   
   @router.get("/health")
   async def legacy_health_check():
       """Legacy health check endpoint."""
       return {"status": "healthy", "version": "1.0", "deprecated": True}
   
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
                   "Improved result ranking and fusion"
               ],
               "breaking_changes": [
                   "Response format includes additional fields",
                   "Some field names have changed",
                   "New required fields in request model"
               ],
               "migration_steps": [
                   "1. Update client to use /api/v2/query endpoint",
                   "2. Update request model to EnhancedQueryRequest",
                   "3. Handle new response fields in EnhancedQueryResponse",
                   "4. Test with your existing queries",
                   "5. Gradually adopt new features like graph context"
               ],
               "support_timeline": {
                   "deprecation_date": "2024-01-01",
                   "end_of_support_date": "2024-06-01",
                   "removal_date": "2024-12-01"
               }
           }
       }
   ```

2. **Migration Configuration**
   ```python
   # src/morag_api/config/migration.py
   from typing import Dict, Any, Optional
   from dataclasses import dataclass
   from datetime import datetime, date
   import logging
   
   @dataclass
   class MigrationConfig:
       enable_deprecation_warnings: bool = True
       log_legacy_usage: bool = True
       legacy_endpoint_enabled: bool = True
       migration_deadline: Optional[date] = None
       
       # Feature flags for gradual migration
       enable_graph_context_in_legacy: bool = False
       enable_enhanced_scoring_in_legacy: bool = True
       
       # Rate limiting for legacy endpoints
       legacy_rate_limit_per_minute: int = 100
       
   class MigrationManager:
       def __init__(self, config: MigrationConfig):
           self.config = config
           self.logger = logging.getLogger(__name__)
           self.legacy_usage_stats = {
               "total_requests": 0,
               "unique_clients": set(),
               "first_seen": datetime.now(),
               "last_seen": None
           }
       
       def log_legacy_usage(self, client_id: str, endpoint: str):
           """Log usage of legacy endpoints."""
           if not self.config.log_legacy_usage:
               return
           
           self.legacy_usage_stats["total_requests"] += 1
           self.legacy_usage_stats["unique_clients"].add(client_id)
           self.legacy_usage_stats["last_seen"] = datetime.now()
           
           self.logger.info(
               f"Legacy API usage - Client: {client_id}, Endpoint: {endpoint}, "
               f"Total requests: {self.legacy_usage_stats['total_requests']}"
           )
       
       def should_show_deprecation_warning(self) -> bool:
           """Determine if deprecation warning should be shown."""
           return self.config.enable_deprecation_warnings
       
       def is_migration_deadline_passed(self) -> bool:
           """Check if migration deadline has passed."""
           if not self.config.migration_deadline:
               return False
           return date.today() > self.config.migration_deadline
       
       def get_migration_status(self) -> Dict[str, Any]:
           """Get current migration status."""
           return {
               "legacy_endpoint_enabled": self.config.legacy_endpoint_enabled,
               "migration_deadline": self.config.migration_deadline.isoformat() if self.config.migration_deadline else None,
               "days_until_deadline": (
                   (self.config.migration_deadline - date.today()).days
                   if self.config.migration_deadline else None
               ),
               "legacy_usage_stats": {
                   "total_requests": self.legacy_usage_stats["total_requests"],
                   "unique_clients": len(self.legacy_usage_stats["unique_clients"]),
                   "first_seen": self.legacy_usage_stats["first_seen"].isoformat(),
                   "last_seen": self.legacy_usage_stats["last_seen"].isoformat() if self.legacy_usage_stats["last_seen"] else None
               }
           }
   ```

#### Deliverables
- Legacy endpoint wrapper maintaining API contracts
- Migration configuration and management
- Deprecation warnings and guidance
- Usage tracking for legacy endpoints

## Testing Requirements

### Unit Tests
```python
# tests/test_enhanced_api.py
import pytest
from fastapi.testclient import TestClient
from morag_api.models.enhanced_query import (
    EnhancedQueryRequest, QueryType, ExpansionStrategy
)

class TestEnhancedQueryEndpoints:
    def test_enhanced_query_basic(self, test_client: TestClient):
        request_data = {
            "query": "What is machine learning?",
            "query_type": "simple",
            "max_results": 5
        }
        
        response = test_client.post("/api/v2/query", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "query_id" in data
        assert "results" in data
        assert "graph_context" in data
        assert len(data["results"]) <= 5
    
    def test_enhanced_query_with_graph_context(self, test_client: TestClient):
        request_data = {
            "query": "Tell me about neural networks",
            "query_type": "entity_focused",
            "expansion_strategy": "breadth_first",
            "expansion_depth": 2,
            "include_graph_context": True
        }
        
        response = test_client.post("/api/v2/query", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["graph_context"]["entities"]
        assert "expansion_path" in data["graph_context"]
    
    def test_entity_query(self, test_client: TestClient):
        request_data = {
            "entity_name": "machine learning",
            "include_relations": True,
            "relation_depth": 2
        }
        
        response = test_client.post("/api/v2/entity/query", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "entity" in data
        assert "relations" in data
    
    def test_graph_traversal(self, test_client: TestClient):
        request_data = {
            "start_entity": "entity_1",
            "end_entity": "entity_2",
            "traversal_type": "shortest_path",
            "max_depth": 3
        }
        
        response = test_client.post("/api/v2/graph/traverse", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "paths" in data
        assert "processing_time_ms" in data

class TestBackwardCompatibility:
    def test_legacy_query_compatibility(self, test_client: TestClient):
        request_data = {
            "query": "What is AI?",
            "max_results": 10
        }
        
        response = test_client.post("/api/v1/query", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "query" in data
        assert "results" in data
        assert "total_results" in data
        
        # Check legacy format
        for result in data["results"]:
            assert "id" in result
            assert "content" in result
            assert "score" in result
    
    def test_migration_guide(self, test_client: TestClient):
        response = test_client.get("/api/v1/migration-guide")
        assert response.status_code == 200
        
        data = response.json()
        assert "migration_guide" in data
        assert "migration_steps" in data["migration_guide"]
```

### Integration Tests
```python
# tests/integration/test_api_integration.py
class TestAPIIntegration:
    @pytest.mark.asyncio
    async def test_end_to_end_enhanced_query(self, test_system):
        """Test complete enhanced query pipeline."""
        # Test with real system components
        request = EnhancedQueryRequest(
            query="How does deep learning work?",
            query_type=QueryType.MULTI_HOP,
            expansion_strategy=ExpansionStrategy.ADAPTIVE,
            include_graph_context=True,
            include_reasoning_path=True
        )
        
        response = await test_system.process_enhanced_query(request)
        
        # Validate response structure
        assert response.query_id
        assert len(response.results) > 0
        assert response.graph_context.entities
        assert response.confidence_score > 0
        
        # Validate graph context
        for entity_id, entity in response.graph_context.entities.items():
            assert entity.relevance_score >= 0
            assert entity.type
        
        # Validate reasoning path if included
        if response.graph_context.reasoning_steps:
            assert len(response.graph_context.reasoning_steps) > 0
```

## Success Criteria

- [ ] Enhanced API endpoints provide comprehensive graph-augmented query capabilities
- [ ] Backward compatibility maintained for existing clients
- [ ] Request/response models are well-documented and validated
- [ ] Streaming query support works correctly
- [ ] Entity and graph traversal endpoints function properly
- [ ] Migration guidance is clear and helpful
- [ ] Performance meets requirements (< 2s for standard queries)
- [ ] Unit test coverage > 90%
- [ ] Integration tests validate end-to-end functionality

## Performance Targets

- **Enhanced Query Response**: < 2 seconds for standard queries
- **Entity Query Response**: < 500ms for single entity queries
- **Graph Traversal**: < 1 second for paths up to depth 3
- **Streaming Latency**: < 100ms for first result
- **Legacy Compatibility**: No performance degradation
- **Concurrent Requests**: Support 100+ concurrent enhanced queries

## Next Steps

After completing this task:
1. Deploy enhanced API endpoints to staging environment
2. Update API documentation with new endpoints
3. Create client SDK examples for enhanced features
4. Begin migration communication to existing API users

## Dependencies

**Requires**:
- Task 3.1: Hybrid Retrieval
- FastAPI framework
- Pydantic for request/response models

**Enables**:
- Client applications to use graph-augmented retrieval
- Advanced query capabilities for end users
- Gradual migration from legacy API