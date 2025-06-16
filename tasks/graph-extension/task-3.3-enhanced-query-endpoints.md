# Task 3.3: Enhanced Query Endpoints

**Phase**: 3 - Retrieval Integration  
**Priority**: High  
**Estimated Time**: 5-7 days total  
**Dependencies**: Task 3.1 (Hybrid Retrieval), Task 3.2 (Sparse Vector Integration)

## Overview

This task implements new API endpoints that expose the graph-enhanced retrieval capabilities to users while maintaining backward compatibility with existing endpoints. The enhanced endpoints provide access to hybrid retrieval, graph traversal, entity extraction, and advanced query features.

## Subtasks

### 3.3.1: New API Endpoints
**Estimated Time**: 3-4 days  
**Priority**: High

#### Implementation Steps

1. **Enhanced Query Models**
   ```python
   # src/morag_api/models/enhanced_query.py
   from typing import List, Dict, Any, Optional, Union
   from pydantic import BaseModel, Field
   from enum import Enum
   
   class RetrievalMethod(str, Enum):
       DENSE = "dense"
       SPARSE = "sparse"
       HYBRID = "hybrid"
       GRAPH = "graph"
       AUTO = "auto"
   
   class ContextExpansionStrategy(str, Enum):
       NONE = "none"
       DIRECT_NEIGHBORS = "direct_neighbors"
       BREADTH_FIRST = "breadth_first"
       SHORTEST_PATH = "shortest_path"
       ADAPTIVE = "adaptive"
   
   class FusionMethod(str, Enum):
       WEIGHTED = "weighted"
       RECIPROCAL_RANK = "reciprocal_rank"
       ROUND_ROBIN = "round_robin"
       ADAPTIVE = "adaptive"
   
   class EnhancedQueryRequest(BaseModel):
       query: str = Field(..., description="The search query")
       retrieval_method: RetrievalMethod = Field(
           default=RetrievalMethod.AUTO,
           description="Retrieval method to use"
       )
       context_expansion: ContextExpansionStrategy = Field(
           default=ContextExpansionStrategy.ADAPTIVE,
           description="Context expansion strategy for graph-based retrieval"
       )
       fusion_method: FusionMethod = Field(
           default=FusionMethod.ADAPTIVE,
           description="Method for fusing results from different retrievers"
       )
       max_results: int = Field(
           default=20,
           ge=1,
           le=100,
           description="Maximum number of results to return"
       )
       include_entities: bool = Field(
           default=True,
           description="Whether to include extracted entities in response"
       )
       include_relations: bool = Field(
           default=False,
           description="Whether to include extracted relations in response"
       )
       include_graph_context: bool = Field(
           default=True,
           description="Whether to include graph context information"
       )
       filters: Optional[Dict[str, Any]] = Field(
           default=None,
           description="Additional filters to apply"
       )
       weights: Optional[Dict[str, float]] = Field(
           default=None,
           description="Custom weights for different retrieval methods"
       )
   
   class EntityInfo(BaseModel):
       text: str
       label: str
       confidence: float
       start_pos: int
       end_pos: int
       metadata: Optional[Dict[str, Any]] = None
   
   class RelationInfo(BaseModel):
       subject: str
       predicate: str
       object: str
       confidence: float
       metadata: Optional[Dict[str, Any]] = None
   
   class GraphContext(BaseModel):
       related_entities: List[str]
       expansion_paths: List[List[str]]
       centrality_scores: Optional[Dict[str, float]] = None
       community_info: Optional[Dict[str, Any]] = None
   
   class EnhancedResult(BaseModel):
       id: str
       content: str
       score: float
       source: str  # "dense", "sparse", "graph", "hybrid"
       metadata: Dict[str, Any]
       entities: Optional[List[EntityInfo]] = None
       relations: Optional[List[RelationInfo]] = None
       graph_context: Optional[GraphContext] = None
   
   class EnhancedQueryResponse(BaseModel):
       query: str
       results: List[EnhancedResult]
       total_results: int
       retrieval_method_used: str
       processing_time_ms: float
       query_entities: Optional[List[EntityInfo]] = None
       query_intent: Optional[str] = None
       expansion_strategy_used: Optional[str] = None
       fusion_method_used: Optional[str] = None
   
   class EntityQueryRequest(BaseModel):
       entity: str = Field(..., description="Entity to search for")
       entity_type: Optional[str] = Field(
           default=None,
           description="Type of entity (PERSON, ORG, etc.)"
       )
       max_results: int = Field(
           default=20,
           ge=1,
           le=100,
           description="Maximum number of results"
       )
       include_related: bool = Field(
           default=True,
           description="Include related entities"
       )
       max_hops: int = Field(
           default=2,
           ge=1,
           le=5,
           description="Maximum hops for related entities"
       )
   
   class GraphTraversalRequest(BaseModel):
       start_entity: str = Field(..., description="Starting entity for traversal")
       end_entity: Optional[str] = Field(
           default=None,
           description="Target entity (for path finding)"
       )
       traversal_type: str = Field(
           default="bfs",
           description="Traversal algorithm (bfs, dfs, shortest_path)"
       )
       max_depth: int = Field(
           default=3,
           ge=1,
           le=10,
           description="Maximum traversal depth"
       )
       filters: Optional[Dict[str, Any]] = Field(
           default=None,
           description="Filters for entities and relations"
       )
   
   class GraphPath(BaseModel):
       entities: List[str]
       relations: List[str]
       total_weight: float
       confidence: float
   
   class GraphTraversalResponse(BaseModel):
       start_entity: str
       end_entity: Optional[str]
       paths: List[GraphPath]
       visited_entities: List[str]
       processing_time_ms: float
   ```

2. **Enhanced Query Endpoints**
   ```python
   # src/morag_api/routers/enhanced_query.py
   from fastapi import APIRouter, HTTPException, Depends
   from typing import List, Dict, Any
   import time
   import logging
   
   from morag_api.models.enhanced_query import (
       EnhancedQueryRequest, EnhancedQueryResponse, EnhancedResult,
       EntityQueryRequest, GraphTraversalRequest, GraphTraversalResponse
   )
   from morag_retrieval.hybrid import HybridRetrievalCoordinator
   from morag_graph.query import GraphQueryEngine
   from morag_nlp.extractors import QueryEntityExtractor
   
   router = APIRouter(prefix="/api/v2", tags=["enhanced-query"])
   logger = logging.getLogger(__name__)
   
   @router.post("/query/enhanced", response_model=EnhancedQueryResponse)
   async def enhanced_query(
       request: EnhancedQueryRequest,
       retrieval_coordinator: HybridRetrievalCoordinator = Depends(get_retrieval_coordinator),
       entity_extractor: QueryEntityExtractor = Depends(get_entity_extractor)
   ):
       """Enhanced query endpoint with graph-augmented retrieval."""
       start_time = time.time()
       
       try:
           # Extract entities from query
           query_entities = None
           if request.include_entities:
               query_entities = await entity_extractor.extract_entities(request.query)
           
           # Perform retrieval
           retrieval_result = await retrieval_coordinator.search(
               query=request.query,
               method=request.retrieval_method,
               context_expansion=request.context_expansion,
               fusion_method=request.fusion_method,
               max_results=request.max_results,
               filters=request.filters,
               weights=request.weights
           )
           
           # Process results
           enhanced_results = []
           for result in retrieval_result.results:
               enhanced_result = EnhancedResult(
                   id=result['id'],
                   content=result['content'],
                   score=result['score'],
                   source=result.get('source', 'unknown'),
                   metadata=result.get('metadata', {})
               )
               
               # Add entities if requested
               if request.include_entities and 'entities' in result:
                   enhanced_result.entities = result['entities']
               
               # Add relations if requested
               if request.include_relations and 'relations' in result:
                   enhanced_result.relations = result['relations']
               
               # Add graph context if requested
               if request.include_graph_context and 'graph_context' in result:
                   enhanced_result.graph_context = result['graph_context']
               
               enhanced_results.append(enhanced_result)
           
           processing_time = (time.time() - start_time) * 1000
           
           return EnhancedQueryResponse(
               query=request.query,
               results=enhanced_results,
               total_results=len(enhanced_results),
               retrieval_method_used=retrieval_result.method_used,
               processing_time_ms=processing_time,
               query_entities=query_entities,
               query_intent=retrieval_result.query_intent,
               expansion_strategy_used=retrieval_result.expansion_strategy,
               fusion_method_used=retrieval_result.fusion_method
           )
       
       except Exception as e:
           logger.error(f"Error in enhanced query: {str(e)}")
           raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")
   
   @router.post("/query/entity", response_model=EnhancedQueryResponse)
   async def entity_query(
       request: EntityQueryRequest,
       graph_engine: GraphQueryEngine = Depends(get_graph_engine)
   ):
       """Query for documents related to a specific entity."""
       start_time = time.time()
       
       try:
           # Find entity in graph
           entity_results = await graph_engine.find_entity_documents(
               entity=request.entity,
               entity_type=request.entity_type,
               max_results=request.max_results,
               include_related=request.include_related,
               max_hops=request.max_hops
           )
           
           # Convert to enhanced results
           enhanced_results = []
           for result in entity_results:
               enhanced_result = EnhancedResult(
                   id=result['document_id'],
                   content=result['content'],
                   score=result['relevance_score'],
                   source="graph",
                   metadata=result.get('metadata', {}),
                   graph_context=result.get('graph_context')
               )
               enhanced_results.append(enhanced_result)
           
           processing_time = (time.time() - start_time) * 1000
           
           return EnhancedQueryResponse(
               query=f"Entity: {request.entity}",
               results=enhanced_results,
               total_results=len(enhanced_results),
               retrieval_method_used="graph_entity",
               processing_time_ms=processing_time
           )
       
       except Exception as e:
           logger.error(f"Error in entity query: {str(e)}")
           raise HTTPException(status_code=500, detail=f"Entity query failed: {str(e)}")
   
   @router.post("/graph/traverse", response_model=GraphTraversalResponse)
   async def graph_traversal(
       request: GraphTraversalRequest,
       graph_engine: GraphQueryEngine = Depends(get_graph_engine)
   ):
       """Perform graph traversal starting from an entity."""
       start_time = time.time()
       
       try:
           if request.end_entity:
               # Path finding between two entities
               paths = await graph_engine.find_paths(
                   start_entity=request.start_entity,
                   end_entity=request.end_entity,
                   algorithm=request.traversal_type,
                   max_depth=request.max_depth,
                   filters=request.filters
               )
               visited_entities = list(set(
                   entity for path in paths for entity in path['entities']
               ))
           else:
               # General traversal from start entity
               traversal_result = await graph_engine.traverse(
                   start_entity=request.start_entity,
                   algorithm=request.traversal_type,
                   max_depth=request.max_depth,
                   filters=request.filters
               )
               paths = traversal_result['paths']
               visited_entities = traversal_result['visited_entities']
           
           processing_time = (time.time() - start_time) * 1000
           
           return GraphTraversalResponse(
               start_entity=request.start_entity,
               end_entity=request.end_entity,
               paths=paths,
               visited_entities=visited_entities,
               processing_time_ms=processing_time
           )
       
       except Exception as e:
           logger.error(f"Error in graph traversal: {str(e)}")
           raise HTTPException(status_code=500, detail=f"Graph traversal failed: {str(e)}")
   
   @router.get("/graph/entities/{entity_id}/related")
   async def get_related_entities(
       entity_id: str,
       max_results: int = 20,
       max_hops: int = 2,
       graph_engine: GraphQueryEngine = Depends(get_graph_engine)
   ):
       """Get entities related to a specific entity."""
       try:
           related_entities = await graph_engine.get_related_entities(
               entity_id=entity_id,
               max_results=max_results,
               max_hops=max_hops
           )
           
           return {
               "entity_id": entity_id,
               "related_entities": related_entities,
               "total_count": len(related_entities)
           }
       
       except Exception as e:
           logger.error(f"Error getting related entities: {str(e)}")
           raise HTTPException(status_code=500, detail=f"Failed to get related entities: {str(e)}")
   
   @router.get("/graph/analytics/centrality")
   async def get_centrality_metrics(
       entity_type: Optional[str] = None,
       top_k: int = 50,
       graph_engine: GraphQueryEngine = Depends(get_graph_engine)
   ):
       """Get centrality metrics for entities in the graph."""
       try:
           centrality_metrics = await graph_engine.calculate_centrality(
               entity_type=entity_type,
               top_k=top_k
           )
           
           return {
               "centrality_metrics": centrality_metrics,
               "entity_type_filter": entity_type,
               "top_k": top_k
           }
       
       except Exception as e:
           logger.error(f"Error calculating centrality: {str(e)}")
           raise HTTPException(status_code=500, detail=f"Failed to calculate centrality: {str(e)}")
   ```

#### Deliverables
- Enhanced query request/response models
- New API endpoints for graph-enhanced queries
- Entity-specific query endpoints
- Graph traversal endpoints
- Analytics endpoints for graph metrics

### 3.3.2: Backward Compatibility
**Estimated Time**: 2-3 days  
**Priority**: High

#### Implementation Steps

1. **Legacy Endpoint Wrapper**
   ```python
   # src/morag_api/routers/legacy_compatibility.py
   from fastapi import APIRouter, Depends
   from morag_api.models.query import QueryRequest, QueryResponse  # Original models
   from morag_api.models.enhanced_query import EnhancedQueryRequest, RetrievalMethod
   from morag_api.routers.enhanced_query import enhanced_query
   
   router = APIRouter(prefix="/api/v1", tags=["legacy-query"])
   
   @router.post("/query", response_model=QueryResponse)
   async def legacy_query(
       request: QueryRequest,
       retrieval_coordinator = Depends(get_retrieval_coordinator)
   ):
       """Legacy query endpoint with backward compatibility."""
       # Convert legacy request to enhanced request
       enhanced_request = EnhancedQueryRequest(
           query=request.query,
           retrieval_method=RetrievalMethod.HYBRID,  # Default to hybrid for better results
           max_results=request.limit or 20,
           include_entities=False,  # Legacy doesn't include entities
           include_relations=False,
           include_graph_context=False
       )
       
       # Call enhanced query
       enhanced_response = await enhanced_query(enhanced_request, retrieval_coordinator)
       
       # Convert enhanced response to legacy format
       legacy_results = []
       for result in enhanced_response.results:
           legacy_results.append({
               'id': result.id,
               'content': result.content,
               'score': result.score,
               'metadata': result.metadata
           })
       
       return QueryResponse(
           query=request.query,
           results=legacy_results,
           total=enhanced_response.total_results
       )
   ```

2. **Migration Guide and Configuration**
   ```python
   # src/morag_api/config/migration.py
   from typing import Dict, Any
   from pydantic import BaseModel
   
   class MigrationConfig(BaseModel):
       enable_legacy_endpoints: bool = True
       legacy_default_retrieval_method: str = "hybrid"
       legacy_include_graph_features: bool = False
       migration_warnings: bool = True
       deprecation_date: Optional[str] = None
   
   class APIVersionManager:
       def __init__(self, config: MigrationConfig):
           self.config = config
           self.logger = logging.getLogger(__name__)
       
       def should_show_migration_warning(self, endpoint: str) -> bool:
           """Determine if migration warning should be shown."""
           return (
               self.config.migration_warnings and 
               endpoint.startswith("/api/v1/")
           )
       
       def get_migration_headers(self) -> Dict[str, str]:
           """Get headers to include in legacy responses."""
           headers = {}
           
           if self.config.deprecation_date:
               headers["X-API-Deprecation-Date"] = self.config.deprecation_date
           
           headers["X-API-Migration-Guide"] = "/docs/migration-guide"
           headers["X-API-Latest-Version"] = "v2"
           
           return headers
   ```

#### Deliverables
- Legacy endpoint wrappers maintaining original API contracts
- Migration configuration and management
- Deprecation warnings and migration guidance
- Comprehensive documentation for API migration

## Testing Requirements

### Unit Tests
```python
# tests/test_enhanced_endpoints.py
import pytest
from fastapi.testclient import TestClient
from morag_api.main import app

class TestEnhancedQueryEndpoints:
    def setup_method(self):
        self.client = TestClient(app)
    
    def test_enhanced_query_basic(self):
        request_data = {
            "query": "machine learning algorithms",
            "retrieval_method": "hybrid",
            "max_results": 10
        }
        
        response = self.client.post("/api/v2/query/enhanced", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "query" in data
        assert "results" in data
        assert "processing_time_ms" in data
        assert len(data["results"]) <= 10
    
    def test_enhanced_query_with_entities(self):
        request_data = {
            "query": "Apple Inc. and Microsoft collaboration",
            "include_entities": True,
            "include_relations": True
        }
        
        response = self.client.post("/api/v2/query/enhanced", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "query_entities" in data
        # Check if entities are extracted from results
        if data["results"]:
            result = data["results"][0]
            assert "entities" in result or result["entities"] is None
    
    def test_entity_query(self):
        request_data = {
            "entity": "Apple Inc.",
            "entity_type": "ORG",
            "max_results": 5,
            "include_related": True
        }
        
        response = self.client.post("/api/v2/query/entity", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["retrieval_method_used"] == "graph_entity"
        assert len(data["results"]) <= 5
    
    def test_graph_traversal(self):
        request_data = {
            "start_entity": "Apple Inc.",
            "traversal_type": "bfs",
            "max_depth": 2
        }
        
        response = self.client.post("/api/v2/graph/traverse", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "start_entity" in data
        assert "paths" in data
        assert "visited_entities" in data

class TestBackwardCompatibility:
    def setup_method(self):
        self.client = TestClient(app)
    
    def test_legacy_query_compatibility(self):
        # Test that legacy v1 endpoint still works
        request_data = {
            "query": "test query",
            "limit": 10
        }
        
        response = self.client.post("/api/v1/query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check legacy response format
        assert "query" in data
        assert "results" in data
        assert "total" in data
        
        # Check migration headers
        assert "X-API-Latest-Version" in response.headers
    
    def test_legacy_response_format(self):
        # Ensure legacy responses don't include new fields
        request_data = {"query": "test", "limit": 5}
        
        response = self.client.post("/api/v1/query", json=request_data)
        data = response.json()
        
        if data["results"]:
            result = data["results"][0]
            # Legacy format should not include new fields
            assert "entities" not in result
            assert "relations" not in result
            assert "graph_context" not in result
```

### Integration Tests
```python
# tests/integration/test_enhanced_api_integration.py
class TestEnhancedAPIIntegration:
    @pytest.mark.asyncio
    async def test_end_to_end_enhanced_query(self, test_corpus, test_graph):
        """Test complete enhanced query pipeline."""
        # Set up test data
        await self.setup_test_data(test_corpus, test_graph)
        
        # Test various query types
        test_cases = [
            {
                "query": "machine learning research",
                "retrieval_method": "hybrid",
                "expected_sources": ["dense", "sparse", "graph"]
            },
            {
                "query": "Apple Inc. partnerships",
                "include_entities": True,
                "expected_entities": ["Apple Inc."]
            },
            {
                "query": "neural networks deep learning",
                "context_expansion": "adaptive",
                "include_graph_context": True
            }
        ]
        
        for test_case in test_cases:
            response = await self.client.post("/api/v2/query/enhanced", json=test_case)
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify response structure
            assert "results" in data
            assert "processing_time_ms" in data
            assert data["processing_time_ms"] < 5000  # Should be fast
            
            # Verify expected sources if specified
            if "expected_sources" in test_case:
                result_sources = set(r["source"] for r in data["results"])
                expected_sources = set(test_case["expected_sources"])
                assert len(result_sources.intersection(expected_sources)) > 0
            
            # Verify entities if requested
            if test_case.get("include_entities") and "expected_entities" in test_case:
                if data.get("query_entities"):
                    entity_texts = [e["text"] for e in data["query_entities"]]
                    for expected_entity in test_case["expected_entities"]:
                        assert any(expected_entity in text for text in entity_texts)
```

## Success Criteria

- [ ] All new API endpoints function correctly
- [ ] Enhanced query models validate input properly
- [ ] Backward compatibility maintained for v1 endpoints
- [ ] Response times meet performance targets
- [ ] Error handling provides meaningful feedback
- [ ] API documentation is comprehensive and accurate
- [ ] Migration path is clear and well-documented
- [ ] Unit test coverage > 90%
- [ ] Integration tests pass
- [ ] Load testing shows acceptable performance

## Performance Targets

- **Enhanced Query Response**: < 3 seconds for complex queries
- **Entity Query Response**: < 1 second for typical requests
- **Graph Traversal**: < 2 seconds for depth ≤ 3
- **Legacy Compatibility**: No performance degradation
- **Concurrent Requests**: Handle 100+ concurrent requests

## Next Steps

After completing this task:
1. Proceed to **Task 4.1**: LLM-Guided Path Selection
2. Implement comprehensive API documentation
3. Set up monitoring and analytics for new endpoints

## Dependencies

**Requires**:
- Task 3.1: Hybrid Retrieval System
- Task 3.2: Sparse Vector Integration

**Enables**:
- Task 4.1: LLM-Guided Path Selection
- Task 4.3: Caching Strategy
- Task 4.4: Parallel Processing