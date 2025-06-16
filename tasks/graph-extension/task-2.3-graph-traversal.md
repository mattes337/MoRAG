# Task 2.3: Basic Graph Traversal

**Phase**: 2 - Core Graph Features  
**Priority**: High  
**Estimated Time**: 7-9 days total  
**Dependencies**: Task 1.2.2 (Graph Storage Implementation)

## Overview

This task implements the core graph traversal and analytics capabilities that enable querying the knowledge graph to find related entities, compute paths, and analyze graph structure. These capabilities form the foundation for graph-guided retrieval and multi-hop reasoning.

## Subtasks

### 2.3.1: Graph Query Engine
**Estimated Time**: 4-5 days  
**Priority**: High

#### Implementation Steps

1. **Core Query Interface**
   ```python
   # src/morag_graph/query/query_engine.py
   from typing import List, Optional, Dict, Any, Set
   from dataclasses import dataclass
   from morag_graph.models import Entity, Relation, GraphPath
   from morag_graph.storage.base import BaseGraphStorage
   
   class GraphQueryEngine:
       def __init__(self, graph_storage: BaseGraphStorage):
           self.graph_storage = graph_storage
           self.logger = logging.getLogger(__name__)
       
       async def find_related_entities(
           self, 
           entity_id: str, 
           relation_types: Optional[List[str]] = None,
           max_depth: int = 2,
           max_results: int = 50
       ) -> List[GraphPath]:
           """Find entities related to the given entity within specified depth."""
           try:
               paths = await self._traverse_from_entity(
                   entity_id, relation_types, max_depth, max_results
               )
               return self._rank_paths(paths)
           except Exception as e:
               self.logger.error(f"Error finding related entities for {entity_id}: {str(e)}")
               raise QueryExecutionError(f"Failed to find related entities: {str(e)}")
       
       async def find_shortest_path(
           self, 
           start_entity: str, 
           end_entity: str,
           max_depth: int = 5
       ) -> Optional[GraphPath]:
           """Find shortest path between two entities using BFS."""
           if start_entity == end_entity:
               start = await self.graph_storage.get_entity(start_entity)
               return GraphPath(entities=[start], relations=[], total_weight=0.0)
           
           visited = set()
           queue = [(start_entity, GraphPath(entities=[], relations=[], total_weight=0.0))]
           
           while queue:
               current_entity_id, current_path = queue.pop(0)
               
               if current_entity_id in visited:
                   continue
               
               visited.add(current_entity_id)
               current_entity = await self.graph_storage.get_entity(current_entity_id)
               
               # Add current entity to path
               new_path = GraphPath(
                   entities=current_path.entities + [current_entity],
                   relations=current_path.relations,
                   total_weight=current_path.total_weight
               )
               
               if current_entity_id == end_entity:
                   return new_path
               
               if len(new_path.entities) >= max_depth:
                   continue
               
               # Get neighbors
               neighbors = await self.graph_storage.get_entity_neighbors(current_entity_id)
               
               for neighbor_id, relation in neighbors:
                   if neighbor_id not in visited:
                       extended_path = GraphPath(
                           entities=new_path.entities,
                           relations=new_path.relations + [relation],
                           total_weight=new_path.total_weight + relation.weight
                       )
                       queue.append((neighbor_id, extended_path))
           
           return None  # No path found
       
       async def find_paths_with_constraints(
           self,
           start_entity: str,
           constraints: PathConstraints
       ) -> List[GraphPath]:
           """Find paths that satisfy specific constraints."""
           paths = []
           visited = set()
           
           await self._dfs_with_constraints(
               start_entity, [], [], 0.0, constraints, paths, visited
           )
           
           return self._rank_paths(paths)
       
       async def _traverse_from_entity(
           self,
           entity_id: str,
           relation_types: Optional[List[str]],
           max_depth: int,
           max_results: int
       ) -> List[GraphPath]:
           """Traverse graph from entity using BFS."""
           paths = []
           visited = set()
           queue = [(entity_id, 0, GraphPath(entities=[], relations=[], total_weight=0.0))]
           
           while queue and len(paths) < max_results:
               current_id, depth, current_path = queue.pop(0)
               
               if current_id in visited or depth > max_depth:
                   continue
               
               visited.add(current_id)
               current_entity = await self.graph_storage.get_entity(current_id)
               
               # Create new path
               new_path = GraphPath(
                   entities=current_path.entities + [current_entity],
                   relations=current_path.relations,
                   total_weight=current_path.total_weight
               )
               
               if depth > 0:  # Don't include the starting entity as a result
                   paths.append(new_path)
               
               # Get neighbors
               neighbors = await self.graph_storage.get_entity_neighbors(
                   current_id, relation_types
               )
               
               for neighbor_id, relation in neighbors:
                   if neighbor_id not in visited:
                       extended_path = GraphPath(
                           entities=new_path.entities,
                           relations=new_path.relations + [relation],
                           total_weight=new_path.total_weight + relation.weight
                       )
                       queue.append((neighbor_id, depth + 1, extended_path))
           
           return paths
       
       def _rank_paths(self, paths: List[GraphPath]) -> List[GraphPath]:
           """Rank paths by relevance score."""
           def calculate_score(path: GraphPath) -> float:
               # Combine multiple factors for scoring
               length_penalty = 1.0 / (len(path.entities) ** 0.5)
               weight_score = 1.0 / (path.total_weight + 1.0)
               entity_importance = sum(e.importance_score for e in path.entities if hasattr(e, 'importance_score'))
               
               return length_penalty * weight_score * (1.0 + entity_importance)
           
           return sorted(paths, key=calculate_score, reverse=True)
   
   @dataclass
   class PathConstraints:
       required_entity_types: Optional[List[str]] = None
       required_relation_types: Optional[List[str]] = None
       forbidden_entity_types: Optional[List[str]] = None
       forbidden_relation_types: Optional[List[str]] = None
       min_path_length: int = 1
       max_path_length: int = 5
       max_total_weight: Optional[float] = None
   
   @dataclass
   class GraphPath:
       entities: List[Entity]
       relations: List[Relation]
       total_weight: float
       
       @property
       def length(self) -> int:
           return len(self.entities)
       
       def to_dict(self) -> Dict[str, Any]:
           return {
               "entities": [e.to_dict() for e in self.entities],
               "relations": [r.to_dict() for r in self.relations],
               "total_weight": self.total_weight,
               "length": self.length
           }
   ```

2. **Advanced Path Finding Algorithms**
   ```python
   # src/morag_graph/query/algorithms.py
   import heapq
   from typing import Dict, List, Tuple
   
   class PathFindingAlgorithms:
       def __init__(self, graph_storage: BaseGraphStorage):
           self.graph_storage = graph_storage
       
       async def dijkstra_shortest_path(
           self,
           start_entity: str,
           end_entity: str,
           max_depth: int = 10
       ) -> Optional[GraphPath]:
           """Find shortest weighted path using Dijkstra's algorithm."""
           # Priority queue: (weight, entity_id, path)
           pq = [(0.0, start_entity, GraphPath(entities=[], relations=[], total_weight=0.0))]
           visited = set()
           distances = {start_entity: 0.0}
           
           while pq:
               current_weight, current_id, current_path = heapq.heappop(pq)
               
               if current_id in visited:
                   continue
               
               visited.add(current_id)
               current_entity = await self.graph_storage.get_entity(current_id)
               
               # Update path
               new_path = GraphPath(
                   entities=current_path.entities + [current_entity],
                   relations=current_path.relations,
                   total_weight=current_weight
               )
               
               if current_id == end_entity:
                   return new_path
               
               if len(new_path.entities) >= max_depth:
                   continue
               
               # Explore neighbors
               neighbors = await self.graph_storage.get_entity_neighbors(current_id)
               
               for neighbor_id, relation in neighbors:
                   if neighbor_id not in visited:
                       new_weight = current_weight + relation.weight
                       
                       if neighbor_id not in distances or new_weight < distances[neighbor_id]:
                           distances[neighbor_id] = new_weight
                           extended_path = GraphPath(
                               entities=new_path.entities,
                               relations=new_path.relations + [relation],
                               total_weight=new_weight
                           )
                           heapq.heappush(pq, (new_weight, neighbor_id, extended_path))
           
           return None
       
       async def find_all_paths(
           self,
           start_entity: str,
           end_entity: str,
           max_depth: int = 5,
           max_paths: int = 10
       ) -> List[GraphPath]:
           """Find all paths between two entities up to max_depth."""
           all_paths = []
           
           async def dfs(current_id: str, target_id: str, current_path: GraphPath, depth: int):
               if len(all_paths) >= max_paths or depth > max_depth:
                   return
               
               if current_id == target_id and depth > 0:
                   all_paths.append(current_path)
                   return
               
               if current_id in [e.id for e in current_path.entities]:
                   return  # Avoid cycles
               
               current_entity = await self.graph_storage.get_entity(current_id)
               neighbors = await self.graph_storage.get_entity_neighbors(current_id)
               
               for neighbor_id, relation in neighbors:
                   new_path = GraphPath(
                       entities=current_path.entities + [current_entity],
                       relations=current_path.relations + [relation],
                       total_weight=current_path.total_weight + relation.weight
                   )
                   await dfs(neighbor_id, target_id, new_path, depth + 1)
           
           await dfs(start_entity, end_entity, GraphPath(entities=[], relations=[], total_weight=0.0), 0)
           return all_paths
   ```

3. **Query Optimization**
   ```python
   # src/morag_graph/query/optimizer.py
   class QueryOptimizer:
       def __init__(self, graph_storage: BaseGraphStorage):
           self.graph_storage = graph_storage
           self.query_cache = {}
       
       async def optimize_traversal_query(self, query: TraversalQuery) -> OptimizedQuery:
           """Optimize graph traversal queries."""
           # Check cache first
           cache_key = self._generate_cache_key(query)
           if cache_key in self.query_cache:
               return self.query_cache[cache_key]
           
           # Analyze query complexity
           complexity = self._analyze_query_complexity(query)
           
           # Choose optimal algorithm
           if complexity.is_simple:
               algorithm = "bfs"
           elif complexity.has_weights:
               algorithm = "dijkstra"
           else:
               algorithm = "dfs"
           
           optimized = OptimizedQuery(
               original_query=query,
               algorithm=algorithm,
               estimated_cost=complexity.estimated_cost,
               use_cache=complexity.is_cacheable
           )
           
           if optimized.use_cache:
               self.query_cache[cache_key] = optimized
           
           return optimized
   ```

#### Deliverables
- Graph query engine with BFS, DFS, and Dijkstra algorithms
- Path finding with constraints and ranking
- Query optimization and caching
- Comprehensive error handling

### 2.3.2: Graph Analytics
**Estimated Time**: 3-4 days  
**Priority**: Medium

#### Implementation Steps

1. **Centrality Measures**
   ```python
   # src/morag_graph/analytics/centrality.py
   import networkx as nx
   from typing import Dict, List
   
   class GraphAnalytics:
       def __init__(self, graph_storage: BaseGraphStorage):
           self.graph_storage = graph_storage
           self.logger = logging.getLogger(__name__)
       
       async def calculate_centrality(self, entity_ids: Optional[List[str]] = None) -> Dict[str, CentralityScores]:
           """Calculate various centrality measures for entities."""
           # Build NetworkX graph for analysis
           nx_graph = await self._build_networkx_graph(entity_ids)
           
           # Calculate different centrality measures
           betweenness = nx.betweenness_centrality(nx_graph)
           closeness = nx.closeness_centrality(nx_graph)
           eigenvector = nx.eigenvector_centrality(nx_graph, max_iter=1000)
           pagerank = nx.pagerank(nx_graph)
           
           # Combine results
           centrality_scores = {}
           for node_id in nx_graph.nodes():
               centrality_scores[node_id] = CentralityScores(
                   betweenness=betweenness.get(node_id, 0.0),
                   closeness=closeness.get(node_id, 0.0),
                   eigenvector=eigenvector.get(node_id, 0.0),
                   pagerank=pagerank.get(node_id, 0.0)
               )
           
           return centrality_scores
       
       async def find_communities(self, resolution: float = 1.0) -> List[List[str]]:
           """Detect communities in the graph using Louvain algorithm."""
           nx_graph = await self._build_networkx_graph()
           communities = nx.community.louvain_communities(nx_graph, resolution=resolution)
           return [list(community) for community in communities]
       
       async def calculate_graph_statistics(self) -> GraphStatistics:
           """Calculate overall graph statistics."""
           nx_graph = await self._build_networkx_graph()
           
           stats = GraphStatistics(
               num_nodes=nx_graph.number_of_nodes(),
               num_edges=nx_graph.number_of_edges(),
               density=nx.density(nx_graph),
               average_clustering=nx.average_clustering(nx_graph),
               average_shortest_path_length=self._safe_average_shortest_path(nx_graph),
               diameter=self._safe_diameter(nx_graph),
               connected_components=nx.number_connected_components(nx_graph.to_undirected())
           )
           
           return stats
       
       async def _build_networkx_graph(self, entity_ids: Optional[List[str]] = None) -> nx.DiGraph:
           """Build NetworkX graph from storage."""
           graph = nx.DiGraph()
           
           # Get entities
           if entity_ids:
               entities = await self.graph_storage.get_entities_batch(entity_ids)
           else:
               entities = await self.graph_storage.get_all_entities()
           
           # Add nodes
           for entity in entities:
               graph.add_node(entity.id, **entity.metadata)
           
           # Get relations
           relations = await self.graph_storage.get_all_relations()
           
           # Add edges
           for relation in relations:
               if relation.source_entity_id in graph and relation.target_entity_id in graph:
                   graph.add_edge(
                       relation.source_entity_id,
                       relation.target_entity_id,
                       weight=relation.weight,
                       relation_type=relation.relation_type
                   )
           
           return graph
   
   @dataclass
   class CentralityScores:
       betweenness: float
       closeness: float
       eigenvector: float
       pagerank: float
       
       @property
       def combined_score(self) -> float:
           """Weighted combination of centrality measures."""
           return (0.3 * self.betweenness + 
                   0.2 * self.closeness + 
                   0.2 * self.eigenvector + 
                   0.3 * self.pagerank)
   
   @dataclass
   class GraphStatistics:
       num_nodes: int
       num_edges: int
       density: float
       average_clustering: float
       average_shortest_path_length: Optional[float]
       diameter: Optional[int]
       connected_components: int
   ```

2. **Quality Metrics**
   ```python
   # src/morag_graph/analytics/quality.py
   class GraphQualityAnalyzer:
       def __init__(self, graph_storage: BaseGraphStorage):
           self.graph_storage = graph_storage
       
       async def calculate_quality_metrics(self) -> QualityMetrics:
           """Calculate graph quality metrics."""
           entities = await self.graph_storage.get_all_entities()
           relations = await self.graph_storage.get_all_relations()
           
           # Entity quality
           entity_completeness = self._calculate_entity_completeness(entities)
           entity_consistency = self._calculate_entity_consistency(entities)
           
           # Relation quality
           relation_accuracy = await self._calculate_relation_accuracy(relations)
           relation_coverage = self._calculate_relation_coverage(entities, relations)
           
           # Graph structure quality
           connectivity = await self._calculate_connectivity()
           
           return QualityMetrics(
               entity_completeness=entity_completeness,
               entity_consistency=entity_consistency,
               relation_accuracy=relation_accuracy,
               relation_coverage=relation_coverage,
               graph_connectivity=connectivity
           )
       
       def _calculate_entity_completeness(self, entities: List[Entity]) -> float:
           """Calculate what percentage of entities have complete information."""
           complete_entities = 0
           
           for entity in entities:
               if (entity.name and entity.type and entity.summary and 
                   entity.source_documents):
                   complete_entities += 1
           
           return complete_entities / len(entities) if entities else 0.0
   
   @dataclass
   class QualityMetrics:
       entity_completeness: float
       entity_consistency: float
       relation_accuracy: float
       relation_coverage: float
       graph_connectivity: float
       
       @property
       def overall_quality(self) -> float:
           """Overall quality score."""
           return (self.entity_completeness * 0.25 +
                   self.entity_consistency * 0.2 +
                   self.relation_accuracy * 0.25 +
                   self.relation_coverage * 0.15 +
                   self.graph_connectivity * 0.15)
   ```

#### Deliverables
- Centrality measures (betweenness, closeness, eigenvector, PageRank)
- Community detection algorithms
- Graph statistics and quality metrics
- Performance monitoring and optimization

## Testing Requirements

### Unit Tests
```python
# tests/test_graph_traversal.py
import pytest
from morag_graph.query import GraphQueryEngine, PathFindingAlgorithms
from morag_graph.analytics import GraphAnalytics

class TestGraphQueryEngine:
    @pytest.mark.asyncio
    async def test_find_related_entities(self, mock_graph_storage, sample_entities):
        engine = GraphQueryEngine(mock_graph_storage)
        paths = await engine.find_related_entities("entity_1", max_depth=2)
        
        assert isinstance(paths, list)
        assert all(isinstance(p, GraphPath) for p in paths)
    
    @pytest.mark.asyncio
    async def test_shortest_path(self, mock_graph_storage):
        engine = GraphQueryEngine(mock_graph_storage)
        path = await engine.find_shortest_path("entity_1", "entity_3")
        
        assert path is not None
        assert path.entities[0].id == "entity_1"
        assert path.entities[-1].id == "entity_3"
    
    @pytest.mark.asyncio
    async def test_path_constraints(self, mock_graph_storage):
        engine = GraphQueryEngine(mock_graph_storage)
        constraints = PathConstraints(
            required_entity_types=["PERSON"],
            max_path_length=3
        )
        
        paths = await engine.find_paths_with_constraints("entity_1", constraints)
        
        assert all(len(p.entities) <= 3 for p in paths)
        assert all(any(e.type == "PERSON" for e in p.entities) for p in paths)

class TestGraphAnalytics:
    @pytest.mark.asyncio
    async def test_centrality_calculation(self, mock_graph_storage):
        analytics = GraphAnalytics(mock_graph_storage)
        centrality = await analytics.calculate_centrality()
        
        assert isinstance(centrality, dict)
        assert all(isinstance(scores, CentralityScores) for scores in centrality.values())
    
    @pytest.mark.asyncio
    async def test_graph_statistics(self, mock_graph_storage):
        analytics = GraphAnalytics(mock_graph_storage)
        stats = await analytics.calculate_graph_statistics()
        
        assert isinstance(stats, GraphStatistics)
        assert stats.num_nodes > 0
        assert stats.num_edges >= 0
```

### Performance Tests
```python
# tests/performance/test_traversal_performance.py
class TestTraversalPerformance:
    @pytest.mark.asyncio
    async def test_large_graph_traversal(self, large_graph_storage):
        """Test traversal performance on large graphs."""
        engine = GraphQueryEngine(large_graph_storage)
        
        start_time = time.time()
        paths = await engine.find_related_entities("central_entity", max_depth=3)
        end_time = time.time()
        
        assert end_time - start_time < 5.0  # Should complete within 5 seconds
        assert len(paths) > 0
```

## Success Criteria

- [ ] Graph query engine finds related entities efficiently
- [ ] Shortest path algorithms work correctly (BFS, Dijkstra)
- [ ] Path constraints and ranking function properly
- [ ] Centrality measures calculate accurately
- [ ] Graph statistics provide meaningful insights
- [ ] Query optimization improves performance
- [ ] Performance targets met (< 2 seconds for typical queries)
- [ ] Unit test coverage > 90%
- [ ] Integration tests pass

## Performance Targets

- **Entity Traversal**: < 2 seconds for depth 3, 1000 entities
- **Shortest Path**: < 1 second for paths up to length 10
- **Centrality Calculation**: < 30 seconds for 10,000 entities
- **Memory Usage**: < 2GB for graphs with 100,000 entities

## Next Steps

After completing this task:
1. Proceed to **Task 3.1**: Hybrid Retrieval System
2. Integrate traversal capabilities with query processing
3. Implement caching for frequently accessed paths

## Dependencies

**Requires**:
- Task 1.2.2: Graph Storage Implementation

**Enables**:
- Task 3.1: Hybrid Retrieval System
- Task 4.1: Multi-Hop Reasoning
- Task 4.3: Monitoring & Analytics