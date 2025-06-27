# Task 3.3: Graph-Aware Vector Operations

## Overview

Implement advanced graph-aware vector operations that leverage both Neo4j's graph structure and Qdrant's vector capabilities to enable sophisticated semantic search, entity relationship exploration, and context-aware retrieval.

## Objectives

- Implement graph-constrained vector search operations
- Create relationship-aware similarity calculations
- Enable multi-hop vector traversal with graph context
- Develop entity-centric vector clustering and analysis
- Establish graph-guided vector recommendation systems

## Current State Analysis

### Existing Capabilities

**Neo4j**:
- Rich graph traversal and relationship queries
- Entity relationship mapping and analysis
- Graph algorithms (PageRank, community detection, etc.)
- Limited vector similarity operations

**Qdrant**:
- High-performance vector similarity search
- Vector clustering and recommendation
- Payload filtering and hybrid search
- No graph relationship awareness

**Integration Gaps**:
- No graph-constrained vector operations
- Limited relationship-aware similarity
- No multi-hop vector traversal
- Missing entity-centric vector analysis

## Implementation Plan

### Step 1: Graph-Constrained Vector Search

Implement `src/morag_graph/services/graph_vector_ops.py`:

```python
from typing import Dict, List, Optional, Any, Tuple, Set
import asyncio
import logging
from dataclasses import dataclass
from enum import Enum

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, SearchRequest
from neo4j import AsyncSession
import numpy as np

class GraphConstraintType(Enum):
    DIRECT_NEIGHBORS = "direct_neighbors"
    PATH_EXISTS = "path_exists"
    RELATIONSHIP_TYPE = "relationship_type"
    ENTITY_TYPE = "entity_type"
    COMMUNITY = "community"
    DISTANCE_LIMIT = "distance_limit"

@dataclass
class GraphConstraint:
    """Defines a graph-based constraint for vector search."""
    constraint_type: GraphConstraintType
    parameters: Dict[str, Any]
    weight: float = 1.0

@dataclass
class GraphVectorResult:
    """Result from graph-aware vector operation."""
    vector_id: str
    entity_id: str
    similarity_score: float
    graph_score: float
    combined_score: float
    graph_context: Dict[str, Any]
    relationships: List[Dict[str, Any]]

class GraphAwareVectorOperations:
    """Advanced graph-aware vector operations."""
    
    def __init__(self, neo4j_session: AsyncSession, qdrant_client: QdrantClient):
        self.neo4j_session = neo4j_session
        self.qdrant_client = qdrant_client
        self.logger = logging.getLogger(__name__)
    
    async def graph_constrained_search(self,
                                     query_vector: List[float],
                                     constraints: List[GraphConstraint],
                                     collection_name: str = "morag_vectors",
                                     limit: int = 10,
                                     vector_weight: float = 0.7,
                                     graph_weight: float = 0.3) -> List[GraphVectorResult]:
        """Perform vector search with graph constraints."""
        try:
            # Step 1: Get candidate entities from graph constraints
            candidate_entities = await self._get_constrained_entities(constraints)
            
            if not candidate_entities:
                self.logger.warning("No entities match graph constraints")
                return []
            
            # Step 2: Perform vector search with entity filtering
            vector_filter = Filter(
                must=[
                    FieldCondition(
                        key="entity_id",
                        match=MatchValue(value=entity_id)
                    ) for entity_id in candidate_entities[:1000]  # Limit for performance
                ]
            )
            
            search_results = await self.qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=vector_filter,
                limit=min(limit * 3, 100),  # Get more candidates for reranking
                with_payload=True
            )
            
            # Step 3: Calculate graph scores and combine with vector scores
            results = []
            for result in search_results:
                entity_id = result.payload.get("entity_id")
                if not entity_id:
                    continue
                
                # Calculate graph score based on constraints
                graph_score = await self._calculate_graph_score(entity_id, constraints)
                
                # Get graph context
                graph_context = await self._get_graph_context(entity_id)
                
                # Get relationships
                relationships = await self._get_entity_relationships(entity_id)
                
                # Combine scores
                combined_score = (
                    vector_weight * result.score + 
                    graph_weight * graph_score
                )
                
                results.append(GraphVectorResult(
                    vector_id=str(result.id),
                    entity_id=entity_id,
                    similarity_score=result.score,
                    graph_score=graph_score,
                    combined_score=combined_score,
                    graph_context=graph_context,
                    relationships=relationships
                ))
            
            # Sort by combined score and return top results
            results.sort(key=lambda x: x.combined_score, reverse=True)
            return results[:limit]
            
        except Exception as e:
            self.logger.error(f"Graph-constrained search failed: {e}")
            return []
    
    async def relationship_aware_similarity(self,
                                          entity_id: str,
                                          relationship_types: List[str],
                                          collection_name: str = "morag_vectors",
                                          limit: int = 10,
                                          relationship_weight: float = 0.4) -> List[GraphVectorResult]:
        """Find similar entities considering relationship patterns."""
        try:
            # Get entity's vector
            entity_vector = await self._get_entity_vector(entity_id, collection_name)
            if not entity_vector:
                return []
            
            # Get entities with similar relationship patterns
            similar_pattern_entities = await self._get_entities_with_similar_patterns(
                entity_id, relationship_types
            )
            
            # Perform vector search among pattern-similar entities
            if similar_pattern_entities:
                vector_filter = Filter(
                    must=[
                        FieldCondition(
                            key="entity_id",
                            match=MatchValue(value=eid)
                        ) for eid in similar_pattern_entities
                    ]
                )
                
                search_results = await self.qdrant_client.search(
                    collection_name=collection_name,
                    query_vector=entity_vector,
                    query_filter=vector_filter,
                    limit=limit * 2,
                    with_payload=True
                )
                
                # Calculate relationship similarity scores
                results = []
                for result in search_results:
                    result_entity_id = result.payload.get("entity_id")
                    if result_entity_id == entity_id:
                        continue
                    
                    # Calculate relationship pattern similarity
                    rel_similarity = await self._calculate_relationship_similarity(
                        entity_id, result_entity_id, relationship_types
                    )
                    
                    # Combine vector and relationship scores
                    combined_score = (
                        (1 - relationship_weight) * result.score +
                        relationship_weight * rel_similarity
                    )
                    
                    graph_context = await self._get_graph_context(result_entity_id)
                    relationships = await self._get_entity_relationships(result_entity_id)
                    
                    results.append(GraphVectorResult(
                        vector_id=str(result.id),
                        entity_id=result_entity_id,
                        similarity_score=result.score,
                        graph_score=rel_similarity,
                        combined_score=combined_score,
                        graph_context=graph_context,
                        relationships=relationships
                    ))
                
                results.sort(key=lambda x: x.combined_score, reverse=True)
                return results[:limit]
            
            return []
            
        except Exception as e:
            self.logger.error(f"Relationship-aware similarity failed: {e}")
            return []
    
    async def multi_hop_vector_traversal(self,
                                       start_entity_id: str,
                                       max_hops: int = 3,
                                       relationship_types: Optional[List[str]] = None,
                                       collection_name: str = "morag_vectors",
                                       similarity_threshold: float = 0.7,
                                       max_results_per_hop: int = 5) -> Dict[int, List[GraphVectorResult]]:
        """Perform multi-hop traversal combining graph structure and vector similarity."""
        try:
            results_by_hop = {}
            visited_entities = {start_entity_id}
            current_entities = [start_entity_id]
            
            # Get starting entity vector
            start_vector = await self._get_entity_vector(start_entity_id, collection_name)
            if not start_vector:
                return results_by_hop
            
            for hop in range(1, max_hops + 1):
                next_entities = set()
                hop_results = []
                
                # For each entity in current hop
                for entity_id in current_entities:
                    # Get graph neighbors
                    neighbors = await self._get_graph_neighbors(
                        entity_id, relationship_types
                    )
                    
                    # Filter unvisited neighbors
                    unvisited_neighbors = [
                        neighbor for neighbor in neighbors 
                        if neighbor["entity_id"] not in visited_entities
                    ]
                    
                    if not unvisited_neighbors:
                        continue
                    
                    # Get vectors for neighbors and calculate similarities
                    neighbor_ids = [n["entity_id"] for n in unvisited_neighbors]
                    neighbor_vectors = await self._get_entity_vectors_batch(
                        neighbor_ids, collection_name
                    )
                    
                    for neighbor, vector in zip(unvisited_neighbors, neighbor_vectors):
                        if vector is None:
                            continue
                        
                        # Calculate similarity to start vector
                        similarity = self._cosine_similarity(start_vector, vector)
                        
                        if similarity >= similarity_threshold:
                            graph_context = await self._get_graph_context(neighbor["entity_id"])
                            relationships = await self._get_entity_relationships(neighbor["entity_id"])
                            
                            hop_results.append(GraphVectorResult(
                                vector_id=neighbor["entity_id"],  # Using entity_id as vector_id
                                entity_id=neighbor["entity_id"],
                                similarity_score=similarity,
                                graph_score=1.0 / hop,  # Decay by hop distance
                                combined_score=similarity * (1.0 / hop),
                                graph_context=graph_context,
                                relationships=relationships
                            ))
                            
                            next_entities.add(neighbor["entity_id"])
                            visited_entities.add(neighbor["entity_id"])
                
                # Sort and limit results for this hop
                hop_results.sort(key=lambda x: x.combined_score, reverse=True)
                results_by_hop[hop] = hop_results[:max_results_per_hop]
                
                # Prepare for next hop
                current_entities = list(next_entities)
                
                if not current_entities:
                    break
            
            return results_by_hop
            
        except Exception as e:
            self.logger.error(f"Multi-hop vector traversal failed: {e}")
            return {}
    
    async def entity_centric_clustering(self,
                                      entity_ids: List[str],
                                      collection_name: str = "morag_vectors",
                                      num_clusters: int = 5,
                                      graph_weight: float = 0.3) -> Dict[str, List[str]]:
        """Cluster entities using both vector similarity and graph structure."""
        try:
            # Get entity vectors
            entity_vectors = await self._get_entity_vectors_batch(entity_ids, collection_name)
            
            # Filter entities with valid vectors
            valid_entities = []
            valid_vectors = []
            
            for entity_id, vector in zip(entity_ids, entity_vectors):
                if vector is not None:
                    valid_entities.append(entity_id)
                    valid_vectors.append(vector)
            
            if len(valid_entities) < num_clusters:
                self.logger.warning(f"Not enough entities with vectors for clustering: {len(valid_entities)}")
                return {}
            
            # Calculate vector similarity matrix
            vector_similarity_matrix = self._calculate_similarity_matrix(valid_vectors)
            
            # Calculate graph connectivity matrix
            graph_connectivity_matrix = await self._calculate_graph_connectivity_matrix(valid_entities)
            
            # Combine matrices
            combined_matrix = (
                (1 - graph_weight) * vector_similarity_matrix +
                graph_weight * graph_connectivity_matrix
            )
            
            # Perform clustering (using simple k-means-like approach)
            clusters = await self._perform_hybrid_clustering(
                valid_entities, combined_matrix, num_clusters
            )
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"Entity-centric clustering failed: {e}")
            return {}
    
    async def graph_guided_recommendations(self,
                                         user_entity_id: str,
                                         interaction_types: List[str],
                                         collection_name: str = "morag_vectors",
                                         limit: int = 10,
                                         diversity_factor: float = 0.2) -> List[GraphVectorResult]:
        """Generate recommendations using graph structure and vector similarity."""
        try:
            # Get user's interaction history from graph
            user_interactions = await self._get_user_interactions(
                user_entity_id, interaction_types
            )
            
            if not user_interactions:
                self.logger.warning(f"No interactions found for user {user_entity_id}")
                return []
            
            # Create user profile vector from interactions
            user_profile_vector = await self._create_user_profile_vector(
                user_interactions, collection_name
            )
            
            if not user_profile_vector:
                return []
            
            # Get candidate items (entities not yet interacted with)
            candidate_items = await self._get_recommendation_candidates(
                user_entity_id, interaction_types
            )
            
            if not candidate_items:
                return []
            
            # Calculate recommendation scores
            recommendations = []
            
            for item_id in candidate_items:
                # Vector similarity score
                item_vector = await self._get_entity_vector(item_id, collection_name)
                if not item_vector:
                    continue
                
                vector_similarity = self._cosine_similarity(user_profile_vector, item_vector)
                
                # Graph-based score (collaborative filtering)
                graph_score = await self._calculate_collaborative_score(
                    user_entity_id, item_id, interaction_types
                )
                
                # Diversity score (to avoid over-similar recommendations)
                diversity_score = await self._calculate_diversity_score(
                    item_id, [r.entity_id for r in recommendations], collection_name
                )
                
                # Combined score
                combined_score = (
                    0.5 * vector_similarity +
                    0.3 * graph_score +
                    diversity_factor * diversity_score
                )
                
                graph_context = await self._get_graph_context(item_id)
                relationships = await self._get_entity_relationships(item_id)
                
                recommendations.append(GraphVectorResult(
                    vector_id=item_id,
                    entity_id=item_id,
                    similarity_score=vector_similarity,
                    graph_score=graph_score,
                    combined_score=combined_score,
                    graph_context=graph_context,
                    relationships=relationships
                ))
            
            # Sort and return top recommendations
            recommendations.sort(key=lambda x: x.combined_score, reverse=True)
            return recommendations[:limit]
            
        except Exception as e:
            self.logger.error(f"Graph-guided recommendations failed: {e}")
            return []
    
    # Helper methods
    
    async def _get_constrained_entities(self, constraints: List[GraphConstraint]) -> List[str]:
        """Get entities that satisfy graph constraints."""
        entity_sets = []
        
        for constraint in constraints:
            if constraint.constraint_type == GraphConstraintType.ENTITY_TYPE:
                entity_type = constraint.parameters.get("entity_type")
                query = "MATCH (e:Entity {type: $type}) RETURN e.id as entity_id"
                result = await self.neo4j_session.run(query, type=entity_type)
                entities = [record["entity_id"] async for record in result]
                entity_sets.append(set(entities))
            
            elif constraint.constraint_type == GraphConstraintType.DIRECT_NEIGHBORS:
                center_entity = constraint.parameters.get("center_entity")
                query = """
                MATCH (center:Entity {id: $center_id})-[]-(neighbor:Entity)
                RETURN DISTINCT neighbor.id as entity_id
                """
                result = await self.neo4j_session.run(query, center_id=center_entity)
                entities = [record["entity_id"] async for record in result]
                entity_sets.append(set(entities))
            
            elif constraint.constraint_type == GraphConstraintType.RELATIONSHIP_TYPE:
                rel_type = constraint.parameters.get("relationship_type")
                query = f"""
                MATCH (e1:Entity)-[r:{rel_type}]-(e2:Entity)
                RETURN DISTINCT e1.id as entity_id
                UNION
                RETURN DISTINCT e2.id as entity_id
                """
                result = await self.neo4j_session.run(query)
                entities = [record["entity_id"] async for record in result]
                entity_sets.append(set(entities))
        
        # Intersect all constraint sets
        if entity_sets:
            result_entities = entity_sets[0]
            for entity_set in entity_sets[1:]:
                result_entities = result_entities.intersection(entity_set)
            return list(result_entities)
        
        return []
    
    async def _calculate_graph_score(self, entity_id: str, constraints: List[GraphConstraint]) -> float:
        """Calculate graph-based score for an entity given constraints."""
        total_score = 0.0
        total_weight = 0.0
        
        for constraint in constraints:
            score = 0.0
            
            if constraint.constraint_type == GraphConstraintType.DIRECT_NEIGHBORS:
                center_entity = constraint.parameters.get("center_entity")
                # Check if entity is a direct neighbor
                query = """
                MATCH (center:Entity {id: $center_id})-[]-(entity:Entity {id: $entity_id})
                RETURN count(*) as connection_count
                """
                result = await self.neo4j_session.run(
                    query, center_id=center_entity, entity_id=entity_id
                )
                record = await result.single()
                score = min(1.0, record["connection_count"] / 5.0)  # Normalize
            
            elif constraint.constraint_type == GraphConstraintType.DISTANCE_LIMIT:
                target_entity = constraint.parameters.get("target_entity")
                max_distance = constraint.parameters.get("max_distance", 3)
                
                # Calculate shortest path distance
                query = """
                MATCH path = shortestPath((start:Entity {id: $start_id})-[*..10]-(end:Entity {id: $end_id}))
                RETURN length(path) as distance
                """
                result = await self.neo4j_session.run(
                    query, start_id=target_entity, end_id=entity_id
                )
                record = await result.single()
                
                if record and record["distance"] is not None:
                    distance = record["distance"]
                    if distance <= max_distance:
                        score = 1.0 - (distance / max_distance)
            
            total_score += score * constraint.weight
            total_weight += constraint.weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    async def _get_graph_context(self, entity_id: str) -> Dict[str, Any]:
        """Get graph context information for an entity."""
        query = """
        MATCH (entity:Entity {id: $entity_id})
        OPTIONAL MATCH (entity)-[r]-(connected:Entity)
        WITH entity, type(r) as rel_type, count(connected) as connection_count
        RETURN entity.name as name, entity.type as type,
               collect({relationship_type: rel_type, count: connection_count}) as connections
        """
        
        result = await self.neo4j_session.run(query, entity_id=entity_id)
        record = await result.single()
        
        if record:
            return {
                "name": record["name"],
                "type": record["type"],
                "connections": record["connections"]
            }
        
        return {}
    
    async def _get_entity_relationships(self, entity_id: str) -> List[Dict[str, Any]]:
        """Get relationships for an entity."""
        query = """
        MATCH (entity:Entity {id: $entity_id})-[r]-(connected:Entity)
        RETURN type(r) as relationship_type, connected.id as connected_entity_id,
               connected.name as connected_entity_name, connected.type as connected_entity_type
        LIMIT 20
        """
        
        result = await self.neo4j_session.run(query, entity_id=entity_id)
        relationships = []
        
        async for record in result:
            relationships.append({
                "relationship_type": record["relationship_type"],
                "connected_entity_id": record["connected_entity_id"],
                "connected_entity_name": record["connected_entity_name"],
                "connected_entity_type": record["connected_entity_type"]
            })
        
        return relationships
    
    async def _get_entity_vector(self, entity_id: str, collection_name: str) -> Optional[List[float]]:
        """Get vector for a specific entity."""
        try:
            points = await self.qdrant_client.retrieve(
                collection_name=collection_name,
                ids=[entity_id],
                with_vectors=True
            )
            
            if points and len(points) > 0:
                return points[0].vector
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get vector for entity {entity_id}: {e}")
            return None
    
    async def _get_entity_vectors_batch(self, entity_ids: List[str], collection_name: str) -> List[Optional[List[float]]]:
        """Get vectors for multiple entities."""
        try:
            points = await self.qdrant_client.retrieve(
                collection_name=collection_name,
                ids=entity_ids,
                with_vectors=True
            )
            
            # Create mapping of id to vector
            vector_map = {str(point.id): point.vector for point in points}
            
            # Return vectors in the same order as input entity_ids
            return [vector_map.get(entity_id) for entity_id in entity_ids]
            
        except Exception as e:
            self.logger.error(f"Failed to get vectors for entities: {e}")
            return [None] * len(entity_ids)
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            vec1_np = np.array(vec1)
            vec2_np = np.array(vec2)
            
            dot_product = np.dot(vec1_np, vec2_np)
            norm1 = np.linalg.norm(vec1_np)
            norm2 = np.linalg.norm(vec2_np)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
            
        except Exception:
            return 0.0
    
    def _calculate_similarity_matrix(self, vectors: List[List[float]]) -> np.ndarray:
        """Calculate pairwise similarity matrix for vectors."""
        n = len(vectors)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                similarity = self._cosine_similarity(vectors[i], vectors[j])
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity
        
        return similarity_matrix
    
    async def _calculate_graph_connectivity_matrix(self, entity_ids: List[str]) -> np.ndarray:
        """Calculate graph connectivity matrix for entities."""
        n = len(entity_ids)
        connectivity_matrix = np.zeros((n, n))
        
        # Create entity index mapping
        entity_to_index = {entity_id: i for i, entity_id in enumerate(entity_ids)}
        
        # Get all relationships between these entities
        query = """
        MATCH (e1:Entity)-[r]-(e2:Entity)
        WHERE e1.id IN $entity_ids AND e2.id IN $entity_ids
        RETURN e1.id as entity1, e2.id as entity2, count(r) as connection_strength
        """
        
        result = await self.neo4j_session.run(query, entity_ids=entity_ids)
        
        async for record in result:
            entity1 = record["entity1"]
            entity2 = record["entity2"]
            strength = record["connection_strength"]
            
            if entity1 in entity_to_index and entity2 in entity_to_index:
                i = entity_to_index[entity1]
                j = entity_to_index[entity2]
                connectivity_matrix[i][j] = min(1.0, strength / 5.0)  # Normalize
                connectivity_matrix[j][i] = connectivity_matrix[i][j]
        
        return connectivity_matrix
    
    async def _perform_hybrid_clustering(self, 
                                       entities: List[str], 
                                       similarity_matrix: np.ndarray, 
                                       num_clusters: int) -> Dict[str, List[str]]:
        """Perform clustering using combined similarity matrix."""
        # Simple clustering implementation (in practice, use scikit-learn)
        n = len(entities)
        
        if n <= num_clusters:
            # Each entity in its own cluster
            return {f"cluster_{i}": [entities[i]] for i in range(n)}
        
        # Initialize clusters with most dissimilar entities
        cluster_centers = []
        remaining_entities = list(range(n))
        
        # First center is random
        first_center = remaining_entities.pop(0)
        cluster_centers.append(first_center)
        
        # Select remaining centers to maximize distance
        for _ in range(num_clusters - 1):
            max_min_distance = -1
            best_center = None
            
            for candidate in remaining_entities:
                min_distance = min(
                    1 - similarity_matrix[candidate][center] 
                    for center in cluster_centers
                )
                
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_center = candidate
            
            if best_center is not None:
                cluster_centers.append(best_center)
                remaining_entities.remove(best_center)
        
        # Assign entities to closest cluster centers
        clusters = {f"cluster_{i}": [] for i in range(len(cluster_centers))}
        
        for i in range(n):
            best_cluster = 0
            best_similarity = similarity_matrix[i][cluster_centers[0]]
            
            for j, center in enumerate(cluster_centers[1:], 1):
                similarity = similarity_matrix[i][center]
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster = j
            
            clusters[f"cluster_{best_cluster}"].append(entities[i])
        
        return clusters
```

### Step 2: Advanced Graph Analytics Integration

Implement `src/morag_graph/services/graph_analytics.py`:

```python
from typing import Dict, List, Optional, Any, Tuple
import asyncio
import logging
from dataclasses import dataclass

from neo4j import AsyncSession
from qdrant_client import QdrantClient
import numpy as np

class GraphAnalyticsService:
    """Advanced graph analytics for vector-enhanced operations."""
    
    def __init__(self, neo4j_session: AsyncSession, qdrant_client: QdrantClient):
        self.neo4j_session = neo4j_session
        self.qdrant_client = qdrant_client
        self.logger = logging.getLogger(__name__)
    
    async def calculate_entity_centrality_scores(self, 
                                               entity_ids: List[str],
                                               centrality_type: str = "pagerank") -> Dict[str, float]:
        """Calculate centrality scores for entities in the graph."""
        try:
            if centrality_type == "pagerank":
                query = """
                CALL gds.pageRank.stream({
                    nodeQuery: 'MATCH (e:Entity) WHERE e.id IN $entity_ids RETURN id(e) as id',
                    relationshipQuery: 'MATCH (e1:Entity)-[]-(e2:Entity) WHERE e1.id IN $entity_ids AND e2.id IN $entity_ids RETURN id(e1) as source, id(e2) as target',
                    maxIterations: 20,
                    dampingFactor: 0.85
                })
                YIELD nodeId, score
                MATCH (e:Entity) WHERE id(e) = nodeId
                RETURN e.id as entity_id, score
                """
            
            elif centrality_type == "betweenness":
                query = """
                CALL gds.betweenness.stream({
                    nodeQuery: 'MATCH (e:Entity) WHERE e.id IN $entity_ids RETURN id(e) as id',
                    relationshipQuery: 'MATCH (e1:Entity)-[]-(e2:Entity) WHERE e1.id IN $entity_ids AND e2.id IN $entity_ids RETURN id(e1) as source, id(e2) as target'
                })
                YIELD nodeId, score
                MATCH (e:Entity) WHERE id(e) = nodeId
                RETURN e.id as entity_id, score
                """
            
            elif centrality_type == "closeness":
                query = """
                CALL gds.closeness.stream({
                    nodeQuery: 'MATCH (e:Entity) WHERE e.id IN $entity_ids RETURN id(e) as id',
                    relationshipQuery: 'MATCH (e1:Entity)-[]-(e2:Entity) WHERE e1.id IN $entity_ids AND e2.id IN $entity_ids RETURN id(e1) as source, id(e2) as target'
                })
                YIELD nodeId, score
                MATCH (e:Entity) WHERE id(e) = nodeId
                RETURN e.id as entity_id, score
                """
            
            else:
                self.logger.error(f"Unsupported centrality type: {centrality_type}")
                return {}
            
            result = await self.neo4j_session.run(query, entity_ids=entity_ids)
            centrality_scores = {}
            
            async for record in result:
                centrality_scores[record["entity_id"]] = record["score"]
            
            return centrality_scores
            
        except Exception as e:
            self.logger.error(f"Failed to calculate centrality scores: {e}")
            return {}
    
    async def detect_communities(self, 
                               entity_ids: List[str],
                               algorithm: str = "louvain") -> Dict[str, List[str]]:
        """Detect communities in the entity graph."""
        try:
            if algorithm == "louvain":
                query = """
                CALL gds.louvain.stream({
                    nodeQuery: 'MATCH (e:Entity) WHERE e.id IN $entity_ids RETURN id(e) as id',
                    relationshipQuery: 'MATCH (e1:Entity)-[]-(e2:Entity) WHERE e1.id IN $entity_ids AND e2.id IN $entity_ids RETURN id(e1) as source, id(e2) as target',
                    includeIntermediateCommunities: false
                })
                YIELD nodeId, communityId
                MATCH (e:Entity) WHERE id(e) = nodeId
                RETURN e.id as entity_id, communityId
                """
            
            elif algorithm == "label_propagation":
                query = """
                CALL gds.labelPropagation.stream({
                    nodeQuery: 'MATCH (e:Entity) WHERE e.id IN $entity_ids RETURN id(e) as id',
                    relationshipQuery: 'MATCH (e1:Entity)-[]-(e2:Entity) WHERE e1.id IN $entity_ids AND e2.id IN $entity_ids RETURN id(e1) as source, id(e2) as target',
                    maxIterations: 10
                })
                YIELD nodeId, communityId
                MATCH (e:Entity) WHERE id(e) = nodeId
                RETURN e.id as entity_id, communityId
                """
            
            else:
                self.logger.error(f"Unsupported community detection algorithm: {algorithm}")
                return {}
            
            result = await self.neo4j_session.run(query, entity_ids=entity_ids)
            communities = {}
            
            async for record in result:
                entity_id = record["entity_id"]
                community_id = f"community_{record['communityId']}"
                
                if community_id not in communities:
                    communities[community_id] = []
                communities[community_id].append(entity_id)
            
            return communities
            
        except Exception as e:
            self.logger.error(f"Failed to detect communities: {e}")
            return {}
    
    async def calculate_structural_similarity(self, 
                                            entity1_id: str, 
                                            entity2_id: str,
                                            max_depth: int = 2) -> float:
        """Calculate structural similarity between two entities."""
        try:
            # Get neighborhood structures for both entities
            query = """
            MATCH (e1:Entity {id: $entity1_id})-[*1..$max_depth]-(neighbor1:Entity)
            WITH collect(DISTINCT neighbor1.id) as neighbors1
            MATCH (e2:Entity {id: $entity2_id})-[*1..$max_depth]-(neighbor2:Entity)
            WITH neighbors1, collect(DISTINCT neighbor2.id) as neighbors2
            RETURN neighbors1, neighbors2
            """
            
            result = await self.neo4j_session.run(
                query, 
                entity1_id=entity1_id, 
                entity2_id=entity2_id, 
                max_depth=max_depth
            )
            record = await result.single()
            
            if not record:
                return 0.0
            
            neighbors1 = set(record["neighbors1"] or [])
            neighbors2 = set(record["neighbors2"] or [])
            
            # Calculate Jaccard similarity
            intersection = len(neighbors1.intersection(neighbors2))
            union = len(neighbors1.union(neighbors2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Failed to calculate structural similarity: {e}")
            return 0.0
    
    async def find_influential_entities(self, 
                                      collection_name: str = "morag_vectors",
                                      limit: int = 20,
                                      influence_factors: Dict[str, float] = None) -> List[Dict[str, Any]]:
        """Find most influential entities combining graph and vector metrics."""
        try:
            if influence_factors is None:
                influence_factors = {
                    "pagerank": 0.4,
                    "degree_centrality": 0.3,
                    "vector_centrality": 0.3
                }
            
            # Get all entities with vectors
            query = """
            MATCH (e:Entity)
            WHERE e.qdrant_point_id IS NOT NULL
            RETURN e.id as entity_id, e.name as name, e.type as type
            """
            
            result = await self.neo4j_session.run(query)
            entities = []
            
            async for record in result:
                entities.append({
                    "entity_id": record["entity_id"],
                    "name": record["name"],
                    "type": record["type"]
                })
            
            entity_ids = [e["entity_id"] for e in entities]
            
            if not entity_ids:
                return []
            
            # Calculate various centrality measures
            pagerank_scores = await self.calculate_entity_centrality_scores(
                entity_ids, "pagerank"
            )
            
            degree_scores = await self._calculate_degree_centrality(entity_ids)
            vector_centrality_scores = await self._calculate_vector_centrality(
                entity_ids, collection_name
            )
            
            # Combine scores
            influential_entities = []
            
            for entity in entities:
                entity_id = entity["entity_id"]
                
                pagerank = pagerank_scores.get(entity_id, 0.0)
                degree = degree_scores.get(entity_id, 0.0)
                vector_centrality = vector_centrality_scores.get(entity_id, 0.0)
                
                # Normalize scores (simple min-max normalization)
                normalized_pagerank = pagerank / max(pagerank_scores.values()) if pagerank_scores else 0.0
                normalized_degree = degree / max(degree_scores.values()) if degree_scores else 0.0
                normalized_vector = vector_centrality / max(vector_centrality_scores.values()) if vector_centrality_scores else 0.0
                
                # Calculate combined influence score
                influence_score = (
                    influence_factors["pagerank"] * normalized_pagerank +
                    influence_factors["degree_centrality"] * normalized_degree +
                    influence_factors["vector_centrality"] * normalized_vector
                )
                
                influential_entities.append({
                    "entity_id": entity_id,
                    "name": entity["name"],
                    "type": entity["type"],
                    "influence_score": influence_score,
                    "pagerank": pagerank,
                    "degree_centrality": degree,
                    "vector_centrality": vector_centrality
                })
            
            # Sort by influence score and return top entities
            influential_entities.sort(key=lambda x: x["influence_score"], reverse=True)
            return influential_entities[:limit]
            
        except Exception as e:
            self.logger.error(f"Failed to find influential entities: {e}")
            return []
    
    async def _calculate_degree_centrality(self, entity_ids: List[str]) -> Dict[str, float]:
        """Calculate degree centrality for entities."""
        query = """
        MATCH (e:Entity)
        WHERE e.id IN $entity_ids
        OPTIONAL MATCH (e)-[]-(connected:Entity)
        WITH e, count(connected) as degree
        RETURN e.id as entity_id, degree
        """
        
        result = await self.neo4j_session.run(query, entity_ids=entity_ids)
        degree_scores = {}
        
        async for record in result:
            degree_scores[record["entity_id"]] = float(record["degree"])
        
        return degree_scores
    
    async def _calculate_vector_centrality(self, entity_ids: List[str], collection_name: str) -> Dict[str, float]:
        """Calculate vector-based centrality (average similarity to all other entities)."""
        try:
            # Get vectors for all entities
            points = await self.qdrant_client.retrieve(
                collection_name=collection_name,
                ids=entity_ids,
                with_vectors=True
            )
            
            # Create mapping of entity_id to vector
            entity_vectors = {}
            for point in points:
                if point.vector:
                    entity_vectors[str(point.id)] = point.vector
            
            # Calculate average similarity for each entity
            vector_centrality = {}
            
            for entity_id, vector in entity_vectors.items():
                similarities = []
                
                for other_id, other_vector in entity_vectors.items():
                    if entity_id != other_id:
                        similarity = self._cosine_similarity(vector, other_vector)
                        similarities.append(similarity)
                
                avg_similarity = np.mean(similarities) if similarities else 0.0
                vector_centrality[entity_id] = avg_similarity
            
            return vector_centrality
            
        except Exception as e:
            self.logger.error(f"Failed to calculate vector centrality: {e}")
            return {}
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            vec1_np = np.array(vec1)
            vec2_np = np.array(vec2)
            
            dot_product = np.dot(vec1_np, vec2_np)
            norm1 = np.linalg.norm(vec1_np)
            norm2 = np.linalg.norm(vec2_np)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
            
        except Exception:
            return 0.0
```

## Testing Strategy

### Unit Tests

Create `tests/test_graph_vector_ops.py`:

```python
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from morag_graph.services.graph_vector_ops import (
    GraphAwareVectorOperations, GraphConstraint, GraphConstraintType
)

@pytest.fixture
def mock_neo4j_session():
    return AsyncMock()

@pytest.fixture
def mock_qdrant_client():
    return AsyncMock()

@pytest.fixture
def graph_vector_ops(mock_neo4j_session, mock_qdrant_client):
    return GraphAwareVectorOperations(mock_neo4j_session, mock_qdrant_client)

@pytest.mark.asyncio
async def test_graph_constrained_search(graph_vector_ops, mock_neo4j_session, mock_qdrant_client):
    # Mock constraint entities
    graph_vector_ops._get_constrained_entities = AsyncMock(
        return_value=["entity_1", "entity_2", "entity_3"]
    )
    
    # Mock Qdrant search results
    mock_result = MagicMock()
    mock_result.id = "entity_1"
    mock_result.score = 0.95
    mock_result.payload = {"entity_id": "entity_1"}
    
    mock_qdrant_client.search.return_value = [mock_result]
    
    # Mock helper methods
    graph_vector_ops._calculate_graph_score = AsyncMock(return_value=0.8)
    graph_vector_ops._get_graph_context = AsyncMock(return_value={"name": "Test Entity"})
    graph_vector_ops._get_entity_relationships = AsyncMock(return_value=[])
    
    # Test
    constraints = [
        GraphConstraint(
            constraint_type=GraphConstraintType.ENTITY_TYPE,
            parameters={"entity_type": "PERSON"},
            weight=1.0
        )
    ]
    
    query_vector = [0.1] * 384  # Mock vector
    results = await graph_vector_ops.graph_constrained_search(
        query_vector, constraints, limit=5
    )
    
    assert len(results) == 1
    assert results[0].entity_id == "entity_1"
    assert results[0].similarity_score == 0.95
    assert results[0].graph_score == 0.8

@pytest.mark.asyncio
async def test_multi_hop_vector_traversal(graph_vector_ops, mock_neo4j_session, mock_qdrant_client):
    # Mock entity vector
    start_vector = [0.1] * 384
    graph_vector_ops._get_entity_vector = AsyncMock(return_value=start_vector)
    
    # Mock graph neighbors
    graph_vector_ops._get_graph_neighbors = AsyncMock(
        return_value=[
            {"entity_id": "neighbor_1", "relationship_type": "KNOWS"},
            {"entity_id": "neighbor_2", "relationship_type": "WORKS_WITH"}
        ]
    )
    
    # Mock neighbor vectors
    neighbor_vectors = [[0.2] * 384, [0.15] * 384]
    graph_vector_ops._get_entity_vectors_batch = AsyncMock(return_value=neighbor_vectors)
    
    # Mock helper methods
    graph_vector_ops._get_graph_context = AsyncMock(return_value={})
    graph_vector_ops._get_entity_relationships = AsyncMock(return_value=[])
    
    # Test
    results = await graph_vector_ops.multi_hop_vector_traversal(
        "start_entity", max_hops=2, similarity_threshold=0.5
    )
    
    assert isinstance(results, dict)
    assert 1 in results  # Should have results for hop 1

@pytest.mark.asyncio
async def test_entity_centric_clustering(graph_vector_ops, mock_neo4j_session, mock_qdrant_client):
    # Mock entity vectors
    entity_ids = ["entity_1", "entity_2", "entity_3", "entity_4", "entity_5"]
    mock_vectors = [
        [0.1] * 384,
        [0.2] * 384,
        [0.15] * 384,
        [0.8] * 384,
        [0.85] * 384
    ]
    
    graph_vector_ops._get_entity_vectors_batch = AsyncMock(return_value=mock_vectors)
    graph_vector_ops._calculate_graph_connectivity_matrix = AsyncMock(
        return_value=np.eye(5)  # Identity matrix for simplicity
    )
    
    # Test
    clusters = await graph_vector_ops.entity_centric_clustering(
        entity_ids, num_clusters=2
    )
    
    assert isinstance(clusters, dict)
    assert len(clusters) <= 2  # Should have at most 2 clusters

@pytest.mark.asyncio
async def test_cosine_similarity(graph_vector_ops):
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [0.0, 1.0, 0.0]
    vec3 = [1.0, 0.0, 0.0]
    
    # Test orthogonal vectors
    similarity = graph_vector_ops._cosine_similarity(vec1, vec2)
    assert abs(similarity - 0.0) < 1e-6
    
    # Test identical vectors
    similarity = graph_vector_ops._cosine_similarity(vec1, vec3)
    assert abs(similarity - 1.0) < 1e-6
```

### Integration Tests

Create `tests/integration/test_graph_vector_integration.py`:

```python
import pytest
import asyncio
from qdrant_client import QdrantClient
from neo4j import AsyncGraphDatabase

from morag_graph.services.graph_vector_ops import (
    GraphAwareVectorOperations, GraphConstraint, GraphConstraintType
)

@pytest.mark.integration
@pytest.mark.asyncio
async def test_end_to_end_graph_vector_operations():
    # Setup test databases
    qdrant_client = QdrantClient(":memory:")
    neo4j_driver = AsyncGraphDatabase.driver("bolt://localhost:7687")
    
    async with neo4j_driver.session() as session:
        graph_ops = GraphAwareVectorOperations(session, qdrant_client)
        
        # Create test entities and relationships
        await session.run("""
            CREATE (p1:Entity {id: 'person_1', name: 'Alice', type: 'PERSON'})
            CREATE (p2:Entity {id: 'person_2', name: 'Bob', type: 'PERSON'})
            CREATE (c1:Entity {id: 'company_1', name: 'TechCorp', type: 'COMPANY'})
            CREATE (p1)-[:WORKS_AT]->(c1)
            CREATE (p2)-[:WORKS_AT]->(c1)
            CREATE (p1)-[:KNOWS]->(p2)
        """)
        
        # Test graph-constrained search
        constraints = [
            GraphConstraint(
                constraint_type=GraphConstraintType.ENTITY_TYPE,
                parameters={"entity_type": "PERSON"},
                weight=1.0
            )
        ]
        
        query_vector = [0.1] * 384  # Mock query vector
        results = await graph_ops.graph_constrained_search(
            query_vector, constraints, limit=5
        )
        
        # Should find person entities
        assert len(results) >= 0  # May be 0 if no vectors are stored
        
        # Test multi-hop traversal
        traversal_results = await graph_ops.multi_hop_vector_traversal(
            "person_1", max_hops=2
        )
        
        assert isinstance(traversal_results, dict)
        
        # Cleanup
        await session.run("""
            MATCH (n:Entity) 
            WHERE n.id IN ['person_1', 'person_2', 'company_1']
            DETACH DELETE n
        """)
    
    await neo4j_driver.close()
```

## Performance Considerations

### Optimization Strategies

1. **Vector Caching**: Cache frequently accessed entity vectors
2. **Graph Query Optimization**: Use efficient Cypher queries with proper indexing
3. **Batch Processing**: Process multiple entities in single operations
4. **Parallel Execution**: Use asyncio for concurrent graph and vector operations
5. **Result Caching**: Cache graph constraint results for repeated queries

### Performance Targets

- Graph-constrained search: < 500ms for 1000 candidates
- Multi-hop traversal (3 hops): < 2s
- Entity clustering (100 entities): < 5s
- Relationship-aware similarity: < 200ms
- Graph analytics (centrality): < 10s for 1000 entities

## Success Criteria

- [ ] Graph-constrained vector search implemented and tested
- [ ] Relationship-aware similarity calculations working
- [ ] Multi-hop vector traversal with graph context functional
- [ ] Entity-centric clustering combining graph and vector data
- [ ] Graph-guided recommendation system operational
- [ ] Performance targets met for all operations
- [ ] Comprehensive test coverage (>90%)
- [ ] Integration with existing vector and graph services

## Risk Assessment

**Risk Level**: High

**Key Risks**:
- Complex algorithm performance with large graphs
- Memory usage with large vector datasets
- Graph algorithm scalability limitations
- Vector-graph synchronization complexity

**Mitigation Strategies**:
- Implement result pagination and streaming
- Use efficient graph algorithms and indexing
- Add memory monitoring and optimization
- Design fallback mechanisms for large datasets

## Rollback Plan

1. **Immediate Rollback**: Disable graph-aware vector operations
2. **Service Isolation**: Revert to separate graph and vector operations
3. **Performance Monitoring**: Monitor system performance during rollback
4. **Data Integrity**: Ensure no data corruption during rollback

## Next Steps

- **Task 4.1**: [Hybrid Retrieval Pipeline](./task-4.1-hybrid-retrieval-pipeline.md)
- **Task 4.2**: [Query Coordination Service](./task-4.2-query-coordination-service.md)

## Dependencies

- **Task 3.1**: Neo4j Vector Storage (must be completed)
- **Task 3.2**: Selective Vector Strategy (must be completed)
- **Task 2.1**: Bidirectional Reference Storage (must be completed)
- **Task 2.2**: Metadata Synchronization (must be completed)

## Estimated Time

**Total**: 7-8 days

- Algorithm design: 2 days
- Implementation: 4 days
- Testing and optimization: 1.5 days
- Documentation: 0.5 days

## Status

- [ ] Planning
- [ ] Implementation
- [ ] Testing
- [ ] Documentation
- [ ] Review
- [ ] Deployment