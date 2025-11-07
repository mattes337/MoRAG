"""Graph analytics operations.

This module provides graph analytics capabilities including centrality measures,
clustering analysis, and graph metrics.
"""

import logging
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

from ..models import Entity
from ..storage.base import BaseStorage
from ..storage.neo4j_storage import Neo4jStorage

logger = logging.getLogger(__name__)


class GraphAnalytics:
    """Graph analytics and metrics.

    This class provides various graph analysis capabilities including
    centrality measures, clustering, and statistical analysis.
    """

    def __init__(self, storage: BaseStorage):
        """Initialize GraphAnalytics with a storage backend.

        Args:
            storage: Storage backend (Neo4j, JSON, etc.)
        """
        self.storage = storage
        self.logger = logger.getChild(self.__class__.__name__)

    async def calculate_degree_centrality(
        self, relation_types: Optional[List[str]] = None, top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Calculate degree centrality for all entities.

        Args:
            relation_types: Optional filter for relation types
            top_k: Return only top K entities by centrality

        Returns:
            List of entities with their degree centrality scores
        """
        self.logger.info("Calculating degree centrality")

        if isinstance(self.storage, Neo4jStorage):
            return await self._calculate_degree_centrality_neo4j(relation_types, top_k)
        else:
            return await self._calculate_degree_centrality_generic(
                relation_types, top_k
            )

    async def _calculate_degree_centrality_neo4j(
        self, relation_types: Optional[List[str]], top_k: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Calculate degree centrality using Neo4j queries."""
        # Build relation type filter
        rel_filter = ""
        if relation_types:
            rel_types_str = "|".join(relation_types)
            rel_filter = f":{rel_types_str}"

        query = f"""
        MATCH (e:Entity)
        OPTIONAL MATCH (e)-[r{rel_filter}]-()
        WITH e, count(r) as degree
        RETURN e, degree
        ORDER BY degree DESC
        {f"LIMIT {top_k}" if top_k else ""}
        """

        result = await self.storage._execute_query(query)

        centrality_scores = []
        for record in result:
            try:
                entity = Entity.from_neo4j_node(record["e"])
                degree = record["degree"]
                centrality_scores.append(
                    {
                        "entity": entity,
                        "degree": degree,
                        "centrality": degree,  # For degree centrality, score equals degree
                    }
                )
            except Exception as e:
                self.logger.warning(f"Failed to parse entity for centrality: {e}")

        return centrality_scores

    async def _calculate_degree_centrality_generic(
        self, relation_types: Optional[List[str]], top_k: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Calculate degree centrality using generic graph analysis."""
        # Get all entities and relations
        entities = await self.storage.get_all_entities()
        relations = await self.storage.get_all_relations()

        # Filter relations by type if specified
        if relation_types:
            relations = [r for r in relations if r.type in relation_types]

        # Count degrees for each entity
        degree_counts = defaultdict(int)
        for relation in relations:
            source_id = str(relation.source_id)
            target_id = str(relation.target_id)
            degree_counts[source_id] += 1
            degree_counts[target_id] += 1

        # Build centrality scores
        centrality_scores = []
        for entity in entities:
            entity_id = str(entity.id)
            degree = degree_counts.get(entity_id, 0)
            centrality_scores.append(
                {"entity": entity, "degree": degree, "centrality": degree}
            )

        # Sort by centrality (descending)
        centrality_scores.sort(key=lambda x: x["centrality"], reverse=True)

        # Apply top_k limit if specified
        if top_k:
            centrality_scores = centrality_scores[:top_k]

        return centrality_scores

    async def calculate_betweenness_centrality(
        self, relation_types: Optional[List[str]] = None, top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Calculate betweenness centrality for all entities.

        Betweenness centrality measures how often an entity lies on the
        shortest paths between other entities.

        Args:
            relation_types: Optional filter for relation types
            top_k: Return only top K entities by centrality

        Returns:
            List of entities with their betweenness centrality scores
        """
        self.logger.info("Calculating betweenness centrality")

        # Get all entities and relations
        entities = await self.storage.get_all_entities()
        relations = await self.storage.get_all_relations()

        # Filter relations by type if specified
        if relation_types:
            relations = [r for r in relations if r.type in relation_types]

        # Build adjacency list
        adjacency = defaultdict(list)
        for relation in relations:
            source_id = str(relation.source_id)
            target_id = str(relation.target_id)
            adjacency[source_id].append(target_id)
            adjacency[target_id].append(source_id)  # Undirected

        entity_ids = [str(e.id) for e in entities]
        betweenness_scores = defaultdict(float)

        # Calculate betweenness for each pair of entities
        for i, source_id in enumerate(entity_ids):
            for j, target_id in enumerate(entity_ids):
                if i >= j:  # Avoid duplicate calculations
                    continue

                # Find all shortest paths between source and target
                paths = self._find_all_shortest_paths(source_id, target_id, adjacency)

                if paths:
                    # Each entity on these paths gets credit
                    for path in paths:
                        for entity_id in path[1:-1]:  # Exclude source and target
                            betweenness_scores[entity_id] += 1.0 / len(paths)

        # Normalize by the number of pairs
        n = len(entity_ids)
        normalization_factor = (n - 1) * (n - 2) / 2 if n > 2 else 1

        # Build result
        centrality_scores = []
        entity_map = {str(e.id): e for e in entities}

        for entity in entities:
            entity_id = str(entity.id)
            raw_score = betweenness_scores.get(entity_id, 0)
            normalized_score = (
                raw_score / normalization_factor if normalization_factor > 0 else 0
            )

            centrality_scores.append(
                {
                    "entity": entity,
                    "betweenness": normalized_score,
                    "centrality": normalized_score,
                }
            )

        # Sort by centrality (descending)
        centrality_scores.sort(key=lambda x: x["centrality"], reverse=True)

        # Apply top_k limit if specified
        if top_k:
            centrality_scores = centrality_scores[:top_k]

        return centrality_scores

    def _find_all_shortest_paths(
        self, source_id: str, target_id: str, adjacency: Dict[str, List[str]]
    ) -> List[List[str]]:
        """Find all shortest paths between two entities using BFS."""
        from collections import deque

        if source_id == target_id:
            return [[source_id]]

        # BFS to find shortest distance
        queue = deque([(source_id, [source_id])])
        visited = {source_id: 0}
        shortest_distance = None
        all_paths = []

        while queue:
            current_id, path = queue.popleft()

            # If we've found a longer path than the shortest, stop
            if shortest_distance is not None and len(path) > shortest_distance + 1:
                break

            # Check if we reached the target
            if current_id == target_id:
                if shortest_distance is None:
                    shortest_distance = len(path) - 1
                if len(path) == shortest_distance + 1:
                    all_paths.append(path)
                continue

            # Explore neighbors
            for neighbor_id in adjacency.get(current_id, []):
                new_path = path + [neighbor_id]

                # Only continue if this is a shortest path
                if (
                    neighbor_id not in visited
                    or visited[neighbor_id] == len(new_path) - 1
                ):
                    visited[neighbor_id] = len(new_path) - 1
                    queue.append((neighbor_id, new_path))

        return all_paths

    async def calculate_closeness_centrality(
        self, relation_types: Optional[List[str]] = None, top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Calculate closeness centrality for all entities.

        Closeness centrality measures how close an entity is to all other entities.

        Args:
            relation_types: Optional filter for relation types
            top_k: Return only top K entities by centrality

        Returns:
            List of entities with their closeness centrality scores
        """
        self.logger.info("Calculating closeness centrality")

        # Get all entities and relations
        entities = await self.storage.get_all_entities()
        relations = await self.storage.get_all_relations()

        # Filter relations by type if specified
        if relation_types:
            relations = [r for r in relations if r.type in relation_types]

        # Build adjacency list
        adjacency = defaultdict(list)
        for relation in relations:
            source_id = str(relation.source_id)
            target_id = str(relation.target_id)
            adjacency[source_id].append(target_id)
            adjacency[target_id].append(source_id)  # Undirected

        entity_ids = [str(e.id) for e in entities]
        centrality_scores = []

        for entity in entities:
            entity_id = str(entity.id)

            # Calculate shortest distances to all other entities
            distances = self._calculate_shortest_distances(entity_id, adjacency)

            # Calculate closeness centrality
            reachable_distances = [
                d for d in distances.values() if d > 0 and d != float("inf")
            ]

            if reachable_distances:
                avg_distance = sum(reachable_distances) / len(reachable_distances)
                closeness = 1.0 / avg_distance if avg_distance > 0 else 0
            else:
                closeness = 0

            centrality_scores.append(
                {
                    "entity": entity,
                    "closeness": closeness,
                    "centrality": closeness,
                    "reachable_entities": len(reachable_distances),
                }
            )

        # Sort by centrality (descending)
        centrality_scores.sort(key=lambda x: x["centrality"], reverse=True)

        # Apply top_k limit if specified
        if top_k:
            centrality_scores = centrality_scores[:top_k]

        return centrality_scores

    def _calculate_shortest_distances(
        self, source_id: str, adjacency: Dict[str, List[str]]
    ) -> Dict[str, int]:
        """Calculate shortest distances from source to all other entities using BFS."""
        from collections import deque

        distances = {source_id: 0}
        queue = deque([source_id])

        while queue:
            current_id = queue.popleft()
            current_distance = distances[current_id]

            for neighbor_id in adjacency.get(current_id, []):
                if neighbor_id not in distances:
                    distances[neighbor_id] = current_distance + 1
                    queue.append(neighbor_id)

        return distances

    async def analyze_entity_types(self) -> Dict[str, Any]:
        """Analyze the distribution of entity types in the graph.

        Returns:
            Dictionary containing entity type statistics
        """
        self.logger.info("Analyzing entity types")

        entities = await self.storage.get_all_entities()

        # Count entities by type
        type_counts = Counter(entity.type for entity in entities)

        # Calculate statistics
        total_entities = len(entities)
        unique_types = len(type_counts)

        # Calculate type distribution
        type_distribution = {
            entity_type: {
                "count": count,
                "percentage": (count / total_entities * 100)
                if total_entities > 0
                else 0,
            }
            for entity_type, count in type_counts.items()
        }

        # Sort by count (descending)
        sorted_types = sorted(
            type_distribution.items(), key=lambda x: x[1]["count"], reverse=True
        )

        return {
            "total_entities": total_entities,
            "unique_types": unique_types,
            "type_distribution": dict(sorted_types),
            "most_common_type": sorted_types[0][0] if sorted_types else None,
            "least_common_type": sorted_types[-1][0] if sorted_types else None,
        }

    async def analyze_relation_types(self) -> Dict[str, Any]:
        """Analyze the distribution of relation types in the graph.

        Returns:
            Dictionary containing relation type statistics
        """
        self.logger.info("Analyzing relation types")

        relations = await self.storage.get_all_relations()

        # Count relations by type
        type_counts = Counter(relation.type for relation in relations)

        # Calculate statistics
        total_relations = len(relations)
        unique_types = len(type_counts)

        # Calculate type distribution
        type_distribution = {
            relation_type: {
                "count": count,
                "percentage": (count / total_relations * 100)
                if total_relations > 0
                else 0,
            }
            for relation_type, count in type_counts.items()
        }

        # Sort by count (descending)
        sorted_types = sorted(
            type_distribution.items(), key=lambda x: x[1]["count"], reverse=True
        )

        return {
            "total_relations": total_relations,
            "unique_types": unique_types,
            "type_distribution": dict(sorted_types),
            "most_common_type": sorted_types[0][0] if sorted_types else None,
            "least_common_type": sorted_types[-1][0] if sorted_types else None,
        }

    async def calculate_graph_density(
        self, relation_types: Optional[List[str]] = None
    ) -> float:
        """Calculate the density of the graph.

        Graph density is the ratio of existing edges to possible edges.

        Args:
            relation_types: Optional filter for relation types

        Returns:
            Graph density (0.0 to 1.0)
        """
        self.logger.info("Calculating graph density")

        entities = await self.storage.get_all_entities()
        relations = await self.storage.get_all_relations()

        # Filter relations by type if specified
        if relation_types:
            relations = [r for r in relations if r.type in relation_types]

        num_entities = len(entities)
        num_relations = len(relations)

        if num_entities <= 1:
            return 0.0

        # Maximum possible edges in an undirected graph
        max_possible_edges = num_entities * (num_entities - 1) / 2

        # Calculate density
        density = num_relations / max_possible_edges if max_possible_edges > 0 else 0.0

        return min(density, 1.0)  # Cap at 1.0 in case of directed graph

    async def find_entity_clusters(
        self, min_cluster_size: int = 3
    ) -> List[Dict[str, Any]]:
        """Find clusters of highly connected entities.

        Args:
            min_cluster_size: Minimum number of entities in a cluster

        Returns:
            List of clusters with their properties
        """
        self.logger.info(f"Finding entity clusters (min size: {min_cluster_size})")

        # Get all entities and relations
        entities = await self.storage.get_all_entities()
        relations = await self.storage.get_all_relations()

        # Build adjacency list
        adjacency = defaultdict(set)
        for relation in relations:
            source_id = str(relation.source_id)
            target_id = str(relation.target_id)
            adjacency[source_id].add(target_id)
            adjacency[target_id].add(source_id)

        # Find connected components
        visited = set()
        clusters = []
        entity_map = {str(e.id): e for e in entities}

        def dfs(entity_id: str, cluster: List[str]):
            if entity_id in visited:
                return

            visited.add(entity_id)
            cluster.append(entity_id)

            for neighbor_id in adjacency[entity_id]:
                dfs(neighbor_id, cluster)

        for entity in entities:
            entity_id = str(entity.id)
            if entity_id not in visited:
                cluster = []
                dfs(entity_id, cluster)

                if len(cluster) >= min_cluster_size:
                    # Calculate cluster properties
                    cluster_entities = [
                        entity_map[eid] for eid in cluster if eid in entity_map
                    ]

                    # Count internal edges
                    internal_edges = 0
                    for relation in relations:
                        source_id = str(relation.source_id)
                        target_id = str(relation.target_id)
                        if source_id in cluster and target_id in cluster:
                            internal_edges += 1

                    # Calculate cluster density
                    max_internal_edges = len(cluster) * (len(cluster) - 1) / 2
                    cluster_density = (
                        internal_edges / max_internal_edges
                        if max_internal_edges > 0
                        else 0
                    )

                    # Analyze entity types in cluster
                    type_counts = Counter(entity.type for entity in cluster_entities)

                    clusters.append(
                        {
                            "entities": cluster_entities,
                            "size": len(cluster),
                            "internal_edges": internal_edges,
                            "density": cluster_density,
                            "entity_types": dict(type_counts),
                            "dominant_type": type_counts.most_common(1)[0][0]
                            if type_counts
                            else None,
                        }
                    )

        # Sort clusters by size (descending)
        clusters.sort(key=lambda c: c["size"], reverse=True)

        self.logger.info(f"Found {len(clusters)} clusters")
        return clusters

    async def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics.

        Returns:
            Dictionary containing various graph metrics
        """
        self.logger.info("Calculating comprehensive graph statistics")

        # Get basic counts
        entities = await self.storage.get_all_entities()
        relations = await self.storage.get_all_relations()

        # Calculate basic metrics
        num_entities = len(entities)
        num_relations = len(relations)

        # Calculate degree statistics
        degree_counts = defaultdict(int)
        for relation in relations:
            source_id = str(relation.source_entity_id)
            target_id = str(relation.target_entity_id)
            degree_counts[source_id] += 1
            degree_counts[target_id] += 1

        degrees = list(degree_counts.values())
        avg_degree = sum(degrees) / len(degrees) if degrees else 0
        max_degree = max(degrees) if degrees else 0
        min_degree = min(degrees) if degrees else 0

        # Calculate density
        density = await self.calculate_graph_density()

        # Analyze types
        entity_type_analysis = await self.analyze_entity_types()
        relation_type_analysis = await self.analyze_relation_types()

        # Find connected components
        from .traversal import GraphTraversal

        traversal = GraphTraversal(self.storage)
        components = await traversal.find_connected_components()

        return {
            "basic_metrics": {
                "num_entities": num_entities,
                "num_relations": num_relations,
                "density": density,
            },
            "degree_statistics": {
                "average_degree": avg_degree,
                "max_degree": max_degree,
                "min_degree": min_degree,
                "degree_distribution": dict(Counter(degrees)),
            },
            "connectivity": {
                "num_components": len(components),
                "largest_component_size": len(components[0]) if components else 0,
                "is_connected": len(components) <= 1,
            },
            "entity_analysis": entity_type_analysis,
            "relation_analysis": relation_type_analysis,
        }
