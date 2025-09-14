"""Query and search operations for Neo4j storage."""

import logging
from typing import Dict, List, Optional, Any

from ...models import Entity
from ...models.types import EntityId
from .base_operations import BaseOperations

logger = logging.getLogger(__name__)


class QueryOperations(BaseOperations):
    """Handles search, path finding, and query operations."""
    
    async def get_neighbors(
        self, 
        entity_id: EntityId,
        relation_type: Optional[str] = None,
        max_depth: int = 1
    ) -> List[Entity]:
        """Get neighboring entities.
        
        Args:
            entity_id: ID of the central entity
            relation_type: Optional relation type filter
            max_depth: Maximum depth to search
            
        Returns:
            List of neighboring entities
        """
        # Build query based on parameters
        # Note: Using () instead of (:Entity) to match nodes with any label
        if relation_type:
            query = f"""
            MATCH (e {{id: $entity_id}})-[r*1..{max_depth}]-(neighbor)
            WHERE neighbor.id <> $entity_id
            AND ALL(rel in r WHERE rel.type = $relation_type)
            AND neighbor.id IS NOT NULL
            RETURN DISTINCT neighbor
            """
            parameters = {"entity_id": entity_id, "relation_type": relation_type}
        else:
            query = f"""
            MATCH (e {{id: $entity_id}})-[r*1..{max_depth}]-(neighbor)
            WHERE neighbor.id <> $entity_id
            AND neighbor.id IS NOT NULL
            RETURN DISTINCT neighbor
            """
            parameters = {"entity_id": entity_id}
        
        result = await self._execute_query(query, parameters)
        
        neighbors = []
        for record in result:
            try:
                neighbor_data = dict(record["neighbor"])
                # Only process if it has the required fields for an Entity
                if 'name' in neighbor_data and neighbor_data['name']:
                    neighbors.append(Entity.from_neo4j_node(neighbor_data))
                else:
                    logger.debug(f"Skipping non-entity node: {neighbor_data}")
            except Exception as e:
                logger.warning(f"Failed to parse neighbor entity: {e}")

        return neighbors
    
    async def find_path(
        self, 
        source_entity_id: EntityId,
        target_entity_id: EntityId,
        max_depth: int = 3
    ) -> List[List[EntityId]]:
        """Find paths between two entities.
        
        Args:
            source_entity_id: Source entity ID
            target_entity_id: Target entity ID
            max_depth: Maximum path length
            
        Returns:
            List of paths (each path is a list of entity IDs)
        """
        query = f"""
        MATCH path = (source {{id: $source_id}})-[r*1..{max_depth}]-(target {{id: $target_id}})
        WHERE source.id IS NOT NULL AND target.id IS NOT NULL
        RETURN [node in nodes(path) | node.id] as path_ids
        LIMIT 10
        """
        
        result = await self._execute_query(query, {
            "source_id": source_entity_id,
            "target_id": target_entity_id
        })
        
        return [record["path_ids"] for record in result]
    
    async def search_entities_by_content(
        self,
        search_term: str,
        entity_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Entity]:
        """Search entities by content using full-text search.
        
        Args:
            search_term: Term to search for
            entity_type: Optional entity type filter
            limit: Maximum number of results
            
        Returns:
            List of matching entities
        """
        # Build the query
        # Note: Using () instead of (:Entity) to match nodes with any label
        base_query = """
        MATCH (e)
        WHERE toLower(e.name) CONTAINS toLower($search_term)
        AND e.id IS NOT NULL
        """
        
        parameters = {"search_term": search_term, "limit": limit}
        
        if entity_type:
            base_query += " AND e.type = $entity_type"
            parameters["entity_type"] = entity_type
        
        base_query += """
        RETURN e
        ORDER BY 
            CASE WHEN toLower(e.name) = toLower($search_term) THEN 0 ELSE 1 END,
            e.confidence DESC,
            e.name
        LIMIT $limit
        """
        
        result = await self._execute_query(base_query, parameters)
        
        entities = []
        for record in result:
            try:
                entities.append(Entity.from_neo4j_node(record["e"]))
            except Exception as e:
                logger.warning(f"Failed to parse entity from search: {e}")
        
        return entities
    
    async def find_related_entities(
        self,
        entity_id: EntityId,
        relation_types: Optional[List[str]] = None,
        max_depth: int = 2,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Find entities related to a given entity with relationship context.
        
        Args:
            entity_id: Central entity ID
            relation_types: Optional list of relation types to follow
            max_depth: Maximum relationship depth
            limit: Maximum number of results
            
        Returns:
            List of related entities with relationship metadata
        """
        if relation_types:
            relation_filter = f"WHERE ALL(r in relationships(path) WHERE r.type IN {relation_types})"
        else:
            relation_filter = ""
        
        query = f"""
        MATCH path = (start {{id: $entity_id}})-[r*1..{max_depth}]-(related)
        {relation_filter}
        WHERE related.id <> $entity_id
        AND start.id IS NOT NULL AND related.id IS NOT NULL
        WITH related, path, length(path) as distance
        ORDER BY distance, related.confidence DESC
        LIMIT $limit
        RETURN related, distance, 
               [r in relationships(path) | {{type: r.type, confidence: r.confidence}}] as relationship_path
        """
        
        result = await self._execute_query(query, {
            "entity_id": entity_id,
            "limit": limit
        })
        
        related_entities = []
        for record in result:
            try:
                entity = Entity.from_neo4j_node(record["related"])
                related_entities.append({
                    "entity": entity,
                    "distance": record["distance"],
                    "relationship_path": record["relationship_path"]
                })
            except Exception as e:
                logger.warning(f"Failed to parse related entity: {e}")
        
        return related_entities
    
    async def get_entity_clusters(
        self,
        min_cluster_size: int = 3,
        max_clusters: int = 10
    ) -> List[Dict[str, Any]]:
        """Find clusters of highly connected entities.
        
        Args:
            min_cluster_size: Minimum number of entities in a cluster
            max_clusters: Maximum number of clusters to return
            
        Returns:
            List of entity clusters with metadata
        """
        query = """
        MATCH (e)-[r]-(connected)
        WHERE e.id IS NOT NULL AND connected.id IS NOT NULL
        WITH e, count(connected) as connection_count, collect(connected) as connections
        WHERE connection_count >= $min_cluster_size
        ORDER BY connection_count DESC
        LIMIT $max_clusters
        RETURN e as central_entity, connection_count, connections
        """
        
        result = await self._execute_query(query, {
            "min_cluster_size": min_cluster_size,
            "max_clusters": max_clusters
        })
        
        clusters = []
        for record in result:
            try:
                central_entity = Entity.from_neo4j_node(record["central_entity"])
                connected_entities = []
                for conn in record["connections"]:
                    try:
                        connected_entities.append(Entity.from_neo4j_node(conn))
                    except Exception as e:
                        logger.warning(f"Failed to parse connected entity: {e}")
                
                clusters.append({
                    "central_entity": central_entity,
                    "connection_count": record["connection_count"],
                    "connected_entities": connected_entities
                })
            except Exception as e:
                logger.warning(f"Failed to parse cluster: {e}")
        
        return clusters
    
    async def find_shortest_paths(
        self,
        source_entity_id: EntityId,
        target_entity_id: EntityId,
        max_depth: int = 5
    ) -> List[Dict[str, Any]]:
        """Find shortest paths between two entities with relationship details.
        
        Args:
            source_entity_id: Source entity ID
            target_entity_id: Target entity ID
            max_depth: Maximum path length to consider
            
        Returns:
            List of shortest paths with relationship details
        """
        query = f"""
        MATCH path = shortestPath((source {{id: $source_id}})-[r*1..{max_depth}]-(target {{id: $target_id}}))
        WHERE source.id IS NOT NULL AND target.id IS NOT NULL
        RETURN path,
               length(path) as path_length,
               [node in nodes(path) | {{id: node.id, name: node.name, type: node.type}}] as entities,
               [rel in relationships(path) | {{type: rel.type, confidence: rel.confidence}}] as relationships
        ORDER BY path_length
        LIMIT 5
        """
        
        result = await self._execute_query(query, {
            "source_id": source_entity_id,
            "target_id": target_entity_id
        })
        
        paths = []
        for record in result:
            paths.append({
                "path_length": record["path_length"],
                "entities": record["entities"],
                "relationships": record["relationships"]
            })
        
        return paths
