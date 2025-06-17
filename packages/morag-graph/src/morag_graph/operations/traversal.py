"""Graph traversal operations.

This module provides graph traversal algorithms including path finding,
neighborhood exploration, and multi-hop queries.
"""

import logging
from typing import List, Optional, Dict, Any, Set, Tuple, Union
from uuid import UUID
from collections import deque, defaultdict

from ..models import Entity, Relation
from ..storage.base import BaseStorage
from ..storage.neo4j_storage import Neo4jStorage

logger = logging.getLogger(__name__)


class GraphPath:
    """Represents a path in the graph."""
    
    def __init__(self, entities: List[Entity], relations: List[Relation]):
        """Initialize a graph path.
        
        Args:
            entities: List of entities in the path
            relations: List of relations connecting the entities
        """
        self.entities = entities
        self.relations = relations
        self.length = len(relations)
    
    def __str__(self) -> str:
        """String representation of the path."""
        if not self.entities:
            return "Empty path"
        
        path_str = self.entities[0].name
        for i, relation in enumerate(self.relations):
            if i + 1 < len(self.entities):
                path_str += f" --[{relation.type}]--> {self.entities[i + 1].name}"
        
        return path_str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert path to dictionary representation."""
        return {
            "entities": [{
                "id": str(e.id),
                "name": e.name,
                "type": e.type
            } for e in self.entities],
            "relations": [{
                "id": str(r.id),
                "type": r.type,
                "source_id": str(r.source_id),
                "target_id": str(r.target_id)
            } for r in self.relations],
            "length": self.length
        }


class GraphTraversal:
    """Graph traversal operations.
    
    This class provides various graph traversal algorithms and
    path finding capabilities.
    """
    
    def __init__(self, storage: BaseStorage):
        """Initialize GraphTraversal with a storage backend.
        
        Args:
            storage: Storage backend (Neo4j, JSON, etc.)
        """
        self.storage = storage
        self.logger = logger.getChild(self.__class__.__name__)
    
    async def find_neighbors(self, entity_id: Union[str, UUID], 
                           max_distance: int = 1,
                           relation_types: Optional[List[str]] = None) -> List[Entity]:
        """Find neighboring entities within a given distance.
        
        Args:
            entity_id: Starting entity ID
            max_distance: Maximum distance to traverse
            relation_types: Optional filter for relation types
            
        Returns:
            List of neighboring entities
        """
        self.logger.info(f"Finding neighbors for entity {entity_id} within distance {max_distance}")
        
        if isinstance(self.storage, Neo4jStorage):
            return await self._find_neighbors_neo4j(entity_id, max_distance, relation_types)
        else:
            return await self._find_neighbors_generic(entity_id, max_distance, relation_types)
    
    async def _find_neighbors_neo4j(self, entity_id: Union[str, UUID], 
                                  max_distance: int,
                                  relation_types: Optional[List[str]]) -> List[Entity]:
        """Find neighbors using Neo4j-specific queries."""
        entity_id_str = str(entity_id)
        
        # Build relation type filter
        rel_filter = ""
        if relation_types:
            rel_types_str = "|".join(relation_types)
            rel_filter = f":{rel_types_str}"
        
        query = f"""
        MATCH (start:Entity {{id: $entity_id}})
        MATCH path = (start)-[{rel_filter}*1..{max_distance}]-(neighbor:Entity)
        WHERE start <> neighbor
        RETURN DISTINCT neighbor
        ORDER BY neighbor.name
        """
        
        result = await self.storage._execute_query(query, {"entity_id": entity_id_str})
        
        neighbors = []
        for record in result:
            try:
                entity = Entity.from_neo4j_node(record["neighbor"])
                neighbors.append(entity)
            except Exception as e:
                self.logger.warning(f"Failed to parse neighbor entity: {e}")
        
        return neighbors
    
    async def _find_neighbors_generic(self, entity_id: Union[str, UUID], 
                                    max_distance: int,
                                    relation_types: Optional[List[str]]) -> List[Entity]:
        """Find neighbors using generic graph traversal."""
        entity_id_str = str(entity_id)
        
        # Get all entities and relations
        entities = await self.storage.get_all_entities()
        relations = await self.storage.get_all_relations()
        
        # Build entity lookup
        entity_map = {str(e.id): e for e in entities}
        
        # Filter relations by type if specified
        if relation_types:
            relations = [r for r in relations if r.type in relation_types]
        
        # Build adjacency list
        adjacency = defaultdict(list)
        for relation in relations:
            source_id = str(relation.source_id)
            target_id = str(relation.target_id)
            adjacency[source_id].append(target_id)
            adjacency[target_id].append(source_id)  # Undirected traversal
        
        # BFS to find neighbors within max_distance
        visited = set()
        queue = deque([(entity_id_str, 0)])  # (entity_id, distance)
        neighbors = []
        
        while queue:
            current_id, distance = queue.popleft()
            
            if current_id in visited:
                continue
            
            visited.add(current_id)
            
            # Add to neighbors if not the starting entity
            if distance > 0 and current_id in entity_map:
                neighbors.append(entity_map[current_id])
            
            # Continue traversal if within max distance
            if distance < max_distance:
                for neighbor_id in adjacency[current_id]:
                    if neighbor_id not in visited:
                        queue.append((neighbor_id, distance + 1))
        
        return neighbors
    
    async def find_shortest_path(self, source_id: Union[str, UUID], 
                               target_id: Union[str, UUID],
                               relation_types: Optional[List[str]] = None) -> Optional[GraphPath]:
        """Find the shortest path between two entities.
        
        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            relation_types: Optional filter for relation types
            
        Returns:
            GraphPath if path exists, None otherwise
        """
        self.logger.info(f"Finding shortest path from {source_id} to {target_id}")
        
        if isinstance(self.storage, Neo4jStorage):
            return await self._find_shortest_path_neo4j(source_id, target_id, relation_types)
        else:
            return await self._find_shortest_path_generic(source_id, target_id, relation_types)
    
    async def _find_shortest_path_neo4j(self, source_id: Union[str, UUID], 
                                       target_id: Union[str, UUID],
                                       relation_types: Optional[List[str]]) -> Optional[GraphPath]:
        """Find shortest path using Neo4j-specific queries."""
        source_id_str = str(source_id)
        target_id_str = str(target_id)
        
        # Build relation type filter
        rel_filter = ""
        if relation_types:
            rel_types_str = "|".join(relation_types)
            rel_filter = f":{rel_types_str}"
        
        query = f"""
        MATCH (source:Entity {{id: $source_id}}), (target:Entity {{id: $target_id}})
        MATCH path = shortestPath((source)-[{rel_filter}*]-(target))
        RETURN path, nodes(path) as entities, relationships(path) as relations
        """
        
        result = await self.storage._execute_query(query, {
            "source_id": source_id_str,
            "target_id": target_id_str
        })
        
        if not result:
            return None
        
        record = result[0]
        
        # Parse entities and relations from path
        entities = []
        for node in record["entities"]:
            try:
                entity = Entity.from_neo4j_node(node)
                entities.append(entity)
            except Exception as e:
                self.logger.warning(f"Failed to parse path entity: {e}")
        
        relations = []
        for rel in record["relations"]:
            try:
                relation = Relation.from_neo4j_relationship(rel)
                relations.append(relation)
            except Exception as e:
                self.logger.warning(f"Failed to parse path relation: {e}")
        
        return GraphPath(entities, relations)
    
    async def _find_shortest_path_generic(self, source_id: Union[str, UUID], 
                                        target_id: Union[str, UUID],
                                        relation_types: Optional[List[str]]) -> Optional[GraphPath]:
        """Find shortest path using generic BFS."""
        source_id_str = str(source_id)
        target_id_str = str(target_id)
        
        # Get all entities and relations
        entities = await self.storage.get_all_entities()
        relations = await self.storage.get_all_relations()
        
        # Build lookups
        entity_map = {str(e.id): e for e in entities}
        relation_map = {str(r.id): r for r in relations}
        
        # Filter relations by type if specified
        if relation_types:
            relations = [r for r in relations if r.type in relation_types]
        
        # Build adjacency list with relation info
        adjacency = defaultdict(list)
        for relation in relations:
            source = str(relation.source_id)
            target = str(relation.target_id)
            adjacency[source].append((target, str(relation.id)))
            adjacency[target].append((source, str(relation.id)))  # Undirected
        
        # BFS to find shortest path
        queue = deque([(source_id_str, [source_id_str], [])])  # (current, path_entities, path_relations)
        visited = set()
        
        while queue:
            current_id, path_entities, path_relations = queue.popleft()
            
            if current_id in visited:
                continue
            
            visited.add(current_id)
            
            # Check if we reached the target
            if current_id == target_id_str:
                # Build GraphPath
                path_entity_objects = [entity_map[eid] for eid in path_entities if eid in entity_map]
                path_relation_objects = [relation_map[rid] for rid in path_relations if rid in relation_map]
                return GraphPath(path_entity_objects, path_relation_objects)
            
            # Explore neighbors
            for neighbor_id, relation_id in adjacency[current_id]:
                if neighbor_id not in visited:
                    new_path_entities = path_entities + [neighbor_id]
                    new_path_relations = path_relations + [relation_id]
                    queue.append((neighbor_id, new_path_entities, new_path_relations))
        
        return None  # No path found
    
    async def find_all_paths(self, source_id: Union[str, UUID], 
                           target_id: Union[str, UUID],
                           max_length: int = 5,
                           relation_types: Optional[List[str]] = None) -> List[GraphPath]:
        """Find all paths between two entities up to a maximum length.
        
        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            max_length: Maximum path length
            relation_types: Optional filter for relation types
            
        Returns:
            List of all paths found
        """
        self.logger.info(f"Finding all paths from {source_id} to {target_id} (max length: {max_length})")
        
        source_id_str = str(source_id)
        target_id_str = str(target_id)
        
        # Get all entities and relations
        entities = await self.storage.get_all_entities()
        relations = await self.storage.get_all_relations()
        
        # Build lookups
        entity_map = {str(e.id): e for e in entities}
        relation_map = {str(r.id): r for r in relations}
        
        # Filter relations by type if specified
        if relation_types:
            relations = [r for r in relations if r.type in relation_types]
        
        # Build adjacency list
        adjacency = defaultdict(list)
        for relation in relations:
            source = str(relation.source_id)
            target = str(relation.target_id)
            adjacency[source].append((target, str(relation.id)))
            adjacency[target].append((source, str(relation.id)))  # Undirected
        
        # DFS to find all paths
        all_paths = []
        
        def dfs(current_id: str, path_entities: List[str], path_relations: List[str], visited: Set[str]):
            if len(path_relations) >= max_length:
                return
            
            if current_id == target_id_str and len(path_entities) > 1:
                # Found a path
                path_entity_objects = [entity_map[eid] for eid in path_entities if eid in entity_map]
                path_relation_objects = [relation_map[rid] for rid in path_relations if rid in relation_map]
                all_paths.append(GraphPath(path_entity_objects, path_relation_objects))
                return
            
            for neighbor_id, relation_id in adjacency[current_id]:
                if neighbor_id not in visited:
                    new_visited = visited.copy()
                    new_visited.add(current_id)
                    dfs(neighbor_id, 
                        path_entities + [neighbor_id], 
                        path_relations + [relation_id],
                        new_visited)
        
        dfs(source_id_str, [source_id_str], [], set())
        
        # Sort paths by length
        all_paths.sort(key=lambda p: p.length)
        
        self.logger.info(f"Found {len(all_paths)} paths")
        return all_paths
    
    async def find_connected_components(self, relation_types: Optional[List[str]] = None) -> List[List[Entity]]:
        """Find connected components in the graph.
        
        Args:
            relation_types: Optional filter for relation types
            
        Returns:
            List of connected components (each component is a list of entities)
        """
        self.logger.info("Finding connected components")
        
        # Get all entities and relations
        entities = await self.storage.get_all_entities()
        relations = await self.storage.get_all_relations()
        
        # Build entity lookup
        entity_map = {str(e.id): e for e in entities}
        
        # Filter relations by type if specified
        if relation_types:
            relations = [r for r in relations if r.type in relation_types]
        
        # Build adjacency list
        adjacency = defaultdict(set)
        for relation in relations:
            source_id = str(relation.source_entity_id)
            target_id = str(relation.target_entity_id)
            adjacency[source_id].add(target_id)
            adjacency[target_id].add(source_id)
        
        # Find connected components using DFS
        visited = set()
        components = []
        
        def dfs(entity_id: str, component: List[str]):
            if entity_id in visited:
                return
            
            visited.add(entity_id)
            component.append(entity_id)
            
            for neighbor_id in adjacency[entity_id]:
                dfs(neighbor_id, component)
        
        for entity in entities:
            entity_id = str(entity.id)
            if entity_id not in visited:
                component = []
                dfs(entity_id, component)
                if component:
                    component_entities = [entity_map[eid] for eid in component if eid in entity_map]
                    components.append(component_entities)
        
        # Sort components by size (largest first)
        components.sort(key=len, reverse=True)
        
        self.logger.info(f"Found {len(components)} connected components")
        return components
    
    async def get_entity_degree(self, entity_id: Union[str, UUID], 
                              relation_types: Optional[List[str]] = None) -> Dict[str, int]:
        """Get the degree (number of connections) of an entity.
        
        Args:
            entity_id: Entity ID
            relation_types: Optional filter for relation types
            
        Returns:
            Dictionary with degree information
        """
        entity_id_str = str(entity_id)
        
        # Get all relations
        relations = await self.storage.get_all_relations()
        
        # Filter relations by type if specified
        if relation_types:
            relations = [r for r in relations if r.type in relation_types]
        
        # Count degrees
        in_degree = sum(1 for r in relations if str(r.target_id) == entity_id_str)
        out_degree = sum(1 for r in relations if str(r.source_id) == entity_id_str)
        total_degree = in_degree + out_degree
        
        return {
            "in_degree": in_degree,
            "out_degree": out_degree,
            "total_degree": total_degree
        }