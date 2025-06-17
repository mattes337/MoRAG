"""Neo4J storage backend for graph data."""

import logging
from typing import Dict, List, Optional, Any, Set

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from pydantic import BaseModel

from ..models import Entity, Relation, Graph
from ..models.types import EntityId, RelationId
from .base import BaseStorage

logger = logging.getLogger(__name__)


class Neo4jConfig(BaseModel):
    """Configuration for Neo4J connection."""
    
    uri: str = "neo4j://localhost:7687"
    username: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"
    max_connection_lifetime: int = 3600
    max_connection_pool_size: int = 50
    connection_acquisition_timeout: int = 60


class Neo4jStorage(BaseStorage):
    """Neo4J storage backend for graph data.
    
    This class implements the BaseStorage interface using Neo4J as the backend.
    It provides efficient storage and retrieval of entities and relations.
    """
    
    def __init__(self, config: Neo4jConfig):
        """Initialize Neo4J storage.
        
        Args:
            config: Neo4J configuration
        """
        self.config = config
        self.driver: Optional[AsyncDriver] = None
    
    async def connect(self) -> None:
        """Connect to Neo4J database."""
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password),
                max_connection_lifetime=self.config.max_connection_lifetime,
                max_connection_pool_size=self.config.max_connection_pool_size,
                connection_acquisition_timeout=self.config.connection_acquisition_timeout,
            )
            
            # Test connection
            async with self.driver.session(database=self.config.database) as session:
                await session.run("RETURN 1")
            
            logger.info("Connected to Neo4J database")
            
        except Exception as e:
            logger.error(f"Failed to connect to Neo4J: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from Neo4J database."""
        if self.driver:
            await self.driver.close()
            self.driver = None
            logger.info("Disconnected from Neo4J database")
    
    async def _execute_query(
        self, 
        query: str, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            List of result records
        """
        if not self.driver:
            raise RuntimeError("Not connected to Neo4J database")
        
        async with self.driver.session(database=self.config.database) as session:
            result = await session.run(query, parameters or {})
            records = []
            async for record in result:
                records.append(record.data())
            return records
    
    async def store_entity(self, entity: Entity) -> EntityId:
        """Store an entity in Neo4J.
        
        Args:
            entity: Entity to store
            
        Returns:
            ID of the stored entity
        """
        properties = entity.to_neo4j_node()
        labels = properties.pop('_labels', ['Entity'])
        
        # Create labels string for Cypher
        labels_str = ':'.join(labels)
        
        query = f"""
        MERGE (e:{labels_str} {{id: $id}})
        SET e += $properties
        RETURN e.id as id
        """
        
        parameters = {
            "id": entity.id,
            "properties": properties
        }
        
        result = await self._execute_query(query, parameters)
        return result[0]["id"] if result else entity.id
    
    async def store_entities(self, entities: List[Entity]) -> List[EntityId]:
        """Store multiple entities in Neo4J.
        
        Args:
            entities: List of entities to store
            
        Returns:
            List of IDs of the stored entities
        """
        if not entities:
            return []
        
        # Prepare batch data
        entity_data = []
        for entity in entities:
            properties = entity.to_neo4j_node()
            labels = properties.pop('_labels', ['Entity'])
            entity_data.append({
                "id": entity.id,
                "labels": labels,
                "properties": properties
            })
        
        query = """
        UNWIND $entities as entity_data
        CALL apoc.create.node(entity_data.labels, entity_data.properties) YIELD node
        RETURN node.id as id
        """
        
        try:
            result = await self._execute_query(query, {"entities": entity_data})
            return [record["id"] for record in result]
        except Exception as e:
            # Fallback to individual inserts if APOC is not available
            logger.warning(f"Batch insert failed, falling back to individual inserts: {e}")
            return [await self.store_entity(entity) for entity in entities]
    
    async def get_entity(self, entity_id: EntityId) -> Optional[Entity]:
        """Get an entity by ID.
        
        Args:
            entity_id: ID of the entity to get
            
        Returns:
            Entity if found, None otherwise
        """
        query = """
        MATCH (e:Entity {id: $entity_id})
        RETURN e
        """
        
        result = await self._execute_query(query, {"entity_id": entity_id})
        
        if result:
            node_data = result[0]["e"]
            return Entity.from_neo4j_node(node_data)
        
        return None
    
    async def get_entities(self, entity_ids: List[EntityId]) -> List[Entity]:
        """Get multiple entities by IDs.
        
        Args:
            entity_ids: List of entity IDs to get
            
        Returns:
            List of entities
        """
        if not entity_ids:
            return []
        
        query = """
        MATCH (e:Entity)
        WHERE e.id IN $entity_ids
        RETURN e
        """
        
        result = await self._execute_query(query, {"entity_ids": entity_ids})
        
        entities = []
        for record in result:
            try:
                entity = Entity.from_neo4j_node(record["e"])
                entities.append(entity)
            except Exception as e:
                logger.warning(f"Failed to parse entity from Neo4J: {e}")
        
        return entities
    
    async def get_all_entities(self) -> List[Entity]:
        """Get all entities from the storage.
        
        Returns:
            List of all entities
        """
        query = "MATCH (e:Entity) RETURN e"
        result = await self._execute_query(query)
        
        entities = []
        for record in result:
            try:
                entity = Entity.from_neo4j_node(record["e"])
                entities.append(entity)
            except Exception as e:
                logger.warning(f"Failed to parse entity: {e}")
        
        return entities
    
    async def search_entities(
        self, 
        query: str, 
        entity_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Entity]:
        """Search for entities by name or attributes.
        
        Args:
            query: Search query
            entity_type: Optional entity type filter
            limit: Maximum number of results
            
        Returns:
            List of matching entities
        """
        cypher_query = """
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower($query)
        """
        
        parameters = {"query": query, "limit": limit}
        
        if entity_type:
            cypher_query += " AND e.type = $entity_type"
            parameters["entity_type"] = entity_type
        
        cypher_query += """
        RETURN e
        ORDER BY e.name
        LIMIT $limit
        """
        
        result = await self._execute_query(cypher_query, parameters)
        
        entities = []
        for record in result:
            try:
                entity = Entity.from_neo4j_node(record["e"])
                entities.append(entity)
            except Exception as e:
                logger.warning(f"Failed to parse entity from search: {e}")
        
        return entities
    
    async def update_entity(self, entity: Entity) -> bool:
        """Update an existing entity.
        
        Args:
            entity: Entity with updated data
            
        Returns:
            True if entity was updated, False if not found
        """
        properties = entity.to_neo4j_node()
        properties.pop('_labels', None)  # Don't update labels
        
        query = """
        MATCH (e:Entity {id: $id})
        SET e += $properties
        RETURN e.id as id
        """
        
        result = await self._execute_query(query, {
            "id": entity.id,
            "properties": properties
        })
        
        return len(result) > 0
    
    async def delete_entity(self, entity_id: EntityId) -> bool:
        """Delete an entity and all its relations.
        
        Args:
            entity_id: ID of the entity to delete
            
        Returns:
            True if entity was deleted, False if not found
        """
        query = """
        MATCH (e:Entity {id: $entity_id})
        DETACH DELETE e
        RETURN count(e) as deleted_count
        """
        
        result = await self._execute_query(query, {"entity_id": entity_id})
        return result[0]["deleted_count"] > 0 if result else False
    
    async def store_relation(self, relation: Relation) -> RelationId:
        """Store a relation in Neo4J.
        
        Args:
            relation: Relation to store
            
        Returns:
            ID of the stored relation
        """
        properties = relation.to_neo4j_relationship()
        relation_type = relation.get_neo4j_type()
        
        query = f"""
        MATCH (source:Entity {{id: $source_id}})
        MATCH (target:Entity {{id: $target_id}})
        MERGE (source)-[r:{relation_type} {{id: $relation_id}}]->(target)
        SET r += $properties
        RETURN r.id as id
        """
        
        parameters = {
            "source_id": relation.source_entity_id,
            "target_id": relation.target_entity_id,
            "relation_id": relation.id,
            "properties": properties
        }
        
        result = await self._execute_query(query, parameters)
        return result[0]["id"] if result else relation.id
    
    async def store_relations(self, relations: List[Relation]) -> List[RelationId]:
        """Store multiple relations in Neo4J.
        
        Args:
            relations: List of relations to store
            
        Returns:
            List of IDs of the stored relations
        """
        if not relations:
            return []
        
        # For now, use individual inserts (could be optimized with batch operations)
        return [await self.store_relation(relation) for relation in relations]
    
    async def get_relation(self, relation_id: RelationId) -> Optional[Relation]:
        """Get a relation by ID.
        
        Args:
            relation_id: ID of the relation to get
            
        Returns:
            Relation if found, None otherwise
        """
        query = """
        MATCH (source)-[r {id: $relation_id}]->(target)
        RETURN r, source.id as source_id, target.id as target_id
        """
        
        result = await self._execute_query(query, {"relation_id": relation_id})
        
        if result:
            record = result[0]
            return Relation.from_neo4j_relationship(
                record["r"], 
                record["source_id"], 
                record["target_id"]
            )
        
        return None
    
    async def get_relations(self, relation_ids: List[RelationId]) -> List[Relation]:
        """Get multiple relations by IDs.
        
        Args:
            relation_ids: List of relation IDs to get
            
        Returns:
            List of relations
        """
        if not relation_ids:
            return []
        
        query = """
        MATCH (source)-[r]->(target)
        WHERE r.id IN $relation_ids
        RETURN r, source.id as source_id, target.id as target_id
        """
        
        result = await self._execute_query(query, {"relation_ids": relation_ids})
        
        relations = []
        for record in result:
            try:
                relation = Relation.from_neo4j_relationship(
                    record["r"], 
                    record["source_id"], 
                    record["target_id"]
                )
                relations.append(relation)
            except Exception as e:
                logger.warning(f"Failed to parse relation from Neo4J: {e}")
        
        return relations
    
    async def get_entity_relations(
        self, 
        entity_id: EntityId,
        relation_type: Optional[str] = None,
        direction: str = "both"
    ) -> List[Relation]:
        """Get all relations for an entity.
        
        Args:
            entity_id: ID of the entity
            relation_type: Optional relation type filter
            direction: Direction of relations ("in", "out", "both")
            
        Returns:
            List of relations involving the entity
        """
        if direction == "out":
            pattern = "(e)-[r]->(target)"
            return_clause = "r, e.id as source_id, target.id as target_id"
        elif direction == "in":
            pattern = "(source)-[r]->(e)"
            return_clause = "r, source.id as source_id, e.id as target_id"
        else:  # both
            pattern = "(n1)-[r]-(e)-[r2]-(n2)"
            return_clause = "r, n1.id as source_id, e.id as target_id"
        
        query = f"""
        MATCH (e:Entity {{id: $entity_id}})
        MATCH {pattern}
        """
        
        parameters = {"entity_id": entity_id}
        
        if relation_type:
            query = query.replace("[r]", f"[r:{relation_type}]")
        
        query += f" RETURN {return_clause}"
        
        result = await self._execute_query(query, parameters)
        
        relations = []
        for record in result:
            try:
                relation = Relation.from_neo4j_relationship(
                    record["r"], 
                    record["source_id"], 
                    record["target_id"]
                )
                relations.append(relation)
            except Exception as e:
                logger.warning(f"Failed to parse relation: {e}")
        
        return relations
    
    async def get_all_relations(self) -> List[Relation]:
        """Get all relations from the storage.
        
        Returns:
            List of all relations
        """
        query = """
        MATCH (source:Entity)-[r]->(target:Entity)
        RETURN r, source.id as source_id, target.id as target_id
        """
        
        result = await self._execute_query(query)
        
        relations = []
        for record in result:
            try:
                relation = Relation.from_neo4j_relationship(
                    record["r"], 
                    record["source_id"], 
                    record["target_id"]
                )
                relations.append(relation)
            except Exception as e:
                logger.warning(f"Failed to parse relation: {e}")
        
        return relations
    
    async def update_relation(self, relation: Relation) -> bool:
        """Update an existing relation.
        
        Args:
            relation: Relation with updated data
            
        Returns:
            True if relation was updated, False if not found
        """
        properties = relation.to_neo4j_relationship()
        
        query = """
        MATCH ()-[r {id: $id}]->()
        SET r += $properties
        RETURN r.id as id
        """
        
        result = await self._execute_query(query, {
            "id": relation.id,
            "properties": properties
        })
        
        return len(result) > 0
    
    async def delete_relation(self, relation_id: RelationId) -> bool:
        """Delete a relation.
        
        Args:
            relation_id: ID of the relation to delete
            
        Returns:
            True if relation was deleted, False if not found
        """
        query = """
        MATCH ()-[r {id: $relation_id}]->()
        DELETE r
        RETURN count(r) as deleted_count
        """
        
        result = await self._execute_query(query, {"relation_id": relation_id})
        return result[0]["deleted_count"] > 0 if result else False
    
    async def get_neighbors(
        self, 
        entity_id: EntityId,
        relation_type: Optional[str] = None,
        max_depth: int = 1
    ) -> List[Entity]:
        """Get neighboring entities.
        
        Args:
            entity_id: ID of the entity
            relation_type: Optional relation type filter
            max_depth: Maximum traversal depth
            
        Returns:
            List of neighboring entities
        """
        relation_filter = f":{relation_type}" if relation_type else ""
        
        query = f"""
        MATCH (e:Entity {{id: $entity_id}})
        MATCH (e)-[{relation_filter}*1..{max_depth}]-(neighbor:Entity)
        WHERE neighbor.id <> e.id
        RETURN DISTINCT neighbor
        """
        
        result = await self._execute_query(query, {"entity_id": entity_id})
        
        neighbors = []
        for record in result:
            try:
                entity = Entity.from_neo4j_node(record["neighbor"])
                neighbors.append(entity)
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
            source_entity_id: ID of the source entity
            target_entity_id: ID of the target entity
            max_depth: Maximum path length
            
        Returns:
            List of paths (each path is a list of entity IDs)
        """
        query = f"""
        MATCH path = (source:Entity {{id: $source_id}})
        -[*1..{max_depth}]-
        (target:Entity {{id: $target_id}})
        RETURN [node in nodes(path) | node.id] as path_ids
        LIMIT 10
        """
        
        result = await self._execute_query(query, {
            "source_id": source_entity_id,
            "target_id": target_entity_id
        })
        
        return [record["path_ids"] for record in result]
    
    async def store_graph(self, graph: Graph) -> None:
        """Store an entire graph.
        
        Args:
            graph: Graph to store
        """
        # Store entities first
        if graph.entities:
            await self.store_entities(list(graph.entities.values()))
        
        # Then store relations
        if graph.relations:
            await self.store_relations(list(graph.relations.values()))
    
    async def get_graph(
        self, 
        entity_ids: Optional[List[EntityId]] = None
    ) -> Graph:
        """Get a graph or subgraph.
        
        Args:
            entity_ids: Optional list of entity IDs to include (None for all)
            
        Returns:
            Graph containing the requested entities and their relations
        """
        graph = Graph()
        
        # Get entities
        if entity_ids:
            entities = await self.get_entities(entity_ids)
        else:
            # Get all entities
            query = "MATCH (e:Entity) RETURN e"
            result = await self._execute_query(query)
            entities = []
            for record in result:
                try:
                    entity = Entity.from_neo4j_node(record["e"])
                    entities.append(entity)
                except Exception as e:
                    logger.warning(f"Failed to parse entity: {e}")
        
        # Add entities to graph
        for entity in entities:
            graph.add_entity(entity)
        
        # Get relations between these entities
        if graph.entities:
            entity_id_list = list(graph.entities.keys())
            query = """
            MATCH (source:Entity)-[r]->(target:Entity)
            WHERE source.id IN $entity_ids AND target.id IN $entity_ids
            RETURN r, source.id as source_id, target.id as target_id
            """
            
            result = await self._execute_query(query, {"entity_ids": entity_id_list})
            
            for record in result:
                try:
                    relation = Relation.from_neo4j_relationship(
                        record["r"], 
                        record["source_id"], 
                        record["target_id"]
                    )
                    graph.add_relation(relation)
                except Exception as e:
                    logger.warning(f"Failed to parse relation: {e}")
        
        return graph
    
    async def clear(self) -> None:
        """Clear all data from the storage."""
        query = "MATCH (n) DETACH DELETE n"
        await self._execute_query(query)
        logger.info("Cleared all data from Neo4J database")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics.
        
        Returns:
            Dictionary containing statistics
        """
        queries = {
            "entity_count": "MATCH (e:Entity) RETURN count(e) as count",
            "relation_count": "MATCH ()-[r]->() RETURN count(r) as count",
            "entity_types": """
                MATCH (e:Entity) 
                RETURN e.type as type, count(e) as count 
                ORDER BY count DESC
            """,
            "relation_types": """
                MATCH ()-[r]->() 
                RETURN type(r) as type, count(r) as count 
                ORDER BY count DESC
            """
        }
        
        stats = {}
        
        for stat_name, query in queries.items():
            try:
                result = await self._execute_query(query)
                if stat_name in ["entity_count", "relation_count"]:
                    stats[stat_name] = result[0]["count"] if result else 0
                else:
                    stats[stat_name] = result
            except Exception as e:
                logger.warning(f"Failed to get {stat_name}: {e}")
                stats[stat_name] = 0 if "count" in stat_name else []
        
        return stats