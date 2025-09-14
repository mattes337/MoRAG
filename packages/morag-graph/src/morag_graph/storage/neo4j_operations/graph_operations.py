"""Graph-level operations for Neo4j storage."""

import logging
from typing import Dict, List, Optional, Any

from ...models import Graph, Entity, Relation
from ...models.types import EntityId
from .base_operations import BaseOperations

logger = logging.getLogger(__name__)


class GraphOperations(BaseOperations):
    """Handles graph-level operations and statistics."""
    
    async def store_graph(self, graph: Graph) -> None:
        """Store an entire graph.
        
        Args:
            graph: Graph to store
        """
        # First store entities
        if graph.entities:
            # We need to import EntityOperations to store entities
            # This creates a circular import, so we'll use direct queries
            for entity in graph.entities.values():
                await self._store_entity_direct(entity)
        
        # Then store relations
        if graph.relations:
            # Similarly for relations
            for relation in graph.relations.values():
                await self._store_relation_direct(relation)
    
    async def _store_entity_direct(self, entity: Entity) -> str:
        """Store entity directly without using EntityOperations."""
        # Get the normalized Neo4j label from the entity type
        neo4j_label = entity.get_neo4j_label()

        # Create normalized name for case-insensitive deduplication
        normalized_name = entity.name.lower().strip()

        # Create dynamic query with case-insensitive normalized name for deduplication
        query = f"""
        MERGE (e:{neo4j_label} {{normalized_name: $normalized_name}})
        ON CREATE SET
            e.id = $id,
            e.name = $name,
            e.type = $type,
            e.confidence = $confidence,
            e.metadata = $metadata,
            e.created_at = datetime(),
            e.updated_at = datetime()
        ON MATCH SET
            e.name = CASE
                WHEN $confidence > coalesce(e.confidence, 0.0) THEN $name
                ELSE e.name
            END,
            e.type = CASE
                WHEN $confidence > coalesce(e.confidence, 0.0) THEN $type
                ELSE e.type
            END,
            e.confidence = CASE
                WHEN $confidence > coalesce(e.confidence, 0.0) THEN $confidence
                ELSE e.confidence
            END,
            e.metadata = $metadata,
            e.updated_at = datetime()
        RETURN e.id as id
        """

        properties = entity.metadata.copy() if entity.metadata else {}

        result = await self._execute_query(query, {
            "id": entity.id,
            "name": entity.name,
            "normalized_name": normalized_name,
            "type": entity.type,
            "confidence": entity.confidence,
            "metadata": properties
        })

        return result[0]["id"] if result else entity.id
    
    async def _store_relation_direct(self, relation: Relation) -> str:
        """Store relation directly without using RelationOperations."""
        # Get the normalized Neo4j relationship type
        neo4j_rel_type = relation.get_neo4j_type()

        # Store relation with dynamic relationship type
        # Note: We need to match entities by any label since they now have specific labels
        query = f"""
        MATCH (source {{id: $source_id}}), (target {{id: $target_id}})
        MERGE (source)-[r:{neo4j_rel_type} {{
            id: $relation_id
        }}]->(target)
        SET r.type = $relation_type,
            r.confidence = $confidence,
            r.metadata = $metadata,
            r.created_at = coalesce(r.created_at, datetime()),
            r.updated_at = datetime()
        RETURN r.id as id
        """

        properties = relation.metadata.copy() if relation.metadata else {}

        result = await self._execute_query(query, {
            "source_id": relation.source_entity_id,
            "target_id": relation.target_entity_id,
            "relation_id": relation.id,
            "relation_type": relation.type,
            "confidence": relation.confidence,
            "metadata": properties
        })

        return result[0]["id"] if result else relation.id
    
    async def get_graph(
        self, 
        entity_ids: Optional[List[EntityId]] = None
    ) -> Graph:
        """Get a graph or subgraph.
        
        Args:
            entity_ids: Optional list of entity IDs to include (if None, gets all)
            
        Returns:
            Graph containing the requested entities and their relations
        """
        graph = Graph()
        
        # Get entities
        if entity_ids:
            entity_query = """
            MATCH (e:Entity)
            WHERE e.id IN $entity_ids
            RETURN e
            """
            entity_result = await self._execute_query(entity_query, {"entity_ids": entity_ids})
        else:
            entity_query = "MATCH (e:Entity) RETURN e"
            entity_result = await self._execute_query(entity_query)
        
        # Parse entities
        for record in entity_result:
            try:
                entity = Entity.from_neo4j_node(record["e"])
                graph.entities[entity.id] = entity
            except Exception as e:
                logger.warning(f"Failed to parse entity: {e}")
        
        # Get relations between the entities
        if graph.entities:
            entity_id_list = list(graph.entities.keys())
            relation_query = """
            MATCH (source:Entity)-[r:RELATION]->(target:Entity)
            WHERE source.id IN $entity_ids AND target.id IN $entity_ids
            RETURN r, source.id as source_id, target.id as target_id
            """
            relation_result = await self._execute_query(relation_query, {"entity_ids": entity_id_list})
            
            # Parse relations
            for record in relation_result:
                try:
                    rel_data = record["r"]
                    relation = Relation(
                        id=rel_data["id"],
                        type=rel_data["type"],
                        source_entity_id=record["source_id"],
                        target_entity_id=record["target_id"],
                        confidence=rel_data.get("confidence", 1.0),
                        metadata=rel_data.get("metadata", {})
                    )
                    graph.relations[relation.id] = relation
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
        stats = {}
        
        # Define all statistics queries
        stat_queries = {
            "entity_count": "MATCH (e:Entity) RETURN count(e) as count",
            "relation_count": "MATCH ()-[r:RELATION]->() RETURN count(r) as count",
            "document_count": "MATCH (d:Document) RETURN count(d) as count",
            "chunk_count": "MATCH (c:DocumentChunk) RETURN count(c) as count",
            "entity_types": """
                MATCH (e:Entity) 
                RETURN e.type as type, count(*) as count 
                ORDER BY count DESC
            """,
            "relation_types": """
                MATCH ()-[r:RELATION]->() 
                RETURN r.type as type, count(*) as count 
                ORDER BY count DESC
            """,
            "avg_entity_connections": """
                MATCH (e:Entity)
                OPTIONAL MATCH (e)-[r:RELATION]-()
                WITH e, count(r) as connections
                RETURN avg(connections) as avg_connections
            """,
            "most_connected_entities": """
                MATCH (e:Entity)-[r:RELATION]-()
                WITH e, count(r) as connections
                ORDER BY connections DESC
                LIMIT 10
                RETURN e.id as entity_id, e.name as name, connections
            """
        }
        
        # Execute each query and collect results
        for stat_name, query in stat_queries.items():
            try:
                result = await self._execute_query(query)
                
                if stat_name in ["entity_count", "relation_count", "document_count", "chunk_count"]:
                    stats[stat_name] = result[0]["count"] if result else 0
                elif stat_name == "avg_entity_connections":
                    stats[stat_name] = float(result[0]["avg_connections"]) if result and result[0]["avg_connections"] else 0.0
                elif stat_name in ["entity_types", "relation_types"]:
                    stats[stat_name] = [{"type": r["type"], "count": r["count"]} for r in result]
                elif stat_name == "most_connected_entities":
                    stats[stat_name] = [
                        {
                            "entity_id": r["entity_id"],
                            "name": r["name"],
                            "connections": r["connections"]
                        } for r in result
                    ]
                    
            except Exception as e:
                logger.warning(f"Failed to get {stat_name}: {e}")
                stats[stat_name] = 0 if "count" in stat_name else {}
        
        return stats
    
    async def get_graph_metrics(self) -> Dict[str, Any]:
        """Get advanced graph metrics.
        
        Returns:
            Dictionary containing graph metrics
        """
        metrics = {}
        
        try:
            # Graph density
            entity_count_query = "MATCH (e:Entity) RETURN count(e) as count"
            relation_count_query = "MATCH ()-[r:RELATION]->() RETURN count(r) as count"
            
            entity_result = await self._execute_query(entity_count_query)
            relation_result = await self._execute_query(relation_count_query)
            
            entity_count = entity_result[0]["count"] if entity_result else 0
            relation_count = relation_result[0]["count"] if relation_result else 0
            
            if entity_count > 1:
                max_possible_relations = entity_count * (entity_count - 1)
                density = relation_count / max_possible_relations
                metrics["graph_density"] = density
            else:
                metrics["graph_density"] = 0.0
            
            # Average clustering coefficient (simplified)
            clustering_query = """
                MATCH (e:Entity)-[:RELATION]-(neighbor:Entity)
                WITH e, collect(neighbor) as neighbors
                WHERE size(neighbors) > 1
                UNWIND neighbors as n1
                UNWIND neighbors as n2
                WITH e, n1, n2
                WHERE n1 <> n2
                OPTIONAL MATCH (n1)-[:RELATION]-(n2)
                WITH e, count(DISTINCT [n1, n2]) as possible_connections, 
                     count(DISTINCT CASE WHEN (n1)-[:RELATION]-(n2) THEN [n1, n2] END) as actual_connections
                WHERE possible_connections > 0
                RETURN avg(toFloat(actual_connections) / possible_connections) as avg_clustering
            """
            
            clustering_result = await self._execute_query(clustering_query)
            metrics["avg_clustering_coefficient"] = float(clustering_result[0]["avg_clustering"]) if clustering_result and clustering_result[0]["avg_clustering"] else 0.0
            
            # Connected components count (simplified)
            components_query = """
                MATCH (e:Entity)
                WHERE NOT (e)-[:RELATION]-()
                RETURN count(e) as isolated_entities
            """
            
            components_result = await self._execute_query(components_query)
            metrics["isolated_entities"] = components_result[0]["isolated_entities"] if components_result else 0
            
        except Exception as e:
            logger.warning(f"Failed to calculate graph metrics: {e}")
            metrics = {
                "graph_density": 0.0,
                "avg_clustering_coefficient": 0.0,
                "isolated_entities": 0
            }
        
        return metrics
    
    async def optimize_database(self) -> Dict[str, Any]:
        """Optimize database performance by creating indexes and constraints.
        
        Returns:
            Dictionary with optimization results
        """
        optimization_results = {}
        
        # Define indexes and constraints to create
        optimizations = [
            {
                "name": "entity_id_index",
                "query": "CREATE INDEX entity_id_index IF NOT EXISTS FOR (e:Entity) ON (e.id)"
            },
            {
                "name": "entity_name_index", 
                "query": "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)"
            },
            {
                "name": "relation_id_index",
                "query": "CREATE INDEX relation_id_index IF NOT EXISTS FOR ()-[r:RELATION]-() ON (r.id)"
            },
            {
                "name": "document_id_index",
                "query": "CREATE INDEX document_id_index IF NOT EXISTS FOR (d:Document) ON (d.id)"
            },
            {
                "name": "chunk_id_index",
                "query": "CREATE INDEX chunk_id_index IF NOT EXISTS FOR (c:DocumentChunk) ON (c.id)"
            },
            {
                "name": "chunk_document_id_index",
                "query": "CREATE INDEX chunk_document_id_index IF NOT EXISTS FOR (c:DocumentChunk) ON (c.document_id)"
            }
        ]
        
        for optimization in optimizations:
            try:
                await self._execute_query(optimization["query"])
                optimization_results[optimization["name"]] = "created"
                logger.info(f"Created {optimization['name']}")
            except Exception as e:
                optimization_results[optimization["name"]] = f"failed: {str(e)}"
                logger.warning(f"Failed to create {optimization['name']}: {e}")
        
        return optimization_results
