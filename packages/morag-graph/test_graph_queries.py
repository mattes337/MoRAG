#!/usr/bin/env python3
"""Test script for graph queries and traversal in Neo4j.

This script demonstrates various graph query capabilities including:
- Entity retrieval
- Relation traversal
- Multi-hop queries
- Graph statistics
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from morag_graph.models import Entity, Relation
from morag_graph.storage import Neo4jStorage
from morag_graph.storage.neo4j_storage import Neo4jConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def find_entities_by_type(storage: Neo4jStorage, entity_type: str, limit: int = 10) -> List[Entity]:
    """Find entities by type.
    
    Args:
        storage: Neo4j storage instance
        entity_type: Type of entities to find
        limit: Maximum number of entities to return
        
    Returns:
        List of entities
    """
    query = """
    MATCH (e:Entity)
    WHERE e.type = $entity_type
    RETURN e
    ORDER BY e.confidence DESC
    LIMIT $limit
    """
    
    result = await storage._execute_query(query, {
        "entity_type": entity_type,
        "limit": limit
    })
    
    entities = []
    for record in result:
        try:
            entity = Entity.from_neo4j_node(record["e"])
            entities.append(entity)
        except Exception as e:
            logger.warning(f"Failed to parse entity: {e}")
    
    return entities


async def find_related_entities(storage: Neo4jStorage, entity_name: str, max_hops: int = 2) -> Dict[str, Any]:
    """Find entities related to a given entity.
    
    Args:
        storage: Neo4j storage instance
        entity_name: Name of the source entity
        max_hops: Maximum number of hops to traverse
        
    Returns:
        Dictionary containing related entities and paths
    """
    query = f"""
    MATCH (source:Entity {{name: $entity_name}})
    MATCH path = (source)-[*1..{max_hops}]-(related:Entity)
    WHERE source <> related
    RETURN 
        related,
        length(path) as distance,
        [rel in relationships(path) | type(rel)] as relation_types
    ORDER BY distance, related.confidence DESC
    LIMIT 20
    """
    
    result = await storage._execute_query(query, {"entity_name": entity_name})
    
    related_entities = []
    for record in result:
        try:
            entity = Entity.from_neo4j_node(record["related"])
            related_entities.append({
                "entity": entity,
                "distance": record["distance"],
                "relation_types": record["relation_types"]
            })
        except Exception as e:
            logger.warning(f"Failed to parse related entity: {e}")
    
    return {
        "source_entity": entity_name,
        "related_entities": related_entities,
        "total_found": len(related_entities)
    }


async def find_shortest_path(storage: Neo4jStorage, entity1: str, entity2: str) -> Dict[str, Any]:
    """Find shortest path between two entities.
    
    Args:
        storage: Neo4j storage instance
        entity1: Name of first entity
        entity2: Name of second entity
        
    Returns:
        Dictionary containing path information
    """
    query = """
    MATCH (e1:Entity {name: $entity1}), (e2:Entity {name: $entity2})
    MATCH path = shortestPath((e1)-[*]-(e2))
    RETURN 
        path,
        length(path) as distance,
        [node in nodes(path) | node.name] as entity_names,
        [rel in relationships(path) | type(rel)] as relation_types
    """
    
    result = await storage._execute_query(query, {
        "entity1": entity1,
        "entity2": entity2
    })
    
    if not result:
        return {
            "entity1": entity1,
            "entity2": entity2,
            "path_found": False,
            "distance": None,
            "path": []
        }
    
    record = result[0]
    return {
        "entity1": entity1,
        "entity2": entity2,
        "path_found": True,
        "distance": record["distance"],
        "entity_names": record["entity_names"],
        "relation_types": record["relation_types"]
    }


async def find_central_entities(storage: Neo4jStorage, limit: int = 10) -> List[Dict[str, Any]]:
    """Find most connected entities (highest degree centrality).
    
    Args:
        storage: Neo4j storage instance
        limit: Maximum number of entities to return
        
    Returns:
        List of entities with their connection counts
    """
    query = """
    MATCH (e:Entity)
    OPTIONAL MATCH (e)-[r]-()
    WITH e, count(r) as connections
    WHERE connections > 0
    RETURN e, connections
    ORDER BY connections DESC
    LIMIT $limit
    """
    
    result = await storage._execute_query(query, {"limit": limit})
    
    central_entities = []
    for record in result:
        try:
            entity = Entity.from_neo4j_node(record["e"])
            central_entities.append({
                "entity": entity,
                "connections": record["connections"]
            })
        except Exception as e:
            logger.warning(f"Failed to parse central entity: {e}")
    
    return central_entities


async def find_entity_clusters(storage: Neo4jStorage, entity_type: str = None) -> List[Dict[str, Any]]:
    """Find clusters of connected entities.
    
    Args:
        storage: Neo4j storage instance
        entity_type: Optional entity type filter
        
    Returns:
        List of entity clusters
    """
    type_filter = "WHERE e.type = $entity_type" if entity_type else ""
    
    query = f"""
    MATCH (e:Entity)
    {type_filter}
    MATCH path = (e)-[*1..3]-(connected:Entity)
    WITH e, collect(DISTINCT connected) as cluster
    WHERE size(cluster) >= 2
    RETURN e.name as center_entity, e.type as center_type, 
           [entity in cluster | {{name: entity.name, type: entity.type}}] as connected_entities,
           size(cluster) as cluster_size
    ORDER BY cluster_size DESC
    LIMIT 10
    """
    
    params = {"entity_type": entity_type} if entity_type else {}
    result = await storage._execute_query(query, params)
    
    clusters = []
    for record in result:
        clusters.append({
            "center_entity": record["center_entity"],
            "center_type": record["center_type"],
            "connected_entities": record["connected_entities"],
            "cluster_size": record["cluster_size"]
        })
    
    return clusters


async def semantic_search_entities(storage: Neo4jStorage, search_term: str, limit: int = 10) -> List[Entity]:
    """Search entities by name or context (simple text matching).
    
    Args:
        storage: Neo4j storage instance
        search_term: Term to search for
        limit: Maximum number of entities to return
        
    Returns:
        List of matching entities
    """
    query = """
    MATCH (e:Entity)
    WHERE toLower(e.name) CONTAINS toLower($search_term)
       OR toLower(e.source_text) CONTAINS toLower($search_term)
    RETURN e
    ORDER BY e.confidence DESC
    LIMIT $limit
    """
    
    result = await storage._execute_query(query, {
        "search_term": search_term,
        "limit": limit
    })
    
    entities = []
    for record in result:
        try:
            entity = Entity.from_neo4j_node(record["e"])
            entities.append(entity)
        except Exception as e:
            logger.warning(f"Failed to parse entity: {e}")
    
    return entities


async def main():
    """Main function to test graph queries."""
    logger.info("🧠 MoRAG Graph - Query and Traversal Test")
    logger.info("=" * 60)
    
    # Configuration
    config = Neo4jConfig(
        uri="neo4j://localhost:7687",
        username="neo4j",
        password="morag_password",
        database="neo4j"
    )
    
    # Initialize storage
    storage = Neo4jStorage(config)
    
    try:
        # Connect to Neo4j
        await storage.connect()
        logger.info("✅ Connected to Neo4j")
        
        # Test 1: Find entities by type
        logger.info("\n📋 Test 1: Find PERSON entities")
        persons = await find_entities_by_type(storage, "PERSON", 5)
        for person in persons:
            logger.info(f"   • {person.name} (confidence: {person.confidence})")
        
        # Test 2: Find related entities
        if persons:
            test_person = persons[0].name
            logger.info(f"\n🔗 Test 2: Find entities related to '{test_person}'")
            related = await find_related_entities(storage, test_person, max_hops=2)
            logger.info(f"   Found {related['total_found']} related entities:")
            for rel in related['related_entities'][:5]:  # Show first 5
                entity = rel['entity']
                distance = rel['distance']
                rel_types = rel['relation_types']
                logger.info(f"   • {entity.name} ({entity.type}) - distance: {distance}, via: {rel_types}")
        
        # Test 3: Find shortest path
        if len(persons) >= 2:
            entity1, entity2 = persons[0].name, persons[1].name
            logger.info(f"\n🛤️  Test 3: Shortest path between '{entity1}' and '{entity2}'")
            path = await find_shortest_path(storage, entity1, entity2)
            if path['path_found']:
                logger.info(f"   Path found! Distance: {path['distance']}")
                logger.info(f"   Entities: {' -> '.join(path['entity_names'])}")
                logger.info(f"   Relations: {path['relation_types']}")
            else:
                logger.info("   No path found between these entities")
        
        # Test 4: Find central entities
        logger.info("\n⭐ Test 4: Most connected entities")
        central = await find_central_entities(storage, 5)
        for item in central:
            entity = item['entity']
            connections = item['connections']
            logger.info(f"   • {entity.name} ({entity.type}) - {connections} connections")
        
        # Test 5: Find entity clusters
        logger.info("\n🔗 Test 5: Entity clusters")
        clusters = await find_entity_clusters(storage)
        for cluster in clusters[:3]:  # Show first 3 clusters
            center = cluster['center_entity']
            size = cluster['cluster_size']
            logger.info(f"   • Cluster around '{center}' ({size} entities)")
            for connected in cluster['connected_entities'][:3]:  # Show first 3 connected
                logger.info(f"     - {connected['name']} ({connected['type']})")
        
        # Test 6: Semantic search
        logger.info("\n🔍 Test 6: Search for 'virus' related entities")
        virus_entities = await semantic_search_entities(storage, "virus", 5)
        for entity in virus_entities:
            logger.info(f"   • {entity.name} ({entity.type}) - confidence: {entity.confidence}")
        
        # Test 7: Graph statistics
        logger.info("\n📊 Test 7: Graph statistics")
        stats = await storage.get_statistics()
        logger.info(f"   • Total entities: {stats.get('entity_count', 0)}")
        logger.info(f"   • Total relations: {stats.get('relation_count', 0)}")
        logger.info("   • Entity types:")
        for type_info in stats.get('entity_types', [])[:5]:  # Show top 5 types
            logger.info(f"     - {type_info['type']}: {type_info['count']}")
        
        logger.info("\n✅ Graph query tests completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Graph query test failed: {e}")
        raise
    finally:
        # Cleanup
        await storage.disconnect()
        logger.info("Disconnected from Neo4j")


if __name__ == "__main__":
    asyncio.run(main())