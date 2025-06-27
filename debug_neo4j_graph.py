#!/usr/bin/env python3
"""Debug script to inspect Neo4j graph structure and identify clustering issues."""

import asyncio
import os
import sys
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent / "packages" / "morag-graph" / "src"))

from morag_graph.storage.neo4j_storage import Neo4jStorage, Neo4jConfig


async def debug_neo4j_graph():
    """Debug the Neo4j graph structure to identify clustering issues."""
    
    # Connect to Neo4j
    config = Neo4jConfig(
        uri=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
        username=os.getenv('NEO4J_USERNAME', 'neo4j'),
        password=os.getenv('NEO4J_PASSWORD', 'morag_password'),
        database=os.getenv('NEO4J_DATABASE', 'neo4j')
    )
    
    storage = Neo4jStorage(config)
    await storage.connect()
    
    print("🔍 Neo4j Graph Structure Analysis")
    print("=" * 50)
    
    # 1. Count all nodes and relationships
    print("\n📊 Overall Statistics:")
    
    # Count nodes by type
    node_counts = await storage._execute_query("""
        MATCH (n)
        RETURN labels(n) as labels, count(n) as count
        ORDER BY count DESC
    """)
    
    for record in node_counts:
        labels = record['labels']
        count = record['count']
        print(f"  {':'.join(labels)}: {count}")
    
    # Count relationships by type
    rel_counts = await storage._execute_query("""
        MATCH ()-[r]->()
        RETURN type(r) as rel_type, count(r) as count
        ORDER BY count DESC
    """)
    
    print("\n🔗 Relationship Types:")
    for record in rel_counts:
        print(f"  {record['rel_type']}: {record['count']}")
    
    # 2. Check for disconnected components
    print("\n🏝️ Connected Components Analysis:")
    
    # Find connected components
    components = await storage._execute_query("""
        MATCH (n)
        WITH n
        CALL {
            WITH n
            MATCH path = (n)-[*0..10]-(connected)
            RETURN collect(DISTINCT id(connected)) as component_nodes
        }
        RETURN DISTINCT component_nodes, size(component_nodes) as component_size
        ORDER BY component_size DESC
    """)
    
    print(f"  Total components found: {len(components)}")
    for i, record in enumerate(components[:10]):  # Show top 10 components
        size = record['component_size']
        print(f"  Component {i+1}: {size} nodes")
    
    # 3. Check auto_created entities and their connections
    print("\n🤖 Auto-Created Entities Analysis:")

    # First check the structure of attributes field
    attr_sample = await storage._execute_query("""
        MATCH (e:Entity)
        RETURN e.attributes as attrs
        LIMIT 3
    """)

    print("  Sample attributes structure:")
    for record in attr_sample:
        attrs = record['attrs']
        print(f"    Type: {type(attrs)}, Value: {attrs}")

    # Try different approaches to find auto-created entities
    auto_created_stats = await storage._execute_query("""
        MATCH (e:Entity)
        WHERE e.attributes CONTAINS 'auto_created'
        RETURN count(e) as auto_created_count
    """)

    auto_created_count = auto_created_stats[0]['auto_created_count'] if auto_created_stats else 0
    print(f"  Auto-created entities (string search): {auto_created_count}")

    # Check if auto-created entities are connected to chunks
    auto_created_chunk_connections = await storage._execute_query("""
        MATCH (e:Entity)-[r]-(c:DocumentChunk)
        WHERE e.attributes CONTAINS 'auto_created'
        RETURN count(DISTINCT e) as connected_auto_entities, count(r) as total_connections
    """)

    if auto_created_chunk_connections:
        connected = auto_created_chunk_connections[0]['connected_auto_entities']
        total_connections = auto_created_chunk_connections[0]['total_connections']
        print(f"  Auto-created entities connected to chunks: {connected}")
        print(f"  Total auto-created entity-chunk connections: {total_connections}")

        if auto_created_count > 0:
            disconnected = auto_created_count - connected
            print(f"  Disconnected auto-created entities: {disconnected}")
    
    # 4. Check document-chunk-entity relationship chain
    print("\n📄 Document-Chunk-Entity Chain Analysis:")
    
    # Count documents
    doc_count = await storage._execute_query("MATCH (d:Document) RETURN count(d) as count")
    print(f"  Documents: {doc_count[0]['count'] if doc_count else 0}")
    
    # Count chunks and their document connections
    chunk_stats = await storage._execute_query("""
        MATCH (d:Document)-[:CONTAINS]->(c:DocumentChunk)
        RETURN count(DISTINCT d) as docs_with_chunks, count(c) as total_chunks
    """)
    
    if chunk_stats:
        print(f"  Documents with chunks: {chunk_stats[0]['docs_with_chunks']}")
        print(f"  Total chunks: {chunk_stats[0]['total_chunks']}")
    
    # Count chunk-entity connections
    chunk_entity_stats = await storage._execute_query("""
        MATCH (c:DocumentChunk)-[:MENTIONS]->(e:Entity)
        RETURN count(DISTINCT c) as chunks_with_entities, 
               count(DISTINCT e) as entities_mentioned,
               count(*) as total_mentions
    """)
    
    if chunk_entity_stats:
        print(f"  Chunks with entity mentions: {chunk_entity_stats[0]['chunks_with_entities']}")
        print(f"  Entities mentioned in chunks: {chunk_entity_stats[0]['entities_mentioned']}")
        print(f"  Total chunk-entity mentions: {chunk_entity_stats[0]['total_mentions']}")
    
    # 5. Sample disconnected entities
    print("\n🔍 Sample Disconnected Auto-Created Entities:")

    disconnected_entities = await storage._execute_query("""
        MATCH (e:Entity)
        WHERE e.attributes CONTAINS 'auto_created'
        AND NOT EXISTS((e)-[:MENTIONS|MENTIONED_IN]-(:DocumentChunk))
        RETURN e.id, e.name, e.attributes
        LIMIT 5
    """)

    for record in disconnected_entities:
        print(f"  ID: {record['e.id']}")
        print(f"    Name: {record['e.name']}")
        print(f"    Attributes: {record['e.attributes']}")
        print()
    
    # 6. Check entity-entity relationships
    print("\n🔗 Entity-Entity Relationships:")
    
    entity_relations = await storage._execute_query("""
        MATCH (e1:Entity)-[r]->(e2:Entity)
        RETURN type(r) as relation_type, count(*) as count
        ORDER BY count DESC
    """)
    
    for record in entity_relations:
        print(f"  {record['relation_type']}: {record['count']}")
    
    await storage.disconnect()


if __name__ == "__main__":
    asyncio.run(debug_neo4j_graph())
