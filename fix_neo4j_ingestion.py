#!/usr/bin/env python3
"""Fix Neo4j ingestion issues: auto-created entity connections and clustering."""

import asyncio
import os
import sys
import json
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent / "packages" / "morag-graph" / "src"))

from morag_graph.storage.neo4j_storage import Neo4jStorage, Neo4jConfig


async def fix_neo4j_ingestion():
    """Fix the Neo4j ingestion issues."""
    
    # Connect to Neo4j
    config = Neo4jConfig(
        uri=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
        username=os.getenv('NEO4J_USERNAME', 'neo4j'),
        password=os.getenv('NEO4J_PASSWORD', 'morag_password'),
        database=os.getenv('NEO4J_DATABASE', 'neo4j')
    )
    
    storage = Neo4jStorage(config)
    await storage.connect()
    
    print("🔧 Fixing Neo4j Graph Ingestion Issues")
    print("=" * 50)
    
    # 1. Check attributes storage format (they should be JSON strings in Neo4j)
    print("\n1. Checking attributes storage format...")

    # Get sample entities with attributes
    entities_with_attrs = await storage._execute_query("""
        MATCH (e:Entity)
        WHERE e.attributes IS NOT NULL
        AND e.attributes <> ""
        RETURN e.id as entity_id, e.attributes as attrs
        LIMIT 3
    """)

    print(f"Sample entities with attributes:")
    for record in entities_with_attrs:
        entity_id = record['entity_id']
        attrs = record['attrs']
        print(f"  {entity_id}: {type(attrs)} - {attrs}")

    # 2. Connect auto-created entities to document chunks
    print("\n2. Connecting auto-created entities to document chunks...")

    # Find auto-created entities that are not connected to chunks
    # Since attributes are JSON strings, we need to use CONTAINS
    disconnected_auto_entities = await storage._execute_query("""
        MATCH (e:Entity)
        WHERE e.attributes CONTAINS 'auto_created'
        AND NOT EXISTS((e)-[:MENTIONS|MENTIONED_IN]-(:DocumentChunk))
        RETURN e.id as entity_id, e.name as entity_name
    """)
    
    print(f"Found {len(disconnected_auto_entities)} disconnected auto-created entities")
    
    # For each disconnected auto-created entity, find chunks that mention it
    connections_created = 0
    
    for record in disconnected_auto_entities:
        entity_id = record['entity_id']
        entity_name = record['entity_name']
        
        # Find chunks that contain this entity name (case-insensitive)
        matching_chunks = await storage._execute_query("""
            MATCH (c:DocumentChunk)
            WHERE toLower(c.text) CONTAINS toLower($entity_name)
            RETURN c.id as chunk_id, c.text as chunk_text
        """, {"entity_name": entity_name})
        
        # Create MENTIONS relationships
        for chunk_record in matching_chunks:
            chunk_id = chunk_record['chunk_id']
            chunk_text = chunk_record['chunk_text']
            
            # Create chunk -> MENTIONS -> entity relationship
            await storage._execute_query("""
                MATCH (c:DocumentChunk {id: $chunk_id})
                MATCH (e:Entity {id: $entity_id})
                MERGE (c)-[r:MENTIONS]->(e)
                SET r.context = $context
            """, {
                "chunk_id": chunk_id,
                "entity_id": entity_id,
                "context": f"Entity '{entity_name}' mentioned in chunk"
            })
            
            connections_created += 1
            print(f"  Connected entity '{entity_name}' to chunk {chunk_id}")
    
    print(f"Created {connections_created} new chunk-entity connections")
    
    # 3. Verify the fixes
    print("\n3. Verifying fixes...")
    
    # Check connected components again
    components_after = await storage._execute_query("""
        MATCH (n)
        WITH n
        CALL {
            WITH n
            MATCH path = (n)-[*0..5]-(connected)
            RETURN collect(DISTINCT elementId(connected)) as component_nodes
        }
        RETURN DISTINCT component_nodes, size(component_nodes) as component_size
        ORDER BY component_size DESC
        LIMIT 5
    """)
    
    print(f"Top 5 component sizes after fixes:")
    for i, record in enumerate(components_after):
        size = record['component_size']
        print(f"  Component {i+1}: {size} nodes")
    
    # Check auto-created entity connections
    connected_auto_entities = await storage._execute_query("""
        MATCH (e:Entity)-[r]-(c:DocumentChunk)
        WHERE e.attributes CONTAINS 'auto_created'
        RETURN count(DISTINCT e) as connected_count, count(r) as total_connections
    """)
    
    if connected_auto_entities:
        connected = connected_auto_entities[0]['connected_count']
        total_connections = connected_auto_entities[0]['total_connections']
        print(f"Auto-created entities now connected to chunks: {connected}")
        print(f"Total auto-created entity-chunk connections: {total_connections}")
    
    # 4. Check for remaining issues
    print("\n4. Checking for remaining issues...")
    
    # Check if there are still entities with placeholder IDs
    placeholder_entities = await storage._execute_query("""
        MATCH (e:Entity)
        WHERE e.id CONTAINS 'abc123'
        RETURN count(e) as placeholder_count
    """)
    
    if placeholder_entities:
        placeholder_count = placeholder_entities[0]['placeholder_count']
        print(f"Entities with placeholder IDs (abc123): {placeholder_count}")
        
        if placeholder_count > 0:
            print("  Warning: Some entities still have placeholder IDs")
            print("  This indicates the document ID generation needs to be fixed")
    
    await storage.disconnect()
    print("\n✅ Neo4j ingestion fixes completed!")


if __name__ == "__main__":
    asyncio.run(fix_neo4j_ingestion())
