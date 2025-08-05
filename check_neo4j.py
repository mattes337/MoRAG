import asyncio
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), 'packages', 'morag-graph', 'src'))
from morag_graph.storage.neo4j_storage import Neo4jStorage, Neo4jConfig

async def check_neo4j():
    config = Neo4jConfig(
        uri=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
        username=os.getenv('NEO4J_USERNAME', 'neo4j'),
        password=os.getenv('NEO4J_PASSWORD', 'password'),
        database=os.getenv('NEO4J_DATABASE', 'neo4j'),
        verify_ssl=os.getenv("NEO4J_VERIFY_SSL", "false").lower() == "true",
        trust_all_certificates=os.getenv("NEO4J_TRUST_ALL_CERTIFICATES", "true").lower() == "true"
    )
    storage = Neo4jStorage(config)
    await storage.connect()
    
    try:
        # Check database schema
        print("=== Neo4j Database Schema ===")

        # Get all node labels
        query = "CALL db.labels()"
        result = await storage._connection_ops._execute_query(query)
        print(f"Node labels: {[r['label'] for r in result]}")

        # Get all relationship types
        query = "CALL db.relationshipTypes()"
        result = await storage._connection_ops._execute_query(query)
        print(f"Relationship types: {[r['relationshipType'] for r in result]}")

        # Get property keys
        query = "CALL db.propertyKeys()"
        result = await storage._connection_ops._execute_query(query)
        print(f"Property keys: {[r['propertyKey'] for r in result]}")

        # Check DocumentChunk structure
        print("\n=== DocumentChunk Structure ===")
        query = "MATCH (d:DocumentChunk) RETURN d LIMIT 1"
        result = await storage._connection_ops._execute_query(query)
        if result:
            chunk = result[0]['d']
            print(f"DocumentChunk properties: {list(chunk.keys())}")
            print(f"Sample chunk: {chunk}")
        else:
            print("No DocumentChunk nodes found")

        # Check Entity structure
        print("\n=== Entity Structure ===")
        query = "MATCH (e:Entity) RETURN e LIMIT 1"
        result = await storage._connection_ops._execute_query(query)
        if result:
            entity = result[0]['e']
            print(f"Entity properties: {list(entity.keys())}")
            print(f"Sample entity: {entity}")
        else:
            print("No Entity nodes found")

        # Check Fact structure
        print("\n=== Fact Structure ===")
        query = "MATCH (f:Fact) RETURN f LIMIT 1"
        result = await storage._connection_ops._execute_query(query)
        if result:
            fact = result[0]['f']
            print(f"Fact properties: {list(fact.keys())}")
            print(f"Sample fact: {fact}")
        else:
            print("No Fact nodes found")

        # Count nodes
        print("\n=== Node Counts ===")
        for label in ["DocumentChunk", "Entity", "Fact", "Document"]:
            query = f"MATCH (n:{label}) RETURN count(n) as count"
            result = await storage._connection_ops._execute_query(query)
            count = result[0]['count'] if result else 0
            print(f"{label}: {count}")

        # Check for German content
        print("\n=== German Content Check ===")
        query = "MATCH (e:Entity) WHERE e.name =~ '(?i).*(ADHS|Schwermetall|Blei|Quecksilber|Cadmium).*' RETURN e.name LIMIT 10"
        result = await storage._connection_ops._execute_query(query)
        if result:
            print("German entities found:")
            for record in result:
                print(f"  {record['e.name']}")
        else:
            print("No German entities found")

    finally:
        await storage.disconnect()

if __name__ == "__main__":
    asyncio.run(check_neo4j())
