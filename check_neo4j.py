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
    
    # Check entities
    query = 'MATCH (e:Entity) RETURN count(e) as entity_count'
    result = await storage._execute_query(query)
    print(f'Total entities: {result[0]["entity_count"]}')
    
    # Check relations
    query = 'MATCH ()-[r]->() RETURN type(r) as rel_type, count(r) as count ORDER BY count DESC'
    result = await storage._execute_query(query)
    print('Relation types:')
    for record in result:
        print(f'  {record["rel_type"]}: {record["count"]}')
    
    # Check specific entity IDs
    query = 'MATCH (e:Entity) RETURN e.id LIMIT 10'
    result = await storage._execute_query(query)
    print('Sample entity IDs:')
    for record in result:
        print(f'  {record["e.id"]}')
    
    await storage.disconnect()

if __name__ == "__main__":
    asyncio.run(check_neo4j())
