#!/usr/bin/env python3
"""
Database and Collection Creation Utility for MoRAG

This script helps create Neo4j databases and Qdrant collections before running ingestion.
Useful for Neo4j Community Edition which doesn't support automatic database creation.

Usage:
    python create-databases.py --neo4j-database smartcard --qdrant-collection smartcard_docs
    python create-databases.py --neo4j-database test_db
    python create-databases.py --qdrant-collection test_collection
    python create-databases.py --list-existing
"""

import sys
import os
import asyncio
import argparse
from pathlib import Path
from typing import Optional

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
env_path = project_root / '.env'
load_dotenv(env_path, override=True)

def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'-'*40}")
    print(f"  {title}")
    print(f"{'-'*40}")

def print_result(key: str, value: str, indent: int = 0):
    """Print a formatted key-value result."""
    spaces = "  " * indent
    print(f"{spaces}📋 {key}: {value}")

async def create_neo4j_database(database_name: str) -> bool:
    """Create a Neo4j database."""
    try:
        from morag_graph.storage.neo4j_storage import Neo4jStorage, Neo4jConfig
        
        config = Neo4jConfig(
            uri=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
            username=os.getenv('NEO4J_USERNAME', 'neo4j'),
            password=os.getenv('NEO4J_PASSWORD', 'password'),
            database='system'  # Connect to system database to create others
        )
        
        storage = Neo4jStorage(config)
        await storage.connect()
        
        # Use the utility method to create the database
        success = await storage.create_database_if_not_exists(database_name)
        
        await storage.disconnect()
        return success
        
    except Exception as e:
        print(f"❌ Error creating Neo4j database: {e}")
        return False

async def create_qdrant_collection(collection_name: str, vector_size: int = 768) -> bool:
    """Create a Qdrant collection."""
    try:
        from morag_services.storage import QdrantVectorStorage
        
        storage = QdrantVectorStorage(
            host=os.getenv('QDRANT_HOST', 'localhost'),
            port=int(os.getenv('QDRANT_PORT', '6333')),
            api_key=os.getenv('QDRANT_API_KEY'),
            collection_name=collection_name
        )
        
        await storage.connect()
        await storage.create_collection(collection_name, vector_size, force_recreate=False)
        await storage.disconnect()
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating Qdrant collection: {e}")
        return False

async def list_existing_databases() -> None:
    """List existing Neo4j databases and Qdrant collections."""
    print_section("Existing Neo4j Databases")
    
    try:
        from morag_graph.storage.neo4j_storage import Neo4jStorage, Neo4jConfig
        
        config = Neo4jConfig(
            uri=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
            username=os.getenv('NEO4J_USERNAME', 'neo4j'),
            password=os.getenv('NEO4J_PASSWORD', 'password'),
            database='system'
        )
        
        storage = Neo4jStorage(config)
        await storage.connect()
        
        # List databases
        result = await storage._execute_query("SHOW DATABASES YIELD name", {})
        databases = [row['name'] for row in result]
        
        for db in databases:
            print_result("Database", db)
            
        await storage.disconnect()
        
    except Exception as e:
        print(f"❌ Error listing Neo4j databases: {e}")
    
    print_section("Existing Qdrant Collections")
    
    try:
        from qdrant_client import QdrantClient
        
        client = QdrantClient(
            host=os.getenv('QDRANT_HOST', 'localhost'),
            port=int(os.getenv('QDRANT_PORT', '6333')),
            api_key=os.getenv('QDRANT_API_KEY')
        )
        
        collections = client.get_collections()
        for col in collections.collections:
            print_result("Collection", col.name)
            
        client.close()
        
    except Exception as e:
        print(f"❌ Error listing Qdrant collections: {e}")

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Create Neo4j databases and Qdrant collections for MoRAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Create both database and collection:
    python create-databases.py --neo4j-database smartcard --qdrant-collection smartcard_docs
    
  Create only Neo4j database:
    python create-databases.py --neo4j-database test_db
    
  Create only Qdrant collection:
    python create-databases.py --qdrant-collection test_collection
    
  List existing databases and collections:
    python create-databases.py --list-existing

Environment Variables:
  NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
  QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY
        """
    )
    
    parser.add_argument('--neo4j-database', help='Neo4j database name to create')
    parser.add_argument('--qdrant-collection', help='Qdrant collection name to create')
    parser.add_argument('--vector-size', type=int, default=768, help='Vector size for Qdrant collection (default: 768)')
    parser.add_argument('--list-existing', action='store_true', help='List existing databases and collections')
    
    args = parser.parse_args()
    
    if not any([args.neo4j_database, args.qdrant_collection, args.list_existing]):
        parser.print_help()
        return 1
    
    print_header("MoRAG Database and Collection Creation Utility")
    
    if args.list_existing:
        await list_existing_databases()
        return 0
    
    success = True
    
    if args.neo4j_database:
        print_section(f"Creating Neo4j Database: {args.neo4j_database}")
        if await create_neo4j_database(args.neo4j_database):
            print(f"✅ Neo4j database '{args.neo4j_database}' created successfully!")
        else:
            print(f"❌ Failed to create Neo4j database '{args.neo4j_database}'")
            success = False
    
    if args.qdrant_collection:
        print_section(f"Creating Qdrant Collection: {args.qdrant_collection}")
        if await create_qdrant_collection(args.qdrant_collection, args.vector_size):
            print(f"✅ Qdrant collection '{args.qdrant_collection}' created successfully!")
        else:
            print(f"❌ Failed to create Qdrant collection '{args.qdrant_collection}'")
            success = False
    
    if success:
        print("\n🎉 All databases and collections created successfully!")
        return 0
    else:
        print("\n💥 Some operations failed!")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
