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
        from neo4j import AsyncGraphDatabase

        # Connect directly to system database without using the storage class connect method
        uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        username = os.getenv('NEO4J_USERNAME', 'neo4j')
        password = os.getenv('NEO4J_PASSWORD', 'password')

        driver = AsyncGraphDatabase.driver(uri, auth=(username, password))

        try:
            # Try to create database using system database
            async with driver.session(database="system") as session:
                # Check if database exists
                result = await session.run(
                    "SHOW DATABASES YIELD name WHERE name = $db_name",
                    {"db_name": database_name}
                )
                databases = await result.data()

                if not databases:
                    # Database doesn't exist, create it
                    print(f"📋 Creating Neo4j database: {database_name}")
                    await session.run(f"CREATE DATABASE `{database_name}`")
                    print(f"✅ Successfully created Neo4j database: {database_name}")
                    return True
                else:
                    print(f"📋 Neo4j database already exists: {database_name}")
                    return True

        except Exception as e:
            # If we can't create via system database (Community Edition), try direct connection
            print(f"⚠️ Cannot create database via system database (likely Neo4j Community Edition): {e}")
            print(f"📋 Attempting to verify database '{database_name}' exists...")

            try:
                async with driver.session(database=database_name) as session:
                    await session.run("RETURN 1")
                print(f"✅ Database '{database_name}' exists and is accessible")
                return True
            except Exception as direct_error:
                if "DatabaseNotFound" in str(direct_error):
                    print(f"❌ Database '{database_name}' does not exist and cannot be created automatically.")
                    print(f"💡 For Neo4j Community Edition, please:")
                    print(f"   1. Use the default 'neo4j' database, or")
                    print(f"   2. Create the database manually, or")
                    print(f"   3. Upgrade to Neo4j Enterprise Edition")
                    return False
                else:
                    print(f"❌ Error connecting to database '{database_name}': {direct_error}")
                    return False
        finally:
            await driver.close()

    except Exception as e:
        print(f"❌ Error creating Neo4j database: {e}")
        return False

async def create_qdrant_collection(collection_name: str, vector_size: int = 768) -> bool:
    """Create a Qdrant collection."""
    try:
        from morag_services.storage import QdrantVectorStorage

        # Prefer QDRANT_URL if available, otherwise use QDRANT_HOST/PORT
        qdrant_url = os.getenv('QDRANT_URL')
        qdrant_api_key = os.getenv('QDRANT_API_KEY')

        if qdrant_url:
            # Use URL-based connection (supports HTTPS automatically)
            storage = QdrantVectorStorage(
                host=qdrant_url,
                api_key=qdrant_api_key,
                collection_name=collection_name
            )
        else:
            # Fall back to host/port connection
            storage = QdrantVectorStorage(
                host=os.getenv('QDRANT_HOST', 'localhost'),
                port=int(os.getenv('QDRANT_PORT', '6333')),
                api_key=qdrant_api_key,
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
        from neo4j import AsyncGraphDatabase

        uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        username = os.getenv('NEO4J_USERNAME', 'neo4j')
        password = os.getenv('NEO4J_PASSWORD', 'password')

        driver = AsyncGraphDatabase.driver(uri, auth=(username, password))

        try:
            # Try to list databases using system database
            async with driver.session(database="system") as session:
                result = await session.run("SHOW DATABASES YIELD name")
                databases = []
                async for record in result:
                    databases.append(record["name"])

                for db in databases:
                    print_result("Database", db)

        except Exception as e:
            print(f"⚠️ Cannot list databases via system database (likely Neo4j Community Edition): {e}")
            print(f"📋 Attempting to connect to default 'neo4j' database...")

            try:
                async with driver.session(database="neo4j") as session:
                    await session.run("RETURN 1")
                print_result("Database", "neo4j (default)")
            except Exception as direct_error:
                print(f"❌ Error connecting to default database: {direct_error}")
        finally:
            await driver.close()

    except Exception as e:
        print(f"❌ Error listing Neo4j databases: {e}")
    
    print_section("Existing Qdrant Collections")
    
    try:
        from qdrant_client import QdrantClient

        # Use the same logic as the storage classes for HTTPS support
        qdrant_url = os.getenv('QDRANT_URL')
        qdrant_api_key = os.getenv('QDRANT_API_KEY')

        if qdrant_url:
            # Parse URL for connection
            from urllib.parse import urlparse
            parsed = urlparse(qdrant_url)
            hostname = parsed.hostname or "localhost"
            port = parsed.port or (443 if parsed.scheme == 'https' else 6333)
            use_https = parsed.scheme == 'https'

            client = QdrantClient(
                host=hostname,
                port=port,
                https=use_https,
                api_key=qdrant_api_key
            )
        else:
            # Fall back to host/port
            qdrant_host = os.getenv('QDRANT_HOST', 'localhost')
            qdrant_port = int(os.getenv('QDRANT_PORT', '6333'))
            use_https = qdrant_port == 443

            client = QdrantClient(
                host=qdrant_host,
                port=qdrant_port,
                https=use_https,
                api_key=qdrant_api_key
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

Note:
  Neo4j Community Edition only supports the default 'neo4j' database.
  To create custom databases, you need Neo4j Enterprise Edition.
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
