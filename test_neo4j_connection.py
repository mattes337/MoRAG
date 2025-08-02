#!/usr/bin/env python3
"""
Neo4j Connection Test Script
Tests connection to Neo4j database with proper SSL configuration.
"""

import asyncio
import sys
import os
import traceback
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the morag-graph package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'packages', 'morag-graph', 'src'))

from morag_graph.storage.neo4j_storage import Neo4jStorage, Neo4jConfig


async def test_neo4j_connection():
    """Test Neo4j connection with detailed diagnostics."""
    
    print("🔍 Neo4j Connection Test")
    print("=" * 50)
    
    # Read configuration from environment
    uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    username = os.getenv('NEO4J_USERNAME', 'neo4j')
    password = os.getenv('NEO4J_PASSWORD', 'password')
    database = os.getenv('NEO4J_DATABASE', 'neo4j')
    verify_ssl = os.getenv("NEO4J_VERIFY_SSL", "true").lower() == "true"
    trust_all_certificates = os.getenv("NEO4J_TRUST_ALL_CERTIFICATES", "false").lower() == "true"
    
    print(f"📋 Configuration:")
    print(f"   URI: {uri}")
    print(f"   Username: {username}")
    print(f"   Database: {database}")
    print(f"   Verify SSL: {verify_ssl}")
    print(f"   Trust All Certificates: {trust_all_certificates}")
    print()
    
    # Create Neo4j configuration
    config = Neo4jConfig(
        uri=uri,
        username=username,
        password=password,
        database=database,
        verify_ssl=verify_ssl,
        trust_all_certificates=trust_all_certificates
    )
    
    storage = Neo4jStorage(config)
    
    try:
        print("🔌 Attempting to connect...")
        await storage.connect()
        print("✅ Successfully connected to Neo4j!")
        
        # Test basic query using the query operations
        print("\n🔍 Testing basic query...")
        async with storage.driver.session(database=storage.config.database) as session:
            result = await session.run("RETURN 1 as test")
            record = await result.single()
            print(f"✅ Basic query successful: {record['test']}")

        # Get database info
        print("\n📊 Getting database information...")

        # Check entities count
        try:
            async with storage.driver.session(database=storage.config.database) as session:
                result = await session.run('MATCH (e:Entity) RETURN count(e) as entity_count')
                record = await result.single()
                entity_count = record["entity_count"] if record else 0
                print(f"   Total entities: {entity_count}")
        except Exception as e:
            print(f"   Could not count entities: {e}")

        # Check document chunks count
        try:
            async with storage.driver.session(database=storage.config.database) as session:
                result = await session.run('MATCH (d:DocumentChunk) RETURN count(d) as chunk_count')
                record = await result.single()
                chunk_count = record["chunk_count"] if record else 0
                print(f"   Total document chunks: {chunk_count}")
        except Exception as e:
            print(f"   Could not count document chunks: {e}")

        # Check relation types
        try:
            async with storage.driver.session(database=storage.config.database) as session:
                result = await session.run('MATCH ()-[r]->() RETURN type(r) as rel_type, count(r) as count ORDER BY count DESC LIMIT 10')
                records = []
                async for record in result:
                    records.append(record)
                if records:
                    print("   Top relation types:")
                    for record in records:
                        print(f"     {record['rel_type']}: {record['count']}")
                else:
                    print("   No relations found")
        except Exception as e:
            print(f"   Could not get relation types: {e}")

        # Check node labels
        try:
            async with storage.driver.session(database=storage.config.database) as session:
                result = await session.run('CALL db.labels() YIELD label RETURN label')
                records = []
                async for record in result:
                    records.append(record)
                if records:
                    labels = [record['label'] for record in records]
                    print(f"   Node labels: {', '.join(labels)}")
                else:
                    print("   No node labels found")
        except Exception as e:
            print(f"   Could not get node labels: {e}")
        
        print("\n✅ Connection test completed successfully!")
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\n🔍 Detailed error information:")
        print(traceback.format_exc())
        
        # Provide troubleshooting suggestions
        print("\n💡 Troubleshooting suggestions:")
        print("1. Check if the Neo4j server is running and accessible")
        print("2. Verify the URI format (bolt+ssc:// for secure connections)")
        print("3. Check username and password credentials")
        print("4. Verify SSL/TLS configuration")
        print("5. Check firewall settings and network connectivity")
        
        if "ConnectionResetError" in str(e):
            print("6. SSL handshake failed - check certificate configuration")
            print("7. Try setting NEO4J_TRUST_ALL_CERTIFICATES=true for self-signed certificates")
        
        return False
    
    finally:
        try:
            await storage.disconnect()
            print("🔌 Disconnected from Neo4j")
        except Exception as e:
            print(f"⚠️  Error during disconnect: {e}")
    
    return True


async def test_different_ssl_configs():
    """Test different SSL configurations to find working setup."""
    
    print("\n🔧 Testing different SSL configurations...")
    print("=" * 50)
    
    base_config = {
        'uri': os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
        'username': os.getenv('NEO4J_USERNAME', 'neo4j'),
        'password': os.getenv('NEO4J_PASSWORD', 'password'),
        'database': os.getenv('NEO4J_DATABASE', 'neo4j')
    }
    
    # Different SSL configurations to try
    ssl_configs = [
        {'verify_ssl': False, 'trust_all_certificates': True, 'name': 'No SSL verification, trust all'},
        {'verify_ssl': True, 'trust_all_certificates': True, 'name': 'SSL verification, trust all'},
        {'verify_ssl': False, 'trust_all_certificates': False, 'name': 'No SSL verification, no trust all'},
        {'verify_ssl': True, 'trust_all_certificates': False, 'name': 'SSL verification, no trust all'},
    ]
    
    for ssl_config in ssl_configs:
        print(f"\n🧪 Testing: {ssl_config['name']}")
        
        config = Neo4jConfig(
            **base_config,
            verify_ssl=ssl_config['verify_ssl'],
            trust_all_certificates=ssl_config['trust_all_certificates']
        )
        
        storage = Neo4jStorage(config)
        
        try:
            await storage.connect()
            # Test basic query
            async with storage.driver.session(database=storage.config.database) as session:
                result = await session.run("RETURN 1 as test")
                record = await result.single()
            print(f"✅ Success with: {ssl_config['name']}")
            await storage.disconnect()
            return ssl_config
        except Exception as e:
            print(f"❌ Failed with: {ssl_config['name']} - {str(e)[:100]}...")
            try:
                await storage.disconnect()
            except:
                pass
    
    print("❌ No SSL configuration worked")
    return None


if __name__ == "__main__":
    print("Starting Neo4j connection tests...\n")
    
    # Run the main connection test
    success = asyncio.run(test_neo4j_connection())
    
    if not success:
        print("\n" + "=" * 50)
        print("Main test failed, trying different SSL configurations...")
        working_config = asyncio.run(test_different_ssl_configs())
        
        if working_config:
            print(f"\n✅ Found working configuration: {working_config['name']}")
            print("Update your .env file with these settings:")
            print(f"NEO4J_VERIFY_SSL={str(working_config['verify_ssl']).lower()}")
            print(f"NEO4J_TRUST_ALL_CERTIFICATES={str(working_config['trust_all_certificates']).lower()}")
