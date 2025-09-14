#!/usr/bin/env python3
"""Test script for intelligent retrieval with custom database connections."""

import asyncio
import json
import os
import sys
from typing import Optional

# Add the packages to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'morag-reasoning', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'morag-core', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'morag-graph', 'src'))

from morag_reasoning.intelligent_retrieval_models import IntelligentRetrievalRequest
from morag_graph.models.database_config import DatabaseServerConfig, DatabaseType


def create_custom_neo4j_config(
    hostname: str = "neo4j://localhost:7687",
    username: str = "neo4j", 
    password: str = "password",
    database: str = "neo4j"
) -> DatabaseServerConfig:
    """Create a custom Neo4j server configuration."""
    return DatabaseServerConfig(
        type=DatabaseType.NEO4J,
        hostname=hostname,
        username=username,
        password=password,
        database_name=database,
        config_options={
            "verify_ssl": True,
            "trust_all_certificates": False
        }
    )


def create_custom_qdrant_config(
    hostname: str = "localhost",
    port: int = 6333,
    api_key: Optional[str] = None,
    collection: str = "morag_vectors",
    https: bool = False
) -> DatabaseServerConfig:
    """Create a custom Qdrant server configuration."""
    return DatabaseServerConfig(
        type=DatabaseType.QDRANT,
        hostname=hostname,
        port=port,
        password=api_key,  # API key goes in password field
        database_name=collection,
        config_options={
            "https": https,
            "verify_ssl": True
        }
    )


def create_test_request_with_custom_databases() -> IntelligentRetrievalRequest:
    """Create a test request with custom database configurations."""
    
    # Custom Neo4j configuration (example with external server)
    neo4j_config = create_custom_neo4j_config(
        hostname="https://graph.adhs.morag.drydev.de",
        username="neo4j",
        password="your-password-here",
        database="neo4j"
    )
    
    # Custom Qdrant configuration (example with external server)
    qdrant_config = create_custom_qdrant_config(
        hostname="your-qdrant-server.com",
        port=6333,
        api_key="your-api-key-here",
        collection="custom_collection",
        https=True
    )
    
    return IntelligentRetrievalRequest(
        query="What are the main symptoms of ADHD?",
        max_iterations=3,
        max_entities_per_iteration=5,
        max_paths_per_entity=3,
        max_depth=2,
        min_relevance_threshold=0.4,
        include_debug_info=True,
        language="en",
        # Custom database server configurations
        neo4j_server=neo4j_config,
        qdrant_server=qdrant_config
    )


def create_test_request_with_default_databases() -> IntelligentRetrievalRequest:
    """Create a test request using server default configurations."""
    return IntelligentRetrievalRequest(
        query="What are the main symptoms of ADHD?",
        max_iterations=3,
        max_entities_per_iteration=5,
        max_paths_per_entity=3,
        max_depth=2,
        min_relevance_threshold=0.4,
        include_debug_info=True,
        language="en"
        # No custom database configurations - will use server defaults
    )


def create_test_request_with_named_databases() -> IntelligentRetrievalRequest:
    """Create a test request using named database configurations."""
    return IntelligentRetrievalRequest(
        query="What are the main symptoms of ADHD?",
        max_iterations=3,
        max_entities_per_iteration=5,
        max_paths_per_entity=3,
        max_depth=2,
        min_relevance_threshold=0.4,
        include_debug_info=True,
        language="en",
        # Use pre-configured databases by name
        neo4j_database="production",
        qdrant_collection="documents_v2"
    )


async def test_rest_api_with_custom_databases():
    """Test the REST API with custom database configurations."""
    import aiohttp
    
    # Create request with custom database configurations
    request = create_test_request_with_custom_databases()
    
    # Convert to JSON for API call
    request_data = request.model_dump()
    
    print("Testing REST API with custom database configurations...")
    print(f"Request data: {json.dumps(request_data, indent=2)}")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:8000/api/v2/intelligent-query",
                json=request_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"Success! Retrieved {len(result.get('key_facts', []))} key facts")
                    print(f"Response: {json.dumps(result, indent=2)}")
                else:
                    error_text = await response.text()
                    print(f"Error {response.status}: {error_text}")
                    
    except Exception as e:
        print(f"Failed to connect to API: {e}")


def print_example_configurations():
    """Print example configurations for different scenarios."""
    
    print("=== Example 1: Custom Neo4j and Qdrant servers ===")
    request1 = create_test_request_with_custom_databases()
    print(json.dumps(request1.model_dump(), indent=2))
    
    print("\n=== Example 2: Server default configurations ===")
    request2 = create_test_request_with_default_databases()
    print(json.dumps(request2.model_dump(), indent=2))
    
    print("\n=== Example 3: Named database configurations ===")
    request3 = create_test_request_with_named_databases()
    print(json.dumps(request3.model_dump(), indent=2))


def main():
    """Main function to demonstrate custom database configurations."""
    
    if len(sys.argv) > 1 and sys.argv[1] == "examples":
        print_example_configurations()
        return
    
    if len(sys.argv) > 1 and sys.argv[1] == "test-api":
        asyncio.run(test_rest_api_with_custom_databases())
        return
    
    print("Custom Database Configuration Test Script")
    print("========================================")
    print()
    print("This script demonstrates how to use custom database configurations")
    print("with the intelligent retrieval REST API.")
    print()
    print("Usage:")
    print("  python test-custom-database-retrieval.py examples    # Show example configurations")
    print("  python test-custom-database-retrieval.py test-api    # Test REST API with custom config")
    print()
    print("The REST API now supports three ways to specify databases:")
    print("1. Custom server configurations (neo4j_server, qdrant_server)")
    print("2. Named database configurations (neo4j_database, qdrant_collection)")
    print("3. Server default configurations (no database parameters)")
    print()
    print("Custom server configurations allow you to connect to any")
    print("Neo4j or Qdrant server by providing connection details like:")
    print("- hostname/URI")
    print("- port")
    print("- username/password")
    print("- database/collection name")
    print("- SSL and other connection options")


if __name__ == "__main__":
    main()
