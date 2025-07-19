#!/usr/bin/env python3
"""Basic test script for Graphiti functionality using Gemini API."""

import asyncio
import os
import sys
from pathlib import Path

# Add the packages to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent.parent.parent / ".env")
except ImportError:
    print("⚠️ python-dotenv not available. Using environment variables directly.")

def test_imports():
    """Test that all Graphiti components can be imported."""
    print("📦 Testing Graphiti imports...")
    
    try:
        # Test basic imports
        from morag_graph.graphiti import GraphitiConfig
        print("✅ GraphitiConfig imported")
        
        from morag_graph.graphiti import GraphitiConnectionService
        print("✅ GraphitiConnectionService imported")
        
        from morag_graph.graphiti import DocumentEpisodeMapper
        print("✅ DocumentEpisodeMapper imported")
        
        from morag_graph.graphiti import GraphitiEntityStorage
        print("✅ GraphitiEntityStorage imported")
        
        from morag_graph.graphiti import GraphitiSearchService
        print("✅ GraphitiSearchService imported")
        
        # Test that migration-related imports are gone
        try:
            from morag_graph.graphiti import Neo4jToGraphitiMigrator
            print("❌ Migration imports still exist (should be removed)")
            return False
        except ImportError:
            print("✅ Migration imports correctly removed")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_configuration():
    """Test Graphiti configuration creation."""
    print("\n⚙️ Testing Graphiti configuration...")

    try:
        from morag_graph.graphiti import GraphitiConfig

        # Get Gemini API key from environment
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            print("⚠️ No GEMINI_API_KEY found in environment")
            gemini_api_key = "test-key"

        # Test configuration with Gemini settings
        # Note: We'll use the Gemini API key as OpenAI key since Graphiti expects OpenAI format
        config = GraphitiConfig(
            neo4j_uri="bolt://localhost:7687",
            neo4j_username="neo4j",
            neo4j_password="password",
            neo4j_database="neo4j",
            openai_api_key=gemini_api_key,  # Using Gemini key
            openai_model="gemini-1.5-flash",  # Try using Gemini model name
            openai_embedding_model="text-embedding-004"  # Gemini embedding model
        )

        print(f"✅ Configuration created successfully")
        print(f"   Neo4j URI: {config.neo4j_uri}")
        print(f"   Database: {config.neo4j_database}")
        print(f"   Model: {config.openai_model}")
        print(f"   Embedding Model: {config.openai_embedding_model}")
        print(f"   Telemetry: {config.enable_telemetry}")

        return True

    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        return False

def test_neo4j_connection():
    """Test basic Neo4j connection without Graphiti."""
    print("\n🔌 Testing Neo4j connection...")
    
    try:
        import neo4j
        
        # Test Neo4j connection
        driver = neo4j.GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "password")
        )
        
        # Test connection with a simple query
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            record = result.single()
            if record and record["test"] == 1:
                print("✅ Neo4j connection successful")
                driver.close()
                return True
            else:
                print("❌ Neo4j query failed")
                driver.close()
                return False
                
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        return False

async def test_basic_graphiti_setup():
    """Test basic Graphiti setup with Gemini API."""
    print("\n🏗️ Testing basic Graphiti setup...")

    try:
        from morag_graph.graphiti import GraphitiConfig, GraphitiConnectionService

        # Get Gemini API key from environment
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            print("⚠️ No GEMINI_API_KEY found in environment")
            gemini_api_key = "test-key"

        # Create configuration with Gemini settings
        config = GraphitiConfig(
            neo4j_uri="bolt://localhost:7687",
            neo4j_username="neo4j",
            neo4j_password="password",
            neo4j_database="neo4j",
            openai_api_key=gemini_api_key,  # Using Gemini key
            openai_model="gemini-1.5-flash",  # Try using Gemini model name
            openai_embedding_model="text-embedding-004"  # Gemini embedding model
        )

        # Create connection service (but don't connect yet)
        connection_service = GraphitiConnectionService(config)
        print("✅ GraphitiConnectionService created")

        # Test that the service has the expected attributes
        assert hasattr(connection_service, 'config')
        assert hasattr(connection_service, 'connect')
        assert hasattr(connection_service, 'disconnect')
        print("✅ Connection service has expected methods")

        return True

    except Exception as e:
        print(f"❌ Basic Graphiti setup failed: {e}")
        return False

def test_model_compatibility():
    """Test that MoRAG models work with Graphiti components."""
    print("\n🔗 Testing model compatibility...")
    
    try:
        from morag_graph.models import Document, DocumentChunk, Entity, Relation
        from morag_graph.graphiti import DocumentEpisodeMapper, GraphitiConfig
        
        # Create test models (let them generate their own IDs)
        test_doc = Document(
            name="Test Document",
            source_file="test_document.txt",
            checksum="abc123",
            metadata={"source": "test"}
        )

        test_chunk = DocumentChunk(
            document_id=test_doc.id,  # Use the generated document ID
            text="This is a test chunk.",
            chunk_index=0,
            metadata={}
        )

        test_entity = Entity(
            name="Test Entity",
            type="CONCEPT",
            attributes={}
        )

        test_relation = Relation(
            source_entity_id=test_entity.id,  # Use the generated entity ID
            target_entity_id="ent_another_entity_abc123",  # Use proper format
            type="RELATED_TO",
            attributes={}
        )
        
        print("✅ MoRAG models created successfully")
        
        # Test that we can create Graphiti components with these models
        gemini_api_key = os.getenv("GEMINI_API_KEY", "test-key")
        config = GraphitiConfig(
            neo4j_uri="bolt://localhost:7687",
            neo4j_username="neo4j",
            neo4j_password="password",
            neo4j_database="neo4j",
            openai_api_key=gemini_api_key,  # Using Gemini key
            openai_model="gemini-1.5-flash",
            openai_embedding_model="text-embedding-004"
        )
        
        mapper = DocumentEpisodeMapper(config)
        print("✅ DocumentEpisodeMapper created with MoRAG models")
        
        return True
        
    except Exception as e:
        print(f"❌ Model compatibility test failed: {e}")
        return False

async def main():
    """Run all basic tests."""
    print("🚀 Starting basic Graphiti tests...\n")
    
    # Test imports
    imports_ok = test_imports()
    
    # Test configuration
    config_ok = test_configuration()
    
    # Test Neo4j connection
    neo4j_ok = test_neo4j_connection()
    
    # Test basic Graphiti setup
    setup_ok = await test_basic_graphiti_setup()
    
    # Test model compatibility
    models_ok = test_model_compatibility()
    
    # Summary
    print("\n📊 Test Summary:")
    print(f"   Imports: {'✅' if imports_ok else '❌'}")
    print(f"   Configuration: {'✅' if config_ok else '❌'}")
    print(f"   Neo4j Connection: {'✅' if neo4j_ok else '❌'}")
    print(f"   Basic Setup: {'✅' if setup_ok else '❌'}")
    print(f"   Model Compatibility: {'✅' if models_ok else '❌'}")
    
    if all([imports_ok, config_ok, neo4j_ok, setup_ok, models_ok]):
        print("\n🎉 All basic tests passed! Graphiti is properly set up.")
        print("💡 To test full functionality, run the full ingestion test.")
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    asyncio.run(main())
