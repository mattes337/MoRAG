#!/usr/bin/env python3
"""
Simple test of Graphiti with minimal content to debug the vector similarity issue.
"""

import os
import sys
import asyncio
sys.path.append('packages/morag-graph/src')
from morag_graph.graphiti.config import create_graphiti_instance, GraphitiConfig

async def test_simple_graphiti():
    """Test Graphiti with minimal content."""

    # Load environment variables (override existing ones)
    from dotenv import load_dotenv
    load_dotenv(override=True)

    # Get Gemini API key from environment
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print("❌ No GEMINI_API_KEY found in environment")
        return

    print(f"✅ Using Gemini API key: {gemini_api_key[:10]}...")

    # Test the vector patch first
    print("🔧 Testing vector similarity patch...")
    sys.path.append('packages/morag-graph/src')
    from morag_graph.graphiti.vector_patch import test_vector_similarity_patch
    test_vector_similarity_patch()

    # Test if bulk query patch is working
    print("🔧 Testing bulk query patch...")
    import graphiti_core.utils.bulk_utils as bulk
    from graphiti_core.utils.bulk_utils import EntityNode
    node = EntityNode(uuid='test', name='test', labels=['Entity'], group_id='test')
    query = bulk.get_entity_node_save_bulk_query([node], 'neo4j')
    print(f"Generated query: {query}")
    if ":$(" in query:
        print("❌ Bulk query patch NOT working")
    else:
        print("✅ Bulk query patch working")

    # Create configuration
    config = GraphitiConfig(
        neo4j_uri="bolt://localhost:7687",
        neo4j_username="neo4j",
        neo4j_password="password",
        neo4j_database="neo4j",
        openai_api_key=gemini_api_key,
        openai_model="gemini-1.5-flash",
        openai_embedding_model="text-embedding-004"
    )
    
    print("🔄 Creating Graphiti instance...")
    try:
        graphiti = create_graphiti_instance(config)
        print("✅ Graphiti instance created successfully")
        
        print("🔄 Testing simple episode creation...")
        
        # Try to create a very simple episode
        episode_name = "test-episode"
        content = "This is a simple test content for Graphiti."

        # This should trigger the vector similarity error
        # Use the correct Graphiti API
        from datetime import datetime
        result = await graphiti.add_episode(
            name=episode_name,
            episode_body=content,
            source_description="test",
            reference_time=datetime.now()
        )
        
        print(f"✅ Episode created: {result}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        
        # Let's see the exact error
        import traceback
        print("📋 Full traceback:")
        traceback.print_exc()
    
    finally:
        try:
            await graphiti.close()
        except:
            pass

if __name__ == "__main__":
    print("🧪 Testing Graphiti with simple content")
    print("=" * 50)
    asyncio.run(test_simple_graphiti())
