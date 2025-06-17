"""Test script for the new graph operations (CRUD, Traversal, Analytics).

This script tests the newly implemented operations modules.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from morag_graph.storage import Neo4jStorage
from morag_graph.storage.neo4j_storage import Neo4jConfig
from morag_graph.operations import GraphCRUD, GraphTraversal, GraphAnalytics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_operations():
    """Test the new graph operations."""
    
    # Initialize Neo4j storage
    config = Neo4jConfig(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="morag_password",
        database="neo4j"
    )
    
    storage = Neo4jStorage(config)
    
    try:
        # Connect to Neo4j
        await storage.connect()
        logger.info("Connected to Neo4j")
        
        # Initialize operations
        crud = GraphCRUD(storage)
        traversal = GraphTraversal(storage)
        analytics = GraphAnalytics(storage)
        
        # Test CRUD operations
        logger.info("\n=== Testing CRUD Operations ===")
        
        # Get graph summary
        summary = await crud.get_graph_summary()
        logger.info(f"Graph summary: {summary}")
        
        # Test Analytics operations
        logger.info("\n=== Testing Analytics Operations ===")
        
        # Get graph statistics
        stats = await analytics.get_graph_statistics()
        logger.info(f"Graph statistics: {stats}")
        
        # Get entity type analysis
        entity_analysis = await analytics.analyze_entity_types()
        logger.info(f"Entity type analysis: {entity_analysis}")
        
        # Get relation type analysis
        relation_analysis = await analytics.analyze_relation_types()
        logger.info(f"Relation type analysis: {relation_analysis}")
        
        # Calculate graph density
        density = await analytics.calculate_graph_density()
        logger.info(f"Graph density: {density}")
        
        logger.info("\nAll operations tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        raise
    finally:
        # Disconnect
        await storage.disconnect()
        logger.info("Disconnected from Neo4j")


if __name__ == "__main__":
    asyncio.run(test_operations())