#!/usr/bin/env python3
"""Test script for Neo4j ingestion of extracted entities and relations.

This script loads the extracted entities and relations from transcript_extracted.json
and stores them in the Neo4j database to test the graph storage implementation.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from morag_graph.models import Entity, Relation, Graph
from morag_graph.storage import Neo4jStorage
from morag_graph.storage.neo4j_storage import Neo4jConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def load_extracted_data(file_path: str) -> Dict[str, Any]:
    """Load extracted entities and relations from JSON file.
    
    Args:
        file_path: Path to the JSON file containing extracted data
        
    Returns:
        Dictionary containing entities and relations
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data.get('entities', []))} entities and {len(data.get('relations', []))} relations")
        return data
    except Exception as e:
        logger.error(f"Failed to load data from {file_path}: {e}")
        raise


async def test_neo4j_connection(storage: Neo4jStorage) -> bool:
    """Test Neo4j connection.
    
    Args:
        storage: Neo4j storage instance
        
    Returns:
        True if connection successful, False otherwise
    """
    try:
        await storage.connect()
        logger.info("✅ Successfully connected to Neo4j")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to connect to Neo4j: {e}")
        return False


async def ingest_entities(storage: Neo4jStorage, entities_data: List[Dict[str, Any]]) -> List[Entity]:
    """Ingest entities into Neo4j.
    
    Args:
        storage: Neo4j storage instance
        entities_data: List of entity dictionaries
        
    Returns:
        List of Entity objects that were stored
    """
    entities = []
    
    logger.info(f"Converting {len(entities_data)} entities...")
    
    for entity_dict in entities_data:
        try:
            # Create Entity object from dictionary
            entity = Entity(
                id=entity_dict['id'],
                name=entity_dict['name'],
                type=entity_dict['type'],
                confidence=entity_dict.get('confidence', 1.0),
                source_text=entity_dict.get('source_text', ''),
                source_doc_id=entity_dict.get('source_doc_id'),
                attributes=entity_dict.get('attributes', {})
            )
            entities.append(entity)
        except Exception as e:
            logger.warning(f"Failed to create entity from {entity_dict.get('name', 'unknown')}: {e}")
            continue
    
    logger.info(f"Storing {len(entities)} entities in Neo4j...")
    
    try:
        # Store entities in batches for better performance
        batch_size = 50
        stored_entities = []
        
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i + batch_size]
            batch_ids = await storage.store_entities(batch)
            stored_entities.extend(batch)
            logger.info(f"Stored batch {i//batch_size + 1}/{(len(entities) + batch_size - 1)//batch_size}")
        
        logger.info(f"✅ Successfully stored {len(stored_entities)} entities")
        return stored_entities
        
    except Exception as e:
        logger.error(f"❌ Failed to store entities: {e}")
        raise


async def ingest_relations(storage: Neo4jStorage, relations_data: List[Dict[str, Any]]) -> List[Relation]:
    """Ingest relations into Neo4j.
    
    Args:
        storage: Neo4j storage instance
        relations_data: List of relation dictionaries
        
    Returns:
        List of Relation objects that were stored
    """
    relations = []
    
    logger.info(f"Converting {len(relations_data)} relations...")
    
    for relation_dict in relations_data:
        try:
            # Create Relation object from dictionary
            relation = Relation(
                id=relation_dict['id'],
                source_entity_id=relation_dict['source_entity_id'],
                target_entity_id=relation_dict['target_entity_id'],
                type=relation_dict['type'],
                confidence=relation_dict.get('confidence', 1.0),
                source_text=relation_dict.get('source_text', ''),
                source_doc_id=relation_dict.get('source_doc_id'),
                attributes=relation_dict.get('attributes', {})
            )
            relations.append(relation)
        except Exception as e:
            logger.warning(f"Failed to create relation {relation_dict.get('id', 'unknown')}: {e}")
            continue
    
    logger.info(f"Storing {len(relations)} relations in Neo4j...")
    
    try:
        # Store relations in batches
        batch_size = 50
        stored_relations = []
        
        for i in range(0, len(relations), batch_size):
            batch = relations[i:i + batch_size]
            batch_ids = await storage.store_relations(batch)
            stored_relations.extend(batch)
            logger.info(f"Stored batch {i//batch_size + 1}/{(len(relations) + batch_size - 1)//batch_size}")
        
        logger.info(f"✅ Successfully stored {len(stored_relations)} relations")
        return stored_relations
        
    except Exception as e:
        logger.error(f"❌ Failed to store relations: {e}")
        raise


async def query_graph_stats(storage: Neo4jStorage) -> Dict[str, int]:
    """Query basic graph statistics.
    
    Args:
        storage: Neo4j storage instance
        
    Returns:
        Dictionary with graph statistics
    """
    try:
        # Count entities
        entity_count_query = "MATCH (e:Entity) RETURN count(e) as count"
        entity_result = await storage._execute_query(entity_count_query)
        entity_count = entity_result[0]['count'] if entity_result else 0
        
        # Count relations
        relation_count_query = "MATCH ()-[r]->() RETURN count(r) as count"
        relation_result = await storage._execute_query(relation_count_query)
        relation_count = relation_result[0]['count'] if relation_result else 0
        
        # Count entity types
        type_count_query = "MATCH (e:Entity) RETURN e.type as type, count(e) as count ORDER BY count DESC"
        type_result = await storage._execute_query(type_count_query)
        
        stats = {
            'total_entities': entity_count,
            'total_relations': relation_count,
            'entity_types': {record['type']: record['count'] for record in type_result}
        }
        
        logger.info(f"📊 Graph Statistics:")
        logger.info(f"   • Total Entities: {stats['total_entities']}")
        logger.info(f"   • Total Relations: {stats['total_relations']}")
        logger.info(f"   • Entity Types:")
        for entity_type, count in stats['entity_types'].items():
            logger.info(f"     - {entity_type}: {count}")
        
        return stats
        
    except Exception as e:
        logger.error(f"❌ Failed to query graph statistics: {e}")
        return {}


async def main():
    """Main function to test Neo4j ingestion."""
    logger.info("🧠 MoRAG Graph - Neo4j Ingestion Test")
    logger.info("=" * 60)
    
    # Configuration
    config = Neo4jConfig(
        uri="neo4j://localhost:7687",
        username="neo4j",
        password="morag_password",
        database="neo4j"
    )
    
    # Initialize storage
    storage = Neo4jStorage(config)
    
    try:
        # Test connection
        if not await test_neo4j_connection(storage):
            logger.error("Cannot proceed without Neo4j connection")
            return
        
        # Load extracted data
        data_file = "../../examples/transcript_extracted.json"
        logger.info(f"Loading data from {data_file}...")
        data = await load_extracted_data(data_file)
        
        # Clear existing data (optional)
        logger.info("Clearing existing data...")
        await storage.clear()
        
        # Ingest entities
        entities = await ingest_entities(storage, data.get('entities', []))
        
        # Ingest relations
        relations = await ingest_relations(storage, data.get('relations', []))
        
        # Query statistics
        await query_graph_stats(storage)
        
        logger.info("✅ Neo4j ingestion test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Neo4j ingestion test failed: {e}")
        raise
    finally:
        # Cleanup
        await storage.disconnect()
        logger.info("Disconnected from Neo4j")


if __name__ == "__main__":
    asyncio.run(main())