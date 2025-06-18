#!/usr/bin/env python3
"""Test script for the new document structure refactoring.

This script tests the new Document -> DocumentChunk -> Entity structure
using the test_document.json data.
"""

import sys
import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from src.morag_graph.models import Entity, Relation, Document, DocumentChunk
from src.morag_graph.storage.neo4j_storage import Neo4jStorage, Neo4jConfig
from src.morag_graph.ingestion.file_ingestion import FileIngestion

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def clear_database(storage: Neo4jStorage) -> None:
    """Clear all data from the Neo4j database."""
    logger.info("Clearing Neo4j database...")
    await storage._execute_query("MATCH (n) DETACH DELETE n")
    logger.info("Database cleared")


async def load_test_data(test_file: str = "test_document.json") -> tuple[List[Entity], List[Relation]]:
    """Load test data from test_document.json."""
    logger.info(f"Loading test data from {test_file}...")
    test_file = Path(test_file)   
    if not test_file.exists():
        raise FileNotFoundError(f"Test file {test_file} not found")
    
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Parse entities
    entities = []
    for entity_data in data.get('entities', []):
        entities.append(Entity(**entity_data))
    
    # Parse relations
    relations = []
    for relation_data in data.get('relations', []):
        relations.append(Relation(**relation_data))
    
    logger.info(f"Loaded {len(entities)} entities and {len(relations)} relations from test data")
    return entities, relations


async def verify_document_structure(storage: Neo4jStorage) -> Dict[str, Any]:
    """Verify the new document structure in the database."""
    logger.info("Verifying document structure...")
    
    # Count nodes by type
    document_count_query = "MATCH (d:Document) RETURN count(d) as count"
    chunk_count_query = "MATCH (c:DocumentChunk) RETURN count(c) as count"
    entity_count_query = "MATCH (e:Entity) RETURN count(e) as count"
    
    # Count relationships by type
    contains_rel_query = "MATCH ()-[r:CONTAINS]->() RETURN count(r) as count"
    mentions_rel_query = "MATCH ()-[r:MENTIONS]->() RETURN count(r) as count"
    entity_rel_query = "MATCH (e1:Entity)-[r]->(e2:Entity) WHERE type(r) <> 'MENTIONS' AND type(r) <> 'CONTAINS' RETURN count(r) as count"
    
    results = {}
    
    # Execute queries
    queries = {
        'documents': document_count_query,
        'chunks': chunk_count_query,
        'entities': entity_count_query,
        'contains_relationships': contains_rel_query,
        'mentions_relationships': mentions_rel_query,
        'entity_relationships': entity_rel_query
    }
    
    for name, query in queries.items():
        try:
            result = await storage._execute_query(query)
            results[name] = result[0]['count'] if result else 0
        except Exception as e:
            logger.error(f"Error executing query for {name}: {e}")
            results[name] = 0
    
    # Get sample data
    sample_queries = {
        'sample_document': "MATCH (d:Document) RETURN d LIMIT 1",
        'sample_chunk': "MATCH (c:DocumentChunk) RETURN c LIMIT 1",
        'sample_entity': "MATCH (e:Entity) RETURN e LIMIT 1"
    }
    
    for name, query in sample_queries.items():
        try:
            result = await storage._execute_query(query)
            results[name] = result[0] if result else None
        except Exception as e:
            logger.error(f"Error getting {name}: {e}")
            results[name] = None
    
    return results


async def test_graph_traversal(storage: Neo4jStorage) -> Dict[str, Any]:
    """Test graph traversal capabilities with the new structure."""
    logger.info("Testing graph traversal...")
    
    # Test: Find all entities mentioned in a specific document
    entities_in_doc_query = """
    MATCH (d:Document)-[:CONTAINS]->(c:DocumentChunk)-[:MENTIONS]->(e:Entity)
    RETURN d.title as document, c.index as chunk_index, e.name as entity_name, e.type as entity_type
    ORDER BY c.index, e.name
    """
    
    # Test: Find documents that mention a specific entity
    docs_mentioning_entity_query = """
    MATCH (d:Document)-[:CONTAINS]->(c:DocumentChunk)-[:MENTIONS]->(e:Entity {name: 'Apple Inc.'})
    RETURN DISTINCT d.title as document, count(c) as chunks_mentioning
    """
    
    # Test: Find related entities through document co-occurrence
    related_entities_query = """
    MATCH (e1:Entity)<-[:MENTIONS]-(c:DocumentChunk)-[:MENTIONS]->(e2:Entity)
    WHERE e1.name = 'Steve Jobs' AND e1 <> e2
    RETURN e2.name as related_entity, e2.type as entity_type, count(c) as co_occurrences
    ORDER BY co_occurrences DESC
    """
    
    results = {}
    
    queries = {
        'entities_in_document': entities_in_doc_query,
        'documents_mentioning_apple': docs_mentioning_entity_query,
        'entities_related_to_steve_jobs': related_entities_query
    }
    
    for name, query in queries.items():
        try:
            result = await storage._execute_query(query)
            results[name] = result
            logger.info(f"{name}: {len(result)} results")
        except Exception as e:
            logger.error(f"Error executing {name}: {e}")
            results[name] = []
    
    return results


async def main():
    """Main test function."""
    logger.info("Starting document structure test...")

    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        logger.error("No test file provided. Please provide a test file as an argument.")
        return

    
    # Initialize storage with default test configuration
    # Try different common Neo4j configurations
    configs_to_try = [
        Neo4jConfig(uri="neo4j://localhost:7687", username="neo4j", password="neo4j"),
        Neo4jConfig(uri="bolt://localhost:7687", username="neo4j", password="neo4j"),
        Neo4jConfig(uri="neo4j://localhost:7687", username="neo4j", password="morag_password"),
        Neo4jConfig(uri="bolt://localhost:7687", username="neo4j", password="morag_password")
    ]
    
    storage = None
    for config in configs_to_try:
        try:
            storage = Neo4jStorage(config)
            await storage.connect()
            logger.info(f"Successfully connected with config: {config.uri}, user: {config.username}")
            break
        except Exception as e:
            logger.warning(f"Failed to connect with {config.uri}: {e}")
            if storage:
                try:
                    await storage.disconnect()
                except:
                    pass
            storage = None
    
    if not storage:
        logger.error("Could not connect to Neo4j with any configuration. Please ensure Neo4j is running and check credentials.")
        logger.info("Common Neo4j setups:")
        logger.info("- Default: username=neo4j, password=neo4j")
        logger.info("- Docker: username=neo4j, password=password")
        logger.info("- Custom: Check your Neo4j configuration")
        return
    
    try:
        
        # Clear existing data
        await clear_database(storage)
        
        # Load test data
        entities, relations = await load_test_data(test_file)
        
        # Initialize file ingestion
        file_ingestion = FileIngestion(storage)
        
        # Create a dummy file path for testing
        test_file_path = Path("test_document.json")
        
        # Ingest the data using the new structure
        logger.info("Ingesting data with new document structure...")
        result = await file_ingestion.ingest_file_entities_and_relations(
            test_file_path,
            entities,
            relations,
            force_reingest=True
        )
        
        logger.info(f"Ingestion result: {result}")
        
        if result['status'] != 'success':
            logger.error(f"Ingestion failed: {result}")
            return
        
        # Verify the structure
        structure_results = await verify_document_structure(storage)
        logger.info("\n=== Document Structure Verification ===")
        for key, value in structure_results.items():
            if isinstance(value, dict) and value:
                logger.info(f"{key}: {value}")
            else:
                logger.info(f"{key}: {value}")
        
        # Test graph traversal
        traversal_results = await test_graph_traversal(storage)
        logger.info("\n=== Graph Traversal Tests ===")
        for key, value in traversal_results.items():
            logger.info(f"{key}: {len(value)} results")
            if value and len(value) <= 5:  # Show first 5 results
                for item in value:
                    logger.info(f"  {item}")
        
        # Calculate expected vs actual relationship counts
        expected_entity_relations = len(relations)
        expected_mentions_relations = len(entities)  # Each entity should have one MENTIONS relation
        expected_contains_relations = structure_results.get('chunks', 0)  # Each chunk should have one CONTAINS relation
        
        logger.info("\n=== Relationship Count Analysis ===")
        logger.info(f"Expected entity-to-entity relations: {expected_entity_relations}")
        logger.info(f"Actual entity-to-entity relations: {structure_results.get('entity_relationships', 0)}")
        logger.info(f"Expected MENTIONS relations: {expected_mentions_relations}")
        logger.info(f"Actual MENTIONS relations: {structure_results.get('mentions_relationships', 0)}")
        logger.info(f"Expected CONTAINS relations: {expected_contains_relations}")
        logger.info(f"Actual CONTAINS relations: {structure_results.get('contains_relationships', 0)}")
        
        total_expected = expected_entity_relations + expected_mentions_relations + expected_contains_relations
        total_actual = (structure_results.get('entity_relationships', 0) + 
                       structure_results.get('mentions_relationships', 0) + 
                       structure_results.get('contains_relationships', 0))
        
        logger.info(f"\nTotal expected relationships: {total_expected}")
        logger.info(f"Total actual relationships: {total_actual}")
        logger.info(f"Relationship increase: {total_actual - expected_entity_relations} (from {expected_entity_relations} to {total_actual})")
        
        logger.info("\n=== Test completed successfully! ===")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise
    finally:
        await storage.disconnect()


if __name__ == "__main__":
    asyncio.run(main())