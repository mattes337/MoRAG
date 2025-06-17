"""Test script for file ingestion with checksum-based duplicate prevention.

This script demonstrates how to use the FileIngestion class to prevent
duplicate file ingestion using checksums.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from morag_graph.models import Entity, Relation
from morag_graph.storage import Neo4jStorage
from morag_graph.storage.neo4j_storage import Neo4jConfig
from morag_graph.ingestion import FileIngestion

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_entities_and_relations() -> tuple[List[Entity], List[Relation]]:
    """Create sample entities and relations for testing.
    
    Returns:
        Tuple of (entities, relations)
    """
    entities = [
        Entity(
            id="person_1",
            name="Alice Johnson",
            type="PERSON",
            confidence=0.95,
            source_text="Alice Johnson is a software engineer",
            attributes={"profession": "software engineer"}
        ),
        Entity(
            id="person_2",
            name="Bob Smith",
            type="PERSON",
            confidence=0.90,
            source_text="Bob Smith works at TechCorp",
            attributes={"company": "TechCorp"}
        ),
        Entity(
            id="company_1",
            name="TechCorp",
            type="ORGANIZATION",
            confidence=0.98,
            source_text="TechCorp is a technology company",
            attributes={"industry": "technology"}
        )
    ]
    
    relations = [
        Relation(
            id="rel_1",
            source_entity_id="person_1",
            target_entity_id="person_2",
            type="KNOWS",
            confidence=0.85,
            source_text="Alice knows Bob from work",
            attributes={"context": "work"}
        ),
        Relation(
            id="rel_2",
            source_entity_id="person_2",
            target_entity_id="company_1",
            type="WORKS_AT",
            confidence=0.92,
            source_text="Bob works at TechCorp",
            attributes={"role": "employee"}
        )
    ]
    
    return entities, relations


def create_test_file(file_path: Path, content: str) -> None:
    """Create a test file with given content.
    
    Args:
        file_path: Path where to create the file
        content: Content to write to the file
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    logger.info(f"Created test file: {file_path}")


async def test_file_ingestion_with_duplicates():
    """Test file ingestion with duplicate prevention."""
    # Setup Neo4j storage
    config = Neo4jConfig(
        uri="neo4j://localhost:7687",
        username="neo4j",
        password="password",
        database="neo4j"
    )
    
    storage = Neo4jStorage(config)
    
    try:
        await storage.connect()
        logger.info("✅ Connected to Neo4j")
        
        # Clear existing data for clean test
        await storage.clear()
        logger.info("Cleared existing data")
        
        # Initialize file ingestion
        file_ingestion = FileIngestion(storage)
        
        # Create test files
        test_dir = Path("test_files")
        test_file1 = test_dir / "document1.txt"
        test_file2 = test_dir / "document2.txt"
        test_file3 = test_dir / "document1_copy.txt"  # Same content as document1
        
        # Create files with different content
        create_test_file(test_file1, "This is the content of document 1.")
        create_test_file(test_file2, "This is the content of document 2.")
        create_test_file(test_file3, "This is the content of document 1.")  # Same as file1
        
        # Create sample data
        entities, relations = create_sample_entities_and_relations()
        
        # Test 1: Ingest first file
        logger.info("\n=== Test 1: First ingestion ===")
        result1 = await file_ingestion.ingest_file_entities_and_relations(
            test_file1, entities, relations
        )
        logger.info(f"Result 1: {result1['status']} - {result1.get('entities_stored', 0)} entities, {result1.get('relations_stored', 0)} relations")
        
        # Test 2: Ingest second file (different content)
        logger.info("\n=== Test 2: Different file ===")
        result2 = await file_ingestion.ingest_file_entities_and_relations(
            test_file2, entities, relations
        )
        logger.info(f"Result 2: {result2['status']} - {result2.get('entities_stored', 0)} entities, {result2.get('relations_stored', 0)} relations")
        
        # Test 3: Try to ingest duplicate file (same content, different name)
        logger.info("\n=== Test 3: Duplicate content (should be skipped) ===")
        result3 = await file_ingestion.ingest_file_entities_and_relations(
            test_file3, entities, relations
        )
        logger.info(f"Result 3: {result3['status']} - Reason: {result3.get('reason', 'N/A')}")
        
        # Test 4: Force reingest of duplicate
        logger.info("\n=== Test 4: Force reingest ===")
        result4 = await file_ingestion.ingest_file_entities_and_relations(
            test_file3, entities, relations, force_reingest=True
        )
        logger.info(f"Result 4: {result4['status']} - {result4.get('entities_stored', 0)} entities, {result4.get('relations_stored', 0)} relations")
        
        # Show ingestion statistics
        logger.info("\n=== Ingestion Statistics ===")
        stats = file_ingestion.get_ingestion_stats()
        logger.info(f"Total files ingested: {stats['total_files_ingested']}")
        logger.info(f"Unique checksums: {stats['unique_checksums']}")
        logger.info(f"Files by type: {stats['files_by_type']}")
        
        # Show ingested files
        logger.info("\n=== Ingested Files ===")
        ingested_files = await file_ingestion.get_ingested_files()
        for file_meta in ingested_files:
            logger.info(f"File: {file_meta.file_name}, Checksum: {file_meta.checksum[:16]}..., Size: {file_meta.file_size} bytes")
        
        # Test graph statistics
        logger.info("\n=== Graph Statistics ===")
        graph_stats = await storage.get_statistics()
        logger.info(f"Total entities in graph: {graph_stats.get('entity_count', 0)}")
        logger.info(f"Total relations in graph: {graph_stats.get('relation_count', 0)}")
        
        # Test file removal
        logger.info("\n=== Test 5: Remove file from graph ===")
        removal_result = await file_ingestion.remove_file_from_graph(test_file1)
        logger.info(f"Removal result: {removal_result['status']} - {removal_result.get('entities_deleted', 0)} entities deleted")
        
        # Final statistics
        logger.info("\n=== Final Graph Statistics ===")
        final_stats = await storage.get_statistics()
        logger.info(f"Total entities in graph: {final_stats.get('entity_count', 0)}")
        logger.info(f"Total relations in graph: {final_stats.get('relation_count', 0)}")
        
        logger.info("\n✅ File ingestion tests completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise
    finally:
        await storage.disconnect()
        logger.info("Disconnected from Neo4j")
        
        # Clean up test files
        import shutil
        if Path("test_files").exists():
            shutil.rmtree("test_files")
            logger.info("Cleaned up test files")


if __name__ == "__main__":
    asyncio.run(test_file_ingestion_with_duplicates())