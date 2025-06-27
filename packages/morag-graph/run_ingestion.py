#!/usr/bin/env python3
"""Production script for ingesting extracted knowledge graph data into Neo4j.
This script loads entities and relations from a JSON file and uses the FileIngestion
class to properly ingest them with Document and DocumentChunk structure.
"""

import json
import argparse
import asyncio
from pathlib import Path
from typing import Dict, Any, List

from src.morag_graph.storage.neo4j_storage import Neo4jStorage, Neo4jConfig
from src.morag_graph.models import Entity, Relation
from src.morag_graph.ingestion import FileIngestion


def load_extracted_data(file_path: Path) -> Dict[str, Any]:
    """Load extracted entities and relations from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        raise Exception(f"Failed to load data from {file_path}: {e}")


def parse_entities_and_relations(data: Dict[str, Any]) -> tuple[List[Entity], List[Relation]]:
    """Parse entities and relations from loaded data."""
    entities = []
    relations = []
    
    # Parse entities
    if "entities" in data:
        for entity_data in data["entities"]:
            entity = Entity(
                id=entity_data["id"],
                name=entity_data["name"],
                type=entity_data["type"],
                confidence=entity_data.get("confidence", 1.0),
                attributes=entity_data.get("attributes", {})
            )
            entities.append(entity)
    
    # Parse relations
    if "relations" in data:
        for relation_data in data["relations"]:
            relation = Relation(
                id=relation_data.get("id", f"{relation_data['source_entity_id']}-{relation_data['target_entity_id']}"),
                source_entity_id=relation_data["source_entity_id"],
                target_entity_id=relation_data["target_entity_id"],
                type=relation_data["type"],
                confidence=relation_data.get("confidence", 1.0),
                attributes=relation_data.get("attributes", {})
            )
            relations.append(relation)
    
    return entities, relations





async def ingest_to_neo4j(
    file_path: Path,
    entities: List[Entity],
    relations: List[Relation],
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    clear_existing: bool = False
) -> None:
    """Ingest entities and relations into Neo4j using FileIngestion class."""
    config = Neo4jConfig(
        uri=neo4j_uri,
        username=neo4j_user,
        password=neo4j_password
    )
    storage = Neo4jStorage(config)
    
    try:
        # Connect to Neo4j
        await storage.connect()
        print("✓ Neo4j connection successful")
        
        # Clear existing data if requested
        if clear_existing:
            print("Clearing existing data...")
            await storage.clear()
            print("✓ Existing data cleared")
        
        # Use FileIngestion class to handle the ingestion
        file_ingestion = FileIngestion(storage)
        
        # Ingest entities and relations using the standard process
        result = await file_ingestion.ingest_file_entities_and_relations(
            file_path=file_path,
            entities=entities,
            relations=relations,
            force_reingest=True  # Always ingest since we're processing extracted data
        )
        
        if result['status'] == 'success':
            print(f"✓ Document stored: {result['document_id']}")
            print(f"✓ Created {result['chunks_created']} document chunks")
            print(f"✓ Stored {result['entities_stored']} entities")
            print(f"✓ Stored {result['relations_stored']} relations")
        else:
            print(f"✗ Ingestion failed: {result.get('error', 'Unknown error')}")
            return
        
        # Print statistics
        stats = await storage.get_statistics()
        print("\n=== Graph Statistics ===")
        print(f"Total nodes: {stats.get('total_nodes', 0)}")
        print(f"Total relationships: {stats.get('total_relationships', 0)}")
        print(f"Node types: {stats.get('node_types', {})}")
        print(f"Relationship types: {stats.get('relationship_types', {})}")
        
        print(f"\n✓ Successfully ingested file with {result['entities_stored']} entities and {result['relations_stored']} relations")
        
    finally:
        await storage.disconnect()


async def main():
    parser = argparse.ArgumentParser(description="Ingest extracted knowledge graph data into Neo4j")
    parser.add_argument("input_file", help="Path to JSON file containing extracted entities and relations")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--neo4j-user", default="neo4j", help="Neo4j username")
    parser.add_argument("--neo4j-password", required=True, help="Neo4j password")
    parser.add_argument("--clear", action="store_true", help="Clear existing data before ingestion")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist")
        return 1
    
    try:
        # Load data
        print(f"Loading data from {input_path}...")
        data = load_extracted_data(input_path)
        
        # Parse entities and relations
        print("Parsing entities and relations from data...")
        entities, relations = parse_entities_and_relations(data)
        print(f"✓ Parsed {len(entities)} entities and {len(relations)} relations")
        
        # Ingest to Neo4j
        print("\nIngesting to Neo4j...")
        await ingest_to_neo4j(
            file_path=input_path,
            entities=entities,
            relations=relations,
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_password=args.neo4j_password,
            clear_existing=args.clear
        )
        
        print("\n✓ Ingestion completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))