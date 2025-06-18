#!/usr/bin/env python3
"""
Production script for ingesting extracted knowledge graph data into Neo4j.
This script loads entities and relations from a JSON file and stores them in Neo4j.
"""

import json
import argparse
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional

from src.morag_graph.storage.neo4j_storage import Neo4jStorage
from src.morag_graph.models import Entity, Relation, Graph


def load_extracted_data(file_path: Path) -> Dict[str, Any]:
    """Load extracted entities and relations from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        raise Exception(f"Failed to load data from {file_path}: {e}")


def create_graph_from_data(data: Dict[str, Any]) -> Graph:
    """Create a Graph object from loaded data."""
    graph = Graph()
    
    # Add entities
    if "entities" in data:
        for entity_data in data["entities"]:
            entity = Entity(
                id=entity_data["id"],
                name=entity_data["name"],
                type=entity_data["type"],
                confidence=entity_data.get("confidence", 1.0),
                attributes=entity_data.get("attributes", {})
            )
            graph.add_entity(entity)
    
    # Add relations
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
            graph.add_relation(relation)
    
    return graph


async def ingest_to_neo4j(
    graph: Graph,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    clear_existing: bool = False
) -> None:
    """Ingest graph data into Neo4j."""
    storage = Neo4jStorage(
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password
    )
    
    try:
        # Test connection
        await storage.test_connection()
        print("✓ Neo4j connection successful")
        
        # Clear existing data if requested
        if clear_existing:
            print("Clearing existing data...")
            await storage.clear_all()
            print("✓ Existing data cleared")
        
        # Store entities
        print(f"Storing {len(graph.entities)} entities...")
        for entity in graph.entities.values():
            await storage.store_entity(entity)
        print("✓ Entities stored")
        
        # Store relations
        print(f"Storing {len(graph.relations)} relations...")
        for relation in graph.relations.values():
            await storage.store_relation(relation)
        print("✓ Relations stored")
        
        # Print statistics
        stats = await storage.get_graph_statistics()
        print("\n=== Graph Statistics ===")
        print(f"Total nodes: {stats.get('total_nodes', 0)}")
        print(f"Total relationships: {stats.get('total_relationships', 0)}")
        print(f"Node types: {stats.get('node_types', {})}")
        print(f"Relationship types: {stats.get('relationship_types', {})}")
        
    finally:
        await storage.close()


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
        
        # Create graph
        print("Creating graph from data...")
        graph = create_graph_from_data(data)
        print(f"✓ Graph created with {len(graph.entities)} entities and {len(graph.relations)} relations")
        
        # Ingest to Neo4j
        print("\nIngesting to Neo4j...")
        await ingest_to_neo4j(
            graph=graph,
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