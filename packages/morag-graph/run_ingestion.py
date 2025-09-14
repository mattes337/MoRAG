#!/usr/bin/env python3
"""Production script for ingesting extracted fact data into Neo4j.
This script loads facts and relationships from a JSON file and uses the FactExtractionService
to properly ingest them with Document and DocumentChunk structure.
"""

import json
import argparse
import asyncio
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

from morag_graph.storage.neo4j_storage import Neo4jStorage, Neo4jConfig
from morag_graph.models.fact import Fact, FactRelation
from morag_graph.storage.neo4j_operations.fact_operations import FactOperations


def load_extracted_data(file_path: Path) -> Dict[str, Any]:
    """Load extracted facts and relationships from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        raise Exception(f"Failed to load data from {file_path}: {e}")


def parse_facts_and_relationships(data: Dict[str, Any]) -> tuple[List[Fact], List[FactRelation]]:
    """Parse facts and relationships from loaded data."""
    facts = []
    relationships = []

    # Parse facts
    if "facts" in data:
        for fact_data in data["facts"]:
            fact = Fact(
                id=fact_data["id"],
                subject=fact_data["subject"],
                object=fact_data["object"],
                approach=fact_data.get("approach"),
                solution=fact_data.get("solution"),
                remarks=fact_data.get("remarks"),
                source_chunk_id=fact_data["source_chunk_id"],
                source_document_id=fact_data["source_document_id"],
                extraction_confidence=fact_data.get("confidence", 1.0),
                fact_type=fact_data.get("fact_type", "general"),
                domain=fact_data.get("domain"),
                keywords=fact_data.get("keywords", []),
                created_at=datetime.fromisoformat(fact_data.get("created_at", datetime.utcnow().isoformat())),
                language=fact_data.get("language", "en")
            )
            facts.append(fact)

    # Parse relationships
    if "relationships" in data:
        for rel_data in data["relationships"]:
            relationship = FactRelation(
                id=rel_data.get("id", f"{rel_data['source_fact_id']}-{rel_data['target_fact_id']}"),
                source_fact_id=rel_data["source_fact_id"],
                target_fact_id=rel_data["target_fact_id"],
                relationship_type=rel_data.get("relationship_type", "RELATED"),
                confidence=rel_data.get("confidence", 1.0),
                description=rel_data.get("description", ""),
                created_at=datetime.fromisoformat(rel_data.get("created_at", datetime.utcnow().isoformat()))
            )
            relationships.append(relationship)

    return facts, relationships





async def ingest_to_neo4j(
    file_path: Path,
    facts: List[Fact],
    relationships: List[FactRelation],
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    clear_existing: bool = False
) -> None:
    """Ingest facts and relationships into Neo4j using FactOperations."""
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

        # Use FactOperations to handle the ingestion
        fact_operations = FactOperations(storage.driver)

        # Store facts
        if facts:
            fact_ids = await fact_operations.store_facts(facts)
            print(f"✓ Stored {len(fact_ids)} facts")

        # Store relationships
        if relationships:
            await fact_operations.store_fact_relations(relationships)
            print(f"✓ Stored {len(relationships)} fact relationships")

        print("✓ Fact ingestion completed successfully")

        # Print statistics
        stats = await storage.get_statistics()
        print("\n=== Graph Statistics ===")
        print(f"Total nodes: {stats.get('total_nodes', 0)}")
        print(f"Total relationships: {stats.get('total_relationships', 0)}")
        print(f"Node types: {stats.get('node_types', {})}")
        print(f"Relationship types: {stats.get('relationship_types', {})}")

    finally:
        await storage.disconnect()


async def main():
    parser = argparse.ArgumentParser(description="Ingest extracted fact data into Neo4j")
    parser.add_argument("input_file", help="Path to JSON file containing extracted facts and relationships")
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
        
        # Parse facts and relationships
        print("Parsing facts and relationships from data...")
        facts, relationships = parse_facts_and_relationships(data)
        print(f"✓ Parsed {len(facts)} facts and {len(relationships)} relationships")

        # Ingest to Neo4j
        print("\nIngesting to Neo4j...")
        await ingest_to_neo4j(
            file_path=input_path,
            facts=facts,
            relationships=relationships,
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