#!/usr/bin/env python3
"""Production script for ingesting extracted knowledge graph data into Neo4j.
This script loads entities and relations from a JSON file and stores them in Neo4j
with proper Document and DocumentChunk structure, exactly like normal ingestion.
"""

import json
import argparse
import asyncio
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from src.morag_graph.storage.neo4j_storage import Neo4jStorage
from src.morag_graph.models import Entity, Relation, Graph, Document, DocumentChunk


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


def create_document_from_file(file_path: Path, data: Dict[str, Any]) -> Document:
    """Create a Document node from the source file and extracted data."""
    # Calculate file checksum
    file_content = file_path.read_bytes()
    checksum = hashlib.sha256(file_content).hexdigest()
    
    # Get file stats
    file_stats = file_path.stat()
    
    # Create document ID based on file path and checksum
    document_id = f"doc_{checksum[:16]}"
    
    document = Document(
        id=document_id,
        source_file=str(file_path),
        file_name=file_path.name,
        file_size=file_stats.st_size,
        checksum=checksum,
        mime_type="application/json",
        ingestion_timestamp=datetime.now(),
        last_modified=datetime.fromtimestamp(file_stats.st_mtime),
        model=data.get("model", "unknown"),
        metadata={
            'file_name': file_path.name,
            'file_checksum': checksum,
            'ingestion_timestamp': datetime.now().isoformat(),
            'extraction_source': 'json_file'
        }
    )
    
    return document


def group_entities_into_chunks(entities: List[Entity]) -> List[Tuple[str, List[Entity]]]:
    """Group entities by their context to create document chunks.
    
    This mimics the chunking logic from file_ingestion.py but works with
    entities that may not have explicit chunk information.
    """
    # Group entities by their context or attributes
    chunks_data = []
    
    # For now, create chunks based on entity context or group all entities
    # This could be enhanced to use actual text chunks if available
    context_groups = {}
    
    for entity in entities:
        # Try to group by context from attributes
        context = entity.attributes.get('context', 'default')
        if context not in context_groups:
            context_groups[context] = []
        context_groups[context].append(entity)
    
    # Convert groups to chunks
    for context, group_entities in context_groups.items():
        # Create chunk text from entity contexts
        chunk_text = f"Document section containing entities: {', '.join([e.name for e in group_entities])}"
        if context != 'default':
            chunk_text = f"Context: {context}. " + chunk_text
        
        chunks_data.append((chunk_text, group_entities))
    
    # If no context grouping worked, create a single chunk with all entities
    if not chunks_data:
        chunk_text = f"Document content with {len(entities)} entities"
        chunks_data.append((chunk_text, entities))
    
    return chunks_data


async def ingest_to_neo4j(
    file_path: Path,
    data: Dict[str, Any],
    graph: Graph,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    clear_existing: bool = False
) -> None:
    """Ingest graph data into Neo4j with proper Document and DocumentChunk structure."""
    from src.morag_graph.storage.neo4j_storage import Neo4jConfig
    
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
        
        # Create and store Document node
        document = create_document_from_file(file_path, data)
        document_id = await storage.store_document(document)
        print(f"✓ Document stored: {document_id}")
        
        # Group entities into chunks
        entities_list = list(graph.entities.values())
        relations_list = list(graph.relations.values())
        chunks_data = group_entities_into_chunks(entities_list)
        
        # Create and store DocumentChunks with relationships
        chunk_ids = []
        total_entities_stored = 0
        
        for chunk_index, (chunk_text, chunk_entities) in enumerate(chunks_data):
            # Create DocumentChunk
            chunk = DocumentChunk(
                id=f"{document_id}_chunk_{chunk_index}",
                document_id=document_id,
                chunk_index=chunk_index,
                text=chunk_text,
                start_position=0,
                end_position=len(chunk_text),
                chunk_type="text",
                metadata={
                    'entity_count': len(chunk_entities),
                    'extraction_method': 'llm_based'
                }
            )
            
            # Store DocumentChunk
            chunk_id = await storage.store_document_chunk(chunk)
            chunk_ids.append(chunk_id)
            
            # Create Document -> CONTAINS -> DocumentChunk relationship
            await storage.create_document_contains_chunk_relation(document_id, chunk_id)
            
            # Store entities in this chunk
            entity_ids = await storage.store_entities(chunk_entities)
            total_entities_stored += len(chunk_entities)
            
            # Create DocumentChunk -> MENTIONS -> Entity relationships
            for entity in chunk_entities:
                await storage.create_chunk_mentions_entity_relation(
                    chunk_id, 
                    entity.id, 
                    chunk_text  # Use chunk text as context
                )
        
        print(f"✓ Created {len(chunk_ids)} document chunks")
        print(f"✓ Stored {total_entities_stored} entities")
        
        # Store entity-to-entity relations
        relation_ids = await storage.store_relations(relations_list)
        print(f"✓ Stored {len(relations_list)} relations")
        
        # Print statistics
        stats = await storage.get_statistics()
        print("\n=== Graph Statistics ===")
        print(f"Total nodes: {stats.get('total_nodes', 0)}")
        print(f"Total relationships: {stats.get('total_relationships', 0)}")
        print(f"Node types: {stats.get('node_types', {})}")
        print(f"Relationship types: {stats.get('relationship_types', {})}")
        
        print(f"\n✓ Successfully created 1 document, {len(chunk_ids)} chunks, {total_entities_stored} entities, {len(relations_list)} relations")
        
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
        
        # Create graph
        print("Creating graph from data...")
        graph = create_graph_from_data(data)
        print(f"✓ Graph created with {len(graph.entities)} entities and {len(graph.relations)} relations")
        
        # Ingest to Neo4j
        print("\nIngesting to Neo4j...")
        await ingest_to_neo4j(
            file_path=input_path,
            data=data,
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