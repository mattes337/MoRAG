#!/usr/bin/env python3
"""Mock test script for the new document structure refactoring.

This script demonstrates the new Document -> DocumentChunk -> Entity structure
without requiring a Neo4j database connection.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from src.morag_graph.models import Entity, Relation, Document, DocumentChunk
from src.morag_graph.models.types import RelationType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_test_data() -> tuple[List[Entity], List[Relation]]:
    """Load test data from test_document.json."""
    test_file = Path("test_document.json")
    
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


def group_entities_into_chunks(entities: List[Entity]) -> List[tuple[str, List[Entity]]]:
    """Group entities by their source text to create document chunks.
    
    Args:
        entities: List of entities to group
        
    Returns:
        List of tuples (chunk_text, entities_in_chunk)
    """
    # Group entities by source_text
    chunks_map = {}
    for entity in entities:
        source_text = getattr(entity, 'source_text', None) or "Unknown source"
        if source_text not in chunks_map:
            chunks_map[source_text] = []
        chunks_map[source_text].append(entity)
    
    # Convert to list of tuples
    return list(chunks_map.items())


def create_document_structure(entities: List[Entity], relations: List[Relation]) -> Dict[str, Any]:
    """Create the new document structure from entities and relations.
    
    Args:
        entities: List of entities
        relations: List of relations
        
    Returns:
        Dictionary containing the created structure
    """
    # Create Document
    document_id = "file_test_document"
    document = Document(
        id=document_id,
        source_file="test_document.json",
        file_name="test_document.json",
        file_size=1024,
        checksum="mock_checksum_123456",
        mime_type="application/json",
        ingestion_timestamp=datetime.now(),
        metadata={
            'file_name': 'test_document.json',
            'file_checksum': 'mock_checksum_123456',
            'ingestion_timestamp': datetime.now().isoformat()
        }
    )
    
    # Group entities into chunks
    chunks_data = group_entities_into_chunks(entities)
    
    # Create DocumentChunks
    chunks = []
    chunk_entity_mappings = []
    
    for chunk_index, (chunk_text, chunk_entities) in enumerate(chunks_data):
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
        chunks.append(chunk)
        
        # Store chunk-entity mappings
        for entity in chunk_entities:
            chunk_entity_mappings.append({
                'chunk_id': chunk.id,
                'entity_id': entity.id,
                'context': getattr(entity, 'source_text', chunk_text)
            })
    
    # Clean entities (remove document-specific fields)
    clean_entities = []
    for entity in entities:
        entity_dict = entity.model_dump()
        # Remove document-specific fields that are now handled by the document structure
        entity_dict.pop('source_text', None)
        entity_dict.pop('source_doc_id', None)
        clean_entities.append(Entity(**entity_dict))
    
    # Clean relations (remove document-specific fields)
    clean_relations = []
    for relation in relations:
        relation_dict = relation.model_dump()
        # Remove document-specific fields
        relation_dict.pop('source_text', None)
        relation_dict.pop('source_doc_id', None)
        clean_relations.append(Relation(**relation_dict))
    
    return {
        'document': document,
        'chunks': chunks,
        'entities': clean_entities,
        'relations': clean_relations,
        'chunk_entity_mappings': chunk_entity_mappings
    }


def analyze_structure(structure: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze the created document structure.
    
    Args:
        structure: The document structure
        
    Returns:
        Analysis results
    """
    document = structure['document']
    chunks = structure['chunks']
    entities = structure['entities']
    relations = structure['relations']
    chunk_entity_mappings = structure['chunk_entity_mappings']
    
    # Count relationships
    contains_relationships = len(chunks)  # Document -> CONTAINS -> Chunk
    mentions_relationships = len(chunk_entity_mappings)  # Chunk -> MENTIONS -> Entity
    entity_relationships = len(relations)  # Entity -> RELATION -> Entity
    
    total_relationships = contains_relationships + mentions_relationships + entity_relationships
    
    # Analyze entities by type
    entity_types = {}
    for entity in entities:
        entity_type = entity.type
        entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
    
    # Analyze relations by type
    relation_types = {}
    for relation in relations:
        relation_type = relation.type
        relation_types[relation_type] = relation_types.get(relation_type, 0) + 1
    
    # Analyze chunks
    chunk_analysis = []
    for chunk in chunks:
        chunk_entities = [m for m in chunk_entity_mappings if m['chunk_id'] == chunk.id]
        chunk_analysis.append({
            'chunk_id': chunk.id,
            'text_length': len(chunk.text),
            'entity_count': len(chunk_entities),
            'entities': [m['entity_id'] for m in chunk_entities]
        })
    
    return {
        'node_counts': {
            'documents': 1,
            'chunks': len(chunks),
            'entities': len(entities)
        },
        'relationship_counts': {
            'contains': contains_relationships,
            'mentions': mentions_relationships,
            'entity_relations': entity_relationships,
            'total': total_relationships
        },
        'entity_types': entity_types,
        'relation_types': relation_types,
        'chunk_analysis': chunk_analysis
    }


def demonstrate_graph_queries(structure: Dict[str, Any]) -> Dict[str, Any]:
    """Demonstrate graph queries that would be possible with the new structure.
    
    Args:
        structure: The document structure
        
    Returns:
        Query results
    """
    document = structure['document']
    chunks = structure['chunks']
    entities = structure['entities']
    relations = structure['relations']
    chunk_entity_mappings = structure['chunk_entity_mappings']
    
    # Create entity lookup
    entity_lookup = {e.id: e for e in entities}
    
    # Query 1: Find all entities mentioned in the document
    entities_in_document = []
    for mapping in chunk_entity_mappings:
        entity = entity_lookup.get(mapping['entity_id'])
        if entity:
            entities_in_document.append({
                'entity_name': entity.name,
                'entity_type': entity.type,
                'chunk_id': mapping['chunk_id'],
                'context': mapping['context'][:100] + '...' if len(mapping['context']) > 100 else mapping['context']
            })
    
    # Query 2: Find entities that co-occur in the same chunks
    co_occurring_entities = {}
    for chunk in chunks:
        chunk_entities = [m['entity_id'] for m in chunk_entity_mappings if m['chunk_id'] == chunk.id]
        if len(chunk_entities) > 1:
            for i, entity1_id in enumerate(chunk_entities):
                for entity2_id in chunk_entities[i+1:]:
                    entity1 = entity_lookup.get(entity1_id)
                    entity2 = entity_lookup.get(entity2_id)
                    if entity1 and entity2:
                        pair_key = f"{entity1.name} <-> {entity2.name}"
                        if pair_key not in co_occurring_entities:
                            co_occurring_entities[pair_key] = {
                                'entity1': entity1.name,
                                'entity2': entity2.name,
                                'co_occurrences': 0
                            }
                        co_occurring_entities[pair_key]['co_occurrences'] += 1
    
    # Query 3: Find entities related through explicit relations
    related_entities = []
    for relation in relations:
        source_entity = entity_lookup.get(relation.source_entity_id)
        target_entity = entity_lookup.get(relation.target_entity_id)
        if source_entity and target_entity:
            related_entities.append({
                'source': source_entity.name,
                'relation': relation.type,
                'target': target_entity.name,
                'confidence': relation.confidence
            })
    
    return {
        'entities_in_document': entities_in_document,
        'co_occurring_entities': list(co_occurring_entities.values()),
        'related_entities': related_entities
    }


def main():
    """Main test function."""
    logger.info("Starting mock document structure test...")
    
    try:
        # Load test data
        entities, relations = load_test_data()
        
        logger.info(f"Original structure: {len(entities)} entities, {len(relations)} relations")
        
        # Create new document structure
        logger.info("Creating new document structure...")
        structure = create_document_structure(entities, relations)
        
        # Analyze the structure
        analysis = analyze_structure(structure)
        
        logger.info("\n=== Document Structure Analysis ===")
        logger.info(f"Node counts: {analysis['node_counts']}")
        logger.info(f"Relationship counts: {analysis['relationship_counts']}")
        logger.info(f"Entity types: {analysis['entity_types']}")
        logger.info(f"Relation types: {analysis['relation_types']}")
        
        logger.info("\n=== Chunk Analysis ===")
        for chunk_info in analysis['chunk_analysis']:
            logger.info(f"Chunk {chunk_info['chunk_id']}: {chunk_info['entity_count']} entities, {chunk_info['text_length']} chars")
            logger.info(f"  Entities: {chunk_info['entities']}")
        
        # Demonstrate graph queries
        query_results = demonstrate_graph_queries(structure)
        
        logger.info("\n=== Graph Query Demonstrations ===")
        logger.info(f"Entities in document: {len(query_results['entities_in_document'])}")
        for entity_info in query_results['entities_in_document'][:5]:  # Show first 5
            logger.info(f"  {entity_info['entity_name']} ({entity_info['entity_type']}) in {entity_info['chunk_id']}")
        
        logger.info(f"\nCo-occurring entities: {len(query_results['co_occurring_entities'])}")
        for co_occurrence in query_results['co_occurring_entities'][:5]:  # Show first 5
            logger.info(f"  {co_occurrence['entity1']} <-> {co_occurrence['entity2']}: {co_occurrence['co_occurrences']} times")
        
        logger.info(f"\nExplicit relations: {len(query_results['related_entities'])}")
        for relation_info in query_results['related_entities'][:5]:  # Show first 5
            logger.info(f"  {relation_info['source']} --{relation_info['relation']}--> {relation_info['target']} (conf: {relation_info['confidence']})")
        
        # Calculate relationship increase
        original_relations = len(relations)
        new_total_relations = analysis['relationship_counts']['total']
        increase = new_total_relations - original_relations
        
        logger.info("\n=== Relationship Count Comparison ===")
        logger.info(f"Original entity-to-entity relations: {original_relations}")
        logger.info(f"New total relations: {new_total_relations}")
        logger.info(f"  - CONTAINS relations: {analysis['relationship_counts']['contains']}")
        logger.info(f"  - MENTIONS relations: {analysis['relationship_counts']['mentions']}")
        logger.info(f"  - Entity relations: {analysis['relationship_counts']['entity_relations']}")
        logger.info(f"Relationship increase: +{increase} ({((increase/original_relations)*100):.1f}% increase)")
        
        logger.info("\n=== Benefits of New Structure ===")
        logger.info("1. Document-level metadata is now properly separated")
        logger.info("2. Entities are global and can be shared across documents")
        logger.info("3. Document chunks provide context for entity mentions")
        logger.info("4. Enables document-to-document connections through shared entities")
        logger.info("5. Supports more sophisticated graph traversal queries")
        
        logger.info("\n=== Mock test completed successfully! ===")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    main()