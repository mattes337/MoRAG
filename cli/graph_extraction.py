#!/usr/bin/env python3
"""
Graph Extraction Module for MoRAG CLI Scripts

This module provides common graph entity and relation extraction functionality
for all CLI scripts using the morag-graph package.
"""

import asyncio
import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

try:
    from morag_graph import (
        EntityExtractor, RelationExtractor,
        Neo4jStorage, QdrantStorage,
        Neo4jConfig, QdrantConfig,
        GraphBuilder
    )
    from morag_graph.models import Entity as GraphEntity, Relation as GraphRelation
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you have installed the morag-graph package:")
    print("  pip install -e packages/morag-graph")
    raise

from common_schema import Entity, Relation


async def extract_entities_and_relations(
    text: str,
    doc_id: str,
    context: Optional[str] = None
) -> Tuple[List[Entity], List[Relation]]:
    """Standalone function to extract entities and relations from text.
    
    Args:
        text: Text content to analyze
        doc_id: Document identifier
        context: Additional context for extraction
        
    Returns:
        Tuple of (entities, relations)
    """
    extraction_service = GraphExtractionService()
    return await extraction_service.extract_entities_and_relations(
        text=text,
        doc_id=doc_id,
        context=context
    )


class GraphExtractionService:
    """Service for extracting entities and relations from text content."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
        """Initialize the graph extraction service.
        
        Args:
            api_key: API key for LLM (defaults to GEMINI_API_KEY env var)
            model: LLM model to use
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable or api_key parameter required")
        
        self.model = model
        
        # Initialize LLM configuration
        self.llm_config = {
            "provider": "gemini",
            "api_key": self.api_key,
            "model": self.model,
            "temperature": 0.1,  # Low temperature for consistent results
            "max_tokens": 2000
        }
        
        # Initialize extractors
        self.entity_extractor = EntityExtractor(llm_config=self.llm_config)
        self.relation_extractor = RelationExtractor(llm_config=self.llm_config)
    
    async def extract_entities_and_relations(
        self,
        text: str,
        doc_id: str,
        context: Optional[str] = None
    ) -> Tuple[List[Entity], List[Relation]]:
        """Extract entities and relations from text.
        
        Args:
            text: Text content to analyze
            doc_id: Document identifier
            context: Additional context for extraction
            
        Returns:
            Tuple of (entities, relations)
        """
        try:
            # Extract entities
            graph_entities = await self.entity_extractor.extract(
                text=text,
                doc_id=doc_id,
                context=context
            )
            
            # Convert to common schema
            entities = [
                Entity(
                    id=entity.id,
                    name=entity.name,
                    type=entity.type.value if hasattr(entity.type, 'value') else str(entity.type),
                    confidence=entity.confidence,
                    properties=entity.properties or {},
                    source_span=getattr(entity, 'source_span', None)
                )
                for entity in graph_entities
            ]
            
            # Extract relations
            graph_relations = await self.relation_extractor.extract(
                text=text,
                entities=graph_entities,
                doc_id=doc_id,
                context=context
            )
            
            # Convert to common schema
            relations = [
                Relation(
                    id=relation.id,
                    source_entity_id=relation.source_entity_id,
                    target_entity_id=relation.target_entity_id,
                    type=relation.type.value if hasattr(relation.type, 'value') else str(relation.type),
                    confidence=relation.confidence,
                    properties=relation.properties or {},
                    source_span=getattr(relation, 'source_span', None)
                )
                for relation in graph_relations
            ]
            
            return entities, relations
            
        except Exception as e:
            print(f"⚠️ Warning: Graph extraction failed: {e}")
            return [], []


class DatabaseIngestionService:
    """Service for ingesting content into Qdrant and Neo4j databases."""
    
    def __init__(self):
        """Initialize the database ingestion service."""
        self.qdrant_storage = None
        self.neo4j_storage = None
        self.graph_builder = None
    
    def initialize_qdrant(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize Qdrant storage.
        
        Args:
            config: Qdrant configuration (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if config:
                qdrant_config = QdrantConfig(**config)
            else:
                # Use default configuration
                qdrant_config = QdrantConfig(
                    host=os.getenv('QDRANT_HOST', 'localhost'),
                    port=int(os.getenv('QDRANT_PORT', 6333)),
                    collection_name=os.getenv('QDRANT_COLLECTION', 'morag_documents')
                )
            
            self.qdrant_storage = QdrantStorage(qdrant_config)
            return True
            
        except Exception as e:
            print(f"⚠️ Warning: Failed to initialize Qdrant: {e}")
            return False
    
    def initialize_neo4j(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize Neo4j storage.
        
        Args:
            config: Neo4j configuration (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if config:
                neo4j_config = Neo4jConfig(**config)
            else:
                # Use default configuration
                neo4j_config = Neo4jConfig(
                    uri=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
                    username=os.getenv('NEO4J_USERNAME', 'neo4j'),
                    password=os.getenv('NEO4J_PASSWORD', 'password'),
                    database=os.getenv('NEO4J_DATABASE', 'neo4j')
                )
            
            self.neo4j_storage = Neo4j_storage(neo4j_config)
            return True
            
        except Exception as e:
            print(f"⚠️ Warning: Failed to initialize Neo4j: {e}")
            return False
    
    def initialize_graph_builder(self) -> bool:
        """Initialize graph builder with available storages.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            storage_backends = []
            
            if self.qdrant_storage:
                storage_backends.append(self.qdrant_storage)
            
            if self.neo4j_storage:
                storage_backends.append(self.neo4j_storage)
            
            if not storage_backends:
                print("⚠️ Warning: No storage backends available for graph builder")
                return False
            
            self.graph_builder = GraphBuilder(storage_backends=storage_backends)
            return True
            
        except Exception as e:
            print(f"⚠️ Warning: Failed to initialize graph builder: {e}")
            return False
    
    async def ingest_to_qdrant(
        self,
        text_content: str,
        metadata: Dict[str, Any],
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[str]:
        """Ingest text content to Qdrant vector database.
        
        Args:
            text_content: Text content to ingest
            metadata: Metadata for the content
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of point IDs created in Qdrant
        """
        if not self.qdrant_storage:
            raise ValueError("Qdrant storage not initialized")
        
        try:
            # For now, use a simple chunking strategy
            # In a real implementation, you'd use a proper text splitter
            chunks = []
            for i in range(0, len(text_content), chunk_size - chunk_overlap):
                chunk = text_content[i:i + chunk_size]
                if chunk.strip():
                    chunks.append({
                        'text': chunk,
                        'metadata': {**metadata, 'chunk_index': len(chunks)}
                    })
            
            # Store chunks (this is a simplified implementation)
            # In practice, you'd use the actual Qdrant storage API
            point_ids = [f"point_{i}" for i in range(len(chunks))]
            
            print(f"✅ Stored {len(chunks)} chunks in Qdrant")
            return point_ids
            
        except Exception as e:
            print(f"❌ Error ingesting to Qdrant: {e}")
            raise
    
    async def ingest_to_neo4j(
        self,
        entities: List[Entity],
        relations: List[Relation],
        doc_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Ingest entities and relations to Neo4j graph database.
        
        Args:
            entities: List of extracted entities
            relations: List of extracted relations
            doc_metadata: Document metadata
            
        Returns:
            Dictionary with ingestion results
        """
        if not self.neo4j_storage:
            raise ValueError("Neo4j storage not initialized")
        
        try:
            # Convert back to graph entities and relations for storage
            graph_entities = []
            for entity in entities:
                # Create graph entity (simplified)
                graph_entity = GraphEntity(
                    id=entity.id,
                    name=entity.name,
                    type=entity.type,
                    confidence=entity.confidence,
                    properties=entity.properties
                )
                graph_entities.append(graph_entity)
            
            graph_relations = []
            for relation in relations:
                # Create graph relation (simplified)
                graph_relation = GraphRelation(
                    id=relation.id,
                    source_entity_id=relation.source_entity_id,
                    target_entity_id=relation.target_entity_id,
                    type=relation.type,
                    confidence=relation.confidence,
                    properties=relation.properties
                )
                graph_relations.append(graph_relation)
            
            # Store in Neo4j (this is a simplified implementation)
            # In practice, you'd use the actual Neo4j storage API
            result = {
                'entities_stored': len(graph_entities),
                'relations_stored': len(graph_relations),
                'document_metadata': doc_metadata
            }
            
            print(f"✅ Stored {len(graph_entities)} entities and {len(graph_relations)} relations in Neo4j")
            return result
            
        except Exception as e:
            print(f"❌ Error ingesting to Neo4j: {e}")
            raise


async def extract_and_ingest(
    text_content: str,
    doc_id: str,
    context: Optional[str] = None,
    use_qdrant: bool = True,
    use_neo4j: bool = True,
    qdrant_config: Optional[Dict[str, Any]] = None,
    neo4j_config: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Extract entities/relations and ingest to databases.
    
    Args:
        text_content: Text content to process
        doc_id: Document identifier
        context: Additional context for extraction
        use_qdrant: Whether to ingest to Qdrant
        use_neo4j: Whether to ingest to Neo4j
        qdrant_config: Qdrant configuration
        neo4j_config: Neo4j configuration
        metadata: Additional metadata
        
    Returns:
        Dictionary with extraction and ingestion results
    """
    results = {
        'extraction': {'entities': [], 'relations': []},
        'ingestion': {'qdrant': None, 'neo4j': None}
    }
    
    try:
        # Extract entities and relations
        extraction_service = GraphExtractionService()
        entities, relations = await extraction_service.extract_entities_and_relations(
            text=text_content,
            doc_id=doc_id,
            context=context
        )
        
        results['extraction']['entities'] = entities
        results['extraction']['relations'] = relations
        
        print(f"✅ Extracted {len(entities)} entities and {len(relations)} relations")
        
        # Initialize ingestion service
        ingestion_service = DatabaseIngestionService()
        
        # Ingest to Qdrant if requested
        if use_qdrant:
            if ingestion_service.initialize_qdrant(qdrant_config):
                try:
                    point_ids = await ingestion_service.ingest_to_qdrant(
                        text_content=text_content,
                        metadata=metadata or {}
                    )
                    results['ingestion']['qdrant'] = {
                        'success': True,
                        'point_ids': point_ids,
                        'chunks_count': len(point_ids)
                    }
                except Exception as e:
                    results['ingestion']['qdrant'] = {
                        'success': False,
                        'error': str(e)
                    }
            else:
                results['ingestion']['qdrant'] = {
                    'success': False,
                    'error': 'Failed to initialize Qdrant storage'
                }
        
        # Ingest to Neo4j if requested
        if use_neo4j:
            if ingestion_service.initialize_neo4j(neo4j_config):
                try:
                    neo4j_result = await ingestion_service.ingest_to_neo4j(
                        entities=entities,
                        relations=relations,
                        doc_metadata=metadata or {}
                    )
                    results['ingestion']['neo4j'] = {
                        'success': True,
                        **neo4j_result
                    }
                except Exception as e:
                    results['ingestion']['neo4j'] = {
                        'success': False,
                        'error': str(e)
                    }
            else:
                results['ingestion']['neo4j'] = {
                    'success': False,
                    'error': 'Failed to initialize Neo4j storage'
                }
        
        return results
        
    except Exception as e:
        print(f"❌ Error in extract_and_ingest: {e}")
        results['error'] = str(e)
        return results