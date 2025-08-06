"""Atomic storage service for batch database operations."""

import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import structlog

from ..models.atomic_storage_batch import AtomicStorageBatch, AtomicStorageResult
from ..storage.neo4j_storage import Neo4jStorage
from ..storage.qdrant_storage import QdrantStorage


class AtomicStorageService:
    """Service for atomic storage of batched data to databases."""
    
    def __init__(self, neo4j_storage: Optional[Neo4jStorage] = None, qdrant_storage: Optional[QdrantStorage] = None):
        """Initialize the atomic storage service.
        
        Args:
            neo4j_storage: Neo4j storage instance
            qdrant_storage: Qdrant storage instance
        """
        self.neo4j_storage = neo4j_storage
        self.qdrant_storage = qdrant_storage
        self.logger = structlog.get_logger(__name__)
    
    async def store_batch(self, batch: AtomicStorageBatch) -> AtomicStorageResult:
        """Store a complete batch atomically.
        
        Args:
            batch: Batch of data to store
            
        Returns:
            Storage result with success status and details
        """
        start_time = time.time()
        result = AtomicStorageResult(
            success=True,
            batch_id=batch.batch_id,
            stored_counts={}
        )
        
        try:
            # Validate batch consistency first
            validation_errors = batch.validate_consistency()
            if validation_errors:
                for error in validation_errors:
                    result.add_error(f"Validation error: {error}")
                return result
            
            self.logger.info(
                "Starting atomic storage",
                batch_id=batch.batch_id,
                total_items=batch.get_total_items()
            )
            
            # Store to Neo4j if available
            if self.neo4j_storage:
                neo4j_result = await self._store_to_neo4j(batch)
                result.database_results['neo4j'] = neo4j_result
                if not neo4j_result.get('success', False):
                    result.add_error(f"Neo4j storage failed: {neo4j_result.get('error', 'Unknown error')}")
            
            # Store to Qdrant if available
            if self.qdrant_storage:
                qdrant_result = await self._store_to_qdrant(batch)
                result.database_results['qdrant'] = qdrant_result
                if not qdrant_result.get('success', False):
                    result.add_error(f"Qdrant storage failed: {qdrant_result.get('error', 'Unknown error')}")
            
            # Set final counts
            result.stored_counts = batch.get_total_items()
            
        except Exception as e:
            self.logger.error("Atomic storage failed", error=str(e), batch_id=batch.batch_id)
            result.add_error(f"Storage operation failed: {str(e)}")
        
        finally:
            result.storage_duration_seconds = time.time() - start_time
            
        self.logger.info(
            "Atomic storage completed",
            batch_id=batch.batch_id,
            success=result.success,
            duration=result.storage_duration_seconds,
            stored_counts=result.stored_counts
        )
        
        return result
    
    async def _store_to_neo4j(self, batch: AtomicStorageBatch) -> Dict[str, Any]:
        """Store batch data to Neo4j atomically.

        Args:
            batch: Batch to store

        Returns:
            Storage result dictionary
        """
        try:
            counts = {
                'documents': 0,
                'chunks': 0,
                'entities': 0,
                'facts': 0,
                'relations': 0,
                'fact_relations': 0,
                'chunk_fact_relations': 0,
                'document_chunk_relations': 0
            }

            # Store documents using batch operations
            if batch.documents.documents:
                document_ids = await self.neo4j_storage.store_documents(batch.documents.documents)
                counts['documents'] = len(document_ids)

            # Store chunks using batch operations
            if batch.documents.chunks:
                chunk_ids = await self.neo4j_storage.store_document_chunks(batch.documents.chunks)
                counts['chunks'] = len(chunk_ids)

                # Create document-chunk relations
                for chunk in batch.documents.chunks:
                    await self.neo4j_storage.create_document_contains_chunk_relation(
                        chunk.document_id, chunk.id
                    )
                    counts['document_chunk_relations'] += 1

            # Store entities using batch operations
            if batch.entities.entities:
                entity_ids = await self.neo4j_storage.store_entities(batch.entities.entities)
                counts['entities'] = len(entity_ids)

            # Store facts using batch operations
            if batch.facts.facts:
                fact_ids = await self.neo4j_storage.store_facts(batch.facts.facts)
                counts['facts'] = len(fact_ids)

            # Store relations using batch operations
            if batch.relations.relations:
                relation_ids = await self.neo4j_storage.store_relations(batch.relations.relations)
                counts['relations'] = len(relation_ids)

            # Store fact relations
            if batch.facts.fact_relations:
                from ..storage.neo4j_operations.fact_operations import FactOperations
                fact_ops = FactOperations(self.neo4j_storage.driver, self.neo4j_storage.config.database)
                await fact_ops.store_fact_relations(batch.facts.fact_relations)
                counts['fact_relations'] = len(batch.facts.fact_relations)

            # Create chunk-fact relations
            for chunk_id, fact_ids in batch.facts.chunk_mappings.items():
                for fact_id in fact_ids:
                    await self.neo4j_storage.create_chunk_contains_fact_relation(
                        chunk_id, fact_id, context="Fact extracted from chunk"
                    )
                    counts['chunk_fact_relations'] += 1

            return {
                'success': True,
                'stored_counts': counts
            }

        except Exception as e:
            self.logger.error("Neo4j atomic storage failed", error=str(e))
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _store_to_qdrant(self, batch: AtomicStorageBatch) -> Dict[str, Any]:
        """Store batch data to Qdrant atomically.
        
        Args:
            batch: Batch to store
            
        Returns:
            Storage result dictionary
        """
        try:
            counts = {
                'chunk_vectors': 0,
                'entity_vectors': 0,
                'fact_vectors': 0
            }
            
            # Store chunk embeddings
            if batch.documents.chunk_embeddings:
                chunk_vectors = []
                chunk_metadata = []
                
                for chunk in batch.documents.chunks:
                    if chunk.id in batch.documents.chunk_embeddings:
                        chunk_vectors.append(batch.documents.chunk_embeddings[chunk.id])
                        chunk_metadata.append({
                            'chunk_id': chunk.id,
                            'document_id': chunk.document_id,
                            'chunk_index': chunk.chunk_index,
                            'text': chunk.text,
                            'type': 'document_chunk'
                        })
                
                if chunk_vectors:
                    await self.qdrant_storage.store_vectors(chunk_vectors, chunk_metadata)
                    counts['chunk_vectors'] = len(chunk_vectors)
            
            # Store entity embeddings
            if batch.entities.embeddings:
                entity_vectors = []
                entity_metadata = []
                
                for entity in batch.entities.entities:
                    if entity.id in batch.entities.embeddings:
                        entity_vectors.append(batch.entities.embeddings[entity.id])
                        entity_metadata.append({
                            'entity_id': entity.id,
                            'name': entity.name,
                            'type': str(entity.type),
                            'confidence': entity.confidence,
                            'entity_type': 'entity'
                        })
                
                if entity_vectors:
                    await self.qdrant_storage.store_vectors(entity_vectors, entity_metadata)
                    counts['entity_vectors'] = len(entity_vectors)
            
            # Store fact embeddings
            if batch.facts.embeddings:
                fact_vectors = []
                fact_metadata = []
                
                for fact in batch.facts.facts:
                    if fact.id in batch.facts.embeddings:
                        fact_vectors.append(batch.facts.embeddings[fact.id])
                        fact_metadata.append({
                            'fact_id': fact.id,
                            'subject': fact.subject,
                            'object': fact.object,
                            'domain': fact.domain,
                            'type': 'fact'
                        })
                
                if fact_vectors:
                    await self.qdrant_storage.store_vectors(fact_vectors, fact_metadata)
                    counts['fact_vectors'] = len(fact_vectors)
            
            return {
                'success': True,
                'stored_counts': counts
            }
            
        except Exception as e:
            self.logger.error("Qdrant atomic storage failed", error=str(e))
            return {
                'success': False,
                'error': str(e)
            }
    
    def save_batch_to_file(self, batch: AtomicStorageBatch, file_path: Path) -> None:
        """Save batch to JSON file for persistence.
        
        Args:
            batch: Batch to save
            file_path: Path to save the file
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(batch.model_dump(), f, indent=2, default=str)
            
            self.logger.info("Batch saved to file", file_path=str(file_path), batch_id=batch.batch_id)
            
        except Exception as e:
            self.logger.error("Failed to save batch to file", error=str(e), file_path=str(file_path))
            raise
    
    def load_batch_from_file(self, file_path: Path) -> AtomicStorageBatch:
        """Load batch from JSON file.
        
        Args:
            file_path: Path to load the file from
            
        Returns:
            Loaded batch
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            batch = AtomicStorageBatch(**data)
            self.logger.info("Batch loaded from file", file_path=str(file_path), batch_id=batch.batch_id)
            return batch
            
        except Exception as e:
            self.logger.error("Failed to load batch from file", error=str(e), file_path=str(file_path))
            raise
