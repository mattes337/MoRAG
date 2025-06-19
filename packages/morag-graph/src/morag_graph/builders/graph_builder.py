"""Graph builder for constructing knowledge graphs from documents."""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..extraction.entity_extractor import EntityExtractor
from ..extraction.relation_extractor import RelationExtractor
from ..extraction.base import LLMConfig
from ..storage.base import BaseStorage
from ..models.entity import Entity
from ..models.relation import Relation
from ..models.document import Document
from ..models.document_chunk import DocumentChunk


class GraphBuildError(Exception):
    """Exception raised when graph building fails."""
    pass


@dataclass
class GraphBuildResult:
    """Result of graph building operation."""
    document_id: str
    entities_created: int
    relations_created: int
    entity_ids: List[str]
    relation_ids: List[str]
    processing_time: float = 0.0
    chunks_processed: int = 0
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class GraphBuilder:
    """Builder for constructing knowledge graphs from documents.
    
    This class orchestrates the process of extracting entities and relations
    from documents and storing them in a graph database.
    """
    
    def __init__(
        self,
        storage: BaseStorage,
        llm_config: Optional[LLMConfig] = None,
        entity_types: Optional[Dict[str, str]] = None,
        relation_types: Optional[Dict[str, str]] = None
    ):
        """Initialize the graph builder.
        
        Args:
            storage: Graph storage backend
            llm_config: Configuration for LLM-based extraction
            entity_types: Custom entity types for extraction
            relation_types: Custom relation types for extraction
        """
        self.storage = storage
        self.logger = logging.getLogger(__name__)
        
        # Initialize extractors
        self.entity_extractor = EntityExtractor(
            config=llm_config,
            entity_types=entity_types
        )
        
        self.relation_extractor = RelationExtractor(
            config=llm_config,
            relation_types=relation_types
        )
    
    async def process_document(
        self,
        content: str,
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> GraphBuildResult:
        """Process a document and build graph entities and relations.
        
        Args:
            content: Document content to process
            document_id: Unique identifier for the document
            metadata: Optional metadata about the document
            
        Returns:
            GraphBuildResult with processing results
            
        Raises:
            GraphBuildError: If processing fails
        """
        import time
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting graph processing for document {document_id}")
            
            # Extract entities from content
            entities = await self.entity_extractor.extract(
                content,
                source_doc_id=document_id
            )
            
            # Extract relations from content and entities
            relations = await self.relation_extractor.extract(
                content,
                entities=entities
            )
            
            # Note: Entity and Relation models don't have metadata fields
            # Metadata is handled at the document/chunk level
            
            # Store entities and relations
            result = await self._store_entities_and_relations(
                entities, relations, document_id
            )
            
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            
            self.logger.info(
                f"Completed graph processing for document {document_id}: "
                f"{result.entities_created} entities, {result.relations_created} relations "
                f"in {processing_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Failed to process document {document_id}: {str(e)}"
            self.logger.error(error_msg)
            
            return GraphBuildResult(
                document_id=document_id,
                entities_created=0,
                relations_created=0,
                entity_ids=[],
                relation_ids=[],
                processing_time=processing_time,
                errors=[error_msg]
            )
    
    async def process_document_chunks(
        self,
        chunks: List[DocumentChunk],
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> GraphBuildResult:
        """Process document chunks and build graph entities and relations.
        
        Args:
            chunks: List of document chunks to process
            document_id: Unique identifier for the document
            metadata: Optional metadata about the document
            
        Returns:
            GraphBuildResult with processing results
        """
        import time
        start_time = time.time()
        
        try:
            self.logger.info(
                f"Starting graph processing for document {document_id} "
                f"with {len(chunks)} chunks"
            )
            
            all_entities = []
            all_relations = []
            errors = []
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                try:
                    # Extract entities from chunk
                    entities = await self.entity_extractor.extract(
                        chunk.text,
                        source_doc_id=document_id
                    )
                    
                    # Extract relations from chunk
                    relations = await self.relation_extractor.extract(
                        chunk.text,
                        entities=entities
                    )
                    
                    # Add chunk metadata
                    chunk_metadata = {
                        'chunk_id': chunk.id,
                        'chunk_index': i,
                        'chunk_type': chunk.chunk_type
                    }
                    
                    # Add chunk metadata if available
                    if chunk.metadata:
                        chunk_metadata.update(chunk.metadata)
                    
                    if metadata:
                        chunk_metadata.update(metadata)
                    
                    # Note: Entity and Relation models don't have metadata fields
                    # Metadata is handled at the document/chunk level
                    
                    all_entities.extend(entities)
                    all_relations.extend(relations)
                    
                    self.logger.debug(
                        f"Processed chunk {i+1}/{len(chunks)}: "
                        f"{len(entities)} entities, {len(relations)} relations"
                    )
                    
                except Exception as e:
                    error_msg = f"Failed to process chunk {i}: {str(e)}"
                    self.logger.warning(error_msg)
                    errors.append(error_msg)
                    continue
            
            # Store all entities and relations
            result = await self._store_entities_and_relations(
                all_entities, all_relations, document_id
            )
            
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            result.chunks_processed = len(chunks)
            result.errors = errors
            
            self.logger.info(
                f"Completed graph processing for document {document_id}: "
                f"{result.entities_created} entities, {result.relations_created} relations "
                f"from {len(chunks)} chunks in {processing_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Failed to process document chunks {document_id}: {str(e)}"
            self.logger.error(error_msg)
            
            return GraphBuildResult(
                document_id=document_id,
                entities_created=0,
                relations_created=0,
                entity_ids=[],
                relation_ids=[],
                processing_time=processing_time,
                chunks_processed=len(chunks),
                errors=[error_msg]
            )
    
    async def process_documents_batch(
        self,
        documents: List[tuple[str, str, Optional[Dict[str, Any]]]]
    ) -> List[GraphBuildResult]:
        """Process multiple documents in parallel.
        
        Args:
            documents: List of (content, document_id, metadata) tuples
            
        Returns:
            List of GraphBuildResult for each document
        """
        self.logger.info(f"Starting batch processing of {len(documents)} documents")
        
        tasks = [
            self.process_document(content, doc_id, metadata)
            for content, doc_id, metadata in documents
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                doc_id = documents[i][1]
                error_msg = f"Failed to process document {doc_id}: {str(result)}"
                self.logger.error(error_msg)
                processed_results.append(GraphBuildResult(
                    document_id=doc_id,
                    entities_created=0,
                    relations_created=0,
                    entity_ids=[],
                    relation_ids=[],
                    errors=[error_msg]
                ))
            else:
                processed_results.append(result)
        
        self.logger.info(f"Completed batch processing of {len(documents)} documents")
        return processed_results
    
    async def _store_entities_and_relations(
        self,
        entities: List[Entity],
        relations: List[Relation],
        document_id: str
    ) -> GraphBuildResult:
        """Store extracted entities and relations in the graph database.
        
        Args:
            entities: List of entities to store
            relations: List of relations to store
            document_id: Source document ID
            
        Returns:
            GraphBuildResult with storage results
        """
        try:
            # Store entities
            entity_ids = []
            if entities:
                await self.storage.store_entities(entities)
                entity_ids = [entity.id for entity in entities]
            
            # Store relations
            relation_ids = []
            if relations:
                await self.storage.store_relations(relations)
                relation_ids = [relation.id for relation in relations]
            
            return GraphBuildResult(
                document_id=document_id,
                entities_created=len(entities),
                relations_created=len(relations),
                entity_ids=entity_ids,
                relation_ids=relation_ids
            )
            
        except Exception as e:
            error_msg = f"Failed to store entities and relations: {str(e)}"
            self.logger.error(error_msg)
            raise GraphBuildError(error_msg) from e
    
    async def close(self):
        """Close the graph builder and its resources."""
        if self.storage:
            await self.storage.close()