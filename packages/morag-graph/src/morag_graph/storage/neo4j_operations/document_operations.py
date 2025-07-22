"""Document and chunk operations for Neo4j storage."""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from ...models import Document, DocumentChunk
from .base_operations import BaseOperations

logger = logging.getLogger(__name__)


class DocumentOperations(BaseOperations):
    """Handles document and chunk storage/retrieval operations."""
    
    async def store_document_with_unified_id(self, document: Document) -> str:
        """Store document with unified ID format.
        
        Args:
            document: Document instance with unified ID
            
        Returns:
            Document ID
        """
        query = """
        MERGE (d:Document {id: $id})
        SET d.title = $title,
            d.content = $content,
            d.source_type = $source_type,
            d.source_path = $source_path,
            d.checksum = $checksum,
            d.ingestion_timestamp = $ingestion_timestamp,
            d.metadata = $metadata,
            d.updated_at = datetime()
        RETURN d.id as document_id
        """
        
        result = await self._execute_query(
            query,
            {
                "id": document.id,
                "title": document.title,
                "content": document.content,
                "source_type": document.source_type,
                "source_path": document.source_path,
                "checksum": document.checksum,
                "ingestion_timestamp": document.ingestion_timestamp.isoformat(),
                "metadata": document.metadata
            }
        )
        
        return result[0]['document_id']
    
    async def store_chunk_with_unified_id(self, chunk: DocumentChunk) -> str:
        """Store document chunk with unified ID format.
        
        Args:
            chunk: DocumentChunk instance with unified ID
            
        Returns:
            Chunk ID
        """
        query = """
        MERGE (c:DocumentChunk {id: $id})
        SET c.document_id = $document_id,
            c.chunk_index = $chunk_index,
            c.content = $content,
            c.start_char = $start_char,
            c.end_char = $end_char,
            c.metadata = $metadata,
            c.updated_at = datetime()
        RETURN c.id as chunk_id
        """
        
        result = await self._execute_query(
            query,
            {
                "id": chunk.id,
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "metadata": chunk.metadata
            }
        )
        
        return result[0]['chunk_id']
    
    async def get_document_by_unified_id(self, document_id: str) -> Optional[Document]:
        """Retrieve document by unified ID.
        
        Args:
            document_id: Unified document ID
            
        Returns:
            Document instance or None if not found
        """
        query = """
        MATCH (d:Document {id: $document_id})
        RETURN d
        """
        
        result = await self._execute_query(query, {"document_id": document_id})
        
        if not result:
            return None
        
        doc_data = result[0]['d']
        return Document(
            id=doc_data['id'],
            title=doc_data['title'],
            content=doc_data['content'],
            source_type=doc_data['source_type'],
            source_path=doc_data['source_path'],
            checksum=doc_data['checksum'],
            ingestion_timestamp=datetime.fromisoformat(doc_data['ingestion_timestamp']),
            metadata=doc_data['metadata']
        )
    
    async def get_chunks_by_document_id(self, document_id: str) -> List[DocumentChunk]:
        """Get all chunks for a document by unified ID.
        
        Args:
            document_id: Unified document ID
            
        Returns:
            List of DocumentChunk instances
        """
        query = """
        MATCH (c:DocumentChunk {document_id: $document_id})
        RETURN c
        ORDER BY c.chunk_index
        """
        
        result = await self._execute_query(query, {"document_id": document_id})
        
        chunks = []
        for record in result:
            chunk_data = record['c']
            chunks.append(DocumentChunk(
                id=chunk_data['id'],
                document_id=chunk_data['document_id'],
                chunk_index=chunk_data['chunk_index'],
                content=chunk_data['content'],
                start_char=chunk_data['start_char'],
                end_char=chunk_data['end_char'],
                metadata=chunk_data['metadata']
            ))
        
        return chunks
    
    async def store_document(self, document: Document) -> str:
        """Store a document in Neo4J.
        
        Args:
            document: Document to store
            
        Returns:
            Document ID
        """
        query = """
        MERGE (d:Document {id: $id})
        SET d.title = $title,
            d.content = $content,
            d.source_type = $source_type,
            d.source_path = $source_path,
            d.checksum = $checksum,
            d.ingestion_timestamp = $ingestion_timestamp,
            d.metadata = $metadata
        RETURN d.id as id
        """
        
        parameters = {
            "id": document.id,
            "title": document.title,
            "content": document.content,
            "source_type": document.source_type,
            "source_path": document.source_path,
            "checksum": document.checksum,
            "ingestion_timestamp": document.ingestion_timestamp.isoformat(),
            "metadata": document.metadata
        }
        
        result = await self._execute_query(query, parameters)
        return result[0]["id"] if result else document.id
    
    async def store_document_chunk(self, chunk: DocumentChunk) -> str:
        """Store a document chunk in Neo4J.
        
        Args:
            chunk: DocumentChunk to store
            
        Returns:
            Chunk ID
        """
        query = """
        MERGE (c:DocumentChunk {id: $id})
        SET c.document_id = $document_id,
            c.chunk_index = $chunk_index,
            c.content = $content,
            c.start_char = $start_char,
            c.end_char = $end_char,
            c.metadata = $metadata
        RETURN c.id as id
        """
        
        parameters = {
            "id": chunk.id,
            "document_id": chunk.document_id,
            "chunk_index": chunk.chunk_index,
            "content": chunk.content,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            "metadata": chunk.metadata
        }
        
        result = await self._execute_query(query, parameters)
        return result[0]["id"] if result else chunk.id
    
    async def create_document_contains_chunk_relation(self, document_id: str, chunk_id: str) -> None:
        """Create a CONTAINS relationship between a document and a chunk.
        
        Args:
            document_id: ID of the document
            chunk_id: ID of the chunk
        """
        query = """
        MATCH (d:Document {id: $document_id}), (c:DocumentChunk {id: $chunk_id})
        MERGE (d)-[:CONTAINS]->(c)
        """
        await self._execute_query(query, {
            "document_id": document_id,
            "chunk_id": chunk_id
        })
    
    async def get_document_checksum(self, document_id: str) -> Optional[str]:
        """Get stored checksum for a document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Stored checksum or None if not found
        """
        query = """
        MATCH (d:Document {id: $document_id})
        RETURN d.checksum as checksum
        """
        
        result = await self._execute_query(query, {"document_id": document_id})
        return result[0]["checksum"] if result and result[0]["checksum"] else None
    
    async def store_document_checksum(self, document_id: str, checksum: str) -> None:
        """Store document checksum.
        
        Args:
            document_id: Document identifier
            checksum: Document checksum to store
        """
        query = """
        MATCH (d:Document {id: $document_id})
        SET d.checksum = $checksum, d.checksum_updated = datetime()
        """
        await self._execute_query(query, {
            "document_id": document_id,
            "checksum": checksum
        })
    
    async def delete_document_checksum(self, document_id: str) -> None:
        """Remove stored checksum for a document.
        
        Args:
            document_id: Document identifier
        """
        query = """
        MATCH (d:Document {id: $document_id})
        REMOVE d.checksum, d.checksum_updated
        """
        
        await self._execute_query(query, {"document_id": document_id})
    
    async def validate_id_consistency(self) -> Dict[str, Any]:
        """Validate ID consistency across the database.
        
        Returns:
            Validation report
        """
        # Check for duplicate document IDs
        duplicate_docs_query = """
        MATCH (d:Document)
        WITH d.id as doc_id, count(*) as count
        WHERE count > 1
        RETURN count(doc_id) as duplicate_documents, collect(doc_id)[0..5] as sample_ids
        """
        
        # Check for orphaned chunks
        orphan_chunks_query = """
        MATCH (c:DocumentChunk)
        WHERE NOT EXISTS {
            MATCH (d:Document {id: c.document_id})
        }
        RETURN count(c) as orphaned_chunks, collect(c.id)[0..5] as sample_ids
        """
        
        duplicate_result = await self._execute_query(duplicate_docs_query)
        orphan_result = await self._execute_query(orphan_chunks_query)
        
        return {
            "duplicate_documents": {
                "count": duplicate_result[0]['duplicate_documents'],
                "sample_ids": duplicate_result[0]['sample_ids']
            },
            "orphaned_chunks": {
                "count": orphan_result[0]['orphaned_chunks'],
                "sample_ids": orphan_result[0]['sample_ids']
            }
        }
