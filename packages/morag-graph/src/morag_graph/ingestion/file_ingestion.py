"""File ingestion with checksum-based duplicate prevention.

This module provides functionality for ingesting files into the graph database
while preventing duplicates using file checksums and metadata tracking.
"""

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set

from pydantic import BaseModel

from ..models import Entity, Relation, Document, DocumentChunk
from ..storage.base import BaseStorage

logger = logging.getLogger(__name__)


class FileMetadata(BaseModel):
    """Metadata for ingested files."""
    
    file_path: str
    file_name: str
    file_size: int
    checksum: str  # SHA256 hash of file content
    mime_type: Optional[str] = None
    ingestion_timestamp: datetime
    last_modified: Optional[datetime] = None
    source_doc_id: str  # Unique identifier for this file
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class FileIngestion:
    """File ingestion manager with duplicate prevention.
    
    This class handles file ingestion into the graph database with
    checksum-based duplicate detection and metadata tracking.
    """
    
    def __init__(self, storage: BaseStorage):
        """Initialize file ingestion manager.
        
        Args:
            storage: Graph storage backend
        """
        self.storage = storage
        self._ingested_files: Dict[str, FileMetadata] = {}
        self._checksum_to_file: Dict[str, str] = {}  # checksum -> file_path
    
    def calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            SHA256 checksum as hex string
        """
        sha256_hash = hashlib.sha256()
        
        try:
            with open(file_path, "rb") as f:
                # Read file in chunks to handle large files efficiently
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate checksum for {file_path}: {e}")
            raise
    
    def get_file_metadata(self, file_path: Path) -> FileMetadata:
        """Extract metadata from a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            FileMetadata object
        """
        try:
            stat = file_path.stat()
            checksum = self.calculate_file_checksum(file_path)
            
            # Generate unique source_doc_id based on file path and checksum
            source_doc_id = f"file_{checksum[:16]}"
            
            return FileMetadata(
                file_path=str(file_path.absolute()),
                file_name=file_path.name,
                file_size=stat.st_size,
                checksum=checksum,
                mime_type=self._get_mime_type(file_path),
                ingestion_timestamp=datetime.now(),
                last_modified=datetime.fromtimestamp(stat.st_mtime),
                source_doc_id=source_doc_id
            )
        except Exception as e:
            logger.error(f"Failed to extract metadata from {file_path}: {e}")
            raise
    
    def _get_mime_type(self, file_path: Path) -> Optional[str]:
        """Get MIME type based on file extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            MIME type string or None
        """
        extension_to_mime = {
            '.txt': 'text/plain',
            '.pdf': 'application/pdf',
            '.doc': 'application/msword',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.json': 'application/json',
            '.xml': 'application/xml',
            '.html': 'text/html',
            '.md': 'text/markdown',
            '.csv': 'text/csv'
        }
        
        return extension_to_mime.get(file_path.suffix.lower())
    
    async def is_file_already_ingested(self, file_metadata: FileMetadata) -> bool:
        """Check if a file has already been ingested based on checksum.
        
        Args:
            file_metadata: Metadata of the file to check
            
        Returns:
            True if file already ingested, False otherwise
        """
        # Check in-memory cache first
        if file_metadata.checksum in self._checksum_to_file:
            logger.info(f"File {file_metadata.file_name} already ingested (cached)")
            return True
        
        # Check in storage by looking for entities with this source_doc_id
        try:
            entities = await self.storage.search_entities(
                query=file_metadata.source_doc_id,
                limit=1
            )
            
            if entities:
                # File already ingested, update cache
                self._checksum_to_file[file_metadata.checksum] = file_metadata.file_path
                self._ingested_files[file_metadata.file_path] = file_metadata
                logger.info(f"File {file_metadata.file_name} already ingested (found in storage)")
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error checking if file already ingested: {e}")
            return False
    
    def _group_entities_into_chunks(self, entities: List[Entity]) -> List[tuple[str, List[Entity]]]:
        """Group entities by their chunk index to create document chunks.
        
        Args:
            entities: List of entities to group
            
        Returns:
            List of tuples (chunk_text, entities_in_chunk)
        """
        if not entities:
            return []
        
        # Group entities by chunk_index from their attributes
        chunks_map = {}
        entities_without_chunk = []
        
        for entity in entities:
            chunk_index = None
            if entity.attributes and "chunk_index" in entity.attributes:
                chunk_index = entity.attributes["chunk_index"]
            
            if chunk_index is not None:
                if chunk_index not in chunks_map:
                    chunks_map[chunk_index] = []
                chunks_map[chunk_index].append(entity)
            else:
                entities_without_chunk.append(entity)
        
        # Convert to list of tuples, sorted by chunk index
        result = []
        for chunk_index in sorted(chunks_map.keys()):
            chunk_entities = chunks_map[chunk_index]
            # Use source_text from first entity's attributes if available, otherwise create descriptive text
            chunk_text = ""
            if chunk_entities and chunk_entities[0].attributes and 'source_text' in chunk_entities[0].attributes:
                chunk_text = chunk_entities[0].attributes['source_text']
            else:
                chunk_text = f"Document chunk {chunk_index} with {len(chunk_entities)} entities"
            
            result.append((chunk_text, chunk_entities))
        
        # If there are entities without chunk information, create a separate chunk
        if entities_without_chunk:
            chunk_text = f"Document content with {len(entities_without_chunk)} entities (no chunk info)"
            result.append((chunk_text, entities_without_chunk))
        
        # If no chunks were created (no chunk_index info), fall back to single chunk
        if not result:
            chunk_text = f"Document content with {len(entities)} entities"
            result = [(chunk_text, entities)]
        
        return result
    
    async def ingest_file_entities_and_relations(
        self, 
        file_path: Path,
        entities: List[Entity],
        relations: List[Relation],
        force_reingest: bool = False
    ) -> Dict[str, Any]:
        """Ingest entities and relations from a file using the new document structure.
        
        Creates Document -> DocumentChunk -> Entity relationships.
        
        Args:
            file_path: Path to the source file
            entities: List of entities extracted from the file
            relations: List of relations extracted from the file
            force_reingest: If True, reingest even if file already processed
            
        Returns:
            Dictionary with ingestion results
        """
        try:
            # Get file metadata and checksum
            file_metadata = self.get_file_metadata(file_path)
            
            # Check for duplicates unless forced
            if not force_reingest and await self.is_file_already_ingested(file_metadata):
                return {
                    'status': 'skipped',
                    'reason': 'duplicate_file',
                    'file_metadata': file_metadata,
                    'entities_stored': 0,
                    'relations_stored': 0
                }
            
            # Create Document node
            document = Document(
                id=file_metadata.source_doc_id,
                name=file_metadata.file_name,  # Add name property
                source_file=file_metadata.file_path,
                file_name=file_metadata.file_name,
                file_size=file_metadata.file_size,
                checksum=file_metadata.checksum,
                mime_type=file_metadata.mime_type,
                ingestion_timestamp=file_metadata.ingestion_timestamp,
                last_modified=file_metadata.last_modified,
                metadata={
                    'file_name': file_metadata.file_name,
                    'file_checksum': file_metadata.checksum,
                    'ingestion_timestamp': file_metadata.ingestion_timestamp.isoformat()
                }
            )
            
            # Store Document
            document_id = await self.storage.store_document(document)
            logger.info(f"Stored document: {document_id}")
            
            # Group entities by their source text to create chunks
            chunks_data = self._group_entities_into_chunks(entities)
            
            # Create and store DocumentChunks
            chunk_ids = []
            for chunk_index, (chunk_text, chunk_entities) in enumerate(chunks_data):
                chunk = DocumentChunk(
                    id=f"{document_id}_chunk_{chunk_index}",
                    document_id=document_id,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    start_position=0,  # Could be calculated if needed
                    end_position=len(chunk_text),
                    chunk_type="text",
                    metadata={
                        'entity_count': len(chunk_entities),
                        'extraction_method': 'llm_based'
                    }
                )
                
                chunk_id = await self.storage.store_document_chunk(chunk)
                chunk_ids.append(chunk_id)
                
                # Create Document -> CONTAINS -> DocumentChunk relationship
                await self.storage.create_document_contains_chunk_relation(document_id, chunk_id)
                
                # Store entities (they are already clean in the new model)
                entity_ids = await self.storage.store_entities(chunk_entities)
                
                # Create DocumentChunk -> MENTIONS -> Entity relationships
                for entity in chunk_entities:
                    await self.storage.create_chunk_mentions_entity_relation(
                        chunk_id, 
                        entity.id, 
                        chunk_text  # Use chunk text as context
                    )
            
            # Store entity-to-entity relations (they are already clean in the new model)
            relation_ids = await self.storage.store_relations(relations)
            
            # Update cache
            self._ingested_files[file_metadata.file_path] = file_metadata
            self._checksum_to_file[file_metadata.checksum] = file_metadata.file_path
            
            logger.info(f"Successfully ingested file {file_metadata.file_name} with new document structure")
            logger.info(f"Created 1 document, {len(chunk_ids)} chunks, {len(entities)} entities, {len(relations)} relations")
            
            return {
                'status': 'success',
                'file_metadata': file_metadata,
                'document_id': document_id,
                'chunks_created': len(chunk_ids),
                'entities_stored': len(entities),
                'relations_stored': len(relations),
                'chunk_ids': chunk_ids
            }
            
        except Exception as e:
            logger.error(f"Failed to ingest file {file_path}: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'file_path': str(file_path)
            }
    
    async def get_ingested_files(self) -> List[FileMetadata]:
        """Get list of all ingested files.
        
        Returns:
            List of FileMetadata objects
        """
        return list(self._ingested_files.values())
    
    async def remove_file_from_graph(self, file_path: Path) -> Dict[str, Any]:
        """Remove all entities and relations from a specific file.
        
        Args:
            file_path: Path to the file whose data should be removed
            
        Returns:
            Dictionary with removal results
        """
        try:
            file_metadata = self.get_file_metadata(file_path)
            
            # Find all entities from this file
            entities = await self.storage.search_entities(
                query=file_metadata.source_doc_id,
                limit=1000  # Adjust as needed
            )
            
            # Delete entities (this should cascade to relations)
            deleted_entities = 0
            for entity in entities:
                if await self.storage.delete_entity(entity.id):
                    deleted_entities += 1
            
            # Remove from cache
            if file_metadata.file_path in self._ingested_files:
                del self._ingested_files[file_metadata.file_path]
            if file_metadata.checksum in self._checksum_to_file:
                del self._checksum_to_file[file_metadata.checksum]
            
            logger.info(f"Removed {deleted_entities} entities from file {file_metadata.file_name}")
            
            return {
                'status': 'success',
                'entities_deleted': deleted_entities,
                'file_metadata': file_metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to remove file {file_path} from graph: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'file_path': str(file_path)
            }
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics.
        
        Returns:
            Dictionary with ingestion statistics
        """
        return {
            'total_files_ingested': len(self._ingested_files),
            'unique_checksums': len(self._checksum_to_file),
            'files_by_type': self._get_files_by_type()
        }
    
    def _get_files_by_type(self) -> Dict[str, int]:
        """Get count of files by MIME type.
        
        Returns:
            Dictionary mapping MIME types to counts
        """
        type_counts = {}
        for file_metadata in self._ingested_files.values():
            mime_type = file_metadata.mime_type or 'unknown'
            type_counts[mime_type] = type_counts.get(mime_type, 0) + 1
        return type_counts