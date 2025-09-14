"""Document checksum management for efficient change detection."""

import hashlib
import logging
from typing import Optional

from ..models.document import Document
from ..storage.base import BaseStorage


class DocumentChecksumManager:
    """Manages document checksums for efficient change detection.
    
    This class provides checksum-based change detection to avoid
    reprocessing unchanged documents in the graph construction pipeline.
    """
    
    def __init__(self, graph_storage: BaseStorage):
        """Initialize the checksum manager.
        
        Args:
            graph_storage: Graph storage backend
        """
        self.graph_storage = graph_storage
        self.logger = logging.getLogger(__name__)
    
    def calculate_document_checksum(self, content: str, metadata: Optional[dict] = None) -> str:
        """Calculate SHA-256 checksum of document content.
        
        Args:
            content: Document content
            metadata: Optional document metadata
            
        Returns:
            SHA-256 hex digest of the content
        """
        # Combine content and metadata for checksum calculation
        combined_content = content
        if metadata:
            # Sort metadata keys for consistent checksum
            metadata_str = str(sorted(metadata.items()))
            combined_content = f"{content}{metadata_str}"
        
        return hashlib.sha256(combined_content.encode('utf-8')).hexdigest()
    
    async def get_stored_checksum(self, document_id: str) -> Optional[str]:
        """Get stored checksum for a document from graph database.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Stored checksum if found, None otherwise
        """
        try:
            # This method needs to be implemented in the storage backends
            if hasattr(self.graph_storage, 'get_document_checksum'):
                return await self.graph_storage.get_document_checksum(document_id)
            else:
                self.logger.warning("Storage backend does not support checksum operations")
                return None
        except Exception as e:
            self.logger.error(f"Error retrieving checksum for document {document_id}: {str(e)}")
            return None
    
    async def store_document_checksum(self, document_id: str, checksum: str) -> None:
        """Store document checksum in graph database.
        
        Args:
            document_id: Document identifier
            checksum: Document checksum to store
        """
        try:
            # This method needs to be implemented in the storage backends
            if hasattr(self.graph_storage, 'store_document_checksum'):
                await self.graph_storage.store_document_checksum(document_id, checksum)
            else:
                self.logger.warning("Storage backend does not support checksum operations")
        except Exception as e:
            self.logger.error(f"Error storing checksum for document {document_id}: {str(e)}")
            raise
    
    async def needs_update(self, document_id: str, content: str, metadata: Optional[dict] = None) -> bool:
        """Check if document needs to be updated based on checksum comparison.
        
        Args:
            document_id: Document identifier
            content: Document content
            metadata: Optional document metadata
            
        Returns:
            True if document needs processing, False if unchanged
        """
        current_checksum = self.calculate_document_checksum(content, metadata)
        stored_checksum = await self.get_stored_checksum(document_id)
        
        if stored_checksum is None:
            self.logger.info(f"Document {document_id} not found in graph, needs processing")
            return True
        
        if current_checksum != stored_checksum:
            self.logger.info(f"Document {document_id} checksum changed, needs reprocessing")
            return True
        
        self.logger.info(f"Document {document_id} unchanged, skipping")
        return False