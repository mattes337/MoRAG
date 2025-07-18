"""Document to Episode mapping service for Graphiti integration."""

import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import structlog

from morag_core.models import Document, DocumentChunk
from .config import GraphitiConfig
from .connection import GraphitiConnectionService

logger = structlog.get_logger(__name__)


class DocumentEpisodeMapper:
    """Service for mapping MoRAG documents to Graphiti episodes."""

    def __init__(self, config: Optional[GraphitiConfig] = None):
        """Initialize the document episode mapper.
        
        Args:
            config: Optional Graphiti configuration
        """
        self.config = config
        self.connection_service = GraphitiConnectionService(config)

    async def map_document_to_episode(
        self,
        document: Document,
        episode_name: Optional[str] = None,
        source_description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Map a MoRAG document to a Graphiti episode.
        
        Args:
            document: MoRAG document to map
            episode_name: Optional custom episode name
            source_description: Optional source description
            
        Returns:
            Dictionary with episode creation result
        """
        try:
            # Generate episode name if not provided
            if not episode_name:
                episode_name = self._generate_episode_name(document)
            
            # Generate episode content from document
            episode_content = self._generate_episode_content(document)
            
            # Generate source description if not provided
            if not source_description:
                source_description = self._generate_source_description(document)
            
            # Create episode metadata
            episode_metadata = self._generate_episode_metadata(document)
            
            # Connect to Graphiti and create episode
            async with self.connection_service as conn:
                success = await conn.create_episode(
                    name=episode_name,
                    content=episode_content,
                    source_description=source_description,
                    metadata=episode_metadata
                )
                
                if success:
                    logger.info(
                        "Document mapped to episode successfully",
                        document_id=document.id,
                        episode_name=episode_name
                    )
                    return {
                        "success": True,
                        "episode_name": episode_name,
                        "document_id": document.id,
                        "content_length": len(episode_content),
                        "metadata": episode_metadata
                    }
                else:
                    logger.error(
                        "Failed to create episode",
                        document_id=document.id,
                        episode_name=episode_name
                    )
                    return {
                        "success": False,
                        "error": "Failed to create episode in Graphiti",
                        "document_id": document.id
                    }
                    
        except Exception as e:
            logger.error(
                "Document to episode mapping failed",
                document_id=document.id,
                error=str(e)
            )
            return {
                "success": False,
                "error": str(e),
                "document_id": document.id
            }

    async def map_document_chunks_to_episodes(
        self,
        document: Document,
        chunk_episode_prefix: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Map document chunks to individual Graphiti episodes.
        
        Args:
            document: MoRAG document with chunks
            chunk_episode_prefix: Optional prefix for chunk episode names
            
        Returns:
            List of episode creation results
        """
        results = []
        
        if not document.chunks:
            logger.warning("Document has no chunks to map", document_id=document.id)
            return results
        
        try:
            async with self.connection_service as conn:
                for i, chunk in enumerate(document.chunks):
                    try:
                        # Generate episode name for chunk
                        if chunk_episode_prefix:
                            episode_name = f"{chunk_episode_prefix}_chunk_{i+1}"
                        else:
                            episode_name = f"{document.metadata.title or 'document'}_{document.id}_chunk_{i+1}"
                        
                        # Create episode for chunk
                        success = await conn.create_episode(
                            name=episode_name,
                            content=chunk.content,
                            source_description=f"Chunk {i+1} from document: {document.metadata.title or document.id}",
                            metadata=self._generate_chunk_episode_metadata(document, chunk, i)
                        )
                        
                        result = {
                            "success": success,
                            "episode_name": episode_name,
                            "chunk_id": chunk.id,
                            "chunk_index": i,
                            "document_id": document.id
                        }
                        
                        if not success:
                            result["error"] = "Failed to create episode in Graphiti"
                        
                        results.append(result)
                        
                    except Exception as e:
                        logger.error(
                            "Failed to map chunk to episode",
                            document_id=document.id,
                            chunk_index=i,
                            error=str(e)
                        )
                        results.append({
                            "success": False,
                            "error": str(e),
                            "chunk_id": chunk.id,
                            "chunk_index": i,
                            "document_id": document.id
                        })
                        
        except Exception as e:
            logger.error(
                "Document chunks to episodes mapping failed",
                document_id=document.id,
                error=str(e)
            )
            
        return results

    def _generate_episode_name(self, document: Document) -> str:
        """Generate episode name from document.
        
        Args:
            document: MoRAG document
            
        Returns:
            Generated episode name
        """
        # Use document title if available, otherwise use source_name or ID
        if hasattr(document.metadata, 'title') and document.metadata.title:
            base_name = document.metadata.title
        elif hasattr(document.metadata, 'source_name') and document.metadata.source_name:
            base_name = document.metadata.source_name
        else:
            base_name = f"document_{document.id}"
        
        # Add timestamp to ensure uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_{timestamp}"

    def _generate_episode_content(self, document: Document) -> str:
        """Generate episode content from document.
        
        Args:
            document: MoRAG document
            
        Returns:
            Generated episode content
        """
        content_parts = []
        
        # Add document metadata as context
        if hasattr(document.metadata, 'title') and document.metadata.title:
            content_parts.append(f"Document Title: {document.metadata.title}")
        
        if hasattr(document.metadata, 'source_name') and document.metadata.source_name:
            content_parts.append(f"Source File: {document.metadata.source_name}")
        
        # Add document summary if available
        if hasattr(document, 'raw_text') and document.raw_text:
            # Use first 1000 characters of raw text as summary
            summary = document.raw_text[:1000]
            if len(document.raw_text) > 1000:
                summary += "..."
            content_parts.append(f"Content Summary: {summary}")
        
        # Add chunk contents
        if document.chunks:
            content_parts.append(f"Document contains {len(document.chunks)} chunks:")
            for i, chunk in enumerate(document.chunks[:5]):  # Limit to first 5 chunks
                chunk_preview = chunk.content[:200]
                if len(chunk.content) > 200:
                    chunk_preview += "..."
                content_parts.append(f"Chunk {i+1}: {chunk_preview}")
            
            if len(document.chunks) > 5:
                content_parts.append(f"... and {len(document.chunks) - 5} more chunks")
        
        return "\n\n".join(content_parts)

    def _generate_source_description(self, document: Document) -> str:
        """Generate source description from document.
        
        Args:
            document: MoRAG document
            
        Returns:
            Generated source description
        """
        parts = ["MoRAG document"]
        
        if hasattr(document.metadata, 'source_type'):
            parts.append(f"type: {document.metadata.source_type}")
        
        if hasattr(document.metadata, 'source_name') and document.metadata.source_name:
            parts.append(f"file: {document.metadata.source_name}")
        
        return " - ".join(parts)

    def _generate_episode_metadata(self, document: Document) -> Dict[str, Any]:
        """Generate episode metadata from document.
        
        Args:
            document: MoRAG document
            
        Returns:
            Generated episode metadata
        """
        metadata = {
            "morag_document_id": document.id,
            "morag_source": "document_episode_mapper",
            "created_at": datetime.now().isoformat(),
            "chunk_count": len(document.chunks) if document.chunks else 0
        }
        
        # Add document metadata
        if hasattr(document, 'metadata') and document.metadata:
            doc_meta = document.metadata
            if hasattr(doc_meta, 'title') and doc_meta.title:
                metadata["document_title"] = doc_meta.title
            if hasattr(doc_meta, 'source_name') and doc_meta.source_name:
                metadata["document_filename"] = doc_meta.source_name
            if hasattr(doc_meta, 'source_type'):
                metadata["document_source_type"] = doc_meta.source_type
            if hasattr(doc_meta, 'file_size') and doc_meta.file_size:
                metadata["document_file_size"] = doc_meta.file_size
        
        # Add processing metadata
        if hasattr(document, 'processed_at') and document.processed_at:
            metadata["document_processed_at"] = document.processed_at.isoformat()
        
        return metadata

    def _generate_chunk_episode_metadata(
        self,
        document: Document,
        chunk: DocumentChunk,
        chunk_index: int
    ) -> Dict[str, Any]:
        """Generate episode metadata for a document chunk.
        
        Args:
            document: Parent MoRAG document
            chunk: Document chunk
            chunk_index: Index of the chunk
            
        Returns:
            Generated episode metadata
        """
        metadata = {
            "morag_document_id": document.id,
            "morag_chunk_id": chunk.id,
            "morag_source": "chunk_episode_mapper",
            "created_at": datetime.now().isoformat(),
            "chunk_index": chunk_index,
            "chunk_length": len(chunk.content)
        }
        
        # Add chunk-specific metadata
        if hasattr(chunk, 'chunk_index'):
            metadata["original_chunk_index"] = chunk.chunk_index
        if hasattr(chunk, 'page_number') and chunk.page_number:
            metadata["page_number"] = chunk.page_number
        if hasattr(chunk, 'section') and chunk.section:
            metadata["section"] = chunk.section
        
        # Add document context
        if hasattr(document.metadata, 'title') and document.metadata.title:
            metadata["document_title"] = document.metadata.title
        if hasattr(document.metadata, 'source_name') and document.metadata.source_name:
            metadata["document_filename"] = document.metadata.source_name
        
        return metadata


# Convenience function for creating mapper
def create_episode_mapper(config: Optional[GraphitiConfig] = None) -> DocumentEpisodeMapper:
    """Create a document episode mapper.
    
    Args:
        config: Optional Graphiti configuration
        
    Returns:
        DocumentEpisodeMapper instance
    """
    return DocumentEpisodeMapper(config)
