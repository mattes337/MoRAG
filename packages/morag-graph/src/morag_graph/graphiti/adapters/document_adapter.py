"""Document adapter for MoRAG-Graphiti integration."""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

try:
    from graphiti_core.nodes import EpisodeType
    GRAPHITI_AVAILABLE = True
except ImportError:
    # Graceful degradation when graphiti-core is not installed
    class EpisodeType:
        text = "text"
        json = "json"
    GRAPHITI_AVAILABLE = False

from morag_core.models import Document, DocumentChunk
from .core import BaseAdapter, ConversionResult, ConversionDirection, ValidationError

logger = logging.getLogger(__name__)


class DocumentAdapter(BaseAdapter[Document, Dict[str, Any]]):
    """Adapter for converting Documents between MoRAG and Graphiti formats."""
    
    def __init__(self, strict_validation: bool = True, include_chunks: bool = True):
        super().__init__(strict_validation)
        self.include_chunks = include_chunks
        self.episode_type_mapping = {
            'text/plain': EpisodeType.text,
            'text/markdown': EpisodeType.text,
            'application/pdf': EpisodeType.text,
            'text/html': EpisodeType.text,
            'application/json': EpisodeType.json,
            'image/jpeg': EpisodeType.text,  # Graphiti doesn't have image type
            'image/png': EpisodeType.text,
            'default': EpisodeType.text
        }
    
    def to_graphiti(self, morag_model: Document) -> ConversionResult:
        """Convert MoRAG Document to Graphiti episode format.
        
        Args:
            morag_model: MoRAG Document instance
            
        Returns:
            ConversionResult with episode data
        """
        warnings = []
        
        try:
            # Validate input
            validation_errors = self.validate_input(morag_model, ConversionDirection.MORAG_TO_GRAPHITI)
            if validation_errors and self.strict_validation:
                self._record_conversion(False)
                return ConversionResult(
                    success=False,
                    error=f"Validation failed: {'; '.join(validation_errors)}"
                )
            elif validation_errors:
                warnings.extend(validation_errors)
            
            # Generate episode name
            episode_name = self._generate_episode_name(morag_model)
            
            # Generate episode content
            episode_content = self._generate_episode_content(morag_model)
            
            # Generate source description
            source_description = self._generate_source_description(morag_model)
            
            # Generate metadata
            metadata = self._generate_episode_metadata(morag_model)
            
            # Determine episode type
            episode_type = self.episode_type_mapping.get(
                morag_model.metadata.mime_type,
                self.episode_type_mapping['default']
            )
            
            # Create episode data structure
            episode_data = {
                "name": episode_name,
                "content": episode_content,
                "source_description": source_description,
                "episode_type": episode_type,
                "metadata": metadata
            }
            
            # Include chunks if requested
            if self.include_chunks and morag_model.chunks:
                chunk_adapter = DocumentChunkAdapter(self.strict_validation)
                chunk_episodes = []
                
                for i, chunk in enumerate(morag_model.chunks):
                    chunk_result = chunk_adapter.to_graphiti(chunk)
                    if chunk_result.success:
                        chunk_episodes.append(chunk_result.data)
                    else:
                        warnings.append(f"Failed to convert chunk {i}: {chunk_result.error}")
                
                episode_data["chunk_episodes"] = chunk_episodes
            
            self._record_conversion(True, len(warnings))
            return ConversionResult(
                success=True,
                data=episode_data,
                warnings=warnings,
                metadata={
                    "original_document_id": morag_model.id,
                    "conversion_timestamp": datetime.utcnow().isoformat(),
                    "chunks_included": self.include_chunks,
                    "chunk_count": len(morag_model.chunks) if morag_model.chunks else 0
                }
            )
            
        except Exception as e:
            self._record_conversion(False)
            logger.error(f"Document conversion failed: {str(e)}")
            return ConversionResult(
                success=False,
                error=f"Conversion error: {str(e)}"
            )
    
    def from_graphiti(self, graphiti_data: Dict[str, Any]) -> ConversionResult:
        """Convert Graphiti episode data to MoRAG Document.
        
        Args:
            graphiti_data: Graphiti episode data
            
        Returns:
            ConversionResult with MoRAG Document
        """
        warnings = []
        
        try:
            # Validate input
            validation_errors = self.validate_input(graphiti_data, ConversionDirection.GRAPHITI_TO_MORAG)
            if validation_errors and self.strict_validation:
                self._record_conversion(False)
                return ConversionResult(
                    success=False,
                    error=f"Validation failed: {'; '.join(validation_errors)}"
                )
            elif validation_errors:
                warnings.extend(validation_errors)
            
            # Extract metadata
            metadata = graphiti_data.get("metadata", {})
            
            # Create Document instance
            # Note: This is a simplified conversion - in practice, you'd need
            # to reconstruct the full Document structure from episode data
            document_data = {
                "id": metadata.get("original_document_id", graphiti_data.get("name", "unknown")),
                "content": graphiti_data.get("content", ""),
                "metadata": self._extract_document_metadata(graphiti_data),
                "chunks": []
            }
            
            # Convert chunk episodes if present
            if "chunk_episodes" in graphiti_data:
                chunk_adapter = DocumentChunkAdapter(self.strict_validation)
                for chunk_data in graphiti_data["chunk_episodes"]:
                    chunk_result = chunk_adapter.from_graphiti(chunk_data)
                    if chunk_result.success:
                        document_data["chunks"].append(chunk_result.data)
                    else:
                        warnings.append(f"Failed to convert chunk: {chunk_result.error}")
            
            # Create Document instance (simplified - would need proper constructor)
            # This is a placeholder - actual implementation would depend on Document class structure
            document = Document(**document_data)
            
            self._record_conversion(True, len(warnings))
            return ConversionResult(
                success=True,
                data=document,
                warnings=warnings,
                metadata={
                    "conversion_timestamp": datetime.utcnow().isoformat(),
                    "source_episode_name": graphiti_data.get("name"),
                    "chunks_converted": len(document_data["chunks"])
                }
            )
            
        except Exception as e:
            self._record_conversion(False)
            logger.error(f"Graphiti to Document conversion failed: {str(e)}")
            return ConversionResult(
                success=False,
                error=f"Conversion error: {str(e)}"
            )
    
    def validate_input(self, data: Any, direction: ConversionDirection) -> List[str]:
        """Validate input data for document conversion."""
        errors = super().validate_input(data, direction)
        
        if direction == ConversionDirection.MORAG_TO_GRAPHITI:
            if not isinstance(data, Document):
                errors.append("Input must be a Document instance")
            elif not data.id:
                errors.append("Document must have an ID")
        
        elif direction == ConversionDirection.GRAPHITI_TO_MORAG:
            if not isinstance(data, dict):
                errors.append("Input must be a dictionary")
            elif "content" not in data:
                errors.append("Episode data must contain 'content' field")
        
        return errors
    
    def _generate_episode_name(self, document: Document) -> str:
        """Generate episode name from document."""
        if document.metadata.title:
            base_name = document.metadata.title
        elif document.metadata.source_path:
            base_name = Path(document.metadata.source_path).stem
        else:
            base_name = f"document_{document.id}"
        
        # Add timestamp for uniqueness
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_{timestamp}"
    
    def _generate_episode_content(self, document: Document) -> str:
        """Generate episode content from document."""
        content_parts = []
        
        # Add document metadata as context
        if document.metadata.title:
            content_parts.append(f"Document Title: {document.metadata.title}")
        
        if document.metadata.source_path:
            content_parts.append(f"Source File: {Path(document.metadata.source_path).name}")
        
        if document.metadata.author:
            content_parts.append(f"Author: {document.metadata.author}")
        
        # Add content summary
        content_parts.append("Content Summary:")
        
        # Add chunk summaries if available
        if document.chunks:
            content_parts.append(f"Document contains {len(document.chunks)} chunks:")
            for i, chunk in enumerate(document.chunks[:5]):  # Limit to first 5 chunks
                content_parts.append(f"Chunk {i+1}: {chunk.content[:200]}...")
            
            if len(document.chunks) > 5:
                content_parts.append(f"... and {len(document.chunks) - 5} more chunks")
        else:
            # Use document content directly if no chunks
            content_parts.append(document.content[:1000] + "..." if len(document.content) > 1000 else document.content)
        
        return "\n\n".join(content_parts)
    
    def _generate_source_description(self, document: Document) -> str:
        """Generate source description from document."""
        parts = []
        
        if document.metadata.source_path:
            parts.append(f"File: {document.metadata.source_path}")
        
        if document.metadata.mime_type:
            parts.append(f"Type: {document.metadata.mime_type}")
        
        if document.metadata.created_at:
            parts.append(f"Created: {document.metadata.created_at}")
        
        return "MoRAG Document - " + ", ".join(parts) if parts else "MoRAG Document"
    
    def _generate_episode_metadata(self, document: Document) -> Dict[str, Any]:
        """Generate episode metadata from document."""
        metadata = {
            "morag_document_id": document.id,
            "source_path": document.metadata.source_path,
            "mime_type": document.metadata.mime_type,
            "title": document.metadata.title,
            "author": document.metadata.author,
            "created_at": document.metadata.created_at.isoformat() if document.metadata.created_at else None,
            "updated_at": document.metadata.updated_at.isoformat() if document.metadata.updated_at else None,
            "chunk_count": len(document.chunks) if document.chunks else 0,
            "conversion_timestamp": datetime.utcnow().isoformat()
        }
        
        # Add custom metadata if present
        if hasattr(document.metadata, 'custom_metadata') and document.metadata.custom_metadata:
            metadata["custom_metadata"] = document.metadata.custom_metadata
        
        return metadata
    
    def _extract_document_metadata(self, graphiti_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract document metadata from Graphiti episode data."""
        episode_metadata = graphiti_data.get("metadata", {})
        
        # Map back to document metadata structure
        document_metadata = {
            "title": episode_metadata.get("title"),
            "author": episode_metadata.get("author"),
            "source_path": episode_metadata.get("source_path"),
            "mime_type": episode_metadata.get("mime_type"),
            "created_at": episode_metadata.get("created_at"),
            "updated_at": episode_metadata.get("updated_at")
        }
        
        # Add custom metadata if present
        if "custom_metadata" in episode_metadata:
            document_metadata["custom_metadata"] = episode_metadata["custom_metadata"]
        
        return document_metadata


class DocumentChunkAdapter(BaseAdapter[DocumentChunk, Dict[str, Any]]):
    """Adapter for converting DocumentChunks between MoRAG and Graphiti formats."""
    
    def to_graphiti(self, morag_model: DocumentChunk) -> ConversionResult:
        """Convert MoRAG DocumentChunk to Graphiti episode format."""
        try:
            episode_data = {
                "name": f"chunk_{morag_model.id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "content": morag_model.content,
                "source_description": f"Document chunk from {morag_model.document_id}",
                "episode_type": EpisodeType.text,
                "metadata": {
                    "morag_chunk_id": morag_model.id,
                    "document_id": morag_model.document_id,
                    "chunk_index": getattr(morag_model, 'chunk_index', None),
                    "start_char": getattr(morag_model, 'start_char', None),
                    "end_char": getattr(morag_model, 'end_char', None),
                    "conversion_timestamp": datetime.utcnow().isoformat()
                }
            }
            
            self._record_conversion(True)
            return ConversionResult(success=True, data=episode_data)
            
        except Exception as e:
            self._record_conversion(False)
            return ConversionResult(success=False, error=str(e))
    
    def from_graphiti(self, graphiti_data: Dict[str, Any]) -> ConversionResult:
        """Convert Graphiti episode data to MoRAG DocumentChunk."""
        try:
            metadata = graphiti_data.get("metadata", {})
            
            # Create DocumentChunk instance (simplified)
            chunk_data = {
                "id": metadata.get("morag_chunk_id", graphiti_data.get("name")),
                "content": graphiti_data.get("content", ""),
                "document_id": metadata.get("document_id"),
                "chunk_index": metadata.get("chunk_index"),
                "start_char": metadata.get("start_char"),
                "end_char": metadata.get("end_char")
            }
            
            # Create DocumentChunk instance (placeholder - would need proper constructor)
            chunk = DocumentChunk(**chunk_data)
            
            self._record_conversion(True)
            return ConversionResult(success=True, data=chunk)
            
        except Exception as e:
            self._record_conversion(False)
            return ConversionResult(success=False, error=str(e))
