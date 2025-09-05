"""Refactored chunk processor for MoRAG ingestion system.

This module coordinates chunking, embedding, and result processing using
separate specialized components.
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
import uuid

from morag_core.models import ProcessingResult
from morag_services.embedding import GeminiEmbeddingService
from morag_graph.services.fact_extraction_service import FactExtractionService
from morag_graph.services.enhanced_fact_processing_service import EnhancedFactProcessingService
from morag_graph.utils.id_generation import UnifiedIDGenerator
from morag_core.utils.logging import get_logger

from .chunkers import ContentChunkers
from .embedding_processor import EmbeddingProcessor
from .result_processor import ResultProcessor

logger = get_logger(__name__)


class ChunkProcessor:
    """Refactored chunk processor that coordinates chunking, embedding, and result processing."""
    
    def __init__(self):
        """Initialize the chunk processor with specialized components."""
        self.chunkers = ContentChunkers()
        self.embedding_processor = EmbeddingProcessor()
        self.result_processor = ResultProcessor()
        self.fact_extractor = None
    
    async def initialize(self):
        """Initialize all services and components."""
        try:
            # Initialize embedding processor
            await self.embedding_processor.initialize()
            
            # Initialize fact extraction service
            self.fact_extractor = FactExtractionService()
            
            logger.info("Chunk processor initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize chunk processor", error=str(e))
            raise

    def create_chunks(self, content: str, chunk_size: int, chunk_overlap: int, content_type: str = 'document', metadata: Dict[str, Any] = None) -> List[str]:
        """Create chunks from content based on content type and metadata."""
        if metadata is None:
            metadata = {}

        # Determine the best chunking strategy for this content
        chunk_type = self._determine_chunk_type(content_type, metadata)
        
        try:
            if chunk_type == 'topic':
                return self.chunkers.create_topic_based_chunks(content, chunk_size, chunk_overlap, metadata)
            elif chunk_type == 'timestamp':
                return self.chunkers.create_timestamp_chunks(content, chunk_size, chunk_overlap)
            elif chunk_type == 'image_section':
                return self.chunkers.create_image_section_chunks(content, chunk_size, chunk_overlap)
            elif chunk_type == 'web_article':
                return self.chunkers.create_web_article_chunks(content, chunk_size, chunk_overlap)
            elif chunk_type == 'text_semantic':
                return self.chunkers.create_text_semantic_chunks(content, chunk_size, chunk_overlap)
            elif chunk_type == 'code_structural':
                return self.chunkers.create_code_structural_chunks(content, chunk_size, chunk_overlap)
            elif chunk_type == 'archive':
                return self.chunkers.create_archive_file_chunks(content, chunk_size, chunk_overlap)
            elif chunk_type == 'document':
                return self.chunkers.create_document_chunks(content, chunk_size, chunk_overlap)
            else:
                # Default to character-based chunking
                return self.chunkers.create_character_chunks(content, chunk_size, chunk_overlap)
                
        except Exception as e:
            logger.error("Chunking failed, falling back to character chunking", 
                        chunk_type=chunk_type, error=str(e))
            return self.chunkers.create_character_chunks(content, chunk_size, chunk_overlap)

    def _determine_chunk_type(self, content_type: str, metadata: Dict[str, Any]) -> str:
        """Determine the best chunking strategy based on content type and metadata."""
        # Audio/video content with topics
        if content_type in ['audio', 'video']:
            if metadata.get('topics'):
                return 'topic'
            else:
                return 'timestamp'
        
        # Image content
        if content_type == 'image':
            return 'image_section'
        
        # Web content
        if content_type == 'web' or metadata.get('source_type') == 'web':
            return 'web_article'
        
        # Code content
        if content_type == 'code' or metadata.get('file_extension') in ['.py', '.js', '.java', '.cpp', '.c']:
            return 'code_structural'
        
        # Archive content
        if content_type == 'archive' or metadata.get('file_extension') in ['.zip', '.tar', '.gz']:
            return 'archive'
        
        # Document content (PDF, Word, etc.)
        if content_type in ['document', 'pdf', 'docx', 'doc']:
            return 'document'
        
        # Default to semantic text chunking
        return 'text_semantic'

    async def generate_embeddings_and_metadata(
        self,
        chunks: List[str],
        content_type: str = 'document',
        base_metadata: Dict[str, Any] = None
    ) -> Tuple[List[str], List[List[float]], List[Dict[str, Any]]]:
        """Generate embeddings and metadata for chunks."""
        return await self.embedding_processor.generate_embeddings_and_metadata(
            chunks, content_type, base_metadata
        )

    async def _generate_document_summary(self, content: str, content_type: str) -> str:
        """Generate a summary of the document content."""
        return await self.embedding_processor._generate_document_summary(content, content_type)

    def create_ingest_result(
        self,
        source_path: str,
        content: str,
        chunks: List[str],
        embeddings: List[List[float]],
        chunk_metadata_list: List[Dict[str, Any]],
        summary: str,
        content_type: str = 'document',
        base_metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Create comprehensive ingest result."""
        return self.result_processor.create_ingest_result(
            source_path, content, chunks, embeddings, chunk_metadata_list,
            summary, content_type, base_metadata
        )

    def write_ingest_result_file(self, source_path: str, ingest_result: Dict[str, Any]) -> str:
        """Write ingest result to JSON file."""
        return self.result_processor.write_ingest_result_file(source_path, ingest_result)

    def create_ingest_data(
        self,
        source_path: str,
        chunks: List[str],
        embeddings: List[List[float]],
        chunk_metadata_list: List[Dict[str, Any]],
        document_id: str = None,
        collection_name: str = None
    ) -> Dict[str, Any]:
        """Create ingest data for database storage."""
        return self.result_processor.create_ingest_data(
            source_path, chunks, embeddings, chunk_metadata_list,
            document_id, collection_name
        )

    def write_ingest_data_file(self, source_path: str, ingest_data: Dict[str, Any]) -> str:
        """Write ingest data to JSON file."""
        return self.result_processor.write_ingest_data_file(source_path, ingest_data)


__all__ = ["ChunkProcessor"]