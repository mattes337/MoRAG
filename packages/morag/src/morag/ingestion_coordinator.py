"""
Comprehensive ingestion coordinator for MoRAG system.

This module handles the complete ingestion flow including:
1. Database detection and configuration
2. Embedding and metadata generation
3. Complete ingest_result.json file creation
4. Database initialization (collections/databases)
5. Data writing to databases using ingest_data.json
"""

import json
import asyncio
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
import uuid

from morag_core.models.config import ProcessingResult
from morag_services.embedding import GeminiEmbeddingService
from morag_services.storage import QdrantVectorStorage
# Chunking is implemented directly in this module
from morag_graph.storage.neo4j_storage import Neo4jStorage, Neo4jConfig
from morag_graph.storage.qdrant_storage import QdrantStorage, QdrantConfig
from morag_graph.models.entity import Entity
from morag_graph.models.relation import Relation
from morag_graph.models.fact import Fact, FactRelation
from morag_graph.models.document import Document
from morag_graph.models.document_chunk import DocumentChunk
from morag_graph.models.database_config import DatabaseConfig, DatabaseType
from morag_graph.services.fact_extraction_service import FactExtractionService
from morag_graph.services.enhanced_fact_processing_service import EnhancedFactProcessingService
from morag_graph.utils.id_generation import UnifiedIDGenerator

import structlog

logger = structlog.get_logger(__name__)


class IngestionCoordinator:
    """Coordinates the complete ingestion process across multiple databases."""

    def __init__(self):
        """Initialize the ingestion coordinator."""
        self.embedding_service = None
        self.vector_storage = None
        self.fact_extractor = None
        self.logger = logger  # Initialize logger attribute
        
    async def initialize(self):
        """Initialize all services."""
        import os

        # Initialize embedding service
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")

        self.embedding_service = GeminiEmbeddingService(
            api_key=api_key,
            embedding_model=os.getenv('GEMINI_EMBEDDING_MODEL', 'text-embedding-004')
        )

        # Chunking is implemented directly in this class

        # Initialize vector storage
        # Prefer QDRANT_URL if available, otherwise use QDRANT_HOST/PORT
        qdrant_url = os.getenv('QDRANT_URL')
        qdrant_api_key = os.getenv('QDRANT_API_KEY')
        collection_name = os.getenv('QDRANT_COLLECTION_NAME')
        verify_ssl = os.getenv('QDRANT_VERIFY_SSL', 'true').lower() == 'true'
        if not collection_name:
            raise ValueError("QDRANT_COLLECTION_NAME environment variable is required")

        if qdrant_url:
            # Use URL-based connection (supports HTTPS automatically)
            self.vector_storage = QdrantVectorStorage(
                host=qdrant_url,
                api_key=qdrant_api_key,
                collection_name=collection_name,
                verify_ssl=verify_ssl
            )
        else:
            # Fall back to host/port connection
            qdrant_host = os.getenv('QDRANT_HOST', 'localhost')
            qdrant_port = int(os.getenv('QDRANT_PORT', '6333'))
            self.vector_storage = QdrantVectorStorage(
                host=qdrant_host,
                port=qdrant_port,
                api_key=qdrant_api_key,
                collection_name=collection_name,
                verify_ssl=verify_ssl
            )

        # Initialize vector storage connection and ensure collection exists
        await self.vector_storage.initialize()

        # Initialize fact extractor service
        # We'll create a simple wrapper that mimics the old interface
        self.fact_extractor = FactExtractionWrapper()
        
    async def ingest_content(
        self,
        content: str,
        source_path: str,
        content_type: str,
        metadata: Dict[str, Any],
        processing_result: ProcessingResult,
        databases: Optional[List[DatabaseConfig]] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        document_id: Optional[str] = None,
        replace_existing: bool = False,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive content ingestion.
        
        Args:
            content: Text content to ingest
            source_path: Source file path or URL
            content_type: Type of content (pdf, audio, video, etc.)
            metadata: Additional metadata
            processing_result: Result from content processing
            databases: List of database configurations to use
            chunk_size: Chunk size for text splitting
            chunk_overlap: Overlap between chunks
            document_id: Custom document identifier
            replace_existing: Whether to replace existing document
            
        Returns:
            Complete ingestion result dictionary
        """
        if not self.embedding_service:
            await self.initialize()
            
        start_time = datetime.now(timezone.utc)
        
        # Step 1: Determine which databases to use
        database_configs = self._determine_databases(databases)
        logger.info("Determined databases for ingestion", 
                   databases=[db.type.value for db in database_configs])
        
        # Step 2: Generate document ID if not provided
        if not document_id:
            document_id = self._generate_document_id(source_path, content)
            
        # Step 3: Generate embeddings and metadata for all databases
        embeddings_data = await self._generate_embeddings_and_metadata(
            content, metadata, chunk_size, chunk_overlap, document_id
        )
        
        # Step 4: Determine language for extraction if not provided
        effective_language = language
        if not effective_language:
            # Try to extract language from processing result metadata
            if hasattr(processing_result, 'document') and processing_result.document:
                effective_language = getattr(processing_result.document.metadata, 'language', None)
            # Try to extract from general metadata
            if not effective_language:
                effective_language = metadata.get('language')
            # Try to extract from processing result metadata
            if not effective_language and hasattr(processing_result, 'metadata'):
                effective_language = processing_result.metadata.get('language')

        logger.info("Language determined for extraction",
                   provided_language=language,
                   effective_language=effective_language,
                   source_path=source_path)

        # Step 5: Extract facts and relationships for graph databases
        # Use the same chunk settings as embeddings to ensure consistency
        effective_chunk_size = embeddings_data['chunk_size']
        effective_chunk_overlap = embeddings_data['chunk_overlap']
        graph_data = await self._extract_graph_data(
            content, source_path, document_id, metadata, effective_chunk_size, effective_chunk_overlap, effective_language
        )

        # Step 6: Generate document summary using LLM
        document_summary = await self._generate_document_summary(content, metadata, effective_language)

        # Step 7: Create complete ingest_result.json data
        ingest_result = self._create_ingest_result(
            source_path, content_type, metadata, processing_result,
            embeddings_data, graph_data, database_configs, start_time, document_summary, effective_language
        )
        
        # Step 8: Write ingest_result.json file
        result_file_path = self._write_ingest_result_file(source_path, ingest_result)

        # Step 9: Initialize databases (create collections/databases if needed)
        await self._initialize_databases(database_configs, embeddings_data)

        # Step 10: Create and write ingest_data.json file for database writes
        ingest_data = self._create_ingest_data(
            embeddings_data, graph_data, database_configs, document_id, source_path, metadata
        )
        data_file_path = self._write_ingest_data_file(source_path, ingest_data)

        # Step 11: Write data to databases using the ingest_data
        database_results = await self._write_to_databases(
            database_configs, embeddings_data, graph_data, document_id, replace_existing, document_summary, ingest_data
        )

        # Step 12: Update final result with database write results
        ingest_result['database_results'] = database_results
        ingest_result['processing_time'] = (datetime.now(timezone.utc) - start_time).total_seconds()
        ingest_result['ingest_data_file'] = data_file_path

        # Step 13: Update the ingest_result.json file with final results
        self._write_ingest_result_file(source_path, ingest_result)
        
        logger.info("Ingestion completed successfully",
                   source_path=source_path,
                   databases_used=len(database_configs),
                   chunks_created=len(embeddings_data.get('chunks', [])),
                   facts_extracted=len(graph_data.get('facts', [])),
                   relationships_extracted=len(graph_data.get('relationships', [])),
                   result_file=result_file_path)
        
        return ingest_result
        
    def _determine_databases(self, databases: Optional[List[DatabaseConfig]]) -> List[DatabaseConfig]:
        """Determine which databases to use for ingestion."""
        if databases:
            return databases
            
        # Default configuration based on environment variables
        default_databases = []
        
        # Check for Qdrant configuration
        import os
        qdrant_url = os.getenv('QDRANT_URL')
        qdrant_host = os.getenv('QDRANT_HOST', 'localhost')
        qdrant_collection = os.getenv('QDRANT_COLLECTION_NAME', 'morag_documents')

        if (qdrant_url or qdrant_host) and qdrant_collection:
            # Prefer URL if available, otherwise use host/port
            if qdrant_url:
                default_databases.append(DatabaseConfig(
                    type=DatabaseType.QDRANT,
                    hostname=qdrant_url,  # Store URL in hostname field
                    database_name=qdrant_collection
                ))
            else:
                default_databases.append(DatabaseConfig(
                    type=DatabaseType.QDRANT,
                    hostname=qdrant_host,
                    port=int(os.getenv('QDRANT_PORT', 6333)),
                    database_name=qdrant_collection
                ))
            
        # Check for Neo4j configuration
        neo4j_uri = os.getenv('NEO4J_URI')
        neo4j_username = os.getenv('NEO4J_USERNAME')
        neo4j_password = os.getenv('NEO4J_PASSWORD')
        
        if neo4j_uri and neo4j_username and neo4j_password:
            default_databases.append(DatabaseConfig(
                type=DatabaseType.NEO4J,
                hostname=neo4j_uri,
                username=neo4j_username,
                password=neo4j_password,
                database_name=os.getenv('NEO4J_DATABASE', 'neo4j')
            ))
            
        return default_databases
        
    def _generate_document_id(self, source_path: str, content: str) -> str:
        """Generate a unique document ID."""
        # Generate a checksum from the content for deterministic IDs
        import hashlib
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
        return UnifiedIDGenerator.generate_document_id(source_path, content_hash)
        
    async def _generate_embeddings_and_metadata(
        self,
        content: str,
        metadata: Dict[str, Any],
        chunk_size: Optional[int],
        chunk_overlap: Optional[int],
        document_id: str
    ) -> Dict[str, Any]:
        """Generate embeddings and metadata for vector storage."""
        # Use default chunk settings if not provided
        if chunk_size is None:
            chunk_size = 4000
        if chunk_overlap is None:
            chunk_overlap = 200
            
        # Create chunks with content-type-aware strategy
        content_type = metadata.get('content_type', 'document')
        chunks = self._create_chunks(content, chunk_size, chunk_overlap, content_type, metadata)
        
        # Generate embeddings for all chunks
        embedding_results = await self.embedding_service.generate_embeddings_batch(
            chunks, task_type="retrieval_document"
        )

        # Extract embeddings from results
        embeddings = [result.embedding for result in embedding_results]
        
        # Create chunk metadata with content-type-specific enhancements
        chunk_metadata = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = UnifiedIDGenerator.generate_chunk_id(document_id, i)

            # Base chunk metadata
            chunk_meta = {
                'chunk_id': chunk_id,
                'document_id': document_id,
                'chunk_index': i,
                'chunk_size': len(chunk),
                'chunk_text': chunk,  # Store chunk text for metadata extraction
                'created_at': datetime.now(timezone.utc).isoformat(),
                **metadata
            }

            # Add total chunks count for better context
            chunk_meta['total_chunks'] = len(chunks)

            # Add content-type-specific metadata
            content_type = metadata.get('content_type', 'document')
            chunk_meta['chunk_type'] = self._determine_chunk_type(content_type, metadata)

            # Add position/timestamp information based on content type
            if content_type in ['audio', 'video']:
                chunk_meta.update(self._add_audio_video_chunk_metadata(chunk, i, metadata))
            elif content_type == 'document':
                chunk_meta.update(self._add_document_chunk_metadata(chunk, i, metadata))

            # Ensure we have a location reference for all chunks
            if 'location_reference' not in chunk_meta:
                chunk_meta['location_reference'] = f"Chunk {i + 1} of {len(chunks)}"
                chunk_meta['location_type'] = 'chunk_index'

            chunk_metadata.append(chunk_meta)
            
        return {
            'chunks': chunks,
            'embeddings': embeddings,
            'chunk_metadata': chunk_metadata,
            'document_id': document_id,
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap
        }
        
    def _create_chunks(self, content: str, chunk_size: int, chunk_overlap: int, content_type: str = 'document', metadata: Dict[str, Any] = None) -> List[str]:
        """Create text chunks with content-type-aware strategy.

        Args:
            content: Text content to chunk
            chunk_size: Maximum chunk size
            chunk_overlap: Overlap between chunks
            content_type: Type of content (audio, video, document, image, web, text, code, archive)
            metadata: Additional metadata for chunking decisions

        Returns:
            List of text chunks
        """
        if len(content) <= chunk_size:
            return [content]

        # Use topic-based chunking for audio/video content (line-based with topic awareness)
        if content_type in ['audio', 'video'] and metadata:
            return self._create_topic_based_chunks(content, chunk_size, chunk_overlap, metadata)

        # Use chapter/page-aware semantic chunking for documents
        elif content_type == 'document':
            return self._create_document_chunks(content, chunk_size, chunk_overlap)

        # Use section-based chunking for images
        elif content_type == 'image':
            return self._create_image_section_chunks(content, chunk_size, chunk_overlap)

        # Use article structure chunking for web content
        elif content_type == 'web':
            return self._create_web_article_chunks(content, chunk_size, chunk_overlap)

        # Use paragraph-based semantic chunking for text files
        elif content_type == 'text':
            return self._create_text_semantic_chunks(content, chunk_size, chunk_overlap)

        # Use function/class boundary chunking for code files
        elif content_type == 'code':
            return self._create_code_structural_chunks(content, chunk_size, chunk_overlap)

        # Use file-based chunking for archive content
        elif content_type == 'archive':
            return self._create_archive_file_chunks(content, chunk_size, chunk_overlap)

        # Fallback to character-based chunking with word boundaries
        return self._create_character_chunks(content, chunk_size, chunk_overlap)

    def _create_topic_based_chunks(self, content: str, chunk_size: int, chunk_overlap: int, metadata: Dict[str, Any]) -> List[str]:
        """Create chunks based on topic boundaries for audio/video content.

        Args:
            content: Text content to chunk
            chunk_size: Maximum chunk size
            chunk_overlap: Overlap between chunks
            metadata: Metadata containing topic information

        Returns:
            List of topic-based chunks
        """
        import re

        # Look for topic headers in the content (updated for new format: # Topic Name [timestamp])
        topic_pattern = r'^#\s*(.+?)\s*\[\d+\](?:\n|$)'
        topic_matches = list(re.finditer(topic_pattern, content, re.MULTILINE))

        if not topic_matches:
            # No topics found, fall back to timestamp-based chunking
            return self._create_timestamp_chunks(content, chunk_size, chunk_overlap)

        chunks = []

        for i, match in enumerate(topic_matches):
            # Determine the start and end of this topic section
            topic_start = match.start()
            topic_end = topic_matches[i + 1].start() if i + 1 < len(topic_matches) else len(content)

            topic_content = content[topic_start:topic_end].strip()

            # If topic content is too large, split it further
            if len(topic_content) > chunk_size:
                # Split large topics at timestamp boundaries
                sub_chunks = self._split_topic_at_timestamps(topic_content, chunk_size, chunk_overlap)
                chunks.extend(sub_chunks)
            else:
                chunks.append(topic_content)

        return chunks

    def _create_timestamp_chunks(self, content: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Create chunks based on timestamp boundaries for audio/video content.

        Args:
            content: Text content to chunk
            chunk_size: Maximum chunk size
            chunk_overlap: Overlap between chunks

        Returns:
            List of timestamp-based chunks
        """
        import re

        # Look for timestamp patterns (updated for new format: [MM:SS][SPEAKER] or [HH:MM:SS][SPEAKER])
        # This pattern matches lines starting with timestamps, preserving the full line structure
        timestamp_pattern = r'^(\[\d{1,2}:\d{2}(?::\d{2})?\](?:\[[^\]]+\])?\s*.+?)(?=\n\[\d{1,2}:\d{2}|\n#|\Z)'
        timestamp_matches = list(re.finditer(timestamp_pattern, content, re.MULTILINE | re.DOTALL))

        if not timestamp_matches:
            # No timestamps found, fall back to character chunking
            return self._create_character_chunks(content, chunk_size, chunk_overlap)

        chunks = []
        current_chunk = ""

        for match in timestamp_matches:
            # Each match is a complete line with [timecode][speaker] text format
            line = match.group(1).strip()

            # If adding this line would exceed chunk size, finalize current chunk
            if current_chunk and len(current_chunk) + len(line) + 1 > chunk_size:  # +1 for newline
                chunks.append(current_chunk.strip())
                current_chunk = line
            else:
                current_chunk += "\n" + line if current_chunk else line

        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _split_topic_at_timestamps(self, topic_content: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Split a large topic at timestamp boundaries.

        Args:
            topic_content: Content of a single topic
            chunk_size: Maximum chunk size
            chunk_overlap: Overlap between chunks

        Returns:
            List of sub-chunks
        """
        import re

        # Extract topic header (updated for new format: # Topic Name [timestamp])
        header_pattern = r'^(#\s*.+?\s*\[\d+\])(?:\n|$)'
        header_match = re.search(header_pattern, topic_content, re.MULTILINE)
        topic_header = header_match.group(1) if header_match else ""

        # Get content after header
        content_start = header_match.end() if header_match else 0
        remaining_content = topic_content[content_start:].strip()

        # Split remaining content at timestamps
        timestamp_chunks = self._create_timestamp_chunks(remaining_content, chunk_size - len(topic_header), chunk_overlap)

        # Add topic header to each chunk
        result_chunks = []
        for chunk in timestamp_chunks:
            if topic_header:
                full_chunk = topic_header + "\n\n" + chunk
            else:
                full_chunk = chunk
            result_chunks.append(full_chunk)

        return result_chunks

    def _create_image_section_chunks(self, content: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Create chunks based on content sections for image analysis.

        Args:
            content: Text content to chunk
            chunk_size: Maximum chunk size
            chunk_overlap: Overlap between chunks

        Returns:
            List of text chunks split by sections
        """
        # Look for section headers in image content
        section_pattern = r'^##\s+(.+?)(?=\n##|\Z)'
        sections = re.split(r'^##\s+', content, flags=re.MULTILINE)

        if len(sections) <= 1:
            # No sections found, fall back to character chunking
            return self._create_character_chunks(content, chunk_size, chunk_overlap)

        chunks = []
        current_chunk = ""

        # Process each section
        for i, section in enumerate(sections):
            if not section.strip():
                continue

            # Add section header back (except for first empty split)
            if i > 0:
                section_lines = section.strip().split('\n', 1)
                if len(section_lines) > 1:
                    section_header = f"## {section_lines[0]}"
                    section_content = section_lines[1]
                    full_section = f"{section_header}\n{section_content}"
                else:
                    full_section = f"## {section.strip()}"
            else:
                full_section = section.strip()

            # Check if adding this section would exceed chunk size
            if current_chunk and len(current_chunk) + len(full_section) + 2 > chunk_size:
                # Save current chunk and start new one
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = full_section
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += f"\n\n{full_section}"
                else:
                    current_chunk = full_section

        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # If any chunk is still too large, split it further
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > chunk_size:
                # Split large chunks while preserving structure
                sub_chunks = self._create_character_chunks(chunk, chunk_size, chunk_overlap)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)

        return final_chunks

    def _create_web_article_chunks(self, content: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Create chunks based on article structure for web content.

        Args:
            content: Text content to chunk
            chunk_size: Maximum chunk size
            chunk_overlap: Overlap between chunks

        Returns:
            List of text chunks following article hierarchy
        """
        # Look for hierarchical headers (##, ###, etc.)
        header_pattern = r'^(#{2,})\s+(.+?)$'
        lines = content.split('\n')

        chunks = []
        current_chunk = ""
        current_section = ""

        for line in lines:
            header_match = re.match(header_pattern, line)

            if header_match:
                # Found a header
                header_level = len(header_match.group(1))
                header_text = header_match.group(2)

                # If we have content and this is a major section (## level), start new chunk
                if current_chunk.strip() and header_level == 2:
                    chunks.append(current_chunk.strip())
                    current_chunk = line
                else:
                    # Add header to current chunk
                    if current_chunk:
                        current_chunk += f"\n{line}"
                    else:
                        current_chunk = line
            else:
                # Regular content line
                if current_chunk:
                    current_chunk += f"\n{line}"
                else:
                    current_chunk = line

                # Check if chunk is getting too large
                if len(current_chunk) > chunk_size:
                    # Find a good break point (paragraph boundary)
                    paragraphs = current_chunk.split('\n\n')
                    if len(paragraphs) > 1:
                        # Keep all but the last paragraph in current chunk
                        chunk_content = '\n\n'.join(paragraphs[:-1])
                        chunks.append(chunk_content.strip())
                        current_chunk = paragraphs[-1]
                    else:
                        # No paragraph breaks, split at sentence boundaries
                        sentences = current_chunk.split('. ')
                        if len(sentences) > 1:
                            chunk_content = '. '.join(sentences[:-1]) + '.'
                            chunks.append(chunk_content.strip())
                            current_chunk = sentences[-1]

        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Post-process: merge very small chunks and split very large ones
        final_chunks = []
        for chunk in chunks:
            if len(chunk) < 200 and final_chunks:  # Merge small chunks
                final_chunks[-1] += f"\n\n{chunk}"
            elif len(chunk) > chunk_size * 1.5:  # Split very large chunks
                sub_chunks = self._create_character_chunks(chunk, chunk_size, chunk_overlap)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)

        return final_chunks

    def _create_text_semantic_chunks(self, content: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Create chunks using paragraph boundaries and semantic analysis for text files.

        Args:
            content: Text content to chunk
            chunk_size: Maximum chunk size
            chunk_overlap: Overlap between chunks

        Returns:
            List of text chunks with semantic boundaries
        """
        # Split by paragraphs first
        paragraphs = content.split('\n\n')

        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # Check if adding this paragraph would exceed chunk size
            if current_chunk and len(current_chunk) + len(paragraph) + 2 > chunk_size:
                # Save current chunk
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += f"\n\n{paragraph}"
                else:
                    current_chunk = paragraph

        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Handle chunks that are still too large by splitting at sentence boundaries
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > chunk_size:
                # Split at sentence boundaries
                sentences = re.split(r'(?<=[.!?])\s+', chunk)
                sub_chunk = ""

                for sentence in sentences:
                    if sub_chunk and len(sub_chunk) + len(sentence) + 1 > chunk_size:
                        if sub_chunk.strip():
                            final_chunks.append(sub_chunk.strip())
                        sub_chunk = sentence
                    else:
                        if sub_chunk:
                            sub_chunk += f" {sentence}"
                        else:
                            sub_chunk = sentence

                if sub_chunk.strip():
                    final_chunks.append(sub_chunk.strip())
            else:
                final_chunks.append(chunk)

        return final_chunks

    def _create_code_structural_chunks(self, content: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Create chunks based on function/class boundaries for code files.

        Args:
            content: Text content to chunk
            chunk_size: Maximum chunk size
            chunk_overlap: Overlap between chunks

        Returns:
            List of text chunks preserving code structure
        """
        lines = content.split('\n')
        chunks = []
        current_chunk = ""
        current_function = ""
        indent_level = 0

        # Patterns for different languages
        function_patterns = [
            r'^\s*(def\s+\w+.*?:)',  # Python functions
            r'^\s*(function\s+\w+.*?\{)',  # JavaScript functions
            r'^\s*(class\s+\w+.*?[:{])',  # Class definitions
            r'^\s*(public|private|protected)?\s*(static\s+)?\w+\s+\w+\s*\([^)]*\)\s*\{',  # Java/C# methods
        ]

        for line in lines:
            # Check if this line starts a function/class
            is_function_start = any(re.match(pattern, line) for pattern in function_patterns)

            if is_function_start:
                # If we have accumulated content and this would make chunk too large, save it
                if current_chunk and len(current_chunk) + len(current_function) > chunk_size:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = current_function
                    current_function = line
                else:
                    # Add previous function to chunk
                    if current_function:
                        if current_chunk:
                            current_chunk += f"\n\n{current_function}"
                        else:
                            current_chunk = current_function
                    current_function = line

                # Track indentation for function end detection
                indent_level = len(line) - len(line.lstrip())
            else:
                # Add line to current function
                if current_function:
                    current_function += f"\n{line}"
                else:
                    # Standalone line (imports, comments, etc.)
                    if current_chunk:
                        current_chunk += f"\n{line}"
                    else:
                        current_chunk = line

        # Add final function and chunk
        if current_function:
            if current_chunk:
                current_chunk += f"\n\n{current_function}"
            else:
                current_chunk = current_function

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Handle chunks that are still too large
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > chunk_size * 1.5:
                # Split large chunks while trying to preserve function boundaries
                sub_chunks = self._create_character_chunks(chunk, chunk_size, chunk_overlap)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)

        return final_chunks

    def _create_archive_file_chunks(self, content: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Create chunks based on file boundaries for archive content.

        Args:
            content: Text content to chunk
            chunk_size: Maximum chunk size
            chunk_overlap: Overlap between chunks

        Returns:
            List of text chunks split by file boundaries
        """
        # Look for file boundary markers (typically added during archive extraction)
        file_pattern = r'^---\s*File:\s*(.+?)\s*---'
        sections = re.split(file_pattern, content, flags=re.MULTILINE)

        if len(sections) <= 1:
            # No file boundaries found, fall back to character chunking
            return self._create_character_chunks(content, chunk_size, chunk_overlap)

        chunks = []
        current_chunk = ""

        # Process each file section
        for i in range(0, len(sections), 2):
            if i + 1 < len(sections):
                file_name = sections[i + 1] if i + 1 < len(sections) else "unknown"
                file_content = sections[i + 2] if i + 2 < len(sections) else ""

                # Create file header
                file_header = f"--- File: {file_name} ---"
                full_file = f"{file_header}\n{file_content.strip()}"

                # Check if adding this file would exceed chunk size
                if current_chunk and len(current_chunk) + len(full_file) + 2 > chunk_size:
                    # Save current chunk and start new one
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = full_file
                else:
                    # Add to current chunk
                    if current_chunk:
                        current_chunk += f"\n\n{full_file}"
                    else:
                        current_chunk = full_file
            else:
                # Handle remaining content
                if sections[i].strip():
                    if current_chunk:
                        current_chunk += f"\n{sections[i].strip()}"
                    else:
                        current_chunk = sections[i].strip()

        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # If any chunk is still too large, split it further while preserving file context
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > chunk_size:
                # Try to split at file boundaries within the chunk
                file_sections = re.split(r'^---\s*File:', chunk, flags=re.MULTILINE)
                if len(file_sections) > 1:
                    # Multiple files in chunk, split them
                    sub_chunk = file_sections[0].strip()
                    for j in range(1, len(file_sections)):
                        file_part = f"--- File:{file_sections[j]}"
                        if sub_chunk and len(sub_chunk) + len(file_part) > chunk_size:
                            if sub_chunk.strip():
                                final_chunks.append(sub_chunk.strip())
                            sub_chunk = file_part
                        else:
                            if sub_chunk:
                                sub_chunk += f"\n\n{file_part}"
                            else:
                                sub_chunk = file_part
                    if sub_chunk.strip():
                        final_chunks.append(sub_chunk.strip())
                else:
                    # Single large file, use character chunking
                    sub_chunks = self._create_character_chunks(chunk, chunk_size, chunk_overlap)
                    final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)

        return final_chunks

    def _create_document_chunks(self, content: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Create chunks optimized for document content.

        Args:
            content: Text content to chunk
            chunk_size: Maximum chunk size
            chunk_overlap: Overlap between chunks

        Returns:
            List of document chunks
        """
        import re

        # Try to split at section boundaries first
        section_pattern = r'\n\n+'
        sections = re.split(section_pattern, content)

        chunks = []
        current_chunk = ""

        for section in sections:
            section = section.strip()
            if not section:
                continue

            # If adding this section would exceed chunk size, finalize current chunk
            if current_chunk and len(current_chunk) + len(section) > chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = section
            else:
                current_chunk += "\n\n" + section if current_chunk else section

        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # If any chunk is still too large, split it further
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > chunk_size:
                sub_chunks = self._create_character_chunks(chunk, chunk_size, chunk_overlap)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)

        return final_chunks

    def _create_character_chunks(self, content: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Create chunks using character-based splitting with word boundary preservation.

        Args:
            content: Text content to chunk
            chunk_size: Maximum chunk size
            chunk_overlap: Overlap between chunks

        Returns:
            List of character-based chunks
        """
        chunks = []
        start = 0

        while start < len(content):
            end = start + chunk_size

            if end >= len(content):
                # Last chunk
                chunk = content[start:]
            else:
                # Find word boundary
                chunk = content[start:end]
                last_space = chunk.rfind(' ')
                if last_space > chunk_size // 2:  # Only break on word if reasonable
                    chunk = content[start:start + last_space]
                    end = start + last_space

            if chunk.strip():
                chunks.append(chunk.strip())

            start = end - chunk_overlap
            if start >= len(content):
                break

        return chunks

    async def _extract_graph_data(
        self,
        content: str,
        source_path: str,
        document_id: str,
        metadata: Dict[str, Any],
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract facts and relationships for graph databases using chunk-based approach."""
        try:
            logger.info(f"Extracting fact data from chunks ({len(content)} chars)")

            # Create chunks using the same logic as embeddings
            content_type = metadata.get('content_type', 'document')
            chunks = self._create_chunks(content, chunk_size, chunk_overlap, content_type, metadata)

            logger.info(f"Processing {len(chunks)} chunks for fact extraction")

            all_facts = []
            all_relationships = []
            chunk_fact_mapping = {}

            # Process each chunk individually for fact extraction
            for i, chunk_content in enumerate(chunks):
                try:
                    logger.debug(f"Processing chunk {i+1}/{len(chunks)} for fact extraction")

                    # Create document chunk with metadata using unified ID generator
                    from morag_graph.utils.id_generation import UnifiedIDGenerator
                    chunk_id = UnifiedIDGenerator.generate_chunk_id(document_id, i)
                    chunk_metadata = {
                        'source_file_path': source_path,
                        'source_file_name': Path(source_path).name if source_path else None,
                        'chunk_index': i,
                        'content_type': content_type,
                        **metadata  # Include all original metadata
                    }

                    # Extract facts from this chunk using fact extractor
                    chunk_facts, chunk_relationships = await self.fact_extractor.extract_facts_and_relationships(
                        text=chunk_content,
                        doc_id=document_id,
                        domain=metadata.get('domain', 'general'),
                        context={
                            'language': language or 'en',
                            'chunk_index': i,
                            **chunk_metadata
                        }
                    )

                    # Process facts from this chunk
                    if chunk_facts:
                        for fact in chunk_facts:
                            # Convert fact to dictionary format for storage
                            fact_dict = fact.to_dict()

                            # Add chunk-specific metadata
                            fact_dict['chunk_index'] = i
                            fact_dict['chunk_id'] = chunk_id

                            all_facts.append(fact_dict)

                            # Track facts by chunk for mapping
                            if chunk_id not in chunk_fact_mapping:
                                chunk_fact_mapping[chunk_id] = []
                            chunk_fact_mapping[chunk_id].append(fact.id)

                    # Process relationships from this chunk
                    if chunk_relationships:
                        for relationship in chunk_relationships:
                            # Convert relationship to dictionary format for storage
                            relationship_dict = {
                                'id': relationship.id,
                                'source_fact_id': relationship.source_fact_id,
                                'target_fact_id': relationship.target_fact_id,
                                'relationship_type': relationship.relationship_type,
                                'confidence': relationship.confidence,
                                'description': relationship.description,
                                'created_at': relationship.created_at.isoformat(),
                                'chunk_index': i,
                                'chunk_id': chunk_id
                            }
                            all_relationships.append(relationship_dict)

                    logger.debug(f"Chunk {i+1} processed: {len(chunk_facts)} facts, {len(chunk_relationships)} relationships")

                except Exception as e:
                    logger.warning(f"Failed to process chunk {i+1} for fact extraction", error=str(e))
                    # Continue with other chunks
                    continue

            logger.info(f"Extracted {len(all_facts)} facts and {len(all_relationships)} relationships from {len(chunks)} chunks")
            logger.info(f"Created chunk-fact mapping: {len(chunk_fact_mapping)} chunks with facts")

            # Note: Enhanced fact processing will be done later in _write_to_neo4j when database connection is available

            # Step: Generate embeddings for entities and facts
            entity_embeddings, fact_embeddings = await self._generate_entity_and_fact_embeddings(
                all_facts, all_relationships
            )

            return {
                'facts': all_facts,
                'relationships': all_relationships,
                'chunk_fact_mapping': chunk_fact_mapping,
                'entity_embeddings': entity_embeddings,
                'fact_embeddings': fact_embeddings,
                'enhanced_processing': None,  # Will be done in _write_to_neo4j
                'extraction_metadata': {
                    'total_facts': len(all_facts),
                    'total_relationships': len(all_relationships),
                    'extraction_method': 'chunk_based_facts',
                    'chunks_processed': len(chunks),
                    'chunks_with_facts': len(chunk_fact_mapping),
                    'entities_with_embeddings': len(entity_embeddings),
                    'facts_with_embeddings': len(fact_embeddings),
                    'entities_from_facts': 0,  # Will be populated during Neo4j processing
                    'keyword_entities_created': 0,  # Will be populated during Neo4j processing
                    'fact_entity_relations': 0  # Will be populated during Neo4j processing
                }
            }

        except Exception as e:
            logger.warning("Failed to extract fact data", error=str(e))
            return {
                'facts': [],
                'relationships': [],
                'chunk_fact_mapping': {},
                'extraction_metadata': {'error': str(e)}
            }

    async def _generate_entity_and_fact_embeddings(
        self,
        facts: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Generate embeddings for entities and facts extracted during ingestion.

        Args:
            facts: List of extracted facts
            relationships: List of extracted relationships

        Returns:
            Tuple of (entity_embeddings, fact_embeddings) dictionaries
        """
        entity_embeddings = {}
        fact_embeddings = {}

        try:
            # Extract unique entities from facts
            unique_entities = {}

            # Collect entities from facts (subjects, objects, keywords)
            for fact in facts:
                # Add subject entity - use normalized name as key
                if fact.get('subject'):
                    entity_name = fact['subject'].strip()
                    entity_key = entity_name.lower()  # Use normalized name as key
                    if entity_key not in unique_entities:
                        unique_entities[entity_key] = {
                            'name': entity_name,
                            'type': 'ENTITY',  # Use generic ENTITY type
                            'context': f"Subject entity from fact domain: {fact.get('domain', 'general')}",
                            'original_type': 'SUBJECT'  # Keep original semantic type for reference
                        }

                # Add object entity - use normalized name as key
                if fact.get('object'):
                    entity_name = fact['object'].strip()
                    entity_key = entity_name.lower()  # Use normalized name as key
                    if entity_key not in unique_entities:
                        unique_entities[entity_key] = {
                            'name': entity_name,
                            'type': 'ENTITY',  # Use generic ENTITY type
                            'context': f"Object entity from fact domain: {fact.get('domain', 'general')}",
                            'original_type': 'OBJECT'  # Keep original semantic type for reference
                        }

                # Add keyword entities - use normalized name as key
                if fact.get('keywords'):
                    keywords = fact['keywords'].split(',') if isinstance(fact['keywords'], str) else fact['keywords']
                    if isinstance(keywords, list):
                        for keyword in keywords:
                            keyword = keyword.strip()
                            if keyword:
                                entity_key = keyword.lower()  # Use normalized name as key
                                if entity_key not in unique_entities:
                                    unique_entities[entity_key] = {
                                        'name': keyword,
                                        'type': 'ENTITY',  # Use generic ENTITY type
                                        'context': f"Keyword entity from fact domain: {fact.get('domain', 'general')}",
                                        'original_type': 'KEYWORD'  # Keep original semantic type for reference
                                    }

            logger.info(f"Found {len(unique_entities)} unique entities for embedding generation")

            # Generate embeddings for entities in batches
            if unique_entities:
                entity_embeddings = await self._generate_entity_embeddings_batch(unique_entities)

            # Generate embeddings for facts in batches
            if facts:
                fact_embeddings = await self._generate_fact_embeddings_batch(facts)

            logger.info(f"Generated embeddings: {len(entity_embeddings)} entities, {len(fact_embeddings)} facts")

            return entity_embeddings, fact_embeddings

        except Exception as e:
            logger.error(f"Failed to generate entity and fact embeddings: {e}")
            return {}, {}

    async def _generate_entity_embeddings_batch(
        self,
        entities: Dict[str, Dict[str, Any]],
        batch_size: int = 50
    ) -> Dict[str, Any]:
        """Generate embeddings for entities in batches.

        Args:
            entities: Dictionary of entities to generate embeddings for
            batch_size: Number of entities to process in each batch

        Returns:
            Dictionary mapping entity keys to embedding data
        """
        entity_embeddings = {}
        entity_list = list(entities.items())

        try:
            for i in range(0, len(entity_list), batch_size):
                batch = entity_list[i:i + batch_size]
                logger.info(f"Processing entity embedding batch {i//batch_size + 1}: {len(batch)} entities")

                # Prepare texts for batch embedding
                batch_texts = []
                batch_keys = []

                for entity_key, entity_data in batch:
                    # Create entity text for embedding
                    entity_text = f"{entity_data['name']} ({entity_data['type']})"
                    if entity_data.get('context'):
                        entity_text += f" - {entity_data['context']}"

                    batch_texts.append(entity_text)
                    batch_keys.append(entity_key)

                # Generate embeddings for the batch
                try:
                    batch_embeddings = await self.embedding_service.generate_embeddings_batch(
                        batch_texts, task_type="retrieval_document"
                    )

                    # Store embeddings with metadata
                    for j, (entity_key, entity_data) in enumerate(batch):
                        if j < len(batch_embeddings):
                            embedding = batch_embeddings[j]
                            # Handle different embedding result types
                            if hasattr(embedding, 'embedding'):
                                # EmbeddingResult object
                                embedding_vector = embedding.embedding
                            elif isinstance(embedding, list):
                                # Direct list of floats
                                embedding_vector = embedding
                            else:
                                logger.warning(f"Unexpected embedding type: {type(embedding)}")
                                continue

                            entity_embeddings[entity_key] = {
                                'name': entity_data['name'],
                                'type': entity_data['type'],
                                'embedding': embedding_vector,
                                'embedding_model': 'text-embedding-004',
                                'embedding_dimensions': len(embedding_vector),
                                'context': entity_data.get('context', '')
                            }

                except Exception as e:
                    logger.warning(f"Failed to generate embeddings for entity batch {i//batch_size + 1}: {e}")
                    # Try individual embeddings as fallback
                    for entity_key, entity_data in batch:
                        try:
                            entity_text = f"{entity_data['name']} ({entity_data['type']})"
                            if entity_data.get('context'):
                                entity_text += f" - {entity_data['context']}"

                            embedding = await self.embedding_service.generate_embedding(
                                entity_text, task_type="retrieval_document"
                            )

                            # Handle different embedding result types
                            if hasattr(embedding, 'embedding'):
                                # EmbeddingResult object
                                embedding_vector = embedding.embedding
                            elif isinstance(embedding, list):
                                # Direct list of floats
                                embedding_vector = embedding
                            else:
                                logger.warning(f"Unexpected embedding type: {type(embedding)}")
                                continue

                            entity_embeddings[entity_key] = {
                                'name': entity_data['name'],
                                'type': entity_data['type'],
                                'embedding': embedding_vector,
                                'embedding_model': 'text-embedding-004',
                                'embedding_dimensions': len(embedding_vector),
                                'context': entity_data.get('context', '')
                            }

                        except Exception as individual_e:
                            logger.warning(f"Failed to generate embedding for entity {entity_key}: {individual_e}")

                # Small delay between batches to avoid rate limiting
                if i + batch_size < len(entity_list):
                    await asyncio.sleep(0.1)

            return entity_embeddings

        except Exception as e:
            logger.error(f"Failed to generate entity embeddings in batches: {e}")
            return {}

    async def _generate_fact_embeddings_batch(
        self,
        facts: List[Dict[str, Any]],
        batch_size: int = 50
    ) -> Dict[str, Any]:
        """Generate embeddings for facts in batches.

        Args:
            facts: List of facts to generate embeddings for
            batch_size: Number of facts to process in each batch

        Returns:
            Dictionary mapping fact IDs to embedding data
        """
        fact_embeddings = {}

        try:
            for i in range(0, len(facts), batch_size):
                batch = facts[i:i + batch_size]
                logger.info(f"Processing fact embedding batch {i//batch_size + 1}: {len(batch)} facts")

                # Prepare texts for batch embedding
                batch_texts = []
                batch_ids = []

                for fact in batch:
                    # Create structured fact text for embedding
                    fact_text = self._create_fact_text_for_embedding(fact)
                    batch_texts.append(fact_text)
                    batch_ids.append(fact['id'])

                # Generate embeddings for the batch
                try:
                    batch_embeddings = await self.embedding_service.generate_embeddings_batch(
                        batch_texts, task_type="retrieval_document"
                    )

                    # Store embeddings with metadata
                    for j, fact in enumerate(batch):
                        if j < len(batch_embeddings):
                            embedding = batch_embeddings[j]
                            # Handle different embedding result types
                            if hasattr(embedding, 'embedding'):
                                # EmbeddingResult object
                                embedding_vector = embedding.embedding
                            elif isinstance(embedding, list):
                                # Direct list of floats
                                embedding_vector = embedding
                            else:
                                logger.warning(f"Unexpected embedding type: {type(embedding)}")
                                continue

                            fact_embeddings[fact['id']] = {
                                'fact_id': fact['id'],
                                'embedding': embedding_vector,
                                'embedding_model': 'text-embedding-004',
                                'embedding_dimensions': len(embedding_vector),
                                'fact_text': batch_texts[j],
                                'subject': fact.get('subject'),
                                'approach': fact.get('approach'),
                                'object': fact.get('object'),
                                'solution': fact.get('solution'),
                                'domain': fact.get('domain')
                            }

                except Exception as e:
                    logger.warning(f"Failed to generate embeddings for fact batch {i//batch_size + 1}: {e}")
                    # Try individual embeddings as fallback
                    for fact in batch:
                        try:
                            fact_text = self._create_fact_text_for_embedding(fact)
                            embedding = await self.embedding_service.generate_embedding(
                                fact_text, task_type="retrieval_document"
                            )

                            # Handle different embedding result types
                            if hasattr(embedding, 'embedding'):
                                # EmbeddingResult object
                                embedding_vector = embedding.embedding
                            elif isinstance(embedding, list):
                                # Direct list of floats
                                embedding_vector = embedding
                            else:
                                logger.warning(f"Unexpected embedding type: {type(embedding)}")
                                continue

                            fact_embeddings[fact['id']] = {
                                'fact_id': fact['id'],
                                'embedding': embedding_vector,
                                'embedding_model': 'text-embedding-004',
                                'embedding_dimensions': len(embedding_vector),
                                'fact_text': fact_text,
                                'subject': fact.get('subject'),
                                'approach': fact.get('approach'),
                                'object': fact.get('object'),
                                'solution': fact.get('solution'),
                                'domain': fact.get('domain')
                            }

                        except Exception as individual_e:
                            logger.warning(f"Failed to generate embedding for fact {fact['id']}: {individual_e}")

                # Small delay between batches to avoid rate limiting
                if i + batch_size < len(facts):
                    await asyncio.sleep(0.1)

            return fact_embeddings

        except Exception as e:
            logger.error(f"Failed to generate fact embeddings in batches: {e}")
            return {}

    def _create_fact_text_for_embedding(self, fact: Dict[str, Any]) -> str:
        """Create text representation of fact for embedding generation.

        Args:
            fact: Fact dictionary

        Returns:
            Formatted fact text for embedding
        """
        # Create structured fact text similar to the FactEmbeddingService
        text_parts = []

        if fact.get('subject'):
            text_parts.append(f"Subject: {fact['subject']}")

        if fact.get('approach'):
            text_parts.append(f"Approach: {fact['approach']}")

        if fact.get('object'):
            text_parts.append(f"Object: {fact['object']}")

        if fact.get('solution'):
            text_parts.append(f"Solution: {fact['solution']}")

        # Add keywords if available
        if fact.get('keywords'):
            text_parts.append(f"Keywords: {fact['keywords']}")

        # Add domain context
        domain = fact.get('domain', '')
        if domain and domain != 'general':
            text_parts.append(f"Domain: {domain}")

        # Join all parts
        fact_text = ". ".join(text_parts)

        # Fallback to any available text field if structured approach fails
        if not fact_text.strip():
            for field in ['fact_text', 'text', 'description']:
                if fact.get(field):
                    fact_text = fact[field]
                    break

        return fact_text or "No fact text available"

    async def _process_facts_with_enhanced_processing(self, facts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process facts with enhanced entity creation and relationship management.

        Args:
            facts: List of fact dictionaries

        Returns:
            Enhanced processing results
        """
        if not facts:
            return {
                'facts_processed': 0,
                'entities_created': 0,
                'keyword_entities_created': 0,
                'relations_created': 0
            }

        try:
            # Convert fact dictionaries to Fact objects
            fact_objects = []
            for fact_dict in facts:
                try:
                    # Create Fact object from dictionary
                    fact = Fact(
                        subject=fact_dict.get('subject', ''),
                        object=fact_dict.get('object', ''),
                        approach=fact_dict.get('approach'),
                        solution=fact_dict.get('solution'),
                        condition=fact_dict.get('condition'),
                        remarks=fact_dict.get('remarks'),
                        source_chunk_id=fact_dict.get('chunk_id', ''),
                        source_document_id=fact_dict.get('document_id', ''),
                        extraction_confidence=fact_dict.get('confidence', 0.8),
                        fact_type=fact_dict.get('fact_type', 'definition'),
                        domain=fact_dict.get('domain', 'general'),
                        language=fact_dict.get('language', 'en'),
                        keywords=fact_dict.get('keywords', [])
                    )
                    fact_objects.append(fact)
                except Exception as e:
                    self.logger.warning(f"Failed to convert fact dictionary to object: {e}")
                    continue

            if not fact_objects:
                self.logger.warning("No valid fact objects created from dictionaries")
                return {
                    'facts_processed': 0,
                    'entities_created': 0,
                    'keyword_entities_created': 0,
                    'relations_created': 0
                }

            # Initialize enhanced fact processing service
            from morag_graph.storage.neo4j_storage import Neo4jStorage, Neo4jConfig
            import os

            # Create Neo4j storage for enhanced processing
            neo4j_config = Neo4jConfig(
                uri=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
                username=os.getenv('NEO4J_USERNAME', 'neo4j'),
                password=os.getenv('NEO4J_PASSWORD', 'password'),
                database=os.getenv('NEO4J_DATABASE', 'neo4j')
            )
            neo4j_storage = Neo4jStorage(neo4j_config)
            await neo4j_storage.connect()

            # Initialize entity normalizer for enhanced processing
            from morag_graph.extraction.entity_normalizer import LLMEntityNormalizer

            # Get API key from environment
            import os
            api_key = os.getenv('GEMINI_API_KEY')
            if api_key:
                entity_normalizer = LLMEntityNormalizer(
                    model_name=os.getenv('MORAG_GEMINI_MODEL', 'gemini-2.0-flash'),
                    api_key=api_key,
                    language='de'  # Use German for this document
                )
            else:
                entity_normalizer = None
                logger.warning("GEMINI_API_KEY not found - entity normalization disabled")

            enhanced_processor = EnhancedFactProcessingService(neo4j_storage, entity_normalizer)

            # Process facts with entity creation and relationship management
            result = await enhanced_processor.process_facts_with_entities(
                fact_objects,
                create_keyword_entities=True,
                create_mandatory_relations=True
            )

            await neo4j_storage.disconnect()

            self.logger.info(
                "Enhanced fact processing completed",
                facts_processed=result['facts_processed'],
                entities_created=result['entities_created'],
                keyword_entities=result['keyword_entities_created'],
                relations_created=result['relations_created']
            )

            return result

        except Exception as e:
            self.logger.error(f"Enhanced fact processing failed: {e}")
            return {
                'facts_processed': 0,
                'entities_created': 0,
                'keyword_entities_created': 0,
                'relations_created': 0,
                'error': str(e)
            }





    def _create_chunk_entity_mapping(
        self,
        content: str,
        entities: List[Entity],
        chunk_size: int,
        chunk_overlap: int
    ) -> Dict[str, List[str]]:
        """Create mapping of chunk indices to entity IDs by finding which entities appear in which chunks.

        Args:
            content: Full document content
            entities: List of extracted entities
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks

        Returns:
            Dictionary mapping chunk index (as string) to list of entity IDs
        """
        # Create chunks using the same method as the ingestion coordinator
        chunks = self._create_chunks(content, chunk_size, chunk_overlap, 'document', {})

        chunk_entity_mapping = {}

        logger.info(f"Analyzing {len(chunks)} chunks for entity mentions...")

        for chunk_index, chunk_text in enumerate(chunks):
            entities_in_chunk = []

            # Check each entity to see if it appears in this chunk
            for entity in entities:
                entity_name = entity.name.lower()
                chunk_text_lower = chunk_text.lower()

                # Simple substring matching - could be improved with fuzzy matching
                if entity_name in chunk_text_lower:
                    entities_in_chunk.append(entity.id)
                    logger.debug(f"Found entity '{entity.name}' in chunk {chunk_index}")

            if entities_in_chunk:
                chunk_entity_mapping[str(chunk_index)] = entities_in_chunk
                logger.debug(f"Chunk {chunk_index} contains {len(entities_in_chunk)} entities")

        logger.info(f"Found entities in {len(chunk_entity_mapping)} out of {len(chunks)} chunks")
        return chunk_entity_mapping

    def _enhance_chunk_entity_mapping_with_missing_entities(
        self,
        content: str,
        relations: List,
        chunk_entity_mapping: Dict[str, List[str]],
        chunk_size: int,
        chunk_overlap: int
    ) -> Dict[str, List[str]]:
        """Enhance chunk-entity mapping by adding auto-created entities from relations.

        Args:
            content: Full document content
            relations: List of relations (may contain auto-created entities)
            chunk_entity_mapping: Existing chunk-entity mapping
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks

        Returns:
            Enhanced chunk-entity mapping
        """
        if not relations:
            return chunk_entity_mapping

        # Find auto-created entities (those with creation_reason attribute)
        auto_created_entities = set()
        entity_names_to_ids = {}

        for relation in relations:
            # Check if source or target entities are auto-created
            source_id = relation.source_entity_id
            target_id = relation.target_entity_id

            # Extract entity names from relation attributes
            source_name = relation.attributes.get('source_entity_name', '')
            target_name = relation.attributes.get('target_entity_name', '')

            if source_id and source_name:
                entity_names_to_ids[source_name] = source_id
                # Check if this looks like an auto-created entity (has document hash suffix)
                if '_' in source_id and len(source_id.split('_')[-1]) >= 8:
                    auto_created_entities.add((source_name, source_id))

            if target_id and target_name:
                entity_names_to_ids[target_name] = target_id
                # Check if this looks like an auto-created entity (has document hash suffix)
                if '_' in target_id and len(target_id.split('_')[-1]) >= 8:
                    auto_created_entities.add((target_name, target_id))

        if not auto_created_entities:
            logger.debug("No auto-created entities found in relations")
            return chunk_entity_mapping

        logger.info(f"Found {len(auto_created_entities)} auto-created entities to map to chunks")

        # Create chunks for searching
        chunks = self._create_chunks(content, chunk_size, chunk_overlap, 'document', {})

        # Search for auto-created entities in chunks
        entities_added_to_chunks = 0

        for entity_name, entity_id in auto_created_entities:
            entity_name_lower = entity_name.lower()

            for chunk_index, chunk_text in enumerate(chunks):
                chunk_text_lower = chunk_text.lower()

                # Simple substring matching - same as regular entity mapping
                if entity_name_lower in chunk_text_lower:
                    chunk_index_str = str(chunk_index)

                    # Add to chunk-entity mapping
                    if chunk_index_str not in chunk_entity_mapping:
                        chunk_entity_mapping[chunk_index_str] = []

                    if entity_id not in chunk_entity_mapping[chunk_index_str]:
                        chunk_entity_mapping[chunk_index_str].append(entity_id)
                        entities_added_to_chunks += 1
                        logger.debug(f"Added auto-created entity '{entity_name}' ({entity_id}) to chunk {chunk_index}")

        # For auto-created entities that weren't found in any chunk,
        # connect them to chunks where their related entities are found
        unconnected_entities = []
        for entity_name, entity_id in auto_created_entities:
            # Check if this entity was added to any chunk
            found_in_chunk = False
            for chunk_entities in chunk_entity_mapping.values():
                if entity_id in chunk_entities:
                    found_in_chunk = True
                    break

            if not found_in_chunk:
                unconnected_entities.append((entity_name, entity_id))

        if unconnected_entities:
            logger.info(f"Found {len(unconnected_entities)} unconnected auto-created entities, connecting via relations")

            # For each unconnected entity, find relations it's involved in
            for entity_name, entity_id in unconnected_entities:
                connected_to_chunk = False

                for relation in relations:
                    related_entity_id = None

                    # Check if this entity is source or target in the relation
                    if relation.source_entity_id == entity_id:
                        related_entity_id = relation.target_entity_id
                    elif relation.target_entity_id == entity_id:
                        related_entity_id = relation.source_entity_id

                    if related_entity_id:
                        # Find which chunk the related entity is in
                        for chunk_index_str, chunk_entities in chunk_entity_mapping.items():
                            if related_entity_id in chunk_entities:
                                # Add the unconnected entity to the same chunk
                                if entity_id not in chunk_entities:
                                    chunk_entities.append(entity_id)
                                    entities_added_to_chunks += 1
                                    connected_to_chunk = True
                                    logger.debug(f"Connected auto-created entity '{entity_name}' ({entity_id}) to chunk {chunk_index_str} via related entity {related_entity_id}")
                                break

                    if connected_to_chunk:
                        break

                # If still not connected, add to the first chunk as fallback
                if not connected_to_chunk and chunk_entity_mapping:
                    first_chunk = next(iter(chunk_entity_mapping.keys()))
                    chunk_entity_mapping[first_chunk].append(entity_id)
                    entities_added_to_chunks += 1
                    logger.warning(f"Connected auto-created entity '{entity_name}' ({entity_id}) to first chunk {first_chunk} as fallback")

        logger.info(f"Enhanced chunk-entity mapping: added {entities_added_to_chunks} auto-created entity mappings")
        return chunk_entity_mapping

    def _create_ingest_result(
        self,
        source_path: str,
        content_type: str,
        metadata: Dict[str, Any],
        processing_result: ProcessingResult,
        embeddings_data: Dict[str, Any],
        graph_data: Dict[str, Any],
        database_configs: List[DatabaseConfig],
        start_time: datetime,
        document_summary: str,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create the complete ingest_result.json data structure."""
        # Extract content length from different ProcessingResult types
        content_length = 0
        if hasattr(processing_result, 'content') and processing_result.content:
            content_length = len(processing_result.content)
        elif hasattr(processing_result, 'document') and processing_result.document:
            if hasattr(processing_result.document, 'raw_text'):
                content_length = len(processing_result.document.raw_text or '')
            elif hasattr(processing_result.document, 'content'):
                content_length = len(processing_result.document.content or '')

        return {
            'ingestion_id': str(uuid.uuid4()),
            'timestamp': start_time.isoformat(),
            'language': language,  # Add language information
            'source_info': {
                'source_path': source_path,
                'content_type': content_type,
                'document_id': embeddings_data['document_id']
            },
            'processing_result': {
                'success': processing_result.success,
                'processing_time': processing_result.processing_time,
                'content_length': content_length,
                'metadata': processing_result.metadata
            },
            'databases_configured': [
                {
                    'type': db.type.value,
                    'hostname': db.hostname,
                    'port': db.port,
                    'database_name': db.database_name
                }
                for db in database_configs
            ],
            'embeddings_data': {
                'chunk_count': len(embeddings_data['chunks']),
                'chunk_size': embeddings_data['chunk_size'],
                'chunk_overlap': embeddings_data['chunk_overlap'],
                'embedding_dimension': len(embeddings_data['embeddings'][0]) if embeddings_data['embeddings'] else 0,
                'chunks': [
                    {
                        'chunk_id': meta['chunk_id'],
                        'chunk_index': meta['chunk_index'],
                        'chunk_text': chunk_text,
                        'chunk_size': meta['chunk_size'],
                        'embedding': embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
                        'metadata': {k: v for k, v in meta.items() if k not in ['chunk_text']}
                    }
                    for meta, embedding, chunk_text in zip(embeddings_data['chunk_metadata'], embeddings_data['embeddings'], embeddings_data['chunks'])
                ]
            },
            'graph_data': {
                'language': language,  # Add language information to graph data
                'facts_count': len(graph_data.get('facts', [])),
                'relationships_count': len(graph_data.get('relationships', [])),
                'facts': graph_data.get('facts', []),
                'relationships': graph_data.get('relationships', []),
                'chunk_fact_mapping': graph_data.get('chunk_fact_mapping', {}),
                'extraction_metadata': graph_data['extraction_metadata']
            },
            'metadata': metadata,
            'summary': document_summary,
            'status': 'processing',
            'database_results': {}  # Will be filled after database writes
        }

    def _write_ingest_result_file(self, source_path: str, ingest_result: Dict[str, Any]) -> str:
        """Write the ingest_result.json file."""
        source_path_obj = Path(source_path)
        result_file_path = source_path_obj.parent / f"{source_path_obj.stem}.ingest_result.json"

        with open(result_file_path, 'w', encoding='utf-8') as f:
            json.dump(ingest_result, f, indent=2, ensure_ascii=False)

        return str(result_file_path)

    def _create_ingest_data(
        self,
        embeddings_data: Dict[str, Any],
        graph_data: Dict[str, Any],
        database_configs: List[DatabaseConfig],
        document_id: str,
        source_path: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create the ingest_data.json data structure for database writes."""
        # Extract document metadata from chunk metadata (first chunk contains document metadata)
        document_metadata = {}
        if embeddings_data['chunk_metadata']:
            first_chunk_meta = embeddings_data['chunk_metadata'][0]

            # Extract core document fields from metadata
            source_path_obj = Path(source_path)
            document_metadata = {
                'file_name': first_chunk_meta.get('file_name', source_path_obj.name),
                'source_file': first_chunk_meta.get('source_path', source_path),
                'name': first_chunk_meta.get('source_name', first_chunk_meta.get('file_name', source_path_obj.name)),
                'mime_type': first_chunk_meta.get('mime_type', 'unknown'),
                'file_size': first_chunk_meta.get('file_size'),
                'checksum': first_chunk_meta.get('checksum', first_chunk_meta.get('content_checksum')),
                'summary': first_chunk_meta.get('summary'),
                'metadata': metadata
            }

            # Remove None values
            document_metadata = {k: v for k, v in document_metadata.items() if v is not None}

        return {
            'document_id': document_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'databases': [db.type.value for db in database_configs],
            'document_metadata': document_metadata,
            'vector_data': {
                'chunks': [
                    {
                        'chunk_id': meta['chunk_id'],
                        'chunk_index': meta['chunk_index'],
                        'chunk_text': chunk_text,
                        'embedding': embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
                        'metadata': meta
                    }
                    for meta, embedding, chunk_text in zip(embeddings_data['chunk_metadata'], embeddings_data['embeddings'], embeddings_data['chunks'])
                ]
            },
            'graph_data': {
                'facts': graph_data.get('facts', []),
                'relationships': graph_data.get('relationships', []),
                'chunk_fact_mapping': graph_data.get('chunk_fact_mapping', {}),
                'extraction_metadata': graph_data.get('extraction_metadata', {}),
                'enhanced_processing': self._serialize_enhanced_processing(graph_data.get('enhanced_processing', {})),
                'entity_embeddings': graph_data.get('entity_embeddings', {}),
                'fact_embeddings': graph_data.get('fact_embeddings', {})
            }
        }

    def _write_ingest_data_file(self, source_path: str, ingest_data: Dict[str, Any]) -> str:
        """Write the ingest_data.json file."""
        source_path_obj = Path(source_path)
        data_file_path = source_path_obj.parent / f"{source_path_obj.stem}.ingest_data.json"

        with open(data_file_path, 'w', encoding='utf-8') as f:
            json.dump(ingest_data, f, indent=2, ensure_ascii=False)

        return str(data_file_path)

    async def _initialize_databases(
        self,
        database_configs: List[DatabaseConfig],
        embeddings_data: Dict[str, Any]
    ) -> None:
        """Initialize databases - create collections/databases if they don't exist."""
        for db_config in database_configs:
            try:
                if db_config.type == DatabaseType.QDRANT:
                    await self._initialize_qdrant(db_config, embeddings_data)
                elif db_config.type == DatabaseType.NEO4J:
                    await self._initialize_neo4j(db_config)

            except Exception as e:
                logger.error("Failed to initialize database",
                           database_type=db_config.type.value,
                           error=str(e))
                raise

    async def _initialize_qdrant(
        self,
        db_config: DatabaseConfig,
        embeddings_data: Dict[str, Any]
    ) -> None:
        """Initialize Qdrant collection."""
        host = db_config.hostname or 'localhost'
        port = db_config.port or 6333

        logger.info("Initializing Qdrant with config",
                   hostname=db_config.hostname,
                   port=db_config.port,
                   database_name=db_config.database_name)

        # Check if hostname is a URL and extract components
        if host.startswith(('http://', 'https://')):
            from urllib.parse import urlparse
            parsed = urlparse(host)
            hostname = parsed.hostname or "localhost"
            port = parsed.port or (443 if parsed.scheme == 'https' else port)
            https = parsed.scheme == 'https'
            # Use the full URL for QdrantStorage
            config_host = host
        else:
            hostname = host
            https = port == 443  # Auto-detect HTTPS for port 443
            config_host = hostname

        verify_ssl = os.getenv('QDRANT_VERIFY_SSL', 'true').lower() == 'true'

        logger.info("Creating QdrantVectorStorage with",
                   config_host=config_host,
                   port=port,
                   collection_name=db_config.database_name or 'morag_documents',
                   verify_ssl=verify_ssl)

        # Use QdrantVectorStorage instead of QdrantStorage for better connection handling
        qdrant_storage = QdrantVectorStorage(
            host=config_host,
            port=port,
            api_key=os.getenv('QDRANT_API_KEY'),
            collection_name=db_config.database_name or 'morag_documents',
            verify_ssl=verify_ssl
        )
        await qdrant_storage.connect()

        logger.info("Qdrant collection initialized",
                   collection=db_config.database_name or 'morag_documents',
                   vector_size=embeddings_data.get('embedding_dimension', 768))

        await qdrant_storage.disconnect()

    async def _initialize_neo4j(self, db_config: DatabaseConfig) -> None:
        """Initialize Neo4j database."""
        import os
        neo4j_config = Neo4jConfig(
            uri=db_config.hostname or 'bolt://localhost:7687',
            username=db_config.username or 'neo4j',
            password=db_config.password or 'password',
            database=db_config.database_name or 'neo4j',
            verify_ssl=os.getenv("NEO4J_VERIFY_SSL", "true").lower() == "true",
            trust_all_certificates=os.getenv("NEO4J_TRUST_ALL_CERTIFICATES", "false").lower() == "true"
        )

        neo4j_storage = Neo4jStorage(neo4j_config)
        await neo4j_storage.connect()

        # Test the connection and ensure database exists
        await neo4j_storage.test_connection()

        logger.info("Neo4j database initialized",
                   database=neo4j_config.database,
                   uri=neo4j_config.uri)

        await neo4j_storage.disconnect()

    def _serialize_enhanced_processing(self, enhanced_processing: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize enhanced processing data to be JSON-compatible.

        Converts Entity objects and other non-serializable objects to dictionaries.
        """
        if not enhanced_processing:
            return {}

        serialized = {}

        for key, value in enhanced_processing.items():
            if key == 'entities' and isinstance(value, list):
                # Convert Entity objects to dictionaries
                serialized[key] = []
                for entity in value:
                    if hasattr(entity, 'to_dict'):
                        serialized[key].append(entity.to_dict())
                    elif hasattr(entity, '__dict__'):
                        # Fallback: convert object attributes to dict
                        entity_dict = {}
                        for attr_name, attr_value in entity.__dict__.items():
                            if not attr_name.startswith('_'):  # Skip private attributes
                                try:
                                    # Test if the value is JSON serializable
                                    import json
                                    json.dumps(attr_value)
                                    entity_dict[attr_name] = attr_value
                                except (TypeError, ValueError):
                                    # Skip non-serializable values
                                    continue
                        serialized[key].append(entity_dict)
                    elif isinstance(entity, dict):
                        serialized[key].append(entity)
                    else:
                        # Skip non-serializable entities
                        continue
            elif key == 'relations' and isinstance(value, list):
                # Convert Relation objects to dictionaries
                serialized[key] = []
                for relation in value:
                    if hasattr(relation, 'to_dict'):
                        serialized[key].append(relation.to_dict())
                    elif isinstance(relation, dict):
                        serialized[key].append(relation)
                    else:
                        # Skip non-serializable relations
                        continue
            else:
                # For other values, try to serialize directly
                try:
                    import json
                    json.dumps(value)
                    serialized[key] = value
                except (TypeError, ValueError):
                    # Skip non-serializable values
                    continue

        return serialized

    async def _store_enhanced_processing_data(
        self,
        neo4j_storage,
        graph_data: Dict[str, Any],
        facts: List
    ):
        """Store pre-computed entities and embeddings from enhanced processing data."""
        try:
            enhanced_processing = graph_data.get('enhanced_processing', {})
            entity_embeddings = graph_data.get('entity_embeddings', {})
            fact_embeddings = graph_data.get('fact_embeddings', {})

            # Store facts first
            if facts:
                from morag_graph.storage.neo4j_operations.fact_operations import FactOperations
                fact_operations = FactOperations(neo4j_storage.driver, neo4j_storage.config.database)

                for fact in facts:
                    await fact_operations.store_fact(fact)

                logger.info(f"Stored {len(facts)} facts from enhanced processing data")

            # Store entities from enhanced processing
            entities = enhanced_processing.get('entities', [])
            if entities:
                from morag_graph.models.entity import Entity
                from morag_graph.storage.neo4j_operations.entity_operations import EntityOperations
                entity_operations = EntityOperations(neo4j_storage.driver, neo4j_storage.config.database)

                for entity_data in entities:
                    if isinstance(entity_data, dict):
                        entity = Entity(**entity_data)
                    else:
                        entity = entity_data
                    await entity_operations.store_entity(entity)

                logger.info(f"Stored {len(entities)} entities from enhanced processing data")

            # Store entity embeddings
            if entity_embeddings:
                from morag_graph.services.entity_embedding_service import EntityEmbeddingService
                from morag_services.embedding import GeminiEmbeddingService
                import os

                api_key = os.getenv('GEMINI_API_KEY')
                if api_key:
                    gemini_service = GeminiEmbeddingService(api_key=api_key)
                    entity_embedding_service = EntityEmbeddingService(neo4j_storage, gemini_service)

                    for entity_key, embedding_data in entity_embeddings.items():
                        # Extract just the embedding vector from the embedding data
                        if isinstance(embedding_data, dict) and 'embedding' in embedding_data:
                            embedding_vector = embedding_data['embedding']
                            entity_name = embedding_data['name']
                            entity_type = embedding_data['type']

                            # Find the actual entity ID by looking up the entity by name
                            # The entity_key is now the normalized entity name (no prefixes)
                            actual_entity_id = None

                            # Look through the stored entities to find the matching one
                            # Use case-insensitive name comparison
                            for entity in entities:
                                # Case-insensitive name comparison
                                name_match = entity.name.lower() == entity_name.lower()

                                # All entities are stored with type 'ENTITY', so type matching is straightforward
                                entity_type_str = str(entity.type).lower()

                                # Accept match if name matches and entity type is 'ENTITY'
                                type_match = entity_type_str == 'entity'

                                if name_match and type_match:
                                    actual_entity_id = entity.id
                                    logger.debug(f"Found entity match: {entity.name} ({entity.type}) -> {actual_entity_id}")
                                    break

                            if actual_entity_id:
                                await entity_embedding_service.store_entity_embedding(actual_entity_id, embedding_vector)
                                logger.debug(f"Stored embedding for entity {actual_entity_id} (name: {entity_name})")
                            else:
                                # Enhanced logging to debug the issue
                                logger.warning(f"Could not find actual entity ID for entity key {entity_key} (name: {entity_name}, type: {entity_type})")

                                # Show more entities for debugging and check for partial matches
                                available_entities = [(e.name, str(e.type)) for e in entities[:10]]
                                logger.debug(f"Available entities (first 10): {available_entities}")

                                # Check for partial name matches to help debug
                                partial_matches = [e.name for e in entities if entity_name.lower() in e.name.lower() or e.name.lower() in entity_name.lower()]
                                if partial_matches:
                                    logger.debug(f"Potential partial name matches: {partial_matches[:5]}")

                                logger.debug(f"Total entities available: {len(entities)}, searching for: '{entity_name}' (type: {entity_type})")
                        else:
                            logger.warning(f"Invalid embedding data format for entity {entity_key}: {type(embedding_data)}")
                            continue

                    logger.info(f"Stored {len(entity_embeddings)} entity embeddings")

            # Store fact embeddings
            if fact_embeddings:
                from morag_graph.services.fact_embedding_service import FactEmbeddingService
                from morag_services.embedding import GeminiEmbeddingService
                import os

                api_key = os.getenv('GEMINI_API_KEY')
                if api_key:
                    gemini_service = GeminiEmbeddingService(api_key=api_key)
                    fact_embedding_service = FactEmbeddingService(neo4j_storage, gemini_service)

                    for fact_id, embedding_data in fact_embeddings.items():
                        # Extract just the embedding vector from the embedding data
                        if isinstance(embedding_data, dict) and 'embedding' in embedding_data:
                            embedding_vector = embedding_data['embedding']
                            await fact_embedding_service.store_fact_embedding(fact_id, embedding_vector)
                        else:
                            logger.warning(f"Invalid embedding data format for fact {fact_id}: {type(embedding_data)}")
                            continue

                    logger.info(f"Stored {len(fact_embeddings)} fact embeddings")

        except Exception as e:
            logger.error(f"Failed to store enhanced processing data: {e}")
            import traceback
            traceback.print_exc()

    async def _write_to_databases(
        self,
        database_configs: List[DatabaseConfig],
        embeddings_data: Dict[str, Any],
        graph_data: Dict[str, Any],
        document_id: str,
        replace_existing: bool,
        document_summary: Optional[str] = None,
        ingest_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Write data to all configured databases."""
        results = {}

        for db_config in database_configs:
            try:
                if db_config.type == DatabaseType.QDRANT:
                    result = await self._write_to_qdrant(
                        db_config, embeddings_data, document_id, replace_existing
                    )
                    results['qdrant'] = result

                elif db_config.type == DatabaseType.NEO4J:
                    document_metadata = ingest_data.get('document_metadata', {}) if ingest_data else {}
                    result = await self._write_to_neo4j(
                        db_config, graph_data, embeddings_data, document_id, document_summary, document_metadata
                    )
                    results['neo4j'] = result

            except Exception as e:
                logger.error("Failed to write to database",
                           database_type=db_config.type.value,
                           error=str(e))
                results[db_config.type.value.lower()] = {
                    'success': False,
                    'error': str(e)
                }

        return results

    async def _write_to_qdrant(
        self,
        db_config: DatabaseConfig,
        embeddings_data: Dict[str, Any],
        document_id: str,
        replace_existing: bool
    ) -> Dict[str, Any]:
        """Write vector data to Qdrant."""
        host = db_config.hostname or 'localhost'
        port = db_config.port or 6333

        # Check if hostname is a URL and extract components
        if host.startswith(('http://', 'https://')):
            from urllib.parse import urlparse
            parsed = urlparse(host)
            hostname = parsed.hostname or "localhost"
            port = parsed.port or (443 if parsed.scheme == 'https' else port)
            https = parsed.scheme == 'https'
            # Use the full URL for QdrantStorage
            config_host = host
        else:
            hostname = host
            https = port == 443  # Auto-detect HTTPS for port 443
            config_host = hostname

        verify_ssl = os.getenv('QDRANT_VERIFY_SSL', 'true').lower() == 'true'

        # Use QdrantVectorStorage instead of QdrantStorage for better connection handling
        qdrant_storage = QdrantVectorStorage(
            host=config_host,
            port=port,
            api_key=os.getenv('QDRANT_API_KEY'),
            collection_name=db_config.database_name or 'morag_documents',
            verify_ssl=verify_ssl
        )
        await qdrant_storage.connect()

        try:
            point_ids = []

            # Store each chunk as a vector point
            for i, (chunk_meta, embedding) in enumerate(
                zip(embeddings_data['chunk_metadata'], embeddings_data['embeddings'])
            ):
                # Create a simple integer point ID for Qdrant compatibility
                chunk_id = chunk_meta['chunk_id']
                # Use hash to create a consistent integer ID
                point_id = abs(hash(chunk_id)) % (2**31)  # Ensure positive 32-bit int

                # Store the vector with the hash as point ID but keep original chunk_id in metadata
                # Clean metadata to ensure JSON serialization compatibility
                clean_metadata = {}
                for key, value in chunk_meta.items():
                    if key == 'chunk_text':
                        # Don't store the full text in Qdrant payload to avoid size issues
                        continue
                    elif isinstance(value, (str, int, float, bool, list, dict)) or value is None:
                        clean_metadata[key] = value
                    else:
                        # Convert non-serializable objects to string
                        clean_metadata[key] = str(value)

                enhanced_metadata = {
                    **clean_metadata,
                    'original_chunk_id': chunk_id,
                    'point_id': point_id,
                    'text_length': len(chunk_meta.get('chunk_text', ''))
                }

                try:
                    # Use QdrantVectorStorage's store_vectors method
                    await qdrant_storage.store_vectors(
                        vectors=[embedding.tolist() if hasattr(embedding, 'tolist') else embedding],
                        metadata=[enhanced_metadata]
                    )
                except Exception as e:
                    logger.warning(f"Failed to store chunk in Qdrant: {e}")
                    # If it still fails, skip this chunk but continue with others
                    continue

                point_ids.append(point_id)

            await qdrant_storage.disconnect()

            return {
                'success': True,
                'points_stored': len(point_ids),
                'point_ids': point_ids,
                'collection': db_config.database_name or 'morag_documents'
            }

        except Exception as e:
            await qdrant_storage.disconnect()
            raise

    async def _write_to_neo4j(
        self,
        db_config: DatabaseConfig,
        graph_data: Dict[str, Any],
        embeddings_data: Dict[str, Any],
        document_id: str,
        document_summary: Optional[str] = None,
        document_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Write graph data to Neo4j with proper relationships."""
        import os
        neo4j_config = Neo4jConfig(
            uri=db_config.hostname or 'bolt://localhost:7687',
            username=db_config.username or 'neo4j',
            password=db_config.password or 'password',
            database=db_config.database_name or 'neo4j',
            verify_ssl=os.getenv("NEO4J_VERIFY_SSL", "true").lower() == "true",
            trust_all_certificates=os.getenv("NEO4J_TRUST_ALL_CERTIFICATES", "false").lower() == "true"
        )

        neo4j_storage = Neo4jStorage(neo4j_config)
        await neo4j_storage.connect()

        try:
            # Store document - use document_metadata if available, otherwise fallback to chunk metadata
            if document_metadata:
                # Use the structured document metadata from ingest_data
                source_path = document_metadata.get('source_file', 'Unknown')
                file_name = document_metadata.get('file_name', 'Unknown')
                name = document_metadata.get('name', file_name)
                mime_type = document_metadata.get('mime_type', 'unknown')
                file_size = document_metadata.get('file_size')
                checksum = document_metadata.get('checksum')
                summary = document_summary or document_metadata.get('summary', 'Document processed successfully')
                metadata = document_metadata.get('metadata', {})
            else:
                # Fallback to extracting from first chunk metadata
                chunk_meta = embeddings_data['chunk_metadata'][0]
                source_path = chunk_meta.get('source_path', 'Unknown')

                # Extract filename for name property
                if source_path and source_path != 'Unknown':
                    file_name = Path(source_path).name
                    name = file_name
                else:
                    file_name = chunk_meta.get('source_name', 'Unknown')
                    name = file_name

                mime_type = chunk_meta.get('mime_type', chunk_meta.get('source_type', 'unknown'))
                file_size = chunk_meta.get('file_size')
                checksum = chunk_meta.get('checksum') or chunk_meta.get('content_checksum')
                summary = document_summary or chunk_meta.get('summary', 'Document processed successfully')
                metadata = chunk_meta

            document = Document(
                id=document_id,
                name=name,
                source_file=source_path,
                file_name=file_name,
                file_size=file_size,
                checksum=checksum,
                mime_type=mime_type,
                summary=summary,
                metadata=metadata
            )

            document_id_stored = await neo4j_storage.store_document(document)

            # Store document chunks and create document-chunk relationships
            chunk_ids = []
            chunk_id_to_index = {}  # Map chunk_id to chunk_index for entity relationships

            for i, (chunk_text, chunk_meta) in enumerate(
                zip(embeddings_data['chunks'], embeddings_data['chunk_metadata'])
            ):
                chunk = DocumentChunk(
                    id=chunk_meta['chunk_id'],
                    document_id=document_id,
                    chunk_index=i,
                    text=chunk_text,
                    metadata=chunk_meta
                )

                chunk_id_stored = await neo4j_storage.store_document_chunk(chunk)
                chunk_ids.append(chunk_id_stored)
                chunk_id_to_index[chunk_id_stored] = i

                # Create document -> CONTAINS -> chunk relationship
                await neo4j_storage.create_document_contains_chunk_relation(
                    document_id_stored, chunk_id_stored
                )

            # Store facts and create entities from them
            fact_ids = []
            facts_stored = 0
            entities_from_facts = []
            relations_from_facts = []

            facts_to_process = []
            for fact_data in graph_data.get('facts', []):
                # Convert fact data to Fact object if needed
                if isinstance(fact_data, dict):
                    from morag_graph.models.fact import Fact
                    fact = Fact(**fact_data)
                else:
                    fact = fact_data
                facts_to_process.append(fact)

            # Always run enhanced processing in Neo4j since it wasn't done earlier
            if facts_to_process:
                # Use enhanced fact processing service for entity creation and relationships
                from morag_graph.services.enhanced_fact_processing_service import EnhancedFactProcessingService
                from morag_graph.extraction.entity_normalizer import LLMEntityNormalizer

                # Initialize entity normalizer
                import os
                api_key = os.getenv('GEMINI_API_KEY')
                if api_key:
                    # Get language from first fact, default to 'en'
                    language = facts_to_process[0].language if facts_to_process and hasattr(facts_to_process[0], 'language') else 'en'
                    entity_normalizer = LLMEntityNormalizer(
                        model_name=os.getenv('MORAG_GEMINI_MODEL', 'gemini-2.0-flash'),
                        api_key=api_key,
                        language=language
                    )
                else:
                    entity_normalizer = None
                    logger.warning("GEMINI_API_KEY not found - entity normalization disabled")

                enhanced_processor = EnhancedFactProcessingService(neo4j_storage, entity_normalizer)

                # Process facts with entity creation and relationship management
                result = await enhanced_processor.process_facts_with_entities(
                    facts_to_process,
                    create_keyword_entities=True,
                    create_mandatory_relations=True
                )

                logger.info(f"Enhanced processing: {result['facts_processed']} facts, {result['entities_created']} entities, {result['relations_created']} relations")

                # Facts, entities, and relationships are already stored by the enhanced processor
                facts_stored = result['facts_stored']
            else:
                facts_stored = 0

            # Store fact relationships (if any)
            relationship_ids = []
            relationships_stored = 0

            fact_relationships = []
            for relationship_data in graph_data.get('relationships', []):
                # Convert relationship data to FactRelation object if needed
                if isinstance(relationship_data, dict):
                    from morag_graph.models.fact import FactRelation
                    relationship = FactRelation(**relationship_data)
                else:
                    relationship = relationship_data
                fact_relationships.append(relationship)

            if fact_relationships:
                # Store fact relationships using FactOperations
                from morag_graph.storage.neo4j_operations.fact_operations import FactOperations
                fact_operations = FactOperations(neo4j_storage.driver, neo4j_storage.config.database)
                relationship_ids = await fact_operations.store_fact_relations(fact_relationships)
                relationships_stored = len(relationship_ids)
                logger.info(f"Stored {relationships_stored} fact relationships")

            # Create chunk-fact relationships using chunk_fact_mapping
            chunk_fact_mapping = graph_data.get('chunk_fact_mapping', {})
            chunk_fact_relationships_created = 0

            logger.info(f"Creating chunk-fact relationships: {len(chunk_fact_mapping)} chunks with facts, {len(chunk_ids)} total chunks")

            for chunk_key, fact_ids_in_chunk in chunk_fact_mapping.items():
                # Handle both chunk IDs and chunk indices as keys
                if ':chunk:' in chunk_key:
                    # chunk_key is a chunk ID (e.g., "doc_file_hash:chunk:0")
                    chunk_id = chunk_key
                    chunk_index = int(chunk_key.split(':')[-1])
                elif '_chunk_' in chunk_key:
                    # chunk_key is a chunk ID (e.g., "doc_test_adhs_herbs.md_0b13822c0bb0dde8_chunk_0")
                    chunk_id = chunk_key
                    chunk_index = int(chunk_key.split('_chunk_')[-1])
                else:
                    # chunk_key is a chunk index (e.g., "0")
                    try:
                        chunk_index = int(chunk_key)
                        # Find the chunk_id for this chunk_index
                        chunk_id = None
                        for cid, cidx in chunk_id_to_index.items():
                            if cidx == chunk_index:
                                chunk_id = cid
                                break
                    except ValueError:
                        logger.warning(f"Could not parse chunk key '{chunk_key}' as index or ID")
                        continue

                if chunk_id:
                    # Get the chunk text for context
                    chunk_text = embeddings_data['chunks'][chunk_index] if chunk_index < len(embeddings_data['chunks']) else ""
                    context = chunk_text[:500] + "..." if len(chunk_text) > 500 else chunk_text

                    logger.debug(f"Creating relationships for chunk {chunk_index} (id: {chunk_id}) with {len(fact_ids_in_chunk)} facts")

                    # Create chunk -> CONTAINS -> fact relationships
                    for fact_id in fact_ids_in_chunk:
                        try:
                            await neo4j_storage.create_chunk_contains_fact_relation(chunk_id, fact_id, context)
                            chunk_fact_relationships_created += 1
                            logger.debug(f"Created relationship: chunk {chunk_id} -> fact {fact_id}")

                        except Exception as e:
                            logger.warning(f"Failed to create chunk-fact relationship: chunk {chunk_id} -> fact {fact_id}: {e}")
                else:
                    logger.warning(f"Could not find chunk_id for chunk_key '{chunk_key}' (parsed as chunk_index {chunk_index}). Available chunks: {list(chunk_id_to_index.values())}")

            logger.info(f"Created {chunk_fact_relationships_created} chunk-fact relationships")

            # Note: Fact storage and relationship creation will be implemented when Neo4j fact storage is added
            logger.info(f"Neo4j storage completed: {len(chunk_ids)} chunks stored, {facts_stored} facts processed, {relationships_stored} relationships processed")

            await neo4j_storage.disconnect()

            return {
                'success': True,
                'document_stored': document_id_stored,
                'chunks_stored': len(chunk_ids),
                'facts_processed': facts_stored,
                'relationships_processed': relationships_stored,
                'chunk_fact_relationships': chunk_fact_relationships_created,
                'database': neo4j_config.database
            }

        except Exception as e:
            await neo4j_storage.disconnect()
            raise

    async def _generate_document_summary(
        self,
        content: str,
        metadata: Dict[str, Any],
        language: Optional[str] = None
    ) -> str:
        """Generate a summary of the document using LLM.

        Args:
            content: Full document content
            metadata: Document metadata
            language: Language code for the document

        Returns:
            Document summary
        """
        try:
            if not self.embedding_service:
                logger.warning("No embedding service available for summarization")
                return "Summary not available - no embedding service configured."

            # Extract title from metadata if available
            title = metadata.get('title') or metadata.get('filename', 'Untitled Document')

            # Determine content type for context
            content_type = metadata.get('content_type', 'document')

            # Generate summary with appropriate length
            max_length = 500  # Reasonable summary length

            # Create context for better summarization
            context_parts = [f"Document Title: {title}"]
            if content_type:
                context_parts.append(f"Content Type: {content_type}")
            if language:
                context_parts.append(f"Language: {language}")

            context = "\n".join(context_parts)

            # Use the embedding service's summarization capability
            summary_result = await self.embedding_service.generate_summary(
                content,
                max_length=max_length,
                language=language
            )

            # Extract summary text
            if hasattr(summary_result, 'summary'):
                summary = summary_result.summary
            else:
                summary = str(summary_result)

            # Ensure summary is not empty
            if not summary or summary.strip() == "":
                summary = f"Document about {title} - content processed successfully."

            logger.info("Document summary generated",
                       content_length=len(content),
                       summary_length=len(summary),
                       language=language)

            return summary

        except Exception as e:
            logger.error("Failed to generate document summary",
                        error=str(e),
                        error_type=type(e).__name__)
            # Return a fallback summary instead of failing
            title = metadata.get('title') or metadata.get('filename', 'Document')
            return f"Summary generation failed for {title}. Content processed successfully."

    def _determine_chunk_type(self, content_type: str, metadata: Dict[str, Any]) -> str:
        """Determine the appropriate chunk type based on content type and metadata.

        Args:
            content_type: Type of content (audio, video, document, etc.)
            metadata: Content metadata

        Returns:
            Chunk type string
        """
        if content_type in ['audio', 'video']:
            # Check if topic segmentation was used
            if metadata.get('has_topic_info', False) or metadata.get('num_topics', 0) > 1:
                return 'topic'
            else:
                return 'segment'
        elif content_type == 'document':
            # Check document structure
            if metadata.get('has_chapters', False):
                return 'chapter'
            elif metadata.get('has_sections', False):
                return 'section'
            else:
                return 'paragraph'
        else:
            return 'section'  # Default fallback

    def _add_audio_video_chunk_metadata(self, chunk_text: str, chunk_index: int, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Add audio/video-specific metadata to chunk.

        Args:
            chunk_text: Text content of the chunk
            chunk_index: Index of the chunk
            metadata: Source metadata

        Returns:
            Dictionary with additional metadata
        """
        additional_meta = {}

        # Extract timestamp information from segments data if available
        segments = metadata.get('segments', [])
        if segments and chunk_index < len(segments):
            segment = segments[chunk_index]
            if isinstance(segment, dict):
                # Use segment start/end times for precise location metadata
                if 'start_time' in segment:
                    additional_meta['start_timestamp_seconds'] = segment['start_time']
                    additional_meta['start_timestamp'] = self._seconds_to_timestamp(segment['start_time'])
                if 'end_time' in segment:
                    additional_meta['end_timestamp_seconds'] = segment['end_time']
                    additional_meta['end_timestamp'] = self._seconds_to_timestamp(segment['end_time'])
                if 'duration' in segment:
                    additional_meta['segment_duration'] = segment['duration']
                if 'title' in segment:
                    additional_meta['topic_title'] = segment['title']
                if 'summary' in segment:
                    additional_meta['topic_summary'] = segment['summary']

        # Fallback: Try to extract timestamp information from the chunk text
        if 'start_timestamp_seconds' not in additional_meta:
            import re
            timestamp_pattern = r'\[(\d{2}:\d{2}:\d{2})\]|\[(\d+:\d{2})\]|(\d+:\d{2}:\d{2})|(\d+:\d{2})'
            timestamp_matches = re.findall(timestamp_pattern, chunk_text)

            if timestamp_matches:
                # Use the first timestamp found
                timestamp_str = next(filter(None, timestamp_matches[0]))
                additional_meta['start_timestamp'] = timestamp_str

                # Try to convert to seconds for easier processing
                try:
                    time_parts = timestamp_str.split(':')
                    if len(time_parts) == 3:  # HH:MM:SS
                        seconds = int(time_parts[0]) * 3600 + int(time_parts[1]) * 60 + int(time_parts[2])
                    elif len(time_parts) == 2:  # MM:SS
                        seconds = int(time_parts[0]) * 60 + int(time_parts[1])
                    else:
                        seconds = int(time_parts[0])
                    additional_meta['start_timestamp_seconds'] = seconds
                except (ValueError, IndexError):
                    pass

        # Add topic information if available
        if metadata.get('has_topic_info', False):
            # Try to extract topic title from chunk text
            import re
            topic_title_pattern = r'^###?\s*(.+?)(?:\n|$)'
            topic_match = re.search(topic_title_pattern, chunk_text, re.MULTILINE)
            if topic_match:
                additional_meta['topic_title'] = topic_match.group(1).strip()
                additional_meta['chunk_summary'] = f"Topic: {topic_match.group(1).strip()}"

        # Add speaker information if available
        import re
        speaker_pattern = r'Speaker (\d+):|Speaker ([A-Z]+):|SPEAKER_(\d+)'
        speaker_matches = re.findall(speaker_pattern, chunk_text)
        if speaker_matches:
            speakers = set()
            for match in speaker_matches:
                speaker = match[0] or match[1] or match[2]
                speakers.add(f"SPEAKER_{speaker}")
            additional_meta['speakers'] = list(speakers)

        # Add media type and duration information
        if 'duration' in metadata:
            additional_meta['total_duration'] = metadata['duration']
        if 'content_type' in metadata:
            additional_meta['media_type'] = metadata['content_type']

        # Add location context for easy reference
        if 'start_timestamp_seconds' in additional_meta:
            timestamp_str = additional_meta.get('start_timestamp', self._seconds_to_timestamp(additional_meta['start_timestamp_seconds']))
            additional_meta['location_reference'] = f"[{timestamp_str}]"
            additional_meta['location_type'] = 'timestamp'

        return additional_meta

    def _seconds_to_timestamp(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS timestamp format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _add_document_chunk_metadata(self, chunk_text: str, chunk_index: int, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Add document-specific metadata to chunk.

        Args:
            chunk_text: Text content of the chunk
            chunk_index: Index of the chunk
            metadata: Source metadata

        Returns:
            Dictionary with additional metadata
        """
        additional_meta = {}
        import re

        # Extract page information from multiple sources
        page_number = None

        # 1. Check if page info is in metadata
        if 'page_number' in metadata and metadata['page_number'] is not None:
            page_number = metadata['page_number']

        # 2. Try to extract page number from chunk text
        if page_number is None:
            # Look for page indicators in text
            page_patterns = [
                r'Page\s+(\d+)',
                r'page\s+(\d+)',
                r'PAGE\s+(\d+)',
                r'^(\d+)$',  # Standalone number at beginning of line
                r'^\s*(\d+)\s*$'  # Standalone number with whitespace
            ]

            for pattern in page_patterns:
                page_match = re.search(pattern, chunk_text, re.MULTILINE | re.IGNORECASE)
                if page_match:
                    try:
                        page_number = int(page_match.group(1))
                        break
                    except (ValueError, IndexError):
                        continue

        # 3. Estimate page number based on chunk position and document metadata
        if page_number is None and 'page_count' in metadata and metadata['page_count']:
            # Rough estimation: distribute chunks across pages
            total_chunks = metadata.get('total_chunks', chunk_index + 1)
            page_count = metadata['page_count']
            if total_chunks > 0 and page_count > 0:
                estimated_page = max(1, int((chunk_index / total_chunks) * page_count) + 1)
                page_number = estimated_page
                additional_meta['page_number_estimated'] = True

        if page_number is not None:
            additional_meta['page_number'] = page_number

        # Extract section/chapter information
        section_title = None
        section_level = None

        # Look for various header patterns
        header_patterns = [
            r'^(#{1,6})\s*(.+?)(?:\n|$)',  # Markdown headers
            r'^(\d+\.?\d*\.?)\s+(.+?)(?:\n|$)',  # Numbered sections (1. 1.1. etc.)
            r'^(Chapter|CHAPTER)\s+(\d+)[:\.]?\s*(.+?)(?:\n|$)',  # Chapter X: Title
            r'^(Section|SECTION)\s+(\d+)[:\.]?\s*(.+?)(?:\n|$)',  # Section X: Title
            r'^([A-Z][A-Z\s]{2,})\s*$',  # ALL CAPS titles
        ]

        for pattern in header_patterns:
            header_match = re.search(pattern, chunk_text, re.MULTILINE)
            if header_match:
                groups = header_match.groups()
                if pattern.startswith(r'^(#{1,6})'):  # Markdown headers
                    section_level = len(groups[0])
                    section_title = groups[1].strip()
                elif pattern.startswith(r'^(\d+\.?\d*\.?)'):  # Numbered sections
                    section_level = groups[0].count('.') + 1
                    section_title = groups[1].strip()
                elif 'Chapter' in pattern or 'Section' in pattern:
                    section_level = 1
                    section_title = f"{groups[0]} {groups[1]}: {groups[2].strip()}"
                else:  # ALL CAPS
                    section_level = 1
                    section_title = groups[0].strip()
                break

        if section_title:
            additional_meta['section_title'] = section_title
            additional_meta['section_level'] = section_level
            additional_meta['chunk_summary'] = f"Section: {section_title}"

        # Add document type and format information
        if 'file_extension' in metadata:
            additional_meta['document_format'] = metadata['file_extension']
        if 'content_type' in metadata:
            additional_meta['document_type'] = metadata['content_type']

        # Create location reference for documents
        location_parts = []
        if page_number is not None:
            location_parts.append(f"Page {page_number}")
        if section_title:
            location_parts.append(f"Section: {section_title}")

        if location_parts:
            additional_meta['location_reference'] = " | ".join(location_parts)
            additional_meta['location_type'] = 'document_position'
        else:
            # Fallback to chunk index
            additional_meta['location_reference'] = f"Chunk {chunk_index + 1}"
            additional_meta['location_type'] = 'chunk_index'

        return additional_meta


class FactExtractionWrapper:
    """Wrapper to provide the old graph extractor interface using fact extraction."""

    def __init__(self):
        """Initialize the wrapper."""
        self.fact_extractor = None
        self.graph_builder = None
        self._initialized = False

    async def extract_facts_and_relationships(
        self,
        text: str,
        doc_id: str,
        domain: str = "general",
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Fact], List[FactRelation]]:
        """Extract facts and relationships from text.

        Args:
            text: Text content to analyze
            doc_id: Document identifier
            domain: Domain context for extraction
            context: Additional context for extraction

        Returns:
            Tuple of (facts, relationships)
        """
        # Initialize fact extractor service if needed
        if not self._initialized:
            from morag_graph.extraction.fact_extractor import FactExtractor
            from morag_graph.extraction.fact_graph_builder import FactGraphBuilder

            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                logger.warning("No GEMINI_API_KEY found, fact extraction will fail")
                return [], []

            self.fact_extractor = FactExtractor(
                model_id=os.getenv('MORAG_GEMINI_MODEL', 'gemini-2.0-flash'),
                api_key=api_key,
                domain=domain,
                language=context.get('language', 'en') if context else 'en'
            )

            self.graph_builder = FactGraphBuilder(
                model_id=os.getenv('MORAG_GEMINI_MODEL', 'gemini-2.0-flash'),
                api_key=api_key,
                language=context.get('language', 'en') if context else 'en'
            )

            self._initialized = True
            logger.info("FactExtractionWrapper initialized successfully")

        try:
            # Extract facts
            logger.debug("Extracting facts from text", text_length=len(text), doc_id=doc_id)
            facts = await self.fact_extractor.extract_facts(
                chunk_text=text,
                chunk_id=f"{doc_id}_chunk",
                document_id=doc_id,
                context=context or {}
            )

            logger.info("Facts extracted", facts_count=len(facts), doc_id=doc_id)

            # Extract relationships if we have multiple facts
            relationships = []
            if len(facts) > 1:
                try:
                    fact_graph = await self.graph_builder.build_fact_graph(facts)
                    if hasattr(fact_graph, 'relationships'):
                        relationships = fact_graph.relationships
                    logger.info("Relationships extracted", relationships_count=len(relationships), doc_id=doc_id)
                except Exception as e:
                    logger.warning("Failed to extract fact relationships", error=str(e))

            return facts, relationships

        except Exception as e:
            logger.error("Fact extraction failed", error=str(e), doc_id=doc_id)
            return [], []
