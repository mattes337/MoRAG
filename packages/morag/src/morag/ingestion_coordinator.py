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
from morag_graph.models.document import Document
from morag_graph.models.document_chunk import DocumentChunk
from morag_graph.models.database_config import DatabaseConfig, DatabaseType
from .graph_extractor_wrapper import GraphExtractor
from morag_graph.utils.id_generation import UnifiedIDGenerator
from .ingestion.atomic_ingestion_service import AtomicIngestionService, ValidationError

import structlog

logger = structlog.get_logger(__name__)


class IngestionCoordinator:
    """Coordinates the complete ingestion process across multiple databases."""
    
    def __init__(self):
        """Initialize the ingestion coordinator."""
        self.embedding_service = None
        self.vector_storage = None
        self.graph_extractor = None
        self.atomic_ingestion_service = AtomicIngestionService()
        
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

        # Initialize graph extractor
        self.graph_extractor = GraphExtractor()

    async def ingest_content_atomic(
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
        Perform atomic content ingestion with pre-validation and transaction management.

        This method ensures that either all chunks are successfully ingested or none are,
        preventing inconsistent database states.

        Args:
            content: Text content to ingest
            source_path: Source file path or URL
            content_type: Type of content (document, audio, video, etc.)
            metadata: Content metadata
            processing_result: Processing result from document/media processing
            databases: List of database configurations to use
            chunk_size: Size of text chunks (default: 1000)
            chunk_overlap: Overlap between chunks (default: 200)
            document_id: Document identifier (auto-generated if not provided)
            replace_existing: Whether to replace existing document
            language: Language code for processing

        Returns:
            Dictionary containing ingestion results

        Raises:
            ValidationError: If pre-validation fails
            Exception: If ingestion fails after validation
        """
        start_time = datetime.now(timezone.utc)

        # Set defaults
        chunk_size = chunk_size or 1000
        chunk_overlap = chunk_overlap or 200
        document_id = document_id or str(uuid.uuid4())

        # Detect databases if not provided
        if databases is None:
            databases = self._detect_databases()

        logger.info(
            "Starting atomic content ingestion",
            source_path=source_path,
            content_type=content_type,
            document_id=document_id,
            content_length=len(content),
            databases_count=len(databases),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        try:
            # Use atomic ingestion service
            result = await self.atomic_ingestion_service.ingest_with_validation(
                content=content,
                source_path=source_path,
                document_id=document_id,
                database_configs=databases,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                replace_existing=replace_existing,
                metadata={
                    **metadata,
                    "content_type": content_type,
                    "language": language,
                    "processing_result": processing_result.to_dict() if hasattr(processing_result, 'to_dict') else {}
                }
            )

            # Write result files
            result_file_path = self._write_ingest_result_file(source_path, result)
            result['result_file'] = result_file_path

            logger.info(
                "Atomic ingestion completed successfully",
                document_id=document_id,
                transaction_id=result.get("transaction_id"),
                processing_time=result.get("processing_time"),
                chunks_processed=result.get("chunks_processed"),
                entities_extracted=result.get("entities_extracted"),
                relations_extracted=result.get("relations_extracted")
            )

            return result

        except ValidationError as e:
            logger.error(
                "Atomic ingestion failed during validation",
                document_id=document_id,
                source_path=source_path,
                error=str(e)
            )

            # Write failure result
            failure_result = {
                "success": False,
                "document_id": document_id,
                "source_path": source_path,
                "error": str(e),
                "error_type": "ValidationError",
                "processing_time": (datetime.now(timezone.utc) - start_time).total_seconds()
            }

            result_file_path = self._write_ingest_result_file(source_path, failure_result)
            failure_result['result_file'] = result_file_path

            return failure_result

        except Exception as e:
            logger.error(
                "Atomic ingestion failed",
                document_id=document_id,
                source_path=source_path,
                error=str(e),
                error_type=type(e).__name__
            )

            # Write failure result
            failure_result = {
                "success": False,
                "document_id": document_id,
                "source_path": source_path,
                "error": str(e),
                "error_type": type(e).__name__,
                "processing_time": (datetime.now(timezone.utc) - start_time).total_seconds()
            }

            result_file_path = self._write_ingest_result_file(source_path, failure_result)
            failure_result['result_file'] = result_file_path

            return failure_result

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

        # Step 5: Extract entities and relations for graph databases
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
                   entities_extracted=len(graph_data.get('entities', [])),
                   relations_extracted=len(graph_data.get('relations', [])),
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
                'created_at': datetime.now(timezone.utc).isoformat(),
                **metadata
            }

            # Add content-type-specific metadata
            content_type = metadata.get('content_type', 'document')
            chunk_meta['chunk_type'] = self._determine_chunk_type(content_type, metadata)

            # Add position/timestamp information based on content type
            if content_type in ['audio', 'video']:
                chunk_meta.update(self._add_audio_video_chunk_metadata(chunk, i, metadata))
            elif content_type == 'document':
                chunk_meta.update(self._add_document_chunk_metadata(chunk, i, metadata))

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
            content_type: Type of content (audio, video, document, etc.)
            metadata: Additional metadata for chunking decisions

        Returns:
            List of text chunks
        """
        if len(content) <= chunk_size:
            return [content]

        # Use topic-based chunking for audio/video content
        if content_type in ['audio', 'video'] and metadata:
            return self._create_topic_based_chunks(content, chunk_size, chunk_overlap, metadata)

        # Use document-aware chunking for documents
        elif content_type == 'document':
            return self._create_document_chunks(content, chunk_size, chunk_overlap)

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

        # Look for topic headers in the content
        topic_pattern = r'^###?\s*(.+?)(?:\n|$)'
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

        # Look for timestamp patterns
        timestamp_pattern = r'\[(\d{2}:\d{2}:\d{2})\]|\[(\d+:\d{2})\]|(\d+:\d{2}:\d{2})|(\d+:\d{2})'
        timestamp_matches = list(re.finditer(timestamp_pattern, content))

        if not timestamp_matches:
            # No timestamps found, fall back to character chunking
            return self._create_character_chunks(content, chunk_size, chunk_overlap)

        chunks = []
        current_chunk = ""

        for i, match in enumerate(timestamp_matches):
            # Find the end of this timestamp segment
            next_match_start = timestamp_matches[i + 1].start() if i + 1 < len(timestamp_matches) else len(content)
            segment = content[match.start():next_match_start].strip()

            # If adding this segment would exceed chunk size, finalize current chunk
            if current_chunk and len(current_chunk) + len(segment) > chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = segment
            else:
                current_chunk += "\n" + segment if current_chunk else segment

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

        # Extract topic header
        header_pattern = r'^(###?\s*.+?)(?:\n|$)'
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
        """Extract entities and relations for graph databases using chunk-based approach."""
        try:
            logger.info(f"Extracting graph data from chunks ({len(content)} chars)")

            # Create chunks using the same logic as embeddings
            content_type = metadata.get('content_type', 'document')
            chunks = self._create_chunks(content, chunk_size, chunk_overlap, content_type, metadata)

            logger.info(f"Processing {len(chunks)} chunks for entity and relation extraction")

            all_entities = []
            all_relations = []
            chunk_entity_mapping = {}

            # Process each chunk individually for entity and relation extraction
            for i, chunk_content in enumerate(chunks):
                try:
                    logger.debug(f"Processing chunk {i+1}/{len(chunks)} for entity extraction")

                    # Extract entities and relations from this chunk
                    chunk_extraction_result = await self.graph_extractor.extract_entities_and_relations(
                        chunk_content, source_path, language
                    )

                    chunk_entities = []
                    chunk_relations = []

                    # Process entities from this chunk
                    if chunk_extraction_result and chunk_extraction_result.get('entities'):
                        for entity_data in chunk_extraction_result['entities']:
                            # Clean up attributes - remove duplicate description/context fields
                            entity_attributes = entity_data.get('attributes', {}).copy()

                            # Remove duplicate fields - keep only essential attributes
                            entity_attributes.pop('description', None)  # Remove duplicate description
                            entity_attributes.pop('context', None)      # Remove duplicate context

                            # Keep only meaningful attributes (positions, metadata, etc.)
                            cleaned_attributes = {}
                            for key, value in entity_attributes.items():
                                if key not in ['description', 'context'] and value is not None:
                                    cleaned_attributes[key] = value

                            # Add chunk metadata
                            cleaned_attributes['chunk_index'] = i
                            cleaned_attributes['chunk_content_preview'] = chunk_content[:100] + "..." if len(chunk_content) > 100 else chunk_content

                            entity = Entity(
                                # Let Entity model generate unified ID automatically
                                name=entity_data['name'],
                                type=entity_data.get('type', 'UNKNOWN'),
                                description=entity_data.get('description', ''),
                                attributes=cleaned_attributes,
                                source_doc_id=source_path,
                                confidence=entity_data.get('confidence', 0.8)
                            )
                            chunk_entities.append(entity)

                    # Process relations from this chunk
                    if chunk_extraction_result and chunk_extraction_result.get('relations'):
                        for relation_data in chunk_extraction_result['relations']:
                            # Clean up attributes - remove duplicate description/context fields
                            relation_attributes = relation_data.get('attributes', {}).copy()

                            # Remove duplicate fields from attributes
                            relation_attributes.pop('description', None)  # Remove duplicate description
                            relation_attributes.pop('context', None)      # Remove duplicate context

                            # Keep only meaningful attributes (metadata, etc.)
                            cleaned_attributes = {}
                            for key, value in relation_attributes.items():
                                if key not in ['description', 'context'] and value is not None:
                                    cleaned_attributes[key] = value

                            # Add chunk metadata
                            cleaned_attributes['chunk_index'] = i

                            # Extract description and context as separate fields
                            description = relation_data.get('description', '')
                            context = relation_attributes.get('context', description)  # Use description as fallback for context

                            # If they're the same, make context more specific
                            if context == description and description:
                                context = f"Found in chunk {i+1}: {description}"

                            relation = Relation(
                                # Let Relation model generate unified ID automatically
                                source_entity_id=relation_data['source_entity_id'],
                                target_entity_id=relation_data['target_entity_id'],
                                type=relation_data['relation_type'],
                                description=description,
                                context=context,
                                attributes=cleaned_attributes,
                                source_doc_id=source_path,
                                confidence=relation_data.get('confidence', 0.8)
                            )
                            chunk_relations.append(relation)

                    # Add entities and relations to global lists
                    all_entities.extend(chunk_entities)
                    all_relations.extend(chunk_relations)

                    # Create chunk-entity mapping for this chunk
                    chunk_id = f"{document_id}:chunk:{i}"
                    chunk_entity_mapping[chunk_id] = [entity.id for entity in chunk_entities]

                    logger.debug(f"Chunk {i+1} processed: {len(chunk_entities)} entities, {len(chunk_relations)} relations")

                except Exception as e:
                    logger.warning(f"Failed to process chunk {i+1} for entity extraction", error=str(e))
                    # Continue with other chunks
                    continue

            # Deduplicate entities across chunks (entities with same name should be merged)
            all_entities = self._deduplicate_entities_across_chunks(all_entities)

            # Update chunk-entity mapping after deduplication
            chunk_entity_mapping = self._update_chunk_mapping_after_deduplication(chunk_entity_mapping, all_entities)

            logger.info(f"Extracted {len(all_entities)} entities and {len(all_relations)} relations from {len(chunks)} chunks")
            logger.info(f"Created chunk-entity mapping: {len(chunk_entity_mapping)} chunks with entities")

            return {
                'entities': all_entities,
                'relations': all_relations,
                'chunk_entity_mapping': chunk_entity_mapping,
                'extraction_metadata': {
                    'total_entities': len(all_entities),
                    'total_relations': len(all_relations),
                    'extraction_method': 'chunk_based',
                    'chunks_processed': len(chunks),
                    'chunks_with_entities': len(chunk_entity_mapping)
                }
            }

        except Exception as e:
            logger.warning("Failed to extract graph data", error=str(e))
            return {
                'entities': [],
                'relations': [],
                'chunk_entity_mapping': {},
                'extraction_metadata': {'error': str(e)}
            }

    def _deduplicate_entities_across_chunks(self, entities: List[Entity]) -> List[Entity]:
        """Deduplicate entities across chunks by merging entities with the same name."""
        entity_map = {}

        for entity in entities:
            # Use normalized name as key for deduplication
            key = entity.name.lower().strip()

            if key in entity_map:
                # Merge with existing entity - keep highest confidence
                existing = entity_map[key]
                if entity.confidence > existing.confidence:
                    # Update with higher confidence entity but merge chunk info
                    chunk_indices = set()
                    if 'chunk_index' in existing.attributes:
                        chunk_indices.add(existing.attributes['chunk_index'])
                    if 'chunk_index' in entity.attributes:
                        chunk_indices.add(entity.attributes['chunk_index'])

                    entity.attributes['chunk_indices'] = list(chunk_indices)
                    entity_map[key] = entity
                else:
                    # Keep existing but add chunk info
                    chunk_indices = set()
                    if 'chunk_index' in existing.attributes:
                        chunk_indices.add(existing.attributes['chunk_index'])
                    if 'chunk_index' in entity.attributes:
                        chunk_indices.add(entity.attributes['chunk_index'])

                    existing.attributes['chunk_indices'] = list(chunk_indices)
            else:
                entity_map[key] = entity

        return list(entity_map.values())

    def _update_chunk_mapping_after_deduplication(self, chunk_entity_mapping: Dict[str, List[str]], entities: List[Entity]) -> Dict[str, List[str]]:
        """Update chunk-entity mapping after entity deduplication."""
        # Create a mapping from old entity IDs to new deduplicated entity IDs
        entity_name_to_id = {entity.name.lower().strip(): entity.id for entity in entities}

        updated_mapping = {}
        for chunk_id, entity_ids in chunk_entity_mapping.items():
            updated_entity_ids = []
            for old_entity_id in entity_ids:
                # Find the entity by ID and get its deduplicated version
                for entity in entities:
                    if entity.id == old_entity_id:
                        # Check if this entity appears in this chunk
                        chunk_indices = entity.attributes.get('chunk_indices', [entity.attributes.get('chunk_index')])
                        chunk_index = int(chunk_id.split(':')[-1])
                        if chunk_index in chunk_indices:
                            updated_entity_ids.append(entity.id)
                        break

            if updated_entity_ids:
                updated_mapping[chunk_id] = list(set(updated_entity_ids))  # Remove duplicates

        return updated_mapping

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
                'entities_count': len(graph_data['entities']),
                'relations_count': len(graph_data['relations']),
                'entities': [
                    {
                        'id': entity.id,
                        'name': entity.name,
                        'type': entity.type.value if hasattr(entity.type, 'value') else str(entity.type),
                        'description': getattr(entity, 'description', ''),
                        'attributes': entity.attributes,
                        'confidence': entity.confidence,
                        'embedding': getattr(entity, 'embedding', None)
                    }
                    for entity in graph_data['entities']
                ],
                'relations': [
                    {
                        'id': relation.id,
                        'source_entity_id': relation.source_entity_id,
                        'target_entity_id': relation.target_entity_id,
                        'relation_type': relation.type.value if hasattr(relation.type, 'value') else str(relation.type),
                        'description': getattr(relation, 'description', ''),
                        'context': getattr(relation, 'context', ''),
                        'attributes': relation.attributes,
                        'confidence': relation.confidence
                    }
                    for relation in graph_data['relations']
                ],
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
                'entities': [
                    {
                        'id': entity.id,
                        'name': entity.name,
                        'type': entity.type.value if hasattr(entity.type, 'value') else str(entity.type),
                        'attributes': entity.attributes,
                        'confidence': entity.confidence,
                        'source_doc_id': getattr(entity, 'source_doc_id', document_id)
                    }
                    for entity in graph_data['entities']
                ],
                'relations': [
                    {
                        'id': relation.id,
                        'source_entity_id': relation.source_entity_id,
                        'target_entity_id': relation.target_entity_id,
                        'relation_type': relation.type.value if hasattr(relation.type, 'value') else str(relation.type),
                        'attributes': relation.attributes,
                        'confidence': relation.confidence,
                        'source_doc_id': getattr(relation, 'source_doc_id', document_id)
                    }
                    for relation in graph_data['relations']
                ],
                'chunk_entity_mapping': graph_data.get('chunk_entity_mapping', {})
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
        await neo4j_storage._execute_query("RETURN 1", {})

        logger.info("Neo4j database initialized",
                   database=neo4j_config.database,
                   uri=neo4j_config.uri)

        await neo4j_storage.disconnect()

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

            # Store entities and update their mentioned_in_chunks field
            entity_ids = []
            entity_id_to_entity = {}  # Map entity_id to entity object for later updates

            for entity in graph_data['entities']:
                entity_id_stored = await neo4j_storage.store_entity(entity)
                entity_ids.append(entity_id_stored)
                entity_id_to_entity[entity_id_stored] = entity

            # Store relations
            relation_ids = []
            for relation in graph_data['relations']:
                relation_id_stored = await neo4j_storage.store_relation(relation)
                relation_ids.append(relation_id_stored)

            # Create chunk-entity relationships using chunk_entity_mapping
            chunk_entity_mapping = graph_data.get('chunk_entity_mapping', {})
            chunk_entity_relationships_created = 0

            logger.info(f"Creating chunk-entity relationships: {len(chunk_entity_mapping)} chunks with entities, {len(chunk_ids)} total chunks")

            for chunk_key, entity_ids_in_chunk in chunk_entity_mapping.items():
                # Handle both chunk IDs and chunk indices as keys
                if ':chunk:' in chunk_key:
                    # chunk_key is a chunk ID (e.g., "doc_file_hash:chunk:0")
                    chunk_id = chunk_key
                    chunk_index = int(chunk_key.split(':')[-1])
                else:
                    # chunk_key is a chunk index (e.g., "0")
                    chunk_index = int(chunk_key)
                    # Find the chunk_id for this chunk_index
                    chunk_id = None
                    for cid, cidx in chunk_id_to_index.items():
                        if cidx == chunk_index:
                            chunk_id = cid
                            break

                if chunk_id:
                    # Get the chunk text for context
                    chunk_text = embeddings_data['chunks'][chunk_index] if chunk_index < len(embeddings_data['chunks']) else ""
                    context = chunk_text[:500] + "..." if len(chunk_text) > 500 else chunk_text

                    logger.debug(f"Creating relationships for chunk {chunk_index} (id: {chunk_id}) with {len(entity_ids_in_chunk)} entities")

                    # Create chunk -> MENTIONS -> entity relationships and update entity mentioned_in_chunks
                    for entity_id in entity_ids_in_chunk:
                        try:
                            await neo4j_storage.create_chunk_mentions_entity_relation(
                                chunk_id, entity_id, context
                            )
                            chunk_entity_relationships_created += 1
                            #logger.debug(f"Created relationship: chunk {chunk_id} -> entity {entity_id}")

                            # Update entity's mentioned_in_chunks field
                            entity = None
                            if entity_id in entity_id_to_entity:
                                entity = entity_id_to_entity[entity_id]
                            else:
                                # This might be an auto-created entity, fetch it from Neo4j
                                try:
                                    entity = await neo4j_storage.get_entity(entity_id)
                                    if entity:
                                        logger.debug(f"Fetched auto-created entity {entity_id} from Neo4j")
                                except Exception as fetch_error:
                                    logger.warning(f"Failed to fetch entity {entity_id} from Neo4j: {fetch_error}")

                        except Exception as e:
                            logger.warning(f"Failed to create chunk-entity relationship: chunk {chunk_id} -> entity {entity_id}: {e}")
                else:
                    logger.warning(f"Could not find chunk_id for chunk_key '{chunk_key}' (parsed as chunk_index {chunk_index}). Available chunks: {list(chunk_id_to_index.values())}")

            logger.info(f"Created {chunk_entity_relationships_created} chunk-entity relationships")

            # Fix any unconnected entities
            try:
                fixed_entities = await neo4j_storage.fix_unconnected_entities()
                if fixed_entities > 0:
                    logger.info(f"Fixed {fixed_entities} unconnected entities")
            except Exception as e:
                logger.warning(f"Failed to fix unconnected entities: {e}")

            await neo4j_storage.disconnect()

            return {
                'success': True,
                'document_stored': document_id_stored,
                'chunks_stored': len(chunk_ids),
                'entities_stored': len(entity_ids),
                'relations_stored': len(relation_ids),
                'chunk_entity_relationships': chunk_entity_relationships_created,
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

        # Try to extract timestamp information from the chunk text or metadata
        # Look for timestamp patterns in the chunk text
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
            topic_title_pattern = r'^###?\s*(.+?)(?:\n|$)'
            topic_match = re.search(topic_title_pattern, chunk_text, re.MULTILINE)
            if topic_match:
                additional_meta['topic_title'] = topic_match.group(1).strip()
                additional_meta['chunk_summary'] = f"Topic: {topic_match.group(1).strip()}"

        # Add speaker information if available
        speaker_pattern = r'Speaker (\d+):|Speaker ([A-Z]+):'
        speaker_matches = re.findall(speaker_pattern, chunk_text)
        if speaker_matches:
            speakers = set()
            for match in speaker_matches:
                speaker = match[0] or match[1]
                speakers.add(speaker)
            additional_meta['speakers'] = list(speakers)

        return additional_meta

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

        # Try to extract section/chapter information
        import re

        # Look for chapter/section headers
        header_pattern = r'^(#{1,6})\s*(.+?)(?:\n|$)'
        header_match = re.search(header_pattern, chunk_text, re.MULTILINE)
        if header_match:
            header_level = len(header_match.group(1))
            header_text = header_match.group(2).strip()
            additional_meta['section_title'] = header_text
            additional_meta['section_level'] = header_level
            additional_meta['chunk_summary'] = f"Section: {header_text}"

        # Add page information if available from metadata
        if 'page_number' in metadata:
            additional_meta['page_number'] = metadata['page_number']

        return additional_meta
