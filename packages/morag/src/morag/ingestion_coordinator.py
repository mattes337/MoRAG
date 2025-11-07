"""
Refactored comprehensive ingestion coordinator for MoRAG system.

This module handles the main orchestration of the complete ingestion flow using
separate components for database operations and chunk processing.
"""

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from morag_core.models import ProcessingResult
from morag_core.utils.logging import get_logger
from morag_graph.models.database_config import DatabaseConfig, DatabaseType
from morag_graph.services.enhanced_fact_processing_service import (
    EnhancedFactProcessingService,
)
from morag_graph.services.fact_extraction_service import FactExtractionService
from morag_graph.utils.id_generation import UnifiedIDGenerator
from morag_services.embedding import GeminiEmbeddingService

from .chunk_processor import ChunkProcessor
from .database_handler import DatabaseHandler

logger = get_logger(__name__)


class IngestionCoordinator:
    """Coordinates the complete ingestion process across multiple databases."""

    def __init__(self):
        """Initialize the ingestion coordinator."""
        self.embedding_service = None
        self.fact_extractor = None
        self.database_handler = DatabaseHandler()
        self.chunk_processor = ChunkProcessor()

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        if self.database_handler:
            await self.database_handler.cleanup_connections()
        if self.embedding_service and hasattr(self.embedding_service, "cleanup"):
            try:
                await self.embedding_service.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up embedding service: {e}")

    async def initialize(self):
        """Initialize all services and components."""
        try:
            # Initialize embedding service
            self.embedding_service = GeminiEmbeddingService()
            await self.embedding_service.initialize()

            # Initialize chunk processor
            await self.chunk_processor.initialize()

            # Initialize fact extraction service
            self.fact_extractor = FactExtractionService()

            logger.info("Ingestion coordinator initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize ingestion coordinator", error=str(e))
            raise

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
        language: Optional[str] = None,
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
            language: Language for processing

        Returns:
            Complete ingestion result dictionary
        """
        if not self.embedding_service:
            await self.initialize()

        start_time = datetime.now(timezone.utc)

        try:
            # Step 1: Determine which databases to use
            database_configs = self._determine_databases(databases)
            logger.info(
                "Determined databases for ingestion",
                databases=[db.type.value for db in database_configs],
            )

            # Step 2: Generate document ID if not provided
            if not document_id:
                document_id = self._generate_document_id(source_path, content)

            # Step 3: Set default chunk parameters
            effective_chunk_size = chunk_size or 4000
            effective_chunk_overlap = chunk_overlap or 200

            # Step 4: Create chunks using the chunk processor
            chunks = self.chunk_processor.create_chunks(
                content,
                effective_chunk_size,
                effective_chunk_overlap,
                content_type,
                metadata,
            )

            # Step 5: Generate embeddings and metadata using the chunk processor
            embeddings_data = (
                await self.chunk_processor.generate_embeddings_and_metadata(
                    content, chunks, source_path, content_type, metadata, document_id
                )
            )

            # Step 6: Determine language for extraction if not provided
            effective_language = self._determine_language(
                language, processing_result, metadata
            )

            # Step 7: Extract facts and relationships for graph databases
            graph_data = await self._extract_graph_data(
                content,
                source_path,
                document_id,
                metadata,
                effective_chunk_size,
                effective_chunk_overlap,
                effective_language,
            )

            # Step 8: Create complete ingest result using chunk processor
            ingest_result = self.chunk_processor.create_ingest_result(
                source_path,
                document_id,
                content,
                content_type,
                metadata,
                embeddings_data,
                graph_data,
                processing_result,
            )

            # Step 9: Write ingest result file
            result_file_path = self.chunk_processor.write_ingest_result_file(
                source_path, ingest_result
            )

            # Step 10: Initialize databases using database handler
            await self.database_handler.initialize_databases(
                database_configs, embeddings_data
            )

            # Step 11: Create and write ingest data file using chunk processor
            ingest_data = self.chunk_processor.create_ingest_data(
                document_id, embeddings_data, graph_data
            )
            data_file_path = self.chunk_processor.write_ingest_data_file(
                source_path, ingest_data
            )

            # Step 12: Write data to databases using database handler
            database_results = await self.database_handler.write_to_databases(
                database_configs, ingest_data
            )

            # Step 13: Update final result with database write results
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            ingest_result.update(
                {
                    "database_results": database_results,
                    "processing_time": processing_time,
                    "ingest_data_file": data_file_path,
                    "result_file": result_file_path,
                }
            )

            # Step 14: Update the ingest result file with final results
            self.chunk_processor.write_ingest_result_file(source_path, ingest_result)

            logger.info(
                "Ingestion completed successfully",
                source_path=source_path,
                databases_used=len(database_configs),
                chunks_created=len(embeddings_data.get("chunks_data", [])),
                facts_extracted=len(graph_data.get("facts", []) if graph_data else []),
                processing_time=processing_time,
                result_file=result_file_path,
            )

            return ingest_result

        except Exception as e:
            logger.error(
                "Ingestion failed",
                source_path=source_path,
                error=str(e),
                processing_time=(
                    datetime.now(timezone.utc) - start_time
                ).total_seconds(),
            )
            raise

    def _determine_databases(
        self, databases: Optional[List[DatabaseConfig]]
    ) -> List[DatabaseConfig]:
        """Determine which databases to use for ingestion."""
        if databases:
            return databases

        # Default configuration based on environment variables
        default_databases = []

        # Check for Qdrant configuration
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_collection = os.getenv("QDRANT_COLLECTION_NAME", "morag_documents")

        if (qdrant_url or qdrant_host) and qdrant_collection:
            # Prefer URL if available, otherwise use host/port
            if qdrant_url:
                default_databases.append(
                    DatabaseConfig(
                        type=DatabaseType.QDRANT,
                        hostname=qdrant_url,  # Store URL in hostname field
                        database_name=qdrant_collection,
                    )
                )
            else:
                default_databases.append(
                    DatabaseConfig(
                        type=DatabaseType.QDRANT,
                        hostname=qdrant_host,
                        port=int(os.getenv("QDRANT_PORT", 6333)),
                        database_name=qdrant_collection,
                    )
                )

        # Check for Neo4j configuration
        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_username = os.getenv("NEO4J_USERNAME")
        neo4j_password = os.getenv("NEO4J_PASSWORD")

        if neo4j_uri and neo4j_username and neo4j_password:
            default_databases.append(
                DatabaseConfig(
                    type=DatabaseType.NEO4J,
                    hostname=neo4j_uri,
                    username=neo4j_username,
                    password=neo4j_password,
                    database_name=os.getenv("NEO4J_DATABASE", "neo4j"),
                )
            )

        return default_databases

    def _generate_document_id(self, source_path: str, content: str) -> str:
        """Generate a unique document ID."""
        # Generate a checksum from the content for deterministic IDs
        import hashlib

        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        return UnifiedIDGenerator.generate_document_id(source_path, content_hash)

    def _determine_language(
        self,
        language: Optional[str],
        processing_result: ProcessingResult,
        metadata: Dict[str, Any],
    ) -> Optional[str]:
        """Determine the effective language for processing."""
        effective_language = language

        if not effective_language:
            # Try to extract language from processing result metadata
            if hasattr(processing_result, "document") and processing_result.document:
                effective_language = getattr(
                    processing_result.document.metadata, "language", None
                )

            # Try to extract from general metadata
            if not effective_language:
                effective_language = metadata.get("language")

            # Try to extract from processing result metadata
            if not effective_language and hasattr(processing_result, "metadata"):
                effective_language = processing_result.metadata.get("language")

        logger.info(
            "Language determined for extraction",
            provided_language=language,
            effective_language=effective_language,
        )

        return effective_language

    async def _extract_graph_data(
        self,
        content: str,
        source_path: str,
        document_id: str,
        metadata: Dict[str, Any],
        chunk_size: int,
        chunk_overlap: int,
        language: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Extract graph data (entities, relations, facts) from content."""
        try:
            if not self.fact_extractor:
                logger.warning(
                    "Fact extractor not initialized, skipping graph data extraction"
                )
                return None

            logger.info(
                "Extracting graph data",
                content_length=len(content),
                chunk_size=chunk_size,
                language=language,
            )

            # Extract facts and relationships using the fact extraction service
            extraction_result = (
                await self.fact_extractor.extract_facts_and_relationships(
                    content=content,
                    source_path=source_path,
                    document_id=document_id,
                    metadata=metadata,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    language=language,
                )
            )

            if not extraction_result:
                logger.warning("No graph data extracted")
                return None

            # Generate embeddings for entities and facts if embedding service is available
            if self.embedding_service:
                await self._generate_entity_and_fact_embeddings(extraction_result)

            # Process with enhanced fact processing if available
            try:
                enhanced_processor = EnhancedFactProcessingService()
                enhanced_processing = (
                    await enhanced_processor.process_facts_with_context(
                        facts=extraction_result.get("facts", []),
                        entities=extraction_result.get("entities", []),
                        relations=extraction_result.get("relations", []),
                        content=content,
                        metadata=metadata,
                    )
                )
                extraction_result["enhanced_processing"] = enhanced_processing
            except Exception as e:
                logger.warning("Enhanced fact processing failed", error=str(e))

            logger.info(
                "Graph data extraction completed",
                entities_extracted=len(extraction_result.get("entities", [])),
                relations_extracted=len(extraction_result.get("relations", [])),
                facts_extracted=len(extraction_result.get("facts", [])),
            )

            return extraction_result

        except Exception as e:
            logger.error("Failed to extract graph data", error=str(e))
            # Don't fail the entire ingestion if graph extraction fails
            return None

    async def _generate_entity_and_fact_embeddings(self, graph_data: Dict[str, Any]):
        """Generate embeddings for entities and facts."""
        try:
            if not self.embedding_service:
                return

            # Generate entity embeddings
            entities = graph_data.get("entities", [])
            if entities:
                entity_texts = []
                for entity in entities:
                    # Create text representation for embedding
                    entity_text = (
                        f"{entity.get('name', '')} ({entity.get('type', 'Unknown')})"
                    )
                    if entity.get("description"):
                        entity_text += f": {entity['description']}"
                    entity_texts.append(entity_text)

                entity_embeddings = (
                    await self.embedding_service.generate_embeddings_batch(entity_texts)
                )

                # Add embeddings back to entities
                for i, entity in enumerate(entities):
                    if i < len(entity_embeddings):
                        entity["embedding"] = entity_embeddings[i]

                graph_data["entity_embeddings"] = {
                    entity.get("id", f"entity_{i}"): entity_embeddings[i]
                    for i, entity in enumerate(entities)
                    if i < len(entity_embeddings)
                }

            # Generate fact embeddings
            facts = graph_data.get("facts", [])
            if facts:
                fact_texts = []
                for fact in facts:
                    # Create text representation for embedding
                    fact_text = fact.get("text", "")
                    if (
                        fact.get("subject")
                        and fact.get("predicate")
                        and fact.get("object")
                    ):
                        fact_text = (
                            f"{fact['subject']} {fact['predicate']} {fact['object']}"
                        )
                    fact_texts.append(fact_text)

                fact_embeddings = (
                    await self.embedding_service.generate_embeddings_batch(fact_texts)
                )

                # Add embeddings back to facts
                for i, fact in enumerate(facts):
                    if i < len(fact_embeddings):
                        fact["embedding"] = fact_embeddings[i]

                graph_data["fact_embeddings"] = {
                    fact.get("id", f"fact_{i}"): fact_embeddings[i]
                    for i, fact in enumerate(facts)
                    if i < len(fact_embeddings)
                }

            logger.info(
                "Generated embeddings for graph data",
                entity_embeddings=len(graph_data.get("entity_embeddings", {})),
                fact_embeddings=len(graph_data.get("fact_embeddings", {})),
            )

        except Exception as e:
            logger.error("Failed to generate embeddings for graph data", error=str(e))
            # Don't fail if embedding generation fails


# Wrapper class for backward compatibility
class FactExtractionWrapper:
    """Wrapper to provide backward compatibility for fact extraction."""

    def __init__(self):
        self.service = FactExtractionService()

    async def extract_facts_and_relationships(self, **kwargs):
        """Extract facts and relationships using the fact extraction service."""
        return await self.service.extract_facts_and_relationships(**kwargs)
