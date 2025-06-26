"""Graph processing integration for MoRAG services.

This module provides integration between the main MoRAG processing pipeline
and the morag-graph package for entity and relation extraction.
"""

import asyncio
import os
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import structlog
from pydantic import BaseModel

try:
    from morag_graph import (
        EntityExtractor, RelationExtractor, Neo4jStorage, QdrantStorage,
        Neo4jConfig, QdrantConfig, DatabaseType, DatabaseConfig, DatabaseResult,
        Entity, Relation
    )
    from morag_graph.extraction.base import LLMConfig
    GRAPH_AVAILABLE = True
except ImportError as e:
    logger.warning("morag-graph package not available - graph processing disabled", error=str(e))
    GRAPH_AVAILABLE = False
    EntityExtractor = None
    RelationExtractor = None
    Neo4jStorage = None
    QdrantStorage = None
    Neo4jConfig = None
    QdrantConfig = None
    DatabaseType = None
    DatabaseConfig = None
    DatabaseResult = None
    Entity = None
    Relation = None
    LLMConfig = None

logger = structlog.get_logger(__name__)


class GraphProcessingConfig(BaseModel):
    """Configuration for graph processing."""
    
    enabled: bool = False
    neo4j_uri: Optional[str] = None
    neo4j_username: Optional[str] = None
    neo4j_password: Optional[str] = None
    neo4j_database: Optional[str] = "neo4j"
    
    # LLM configuration for extraction
    llm_provider: str = "gemini"
    llm_api_key: Optional[str] = None
    llm_model: Optional[str] = None
    
    # Processing options
    chunk_by_structure: bool = True
    max_chunk_size: int = 4000
    entity_types: Optional[Dict[str, str]] = None
    relation_types: Optional[Dict[str, str]] = None
    
    @classmethod
    def from_env(cls) -> "GraphProcessingConfig":
        """Create configuration from environment variables."""
        return cls(
            enabled=os.getenv("MORAG_GRAPH_ENABLED", "false").lower() == "true",
            neo4j_uri=os.getenv("NEO4J_URI"),
            neo4j_username=os.getenv("NEO4J_USERNAME"),
            neo4j_password=os.getenv("NEO4J_PASSWORD"),
            neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
            llm_provider=os.getenv("MORAG_GRAPH_LLM_PROVIDER", "gemini"),
            llm_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
            llm_model=os.getenv("MORAG_GRAPH_LLM_MODEL"),
            chunk_by_structure=os.getenv("MORAG_GRAPH_CHUNK_BY_STRUCTURE", "true").lower() == "true",
            max_chunk_size=int(os.getenv("MORAG_GRAPH_MAX_CHUNK_SIZE", "4000"))
        )


class GraphProcessingResult(BaseModel):
    """Result of graph processing."""
    
    success: bool
    entities_count: int = 0
    relations_count: int = 0
    chunks_processed: int = 0
    processing_time: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = {}
    database_results: List[Dict[str, Any]] = []  # Results for each database


class GraphProcessor:
    """Processor for extracting entities and relations from documents."""
    
    def __init__(self, config: Optional[GraphProcessingConfig] = None):
        """Initialize the graph processor.
        
        Args:
            config: Graph processing configuration
        """
        self.config = config or GraphProcessingConfig.from_env()
        self._entity_extractor = None
        self._relation_extractor = None
        self._storage = None
        self._file_ingestion = None
        self._llm_config = None
        
        if not GRAPH_AVAILABLE:
            self.config.enabled = False
            return
            
        if self.config.enabled:
            self._initialize_components()
    
    def _initialize_components(self):
        """Initialize graph processing components."""
        try:
            # Initialize LLM configuration
            self._llm_config = LLMConfig(
                provider=self.config.llm_provider,
                api_key=self.config.llm_api_key,
                model=self.config.llm_model
            )

            # Initialize extractors
            self._entity_extractor = EntityExtractor(
                config=self._llm_config,
                entity_types=self.config.entity_types
            )

            self._relation_extractor = RelationExtractor(
                config=self._llm_config,
                relation_types=self.config.relation_types
            )
            
            # Initialize Neo4j storage
            if all([self.config.neo4j_uri, self.config.neo4j_username, self.config.neo4j_password]):
                self._storage = Neo4jStorage(
                    uri=self.config.neo4j_uri,
                    username=self.config.neo4j_username,
                    password=self.config.neo4j_password,
                    database=self.config.neo4j_database
                )
                
                # Initialize file ingestion
                self._file_ingestion = FileIngestion(self._storage)
                
                logger.info("Graph processing components initialized successfully")
            else:
                logger.warning("Neo4j configuration incomplete - graph processing disabled")
                self.config.enabled = False
                
        except Exception as e:
            logger.error("Failed to initialize graph processing components", error=str(e))
            self.config.enabled = False
    
    def is_enabled(self) -> bool:
        """Check if graph processing is enabled and available."""
        return self.config.enabled and GRAPH_AVAILABLE and self._storage is not None

    async def generate_document_intention(self, content: str, max_length: int = 200) -> Optional[str]:
        """Generate a concise intention summary for the document.

        Args:
            content: Document content to analyze
            max_length: Maximum length for the intention summary

        Returns:
            Document intention summary or None if generation fails
        """
        if not self._llm_config or not self._llm_config.api_key:
            logger.warning("LLM configuration not available for intention generation")
            return None

        try:
            # Import here to avoid circular dependencies
            import google.generativeai as genai

            # Configure the API
            genai.configure(api_key=self._llm_config.api_key)

            # Create the model
            model = genai.GenerativeModel(self._llm_config.model or "gemini-1.5-flash")

            # Create intention analysis prompt
            prompt = f"""
Analyze the following document and provide a concise intention summary that captures the document's primary purpose and domain.

The intention should be a single sentence that describes what the document aims to achieve or communicate.

Examples:
- For medical content: "Heal the pineal gland for spiritual enlightenment"
- For organizational documents: "Document explaining the structure of the organization/company"
- For technical guides: "Guide for implementing software architecture patterns"
- For educational content: "Teach fundamental concepts of machine learning"

Document content:
{content[:2000]}...

Provide only the intention summary (maximum {max_length} characters):
"""

            # Generate intention
            response = model.generate_content(prompt)
            intention = response.text.strip()

            # Ensure it's within max length
            if len(intention) > max_length:
                intention = intention[:max_length-3] + "..."

            logger.debug(f"Generated document intention: {intention}")
            return intention

        except Exception as e:
            logger.warning(f"Failed to generate document intention: {e}")
            return None
    
    def _create_storage_from_config(self, db_config: DatabaseConfig):
        """Create a storage instance from database configuration.
        
        Args:
            db_config: Database configuration
            
        Returns:
            Storage instance
        """
        if db_config.type == DatabaseType.NEO4J:
            # Use provided config or fall back to defaults
            neo4j_config = Neo4jConfig(
                uri=db_config.hostname or self.config.neo4j_uri or "neo4j://localhost:7687",
                username=db_config.username or self.config.neo4j_username or "neo4j",
                password=db_config.password or self.config.neo4j_password or "password",
                database=db_config.database_name or self.config.neo4j_database or "neo4j"
            )
            return Neo4jStorage(neo4j_config)
            
        elif db_config.type == DatabaseType.QDRANT:
            # Use provided config or fall back to defaults
            qdrant_config = QdrantConfig(
                host=db_config.hostname or "localhost",
                port=db_config.port or 6333,
                api_key=db_config.password,  # Use password field for API key
                collection_name=db_config.database_name or "morag_entities"
            )
            return QdrantStorage(qdrant_config)
            
        else:
            raise ValueError(f"Unsupported database type: {db_config.type}")
    
    async def process_document_multi_db(
        self,
        content: str,
        source_doc_id: str,
        database_configs: List[DatabaseConfig],
        metadata: Optional[Dict[str, Any]] = None
    ) -> GraphProcessingResult:
        """Process a document and store in multiple databases.
        
        Args:
            content: Document content to process
            source_doc_id: Unique identifier for the source document
            database_configs: List of database configurations
            metadata: Optional metadata
            
        Returns:
            GraphProcessingResult with results for each database
        """
        if not GRAPH_AVAILABLE:
            return GraphProcessingResult(
                success=False,
                error_message="Graph processing not available - morag-graph package not installed"
            )
        
        if not database_configs:
            # Fall back to single database processing if no configs provided
            if self.is_enabled():
                return await self.process_document(content, source_doc_id, metadata)
            else:
                return GraphProcessingResult(
                    success=False,
                    error_message="No database configurations provided and default graph processing not enabled"
                )
        
        start_time = asyncio.get_event_loop().time()
        database_results = []
        total_entities = 0
        total_relations = 0
        processed_databases = set()  # Track to prevent duplicates
        
        try:
            # Generate document intention for context-aware extraction
            intention = await self.generate_document_intention(content)
            if intention:
                logger.info(f"Generated document intention: {intention}")

            # Extract entities and relations once with intention context
            entities = await self._entity_extractor.extract_entities(content, source_doc_id, intention=intention)
            relations = await self._relation_extractor.extract_relations(content, entities, source_doc_id, intention=intention)
            
            # Process each database configuration
            for db_config in database_configs:
                db_start_time = asyncio.get_event_loop().time()
                
                # Check for duplicate database configurations
                connection_key = db_config.get_connection_key()
                if connection_key in processed_databases:
                    logger.warning(f"Skipping duplicate database configuration: {connection_key}")
                    database_results.append({
                        "database_type": db_config.type.value,
                        "connection_key": connection_key,
                        "success": False,
                        "error_message": "Duplicate database configuration skipped",
                        "entities_count": 0,
                        "relations_count": 0,
                        "processing_time": 0.0
                    })
                    continue
                
                processed_databases.add(connection_key)
                
                try:
                    # Create storage instance for this database
                    storage = self._create_storage_from_config(db_config)
                    
                    # Connect to database
                    await storage.connect()
                    
                    try:
                        # Store entities and relations
                        entity_ids = await storage.store_entities(entities)
                        relation_ids = await storage.store_relations(relations)
                        
                        db_processing_time = asyncio.get_event_loop().time() - db_start_time
                        
                        database_results.append({
                            "database_type": db_config.type.value,
                            "connection_key": connection_key,
                            "success": True,
                            "entities_count": len(entity_ids),
                            "relations_count": len(relation_ids),
                            "processing_time": db_processing_time,
                            "metadata": {
                                "entity_ids": entity_ids[:10],  # Limit for webhook size
                                "relation_ids": relation_ids[:10]
                            }
                        })
                        
                        total_entities += len(entity_ids)
                        total_relations += len(relation_ids)
                        
                        logger.info(
                            f"Successfully processed document in {db_config.type.value} database",
                            entities=len(entity_ids),
                            relations=len(relation_ids),
                            processing_time=db_processing_time
                        )
                        
                    finally:
                        # Always disconnect
                        await storage.disconnect()
                        
                except Exception as e:
                    db_processing_time = asyncio.get_event_loop().time() - db_start_time
                    error_msg = f"Failed to process in {db_config.type.value} database: {str(e)}"
                    logger.error(error_msg)
                    
                    database_results.append({
                        "database_type": db_config.type.value,
                        "connection_key": connection_key,
                        "success": False,
                        "error_message": error_msg,
                        "entities_count": 0,
                        "relations_count": 0,
                        "processing_time": db_processing_time
                    })
            
            total_processing_time = asyncio.get_event_loop().time() - start_time
            
            # Determine overall success
            successful_dbs = sum(1 for result in database_results if result["success"])
            overall_success = successful_dbs > 0
            
            return GraphProcessingResult(
                success=overall_success,
                entities_count=total_entities,
                relations_count=total_relations,
                chunks_processed=1,
                processing_time=total_processing_time,
                database_results=database_results,
                metadata={
                    "successful_databases": successful_dbs,
                    "total_databases": len(database_configs),
                    "processed_databases": list(processed_databases)
                }
            )
            
        except Exception as e:
            total_processing_time = asyncio.get_event_loop().time() - start_time
            error_msg = f"Graph processing failed: {str(e)}"
            logger.error(error_msg)
            
            return GraphProcessingResult(
                success=False,
                processing_time=total_processing_time,
                error_message=error_msg,
                database_results=database_results
            )
    
    def _chunk_by_structure(self, markdown_content: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Chunk markdown content by structural elements.
        
        Args:
            markdown_content: Markdown content to chunk
            
        Returns:
            List of (chunk_content, metadata) tuples
        """
        chunks = []
        lines = markdown_content.split('\n')
        current_chunk = []
        current_section = None
        current_subsection = None
        
        for line in lines:
            # Detect headers
            if line.startswith('#'):
                # Save previous chunk if it exists
                if current_chunk:
                    chunk_content = '\n'.join(current_chunk).strip()
                    if chunk_content:
                        metadata = {
                            'section': current_section,
                            'subsection': current_subsection,
                            'chunk_type': 'structured'
                        }
                        chunks.append((chunk_content, metadata))
                    current_chunk = []
                
                # Update section tracking
                if line.startswith('# '):
                    current_section = line[2:].strip()
                    current_subsection = None
                elif line.startswith('## '):
                    current_subsection = line[3:].strip()
            
            current_chunk.append(line)
            
            # Check if chunk is getting too large
            chunk_text = '\n'.join(current_chunk)
            if len(chunk_text) > self.config.max_chunk_size:
                # Save current chunk
                metadata = {
                    'section': current_section,
                    'subsection': current_subsection,
                    'chunk_type': 'structured'
                }
                chunks.append((chunk_text.strip(), metadata))
                current_chunk = []
        
        # Save final chunk
        if current_chunk:
            chunk_content = '\n'.join(current_chunk).strip()
            if chunk_content:
                metadata = {
                    'section': current_section,
                    'subsection': current_subsection,
                    'chunk_type': 'structured'
                }
                chunks.append((chunk_content, metadata))
        
        return chunks
    
    def _chunk_by_size(self, markdown_content: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Chunk markdown content by size.
        
        Args:
            markdown_content: Markdown content to chunk
            
        Returns:
            List of (chunk_content, metadata) tuples
        """
        chunks = []
        words = markdown_content.split()
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            chunk_text = ' '.join(current_chunk)
            
            if len(chunk_text) > self.config.max_chunk_size:
                # Remove last word and save chunk
                current_chunk.pop()
                if current_chunk:
                    chunk_content = ' '.join(current_chunk)
                    metadata = {'chunk_type': 'size_based'}
                    chunks.append((chunk_content, metadata))
                
                # Start new chunk with the word that exceeded the limit
                current_chunk = [word]
        
        # Save final chunk
        if current_chunk:
            chunk_content = ' '.join(current_chunk)
            metadata = {'chunk_type': 'size_based'}
            chunks.append((chunk_content, metadata))
        
        return chunks
    
    async def process_document(
        self,
        markdown_content: str,
        document_path: Optional[str] = None,
        document_metadata: Optional[Dict[str, Any]] = None
    ) -> GraphProcessingResult:
        """Process a document for graph extraction.
        
        Args:
            markdown_content: Document content in markdown format
            document_path: Optional path to the original document
            document_metadata: Optional metadata about the document
            
        Returns:
            GraphProcessingResult with extraction results
        """
        if not self.is_enabled():
            return GraphProcessingResult(
                success=False,
                error_message="Graph processing not enabled or available"
            )
        
        import time
        start_time = time.time()
        
        try:
            # Generate document intention for context-aware extraction
            intention = await self.generate_document_intention(markdown_content)
            if intention:
                logger.info(f"Generated document intention: {intention}")

            # Chunk the content
            if self.config.chunk_by_structure:
                chunks = self._chunk_by_structure(markdown_content)
            else:
                chunks = self._chunk_by_size(markdown_content)

            logger.info("Document chunked for graph processing",
                       chunks_count=len(chunks),
                       chunk_by_structure=self.config.chunk_by_structure)

            all_entities = []
            all_relations = []
            
            # Process each chunk
            for i, (chunk_content, chunk_metadata) in enumerate(chunks):
                try:
                    # Extract entities with intention context
                    entities = await self._entity_extractor.extract(
                        chunk_content,
                        source_doc_id=document_path,
                        intention=intention
                    )

                    # Extract relations with intention context
                    relations = await self._relation_extractor.extract(
                        chunk_content,
                        entities=entities,
                        intention=intention
                    )
                    
                    # Add chunk metadata to entities and relations
                    for entity in entities:
                        entity.metadata.update(chunk_metadata)
                        entity.metadata['chunk_index'] = i
                    
                    for relation in relations:
                        relation.metadata.update(chunk_metadata)
                        relation.metadata['chunk_index'] = i
                    
                    all_entities.extend(entities)
                    all_relations.extend(relations)
                    
                    logger.debug("Processed chunk for graph extraction",
                               chunk_index=i,
                               entities_count=len(entities),
                               relations_count=len(relations))
                    
                except Exception as e:
                    logger.warning("Failed to process chunk for graph extraction",
                                 chunk_index=i,
                                 error=str(e))
                    continue
            
            # Store in Neo4j
            if all_entities or all_relations:
                await self._storage.store_entities(all_entities)
                await self._storage.store_relations(all_relations)
                
                logger.info("Graph data stored successfully",
                           entities_count=len(all_entities),
                           relations_count=len(all_relations))
            
            processing_time = time.time() - start_time
            
            return GraphProcessingResult(
                success=True,
                entities_count=len(all_entities),
                relations_count=len(all_relations),
                chunks_processed=len(chunks),
                processing_time=processing_time,
                metadata={
                    'document_path': document_path,
                    'chunk_strategy': 'structure' if self.config.chunk_by_structure else 'size',
                    'total_chunks': len(chunks)
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error("Graph processing failed", error=str(e))
            
            return GraphProcessingResult(
                success=False,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    async def close(self):
        """Close graph processing components."""
        if self._storage:
            await self._storage.close()