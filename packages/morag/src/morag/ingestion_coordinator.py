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

import structlog

logger = structlog.get_logger(__name__)


class IngestionCoordinator:
    """Coordinates the complete ingestion process across multiple databases."""
    
    def __init__(self):
        """Initialize the ingestion coordinator."""
        self.embedding_service = None
        self.vector_storage = None
        self.graph_extractor = None
        
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
        qdrant_host = os.getenv('QDRANT_HOST', 'localhost')
        qdrant_port = int(os.getenv('QDRANT_PORT', '6333'))
        qdrant_api_key = os.getenv('QDRANT_API_KEY')
        collection_name = os.getenv('QDRANT_COLLECTION_NAME')
        if not collection_name:
            raise ValueError("QDRANT_COLLECTION_NAME environment variable is required")

        self.vector_storage = QdrantVectorStorage(
            host=qdrant_host,
            port=qdrant_port,
            api_key=qdrant_api_key,
            collection_name=collection_name
        )

        # Initialize graph extractor
        self.graph_extractor = GraphExtractor()
        
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
        replace_existing: bool = False
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
        
        # Step 4: Extract entities and relations for graph databases
        graph_data = await self._extract_graph_data(
            content, source_path, document_id, metadata
        )
        
        # Step 5: Create complete ingest_result.json data
        ingest_result = self._create_ingest_result(
            source_path, content_type, metadata, processing_result,
            embeddings_data, graph_data, database_configs, start_time
        )
        
        # Step 6: Write ingest_result.json file
        result_file_path = self._write_ingest_result_file(source_path, ingest_result)
        
        # Step 7: Initialize databases (create collections/databases if needed)
        await self._initialize_databases(database_configs, embeddings_data)
        
        # Step 8: Write data to databases using the ingest_result data
        database_results = await self._write_to_databases(
            database_configs, embeddings_data, graph_data, document_id, replace_existing
        )
        
        # Step 9: Update final result with database write results
        ingest_result['database_results'] = database_results
        ingest_result['processing_time'] = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        # Step 10: Update the ingest_result.json file with final results
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
        qdrant_host = os.getenv('QDRANT_HOST', 'localhost')
        qdrant_collection = os.getenv('QDRANT_COLLECTION_NAME', 'morag_documents')
        
        if qdrant_host and qdrant_collection:
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
        return UnifiedIDGenerator.generate_document_id(source_path, content)
        
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
            
        # Create chunks
        chunks = self._create_chunks(content, chunk_size, chunk_overlap)
        
        # Generate embeddings for all chunks
        embedding_results = await self.embedding_service.generate_embeddings_batch(
            chunks, task_type="retrieval_document"
        )

        # Extract embeddings from results
        embeddings = [result.embedding for result in embedding_results]
        
        # Create chunk metadata
        chunk_metadata = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = UnifiedIDGenerator.generate_chunk_id(document_id, i)
            chunk_meta = {
                'chunk_id': chunk_id,
                'document_id': document_id,
                'chunk_index': i,
                'chunk_text': chunk,
                'chunk_size': len(chunk),
                'created_at': datetime.now(timezone.utc).isoformat(),
                **metadata
            }
            chunk_metadata.append(chunk_meta)
            
        return {
            'chunks': chunks,
            'embeddings': embeddings,
            'chunk_metadata': chunk_metadata,
            'document_id': document_id,
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap
        }
        
    def _create_chunks(self, content: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Create text chunks with word boundary preservation."""
        if len(content) <= chunk_size:
            return [content]
            
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
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract entities and relations for graph databases."""
        try:
            # Extract entities and relations using graph extractor
            extraction_result = await self.graph_extractor.extract_entities_and_relations(
                content, source_path
            )

            entities = []
            relations = []

            if extraction_result.get('entities'):
                for entity_data in extraction_result['entities']:
                    # Prepare attributes with description if available
                    entity_attributes = entity_data.get('attributes', {})
                    if entity_data.get('description'):
                        entity_attributes['description'] = entity_data['description']

                    entity = Entity(
                        id=entity_data.get('id') or str(uuid.uuid4()),
                        name=entity_data['name'],
                        type=entity_data.get('type', 'UNKNOWN'),
                        attributes=entity_attributes,
                        source_doc_id=document_id,
                        confidence=entity_data.get('confidence', 0.8)
                    )
                    entities.append(entity)

            if extraction_result.get('relations'):
                for relation_data in extraction_result['relations']:
                    # Prepare attributes with description if available
                    relation_attributes = relation_data.get('attributes', {})
                    if relation_data.get('description'):
                        relation_attributes['description'] = relation_data['description']

                    relation = Relation(
                        id=relation_data.get('id') or str(uuid.uuid4()),
                        source_entity_id=relation_data['source_entity_id'],
                        target_entity_id=relation_data['target_entity_id'],
                        type=relation_data['relation_type'],
                        attributes=relation_attributes,
                        source_doc_id=document_id,
                        confidence=relation_data.get('confidence', 0.8)
                    )
                    relations.append(relation)

            return {
                'entities': entities,
                'relations': relations,
                'extraction_metadata': extraction_result.get('metadata', {})
            }

        except Exception as e:
            logger.warning("Failed to extract graph data", error=str(e))
            return {
                'entities': [],
                'relations': [],
                'extraction_metadata': {'error': str(e)}
            }

    def _create_ingest_result(
        self,
        source_path: str,
        content_type: str,
        metadata: Dict[str, Any],
        processing_result: ProcessingResult,
        embeddings_data: Dict[str, Any],
        graph_data: Dict[str, Any],
        database_configs: List[DatabaseConfig],
        start_time: datetime
    ) -> Dict[str, Any]:
        """Create the complete ingest_result.json data structure."""
        return {
            'ingestion_id': str(uuid.uuid4()),
            'timestamp': start_time.isoformat(),
            'source_info': {
                'source_path': source_path,
                'content_type': content_type,
                'document_id': embeddings_data['document_id']
            },
            'processing_result': {
                'success': processing_result.success,
                'processing_time': processing_result.processing_time,
                'content_length': len(processing_result.content or ''),
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
                        'chunk_text': meta['chunk_text'],
                        'chunk_size': meta['chunk_size'],
                        'embedding': embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
                        'metadata': {k: v for k, v in meta.items() if k not in ['chunk_text']}
                    }
                    for meta, embedding in zip(embeddings_data['chunk_metadata'], embeddings_data['embeddings'])
                ]
            },
            'graph_data': {
                'entities_count': len(graph_data['entities']),
                'relations_count': len(graph_data['relations']),
                'entities': [
                    {
                        'id': entity.id,
                        'name': entity.name,
                        'type': entity.type.value if hasattr(entity.type, 'value') else str(entity.type),
                        'description': entity.attributes.get('description', ''),
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
                        'description': relation.attributes.get('description', ''),
                        'attributes': relation.attributes,
                        'confidence': relation.confidence
                    }
                    for relation in graph_data['relations']
                ],
                'extraction_metadata': graph_data['extraction_metadata']
            },
            'metadata': metadata,
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
        qdrant_config = QdrantConfig(
            host=db_config.hostname or 'localhost',
            port=db_config.port or 6333,
            collection_name=db_config.database_name or 'morag_documents',
            vector_size=embeddings_data.get('embedding_dimension', 768)
        )

        qdrant_storage = QdrantStorage(qdrant_config)
        await qdrant_storage.connect()

        logger.info("Qdrant collection initialized",
                   collection=qdrant_config.collection_name,
                   vector_size=qdrant_config.vector_size)

        await qdrant_storage.disconnect()

    async def _initialize_neo4j(self, db_config: DatabaseConfig) -> None:
        """Initialize Neo4j database."""
        neo4j_config = Neo4jConfig(
            uri=db_config.hostname or 'bolt://localhost:7687',
            username=db_config.username or 'neo4j',
            password=db_config.password or 'password',
            database=db_config.database_name or 'neo4j'
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
        replace_existing: bool
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
                    result = await self._write_to_neo4j(
                        db_config, graph_data, embeddings_data, document_id
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
        qdrant_config = QdrantConfig(
            host=db_config.hostname or 'localhost',
            port=db_config.port or 6333,
            collection_name=db_config.database_name or 'morag_documents',
            vector_size=embeddings_data.get('embedding_dimension', 768)
        )

        qdrant_storage = QdrantStorage(qdrant_config)
        await qdrant_storage.connect()

        try:
            point_ids = []

            # Store each chunk as a vector point
            for i, (chunk_meta, embedding) in enumerate(
                zip(embeddings_data['chunk_metadata'], embeddings_data['embeddings'])
            ):
                # Create a hash-based point ID for Qdrant (since it doesn't support colons in IDs)
                import hashlib
                chunk_id = chunk_meta['chunk_id']
                point_id = hashlib.sha256(chunk_id.encode()).hexdigest()[:16]  # Use first 16 chars of hash

                # Store the vector with the hash as point ID but keep original chunk_id in metadata
                enhanced_metadata = {
                    **chunk_meta,
                    'original_chunk_id': chunk_id,
                    'point_id': point_id
                }

                # Use the basic Qdrant client directly with proper format
                try:
                    await qdrant_storage.client.upsert(
                        collection_name=qdrant_storage.config.collection_name,
                        points=[{
                            "id": point_id,
                            "vector": embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
                            "payload": enhanced_metadata
                        }]
                    )
                except Exception as e:
                    logger.warning(f"Failed to store chunk in Qdrant: {e}")
                    # Try with integer ID instead
                    int_point_id = hash(chunk_id) % (2**31)  # Convert to positive 32-bit int
                    await qdrant_storage.client.upsert(
                        collection_name=qdrant_storage.config.collection_name,
                        points=[{
                            "id": int_point_id,
                            "vector": embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
                            "payload": {**enhanced_metadata, 'point_id': int_point_id}
                        }]
                    )
                    point_id = str(int_point_id)

                point_ids.append(point_id)

            await qdrant_storage.disconnect()

            return {
                'success': True,
                'points_stored': len(point_ids),
                'point_ids': point_ids,
                'collection': qdrant_config.collection_name
            }

        except Exception as e:
            await qdrant_storage.disconnect()
            raise

    async def _write_to_neo4j(
        self,
        db_config: DatabaseConfig,
        graph_data: Dict[str, Any],
        embeddings_data: Dict[str, Any],
        document_id: str
    ) -> Dict[str, Any]:
        """Write graph data to Neo4j."""
        neo4j_config = Neo4jConfig(
            uri=db_config.hostname or 'bolt://localhost:7687',
            username=db_config.username or 'neo4j',
            password=db_config.password or 'password',
            database=db_config.database_name or 'neo4j'
        )

        neo4j_storage = Neo4jStorage(neo4j_config)
        await neo4j_storage.connect()

        try:
            # Store document
            source_path = embeddings_data['chunk_metadata'][0].get('source_path', 'Unknown')
            document = Document(
                id=document_id,
                source_file=source_path,
                file_name=Path(source_path).name if source_path else 'Unknown',
                mime_type=embeddings_data['chunk_metadata'][0].get('source_type', 'unknown'),
                metadata=embeddings_data['chunk_metadata'][0]
            )

            document_id_stored = await neo4j_storage.store_document(document)

            # Store document chunks
            chunk_ids = []
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

            # Store entities
            entity_ids = []
            for entity in graph_data['entities']:
                entity_id_stored = await neo4j_storage.store_entity(entity)
                entity_ids.append(entity_id_stored)

            # Store relations
            relation_ids = []
            for relation in graph_data['relations']:
                relation_id_stored = await neo4j_storage.store_relation(relation)
                relation_ids.append(relation_id_stored)

            await neo4j_storage.disconnect()

            return {
                'success': True,
                'document_stored': document_id_stored,
                'chunks_stored': len(chunk_ids),
                'entities_stored': len(entity_ids),
                'relations_stored': len(relation_ids),
                'database': neo4j_config.database
            }

        except Exception as e:
            await neo4j_storage.disconnect()
            raise
