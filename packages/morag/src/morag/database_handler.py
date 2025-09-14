"""Database handler for MoRAG ingestion system.

Handles database initialization and data writing operations for multiple database types.
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path

from morag_services.storage import QdrantVectorStorage
from morag_graph.storage.neo4j_storage import Neo4jStorage, Neo4jConfig
from morag_graph.storage.qdrant_storage import QdrantStorage, QdrantConfig
from morag_graph.models.database_config import DatabaseConfig, DatabaseType
from morag_graph.models.entity import Entity
from morag_graph.models.relation import Relation
from morag_graph.models.fact import Fact, FactRelation
from morag_graph.models.document import Document
from morag_graph.models.document_chunk import DocumentChunk
from morag_core.utils.logging import get_logger

logger = get_logger(__name__)


class DatabaseHandler:
    """Handles database operations with connection pooling."""

    def __init__(self):
        """Initialize the database handler with connection pools."""
        self._qdrant_pool = {}  # Connection pool for Qdrant
        self._neo4j_pool = {}  # Connection pool for Neo4j
        self._pool_lock = asyncio.Lock()

    async def _get_qdrant_connection(self, config_key: str, db_config: DatabaseConfig):
        """Get or create pooled Qdrant connection."""
        async with self._pool_lock:
            if config_key not in self._qdrant_pool:
                host = db_config.hostname or 'localhost'
                port = db_config.port or 6333

                # Handle URL-style hostnames
                if host.startswith(('http://', 'https://')):
                    from urllib.parse import urlparse
                    parsed = urlparse(host)
                    hostname = parsed.hostname or "localhost"
                    port = parsed.port or (443 if parsed.scheme == 'https' else port)
                    config_host = host
                else:
                    hostname = host
                    config_host = hostname

                verify_ssl = os.getenv('QDRANT_VERIFY_SSL', 'true').lower() == 'true'

                storage = QdrantVectorStorage(
                    host=config_host,
                    port=port,
                    api_key=os.getenv('QDRANT_API_KEY'),
                    collection_name=db_config.database_name or 'morag_documents',
                    verify_ssl=verify_ssl
                )
                await storage.connect()
                self._qdrant_pool[config_key] = storage
            return self._qdrant_pool[config_key]

    async def _get_neo4j_connection(self, config_key: str, db_config: DatabaseConfig):
        """Get or create pooled Neo4j connection."""
        async with self._pool_lock:
            if config_key not in self._neo4j_pool:
                neo4j_config = Neo4jConfig(
                    uri=db_config.hostname or 'bolt://localhost:7687',
                    username=db_config.username or 'neo4j',
                    password=db_config.password or 'password',
                    database=db_config.database_name or 'neo4j',
                    verify_ssl=os.getenv("NEO4J_VERIFY_SSL", "true").lower() == "true",
                    trust_all_certificates=os.getenv("NEO4J_TRUST_ALL_CERTIFICATES", "false").lower() == "true"
                )

                storage = Neo4jStorage(neo4j_config)
                await storage.connect()
                self._neo4j_pool[config_key] = storage
            return self._neo4j_pool[config_key]

    def _get_config_key(self, db_config: DatabaseConfig) -> str:
        """Generate unique key for database configuration."""
        return f"{db_config.type.value}_{db_config.hostname}_{db_config.port}_{db_config.database_name}"

    async def cleanup_connections(self):
        """Cleanup all pooled connections."""
        async with self._pool_lock:
            # Close Qdrant connections
            for storage in self._qdrant_pool.values():
                try:
                    await storage.disconnect()
                except Exception as e:
                    logger.warning(f"Error closing Qdrant connection: {e}")

            # Close Neo4j connections
            for storage in self._neo4j_pool.values():
                try:
                    await storage.disconnect()
                except Exception as e:
                    logger.warning(f"Error closing Neo4j connection: {e}")

            self._qdrant_pool.clear()
            self._neo4j_pool.clear()

    async def initialize_databases(
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
        """Initialize Qdrant collection using connection pool."""
        logger.info("Initializing Qdrant with config",
                   hostname=db_config.hostname,
                   port=db_config.port,
                   database_name=db_config.database_name)

        config_key = self._get_config_key(db_config)
        qdrant_storage = await self._get_qdrant_connection(config_key, db_config)

        logger.info("Qdrant collection initialized",
                   collection=db_config.database_name or 'morag_documents',
                   vector_size=embeddings_data.get('embedding_dimension', 768))

        # Connection remains in pool - no disconnect needed

    async def _initialize_neo4j(self, db_config: DatabaseConfig) -> None:
        """Initialize Neo4j database using connection pool."""
        config_key = self._get_config_key(db_config)
        neo4j_storage = await self._get_neo4j_connection(config_key, db_config)

        # Test the connection and ensure database exists
        await neo4j_storage.test_connection()

        logger.info("Neo4j database initialized",
                   database=db_config.database_name or 'neo4j',
                   uri=db_config.hostname or 'bolt://localhost:7687')

        # Connection remains in pool - no disconnect needed

    async def write_to_databases(
        self,
        database_configs: List[DatabaseConfig],
        ingest_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Write data to all configured databases."""
        database_results = {}

        for db_config in database_configs:
            try:
                if db_config.type == DatabaseType.QDRANT:
                    result = await self._write_to_qdrant(db_config, ingest_data)
                    database_results['qdrant'] = result
                elif db_config.type == DatabaseType.NEO4J:
                    result = await self._write_to_neo4j(db_config, ingest_data)
                    database_results['neo4j'] = result

            except Exception as e:
                logger.error("Failed to write to database",
                           database_type=db_config.type.value,
                           error=str(e))
                database_results[db_config.type.value.lower()] = {
                    'success': False,
                    'error': str(e)
                }

        return database_results

    async def _write_to_qdrant(
        self,
        db_config: DatabaseConfig,
        ingest_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Write chunks and embeddings to Qdrant using connection pool."""
        config_key = self._get_config_key(db_config)
        qdrant_storage = await self._get_qdrant_connection(config_key, db_config)

        try:
            # Process chunk data
            chunks_data = ingest_data.get('chunks', [])
            point_ids = []

            for chunk_data in chunks_data:
                if 'embedding' in chunk_data and chunk_data['embedding']:
                    point_id = await qdrant_storage.upsert_document_chunk(
                        chunk_text=chunk_data['text'],
                        embedding=chunk_data['embedding'],
                        metadata=chunk_data.get('metadata', {}),
                        document_id=chunk_data.get('document_id'),
                        chunk_index=chunk_data.get('chunk_index', 0)
                    )
                    point_ids.append(point_id)

            logger.info("Successfully wrote to Qdrant",
                       collection=db_config.database_name or 'morag_documents',
                       points_written=len(point_ids))

            return {
                'success': True,
                'collection': db_config.database_name or 'morag_documents',
                'points_written': len(point_ids),
                'point_ids': point_ids
            }

        except Exception as e:
            logger.error("Failed to write to Qdrant", error=str(e))
            raise
        # No finally block - connection remains in pool

    async def _write_to_neo4j(
        self,
        db_config: DatabaseConfig,
        ingest_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Write graph data to Neo4j using connection pool."""
        config_key = self._get_config_key(db_config)
        neo4j_storage = await self._get_neo4j_connection(config_key, db_config)

        try:

            # Get graph data
            graph_data = ingest_data.get('graph_data', {})
            
            entities_stored = 0
            relations_stored = 0
            facts_stored = 0

            # Store entities
            entities = graph_data.get('entities', [])
            for entity_dict in entities:
                entity = Entity(
                    id=entity_dict['id'],
                    name=entity_dict['name'],
                    type=entity_dict['type'],
                    properties=entity_dict.get('properties', {}),
                    embedding=entity_dict.get('embedding')
                )
                await neo4j_storage.store_entity(entity)
                entities_stored += 1

            # Store relations
            relations = graph_data.get('relations', [])
            for relation_dict in relations:
                relation = Relation(
                    id=relation_dict['id'],
                    source_id=relation_dict['source_id'],
                    target_id=relation_dict['target_id'],
                    type=relation_dict['type'],
                    properties=relation_dict.get('properties', {}),
                    weight=relation_dict.get('weight', 1.0)
                )
                await neo4j_storage.store_relation(relation)
                relations_stored += 1

            # Store facts
            facts = graph_data.get('facts', [])
            for fact_dict in facts:
                fact = Fact(
                    id=fact_dict['id'],
                    text=fact_dict['text'],
                    subject=fact_dict.get('subject', ''),
                    predicate=fact_dict.get('predicate', ''),
                    object=fact_dict.get('object', ''),
                    confidence=fact_dict.get('confidence', 0.0),
                    source_chunk_id=fact_dict.get('source_chunk_id'),
                    embedding=fact_dict.get('embedding')
                )
                await neo4j_storage.store_fact(fact)
                facts_stored += 1

            logger.info("Successfully wrote to Neo4j",
                       database=db_config.database_name or 'neo4j',
                       entities_stored=entities_stored,
                       relations_stored=relations_stored,
                       facts_stored=facts_stored)

            return {
                'success': True,
                'database': db_config.database_name or 'neo4j',
                'entities_stored': entities_stored,
                'relations_stored': relations_stored,
                'facts_stored': facts_stored
            }

        except Exception as e:
            logger.error("Failed to write to Neo4j", error=str(e))
            raise
        # No finally block - connection remains in pool