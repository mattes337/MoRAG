"""ID migration utilities for converting existing IDs to unified format.

This module provides services for migrating existing document, chunk, entity,
and relation IDs from legacy formats to the new unified ID format.
"""

import asyncio
from typing import List, Dict, Any, Optional
from ..storage.neo4j_storage import Neo4jStorage
from ..storage.qdrant_storage import QdrantStorage
from .id_generation import UnifiedIDGenerator, IDValidator


class IDMigrationService:
    """Service for migrating existing IDs to unified format."""

    def __init__(self, neo4j_storage: Neo4jStorage, qdrant_storage: Optional[QdrantStorage] = None):
        self.neo4j = neo4j_storage
        self.qdrant = qdrant_storage
        self.migration_log: List[Dict[str, Any]] = []

    async def migrate_document_ids(self, batch_size: int = 100) -> Dict[str, Any]:
        """Migrate document IDs to unified format.

        Args:
            batch_size: Number of documents to process per batch

        Returns:
            Migration statistics
        """
        # Get all documents with old ID format
        query = """
        MATCH (d:Document)
        WHERE NOT d.id STARTS WITH 'doc_'
        RETURN d.id as old_id, d.source_file as source_file,
               d.checksum as checksum
        LIMIT $batch_size
        """

        migrated_count = 0
        error_count = 0

        while True:
            result = await self.neo4j.execute_query(query, batch_size=batch_size)
            documents = [record.data() for record in result]

            if not documents:
                break

            for doc in documents:
                try:
                    # Generate new unified ID
                    new_id = UnifiedIDGenerator.generate_document_id(
                        source_file=doc['source_file'],
                        checksum=doc.get('checksum')
                    )

                    # Update document ID in Neo4j
                    await self._update_document_id_neo4j(
                        old_id=doc['old_id'],
                        new_id=new_id
                    )

                    # Update references in Qdrant if available
                    if self.qdrant:
                        await self._update_document_id_qdrant(
                            old_id=doc['old_id'],
                            new_id=new_id
                        )

                    migrated_count += 1
                    self.migration_log.append({
                        'type': 'document',
                        'old_id': doc['old_id'],
                        'new_id': new_id,
                        'status': 'success'
                    })

                except Exception as e:
                    error_count += 1
                    self.migration_log.append({
                        'type': 'document',
                        'old_id': doc['old_id'],
                        'error': str(e),
                        'status': 'error'
                    })

        return {
            'migrated': migrated_count,
            'errors': error_count,
            'total_processed': migrated_count + error_count
        }

    async def migrate_chunk_ids(self, batch_size: int = 100) -> Dict[str, Any]:
        """Migrate document chunk IDs to unified format.

        Args:
            batch_size: Number of chunks to process per batch

        Returns:
            Migration statistics
        """
        query = """
        MATCH (c:DocumentChunk)
        WHERE NOT c.id CONTAINS ':chunk:'
        RETURN c.id as old_id, c.document_id as document_id,
               c.chunk_index as chunk_index
        LIMIT $batch_size
        """

        migrated_count = 0
        error_count = 0

        while True:
            result = await self.neo4j.execute_query(query, batch_size=batch_size)
            chunks = [record.data() for record in result]

            if not chunks:
                break

            for chunk in chunks:
                try:
                    # Generate new unified chunk ID
                    new_id = UnifiedIDGenerator.generate_chunk_id(
                        document_id=chunk['document_id'],
                        chunk_index=chunk['chunk_index']
                    )

                    # Update chunk ID in Neo4j
                    await self._update_chunk_id_neo4j(
                        old_id=chunk['old_id'],
                        new_id=new_id
                    )

                    # Update references in Qdrant if available
                    if self.qdrant:
                        await self._update_chunk_id_qdrant(
                            old_id=chunk['old_id'],
                            new_id=new_id
                        )

                    migrated_count += 1
                    self.migration_log.append({
                        'type': 'chunk',
                        'old_id': chunk['old_id'],
                        'new_id': new_id,
                        'status': 'success'
                    })

                except Exception as e:
                    error_count += 1
                    self.migration_log.append({
                        'type': 'chunk',
                        'old_id': chunk['old_id'],
                        'error': str(e),
                        'status': 'error'
                    })

        return {
            'migrated': migrated_count,
            'errors': error_count,
            'total_processed': migrated_count + error_count
        }

    async def migrate_entity_ids(self, batch_size: int = 100) -> Dict[str, Any]:
        """Migrate entity IDs to unified format.

        Args:
            batch_size: Number of entities to process per batch

        Returns:
            Migration statistics
        """
        query = """
        MATCH (e:Entity)
        WHERE NOT e.id STARTS WITH 'ent_'
        RETURN e.id as old_id, e.name as name, e.type as type,
               e.source_doc_id as source_doc_id
        LIMIT $batch_size
        """

        migrated_count = 0
        error_count = 0

        while True:
            result = await self.neo4j.execute_query(query, batch_size=batch_size)
            entities = [record.data() for record in result]

            if not entities:
                break

            for entity in entities:
                try:
                    # Generate new unified entity ID
                    new_id = UnifiedIDGenerator.generate_entity_id(
                        name=entity['name'],
                        entity_type=entity['type'],
                        source_doc_id=entity.get('source_doc_id', '')
                    )

                    # Update entity ID in Neo4j
                    await self._update_entity_id_neo4j(
                        old_id=entity['old_id'],
                        new_id=new_id
                    )

                    migrated_count += 1
                    self.migration_log.append({
                        'type': 'entity',
                        'old_id': entity['old_id'],
                        'new_id': new_id,
                        'status': 'success'
                    })

                except Exception as e:
                    error_count += 1
                    self.migration_log.append({
                        'type': 'entity',
                        'old_id': entity['old_id'],
                        'error': str(e),
                        'status': 'error'
                    })

        return {
            'migrated': migrated_count,
            'errors': error_count,
            'total_processed': migrated_count + error_count
        }

    async def migrate_relation_ids(self, batch_size: int = 100) -> Dict[str, Any]:
        """Migrate relation IDs to unified format.

        Args:
            batch_size: Number of relations to process per batch

        Returns:
            Migration statistics
        """
        query = """
        MATCH (s:Entity)-[r]->(t:Entity)
        WHERE NOT r.id STARTS WITH 'rel_'
        RETURN r.id as old_id, s.id as source_entity_id,
               t.id as target_entity_id, type(r) as relation_type
        LIMIT $batch_size
        """

        migrated_count = 0
        error_count = 0

        while True:
            result = await self.neo4j.execute_query(query, batch_size=batch_size)
            relations = [record.data() for record in result]

            if not relations:
                break

            for relation in relations:
                try:
                    # Generate new unified relation ID
                    new_id = UnifiedIDGenerator.generate_relation_id(
                        source_entity_id=relation['source_entity_id'],
                        target_entity_id=relation['target_entity_id'],
                        relation_type=relation['relation_type']
                    )

                    # Update relation ID in Neo4j
                    await self._update_relation_id_neo4j(
                        old_id=relation['old_id'],
                        new_id=new_id,
                        source_entity_id=relation['source_entity_id'],
                        target_entity_id=relation['target_entity_id'],
                        relation_type=relation['relation_type']
                    )

                    migrated_count += 1
                    self.migration_log.append({
                        'type': 'relation',
                        'old_id': relation['old_id'],
                        'new_id': new_id,
                        'status': 'success'
                    })

                except Exception as e:
                    error_count += 1
                    self.migration_log.append({
                        'type': 'relation',
                        'old_id': relation['old_id'],
                        'error': str(e),
                        'status': 'error'
                    })

        return {
            'migrated': migrated_count,
            'errors': error_count,
            'total_processed': migrated_count + error_count
        }

    async def _update_document_id_neo4j(self, old_id: str, new_id: str):
        """Update document ID in Neo4j."""
        query = """
        MATCH (d:Document {id: $old_id})
        SET d.id = $new_id
        WITH d
        MATCH (d)-[r]-(related)
        RETURN count(r) as relationships_updated
        """
        await self.neo4j.execute_query(query, old_id=old_id, new_id=new_id)

    async def _update_chunk_id_neo4j(self, old_id: str, new_id: str):
        """Update chunk ID in Neo4j."""
        query = """
        MATCH (c:DocumentChunk {id: $old_id})
        SET c.id = $new_id
        """
        await self.neo4j.execute_query(query, old_id=old_id, new_id=new_id)

    async def _update_entity_id_neo4j(self, old_id: str, new_id: str):
        """Update entity ID in Neo4j."""
        query = """
        MATCH (e:Entity {id: $old_id})
        SET e.id = $new_id
        """
        await self.neo4j.execute_query(query, old_id=old_id, new_id=new_id)

    async def _update_relation_id_neo4j(self, old_id: str, new_id: str,
                                       source_entity_id: str, target_entity_id: str,
                                       relation_type: str):
        """Update relation ID in Neo4j."""
        query = f"""
        MATCH (s:Entity {{id: $source_entity_id}})-[r:{relation_type} {{id: $old_id}}]->(t:Entity {{id: $target_entity_id}})
        SET r.id = $new_id
        """
        await self.neo4j.execute_query(
            query,
            old_id=old_id,
            new_id=new_id,
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id
        )

    async def _update_document_id_qdrant(self, old_id: str, new_id: str):
        """Update document ID references in Qdrant."""
        if not self.qdrant:
            return

        try:
            # Search for vectors with old document_id
            search_result = await self.qdrant.client.scroll(
                collection_name=self.qdrant.collection_name,
                scroll_filter={
                    "must": [
                        {
                            "key": "document_id",
                            "match": {"value": old_id}
                        }
                    ]
                },
                limit=1000
            )

            # Update metadata for each vector
            for point in search_result[0]:
                point.payload['document_id'] = new_id
                await self.qdrant.client.upsert(
                    collection_name=self.qdrant.collection_name,
                    points=[point]
                )
        except Exception as e:
            # Log error but don't fail the migration
            self.migration_log.append({
                'type': 'qdrant_document_update',
                'old_id': old_id,
                'new_id': new_id,
                'error': str(e),
                'status': 'error'
            })

    async def _update_chunk_id_qdrant(self, old_id: str, new_id: str):
        """Update chunk ID references in Qdrant."""
        if not self.qdrant:
            return

        try:
            # Search for vectors with old chunk_id
            search_result = await self.qdrant.client.scroll(
                collection_name=self.qdrant.collection_name,
                scroll_filter={
                    "must": [
                        {
                            "key": "chunk_id",
                            "match": {"value": old_id}
                        }
                    ]
                },
                limit=1000
            )

            # Update metadata for each vector
            for point in search_result[0]:
                point.payload['chunk_id'] = new_id
                await self.qdrant.client.upsert(
                    collection_name=self.qdrant.collection_name,
                    points=[point]
                )
        except Exception as e:
            # Log error but don't fail the migration
            self.migration_log.append({
                'type': 'qdrant_chunk_update',
                'old_id': old_id,
                'new_id': new_id,
                'error': str(e),
                'status': 'error'
            })

    def get_migration_report(self) -> Dict[str, Any]:
        """Generate migration report."""
        successful = [log for log in self.migration_log if log['status'] == 'success']
        errors = [log for log in self.migration_log if log['status'] == 'error']

        return {
            'total_migrations': len(self.migration_log),
            'successful': len(successful),
            'errors': len(errors),
            'success_rate': len(successful) / max(len(self.migration_log), 1),
            'error_details': errors
        }

    async def migrate_all(self, batch_size: int = 100) -> Dict[str, Any]:
        """Migrate all ID types to unified format.

        Args:
            batch_size: Number of items to process per batch

        Returns:
            Complete migration statistics
        """
        results = {}

        # Migrate in order: documents -> chunks -> entities -> relations
        results['documents'] = await self.migrate_document_ids(batch_size)
        results['chunks'] = await self.migrate_chunk_ids(batch_size)
        results['entities'] = await self.migrate_entity_ids(batch_size)
        results['relations'] = await self.migrate_relation_ids(batch_size)

        # Generate overall report
        results['overall'] = self.get_migration_report()

        return results
