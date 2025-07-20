"""Enhanced transaction coordinator with actual database operation execution."""

import asyncio
from typing import Dict, List, Any, Optional
import structlog

from morag_core.config import DatabaseConfig, DatabaseType
from morag_core.utils.json_parser import parse_json_response
from morag_graph.storage.neo4j_storage import Neo4jStorage
from morag_graph.storage.qdrant_storage import QdrantVectorStorage
from morag_graph.models import Entity, Relation

from .transaction_coordinator import (
    TransactionCoordinator,
    TransactionOperation,
    IngestionTransaction
)

logger = structlog.get_logger(__name__)


class EnhancedTransactionCoordinator(TransactionCoordinator):
    """Enhanced transaction coordinator that can execute actual database operations."""
    
    def __init__(self):
        super().__init__()
        self.logger = logger.bind(component="enhanced_transaction_coordinator")
        self._storage_cache: Dict[str, Any] = {}
    
    async def _execute_operation(self, operation: TransactionOperation) -> bool:
        """Execute a single operation against the actual database."""
        try:
            self.logger.debug(
                "Executing operation",
                operation_id=operation.operation_id,
                operation_type=operation.operation_type,
                database_type=operation.database_type.value
            )
            
            if operation.database_type == DatabaseType.NEO4J:
                return await self._execute_neo4j_operation(operation)
            elif operation.database_type == DatabaseType.QDRANT:
                return await self._execute_qdrant_operation(operation)
            else:
                self.logger.error(
                    "Unsupported database type for operation execution",
                    database_type=operation.database_type.value,
                    operation_id=operation.operation_id
                )
                return False
                
        except Exception as e:
            self.logger.error(
                "Operation execution failed",
                operation_id=operation.operation_id,
                error=str(e)
            )
            operation.error = str(e)
            return False
    
    async def _execute_neo4j_operation(self, operation: TransactionOperation) -> bool:
        """Execute Neo4j operation."""
        db_config_dict = operation.data.get("db_config")
        if not db_config_dict:
            raise ValueError("Missing db_config in operation data")
        
        # Get or create storage instance
        storage = await self._get_neo4j_storage(db_config_dict)
        
        try:
            if operation.operation_type == "create_entities":
                return await self._create_neo4j_entities(storage, operation)
            elif operation.operation_type == "create_relations":
                return await self._create_neo4j_relations(storage, operation)
            else:
                self.logger.error(
                    "Unsupported Neo4j operation type",
                    operation_type=operation.operation_type
                )
                return False
                
        except Exception as e:
            self.logger.error(
                "Neo4j operation failed",
                operation_type=operation.operation_type,
                error=str(e)
            )
            raise
    
    async def _execute_qdrant_operation(self, operation: TransactionOperation) -> bool:
        """Execute Qdrant operation."""
        db_config_dict = operation.data.get("db_config")
        if not db_config_dict:
            raise ValueError("Missing db_config in operation data")
        
        # Get or create storage instance
        storage = await self._get_qdrant_storage(db_config_dict)
        
        try:
            if operation.operation_type == "store_vectors":
                return await self._store_qdrant_vectors(storage, operation)
            else:
                self.logger.error(
                    "Unsupported Qdrant operation type",
                    operation_type=operation.operation_type
                )
                return False
                
        except Exception as e:
            self.logger.error(
                "Qdrant operation failed",
                operation_type=operation.operation_type,
                error=str(e)
            )
            raise
    
    async def _create_neo4j_entities(self, storage: Neo4jStorage, operation: TransactionOperation) -> bool:
        """Create entities in Neo4j."""
        entities_data = operation.data.get("entities", [])
        document_id = operation.data.get("document_id")
        
        if not entities_data:
            return True  # No entities to create
        
        # Convert dict data back to Entity objects
        entities = []
        for entity_dict in entities_data:
            entity = Entity(
                name=entity_dict["name"],
                type=entity_dict["type"],
                confidence=entity_dict.get("confidence", 0.5),
                attributes=entity_dict.get("attributes", {}),
                source_doc_id=document_id
            )
            entities.append(entity)
        
        # Create entities in Neo4j
        entity_ids = await storage.bulk_create_entities(entities)
        
        # Store entity IDs for rollback
        operation.rollback_data = operation.rollback_data or {}
        operation.rollback_data["created_entity_ids"] = entity_ids
        
        self.logger.info(
            "Created entities in Neo4j",
            operation_id=operation.operation_id,
            entities_count=len(entity_ids)
        )
        
        return True
    
    async def _create_neo4j_relations(self, storage: Neo4jStorage, operation: TransactionOperation) -> bool:
        """Create relations in Neo4j."""
        relations_data = operation.data.get("relations", [])
        document_id = operation.data.get("document_id")
        
        if not relations_data:
            return True  # No relations to create
        
        # Convert dict data back to Relation objects
        relations = []
        for relation_dict in relations_data:
            relation = Relation(
                source_entity=relation_dict["source_entity"],
                target_entity=relation_dict["target_entity"],
                relation_type=relation_dict["relation_type"],
                confidence=relation_dict.get("confidence", 0.5),
                attributes=relation_dict.get("attributes", {}),
                source_doc_id=document_id
            )
            relations.append(relation)
        
        # Create relations in Neo4j
        relation_ids = await storage.bulk_create_relations(relations)
        
        # Store relation IDs for rollback
        operation.rollback_data = operation.rollback_data or {}
        operation.rollback_data["created_relation_ids"] = relation_ids
        
        self.logger.info(
            "Created relations in Neo4j",
            operation_id=operation.operation_id,
            relations_count=len(relation_ids)
        )
        
        return True
    
    async def _store_qdrant_vectors(self, storage: QdrantVectorStorage, operation: TransactionOperation) -> bool:
        """Store vectors in Qdrant."""
        chunks_data = operation.data.get("chunks", [])
        document_id = operation.data.get("document_id")
        replace_existing = operation.data.get("replace_existing", False)
        
        if not chunks_data:
            return True  # No chunks to store
        
        # Extract text chunks for embedding
        texts = [chunk["text"] for chunk in chunks_data]
        
        # Generate embeddings (this would use the actual embedding service)
        # For now, we'll simulate this
        embeddings = await self._generate_embeddings(texts)
        
        # Prepare metadata for each chunk
        metadata_list = []
        for i, chunk in enumerate(chunks_data):
            metadata = {
                "document_id": document_id,
                "chunk_index": chunk["chunk_index"],
                "entities_count": chunk["entities_count"],
                "relations_count": chunk["relations_count"],
                "text": chunk["text"]
            }
            metadata_list.append(metadata)
        
        # Store vectors in Qdrant
        if replace_existing:
            point_ids = await storage.replace_document(
                document_id, embeddings, metadata_list
            )
        else:
            point_ids = await storage.store_vectors(embeddings, metadata_list)
        
        # Store point IDs for rollback
        operation.rollback_data = operation.rollback_data or {}
        operation.rollback_data["created_point_ids"] = point_ids
        
        self.logger.info(
            "Stored vectors in Qdrant",
            operation_id=operation.operation_id,
            points_count=len(point_ids)
        )
        
        return True
    
    async def _rollback_operation(self, operation: TransactionOperation) -> bool:
        """Rollback a completed operation."""
        try:
            self.logger.debug(
                "Rolling back operation",
                operation_id=operation.operation_id,
                operation_type=operation.operation_type,
                database_type=operation.database_type.value
            )
            
            if operation.database_type == DatabaseType.NEO4J:
                return await self._rollback_neo4j_operation(operation)
            elif operation.database_type == DatabaseType.QDRANT:
                return await self._rollback_qdrant_operation(operation)
            else:
                self.logger.error(
                    "Unsupported database type for rollback",
                    database_type=operation.database_type.value
                )
                return False
                
        except Exception as e:
            self.logger.error(
                "Operation rollback failed",
                operation_id=operation.operation_id,
                error=str(e)
            )
            return False
    
    async def _rollback_neo4j_operation(self, operation: TransactionOperation) -> bool:
        """Rollback Neo4j operation."""
        if not operation.rollback_data:
            return True  # Nothing to rollback
        
        db_config_dict = operation.data.get("db_config")
        storage = await self._get_neo4j_storage(db_config_dict)
        
        # Delete created entities
        if "created_entity_ids" in operation.rollback_data:
            entity_ids = operation.rollback_data["created_entity_ids"]
            for entity_id in entity_ids:
                await storage.delete_entity(entity_id)
        
        # Delete created relations
        if "created_relation_ids" in operation.rollback_data:
            relation_ids = operation.rollback_data["created_relation_ids"]
            for relation_id in relation_ids:
                await storage.delete_relation(relation_id)
        
        return True
    
    async def _rollback_qdrant_operation(self, operation: TransactionOperation) -> bool:
        """Rollback Qdrant operation."""
        if not operation.rollback_data:
            return True  # Nothing to rollback
        
        db_config_dict = operation.data.get("db_config")
        storage = await self._get_qdrant_storage(db_config_dict)
        
        # Delete created points
        if "created_point_ids" in operation.rollback_data:
            point_ids = operation.rollback_data["created_point_ids"]
            collection_name = operation.rollback_data.get("collection_name", "morag_documents")
            await storage.delete_points(point_ids, collection_name)
        
        return True
    
    async def _get_neo4j_storage(self, db_config_dict: Dict[str, Any]) -> Neo4jStorage:
        """Get or create Neo4j storage instance."""
        config_key = f"neo4j_{hash(str(sorted(db_config_dict.items())))}"
        
        if config_key not in self._storage_cache:
            # Reconstruct DatabaseConfig from dict
            db_config = DatabaseConfig(**db_config_dict)
            storage = Neo4jStorage(db_config)
            await storage.connect()
            self._storage_cache[config_key] = storage
        
        return self._storage_cache[config_key]
    
    async def _get_qdrant_storage(self, db_config_dict: Dict[str, Any]) -> QdrantVectorStorage:
        """Get or create Qdrant storage instance."""
        config_key = f"qdrant_{hash(str(sorted(db_config_dict.items())))}"
        
        if config_key not in self._storage_cache:
            # Reconstruct DatabaseConfig from dict
            db_config = DatabaseConfig(**db_config_dict)
            storage = QdrantVectorStorage(db_config)
            await storage.connect()
            self._storage_cache[config_key] = storage
        
        return self._storage_cache[config_key]
    
    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        # This is a placeholder - in reality, this would use the embedding service
        # For now, return dummy embeddings
        return [[0.1] * 384 for _ in texts]  # 384-dimensional dummy embeddings
    
    async def cleanup(self):
        """Clean up storage connections."""
        for storage in self._storage_cache.values():
            try:
                if hasattr(storage, 'disconnect'):
                    await storage.disconnect()
                elif hasattr(storage, 'close'):
                    await storage.close()
            except Exception as e:
                self.logger.warning("Error closing storage connection", error=str(e))
        
        self._storage_cache.clear()


# Global enhanced coordinator instance
_enhanced_coordinator = EnhancedTransactionCoordinator()

def get_enhanced_transaction_coordinator() -> EnhancedTransactionCoordinator:
    """Get the global enhanced transaction coordinator instance."""
    return _enhanced_coordinator
