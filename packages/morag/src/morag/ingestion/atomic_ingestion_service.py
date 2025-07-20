"""Atomic ingestion service with transaction management and pre-validation."""

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import structlog

from morag_core.config import DatabaseConfig, DatabaseType
from morag_core.utils.json_parser import parse_json_response, JSONParsingError
from morag_graph.models import Entity, Relation
from morag_graph.storage.neo4j_storage import Neo4jStorage
from morag_graph.storage.qdrant_storage import QdrantVectorStorage

from .transaction_coordinator import (
    TransactionCoordinator,
    TransactionOperation,
    TransactionState,
    get_transaction_coordinator
)
from .enhanced_transaction_coordinator import (
    EnhancedTransactionCoordinator,
    get_enhanced_transaction_coordinator
)
from .state_manager import (
    IngestionStateManager,
    IngestionState,
    get_state_manager
)

logger = structlog.get_logger(__name__)


class ValidationError(Exception):
    """Exception raised when validation fails."""
    pass


class AtomicIngestionService:
    """Service for atomic ingestion with pre-validation and transaction management."""
    
    def __init__(
        self,
        coordinator: Optional[EnhancedTransactionCoordinator] = None,
        state_manager: Optional[IngestionStateManager] = None
    ):
        self.coordinator = coordinator or get_enhanced_transaction_coordinator()
        self.state_manager = state_manager or get_state_manager()
        self.logger = logger.bind(component="atomic_ingestion")
        self._storage_instances: Dict[str, Any] = {}
    
    async def ingest_with_validation(
        self,
        content: str,
        source_path: str,
        document_id: str,
        database_configs: List[DatabaseConfig],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        replace_existing: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Perform atomic ingestion with pre-validation.
        
        Args:
            content: Content to ingest
            source_path: Source file path
            document_id: Document identifier
            database_configs: Database configurations
            chunk_size: Chunk size for text splitting
            chunk_overlap: Overlap between chunks
            replace_existing: Whether to replace existing document
            metadata: Additional metadata
            
        Returns:
            Ingestion result dictionary
            
        Raises:
            ValidationError: If validation fails
            Exception: If ingestion fails
        """
        start_time = datetime.now(timezone.utc)
        
        # Step 1: Begin transaction
        transaction_id = await self.coordinator.begin_transaction(
            document_id=document_id,
            source_path=source_path,
            metadata=metadata or {}
        )

        # Step 1.5: Create ingestion state
        ingestion_state = await self.state_manager.create_state(
            transaction_id=transaction_id,
            document_id=document_id,
            source_path=source_path,
            metadata=metadata
        )
        
        try:
            self.logger.info(
                "Starting atomic ingestion with validation",
                transaction_id=transaction_id,
                document_id=document_id,
                source_path=source_path,
                content_length=len(content)
            )
            
            # Step 2: Pre-validation phase - process all chunks and validate responses
            await self.state_manager.update_state(transaction_id, state="validating")

            validation_result = await self._validate_all_chunks(
                content, chunk_size, chunk_overlap, transaction_id
            )

            if not validation_result["success"]:
                await self.state_manager.update_state(
                    transaction_id,
                    state="failed",
                    error_message=f"Validation failed: {validation_result['error']}"
                )
                await self.coordinator.abort_transaction(transaction_id)
                raise ValidationError(f"Validation failed: {validation_result['error']}")

            # Update state with validation results
            await self.state_manager.update_state(
                transaction_id,
                state="validated",
                chunks_processed=validation_result["data"]["chunks_count"],
                entities_extracted=validation_result["data"]["entities_count"],
                relations_extracted=validation_result["data"]["relations_count"]
            )
            
            # Step 3: Prepare database operations
            await self._prepare_database_operations(
                transaction_id,
                validation_result["data"],
                database_configs,
                document_id,
                replace_existing
            )
            
            # Step 4: Prepare transaction (validate all operations)
            if not await self.coordinator.prepare_transaction(transaction_id):
                transaction = await self.coordinator.get_transaction_status(transaction_id)
                error_msg = transaction.error_message if transaction else "Unknown preparation error"
                raise ValidationError(f"Transaction preparation failed: {error_msg}")
            
            # Step 5: Commit transaction (execute all operations atomically)
            await self.state_manager.update_state(transaction_id, state="committing")

            if not await self.coordinator.commit_transaction(transaction_id):
                transaction = await self.coordinator.get_transaction_status(transaction_id)
                error_msg = transaction.error_message if transaction else "Unknown commit error"
                await self.state_manager.update_state(
                    transaction_id,
                    state="failed",
                    error_message=f"Transaction commit failed: {error_msg}"
                )
                raise Exception(f"Transaction commit failed: {error_msg}")

            # Mark as committed
            await self.state_manager.update_state(transaction_id, state="committed")
            
            # Step 6: Build result
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            result = {
                "success": True,
                "transaction_id": transaction_id,
                "document_id": document_id,
                "source_path": source_path,
                "processing_time": processing_time,
                "chunks_processed": validation_result["data"]["chunks_count"],
                "entities_extracted": validation_result["data"]["entities_count"],
                "relations_extracted": validation_result["data"]["relations_count"],
                "database_results": await self._get_database_results(transaction_id),
                "validation_summary": validation_result["summary"]
            }
            
            self.logger.info(
                "Atomic ingestion completed successfully",
                transaction_id=transaction_id,
                processing_time=processing_time,
                chunks_processed=result["chunks_processed"],
                entities_extracted=result["entities_extracted"],
                relations_extracted=result["relations_extracted"]
            )
            
            return result
            
        except Exception as e:
            # Update state and abort transaction on any error
            await self.state_manager.update_state(
                transaction_id,
                state="failed",
                error_message=str(e)
            )
            await self.coordinator.abort_transaction(transaction_id)

            self.logger.error(
                "Atomic ingestion failed",
                transaction_id=transaction_id,
                error=str(e),
                error_type=type(e).__name__
            )

            raise
    
    async def _validate_all_chunks(
        self,
        content: str,
        chunk_size: int,
        chunk_overlap: int,
        transaction_id: str
    ) -> Dict[str, Any]:
        """Validate all chunks before any database operations.
        
        Args:
            content: Content to process
            chunk_size: Chunk size
            chunk_overlap: Chunk overlap
            transaction_id: Transaction ID
            
        Returns:
            Validation result with processed data
        """
        self.logger.info("Starting pre-validation phase", transaction_id=transaction_id)
        
        try:
            # Step 1: Create chunks
            chunks = self._create_chunks(content, chunk_size, chunk_overlap)
            
            # Step 2: Process each chunk and validate LLM responses
            all_entities = []
            all_relations = []
            chunk_data = []
            validation_errors = []
            
            for i, chunk in enumerate(chunks):
                try:
                    chunk_result = await self._process_and_validate_chunk(chunk, i)
                    
                    all_entities.extend(chunk_result["entities"])
                    all_relations.extend(chunk_result["relations"])
                    chunk_data.append(chunk_result)
                    
                except Exception as e:
                    error_msg = f"Chunk {i} validation failed: {str(e)}"
                    validation_errors.append(error_msg)
                    self.logger.warning(
                        "Chunk validation failed",
                        chunk_index=i,
                        error=str(e),
                        transaction_id=transaction_id
                    )
            
            # Step 3: Check if we have any validation errors
            if validation_errors:
                return {
                    "success": False,
                    "error": f"Validation failed for {len(validation_errors)} chunks",
                    "validation_errors": validation_errors
                }
            
            # Step 4: Deduplicate entities and relations
            deduplicated_entities = self._deduplicate_entities(all_entities)
            deduplicated_relations = self._deduplicate_relations(all_relations)
            
            validation_data = {
                "chunks": chunk_data,
                "chunks_count": len(chunks),
                "entities": deduplicated_entities,
                "entities_count": len(deduplicated_entities),
                "relations": deduplicated_relations,
                "relations_count": len(deduplicated_relations),
                "raw_entities_count": len(all_entities),
                "raw_relations_count": len(all_relations)
            }
            
            self.logger.info(
                "Pre-validation completed successfully",
                transaction_id=transaction_id,
                chunks_count=len(chunks),
                entities_count=len(deduplicated_entities),
                relations_count=len(deduplicated_relations),
                deduplication_savings_entities=len(all_entities) - len(deduplicated_entities),
                deduplication_savings_relations=len(all_relations) - len(deduplicated_relations)
            )
            
            return {
                "success": True,
                "data": validation_data,
                "summary": {
                    "chunks_processed": len(chunks),
                    "entities_extracted": len(deduplicated_entities),
                    "relations_extracted": len(deduplicated_relations),
                    "validation_errors": 0
                }
            }
            
        except Exception as e:
            self.logger.error(
                "Pre-validation failed",
                transaction_id=transaction_id,
                error=str(e)
            )
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _process_and_validate_chunk(self, chunk: str, chunk_index: int) -> Dict[str, Any]:
        """Process a single chunk and validate LLM responses.
        
        Args:
            chunk: Text chunk to process
            chunk_index: Index of the chunk
            
        Returns:
            Processed chunk data
            
        Raises:
            ValidationError: If LLM response validation fails
            JSONParsingError: If JSON parsing fails
        """
        # Import here to avoid circular imports
        from morag_graph.ai.entity_agent import EntityExtractionAgent
        from morag_graph.ai.relation_agent import RelationExtractionAgent
        
        try:
            # Step 1: Extract entities with enhanced JSON parsing
            entity_agent = EntityExtractionAgent()
            entities_result = await entity_agent.extract_entities(chunk)
            
            # Validate entities result
            if not hasattr(entities_result, 'entities') or not isinstance(entities_result.entities, list):
                raise ValidationError(f"Invalid entities result structure for chunk {chunk_index}")
            
            # Step 2: Extract relations with enhanced JSON parsing
            relation_agent = RelationExtractionAgent()
            relations_result = await relation_agent.extract_relations(chunk, entities_result.entities)
            
            # Validate relations result
            if not hasattr(relations_result, 'relations') or not isinstance(relations_result.relations, list):
                raise ValidationError(f"Invalid relations result structure for chunk {chunk_index}")
            
            # Step 3: Convert to graph entities and relations
            graph_entities = []
            for entity in entities_result.entities:
                graph_entity = Entity(
                    name=entity.name,
                    type=entity.type,
                    confidence=entity.confidence,
                    attributes=entity.metadata or {}
                )
                graph_entities.append(graph_entity)
            
            graph_relations = []
            for relation in relations_result.relations:
                graph_relation = Relation(
                    source_entity=relation.source_entity,
                    target_entity=relation.target_entity,
                    relation_type=relation.relation_type,
                    confidence=relation.confidence,
                    attributes=relation.metadata or {}
                )
                graph_relations.append(graph_relation)
            
            return {
                "chunk_index": chunk_index,
                "chunk_text": chunk,
                "entities": graph_entities,
                "relations": graph_relations,
                "entities_count": len(graph_entities),
                "relations_count": len(graph_relations)
            }
            
        except JSONParsingError as e:
            raise ValidationError(f"JSON parsing failed for chunk {chunk_index}: {str(e)}")
        except Exception as e:
            raise ValidationError(f"Chunk processing failed for chunk {chunk_index}: {str(e)}")
    
    def _create_chunks(self, content: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Create text chunks from content."""
        if len(content) <= chunk_size:
            return [content]
        
        chunks = []
        start = 0
        
        while start < len(content):
            end = min(start + chunk_size, len(content))
            
            # Try to break at word boundary
            if end < len(content):
                # Look for the last space within the chunk
                last_space = content.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunk = content[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = max(start + 1, end - chunk_overlap)
            
            # Prevent infinite loop
            if start >= len(content):
                break
        
        return chunks
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Deduplicate entities by name (case-insensitive)."""
        seen = {}
        deduplicated = []
        
        for entity in entities:
            key = entity.name.lower().strip()
            if key not in seen:
                seen[key] = entity
                deduplicated.append(entity)
            else:
                # Merge with existing entity (keep higher confidence)
                existing = seen[key]
                if entity.confidence > existing.confidence:
                    seen[key] = entity
                    # Replace in deduplicated list
                    for i, e in enumerate(deduplicated):
                        if e.name.lower().strip() == key:
                            deduplicated[i] = entity
                            break
        
        return deduplicated
    
    def _deduplicate_relations(self, relations: List[Relation]) -> List[Relation]:
        """Deduplicate relations by source, target, and type."""
        seen = set()
        deduplicated = []
        
        for relation in relations:
            key = (
                relation.source_entity.lower().strip(),
                relation.target_entity.lower().strip(),
                relation.relation_type.lower().strip()
            )
            
            if key not in seen:
                seen.add(key)
                deduplicated.append(relation)
        
        return deduplicated

    async def _prepare_database_operations(
        self,
        transaction_id: str,
        validation_data: Dict[str, Any],
        database_configs: List[DatabaseConfig],
        document_id: str,
        replace_existing: bool
    ) -> None:
        """Prepare database operations for the transaction.

        Args:
            transaction_id: Transaction ID
            validation_data: Validated data from pre-validation
            database_configs: Database configurations
            document_id: Document identifier
            replace_existing: Whether to replace existing document
        """
        self.logger.info(
            "Preparing database operations",
            transaction_id=transaction_id,
            databases_count=len(database_configs)
        )

        for db_config in database_configs:
            try:
                if db_config.type == DatabaseType.NEO4J:
                    await self._prepare_neo4j_operations(
                        transaction_id, validation_data, db_config, document_id
                    )
                elif db_config.type == DatabaseType.QDRANT:
                    await self._prepare_qdrant_operations(
                        transaction_id, validation_data, db_config, document_id, replace_existing
                    )
                else:
                    self.logger.warning(
                        "Unsupported database type",
                        database_type=db_config.type.value,
                        transaction_id=transaction_id
                    )

            except Exception as e:
                self.logger.error(
                    "Failed to prepare operations for database",
                    database_type=db_config.type.value,
                    transaction_id=transaction_id,
                    error=str(e)
                )
                raise

    async def _prepare_neo4j_operations(
        self,
        transaction_id: str,
        validation_data: Dict[str, Any],
        db_config: DatabaseConfig,
        document_id: str
    ) -> None:
        """Prepare Neo4j operations."""
        # Add entity creation operation
        if validation_data["entities"]:
            await self.coordinator.add_operation(
                transaction_id=transaction_id,
                database_type=DatabaseType.NEO4J,
                operation_type="create_entities",
                data={
                    "entities": [entity.to_dict() for entity in validation_data["entities"]],
                    "db_config": db_config.to_dict(),
                    "document_id": document_id
                },
                rollback_data={
                    "entity_ids": [entity.id for entity in validation_data["entities"]],
                    "db_config": db_config.to_dict()
                }
            )

        # Add relation creation operation
        if validation_data["relations"]:
            await self.coordinator.add_operation(
                transaction_id=transaction_id,
                database_type=DatabaseType.NEO4J,
                operation_type="create_relations",
                data={
                    "relations": [relation.to_dict() for relation in validation_data["relations"]],
                    "db_config": db_config.to_dict(),
                    "document_id": document_id
                },
                rollback_data={
                    "relation_ids": [relation.id for relation in validation_data["relations"]],
                    "db_config": db_config.to_dict()
                }
            )

    async def _prepare_qdrant_operations(
        self,
        transaction_id: str,
        validation_data: Dict[str, Any],
        db_config: DatabaseConfig,
        document_id: str,
        replace_existing: bool
    ) -> None:
        """Prepare Qdrant operations."""
        # Prepare vector storage operation
        chunks_data = []
        for chunk_info in validation_data["chunks"]:
            chunks_data.append({
                "text": chunk_info["chunk_text"],
                "chunk_index": chunk_info["chunk_index"],
                "entities_count": chunk_info["entities_count"],
                "relations_count": chunk_info["relations_count"]
            })

        await self.coordinator.add_operation(
            transaction_id=transaction_id,
            database_type=DatabaseType.QDRANT,
            operation_type="store_vectors",
            data={
                "chunks": chunks_data,
                "db_config": db_config.to_dict(),
                "document_id": document_id,
                "replace_existing": replace_existing
            },
            rollback_data={
                "document_id": document_id,
                "db_config": db_config.to_dict(),
                "collection_name": db_config.database_name or "morag_documents"
            }
        )

    async def _get_database_results(self, transaction_id: str) -> Dict[str, Any]:
        """Get database operation results for the transaction."""
        transaction = await self.coordinator.get_transaction_status(transaction_id)
        if not transaction:
            return {}

        results = {}
        for operation in transaction.operations:
            db_type = operation.database_type.value.lower()
            if db_type not in results:
                results[db_type] = {
                    "success": operation.completed and not operation.error,
                    "operations": []
                }

            results[db_type]["operations"].append({
                "operation_id": operation.operation_id,
                "operation_type": operation.operation_type,
                "completed": operation.completed,
                "error": operation.error
            })

        return results
