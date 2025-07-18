"""Migration utilities for transferring Neo4j data to Graphiti."""

import logging
from typing import Dict, Any, List, Optional, Tuple, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime

from .entity_storage import GraphitiEntityStorage, EntityStorageResult, RelationStorageResult
from .config import GraphitiConfig
from morag_graph.models import Entity, Relation
from morag_graph.storage.neo4j_storage import Neo4jStorage

logger = logging.getLogger(__name__)


@dataclass
class MigrationStats:
    """Statistics for migration operation."""
    entities_processed: int = 0
    entities_migrated: int = 0
    entities_failed: int = 0
    entities_deduplicated: int = 0
    relations_processed: int = 0
    relations_migrated: int = 0
    relations_failed: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def duration_seconds(self) -> float:
        """Get migration duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    @property
    def entity_success_rate(self) -> float:
        """Get entity migration success rate."""
        if self.entities_processed == 0:
            return 0.0
        return self.entities_migrated / self.entities_processed
    
    @property
    def relation_success_rate(self) -> float:
        """Get relation migration success rate."""
        if self.relations_processed == 0:
            return 0.0
        return self.relations_migrated / self.relations_processed


@dataclass
class MigrationResult:
    """Result of migration operation."""
    success: bool
    stats: MigrationStats
    errors: List[str]
    warnings: List[str]


class Neo4jToGraphitiMigrator:
    """Utility for migrating data from Neo4j to Graphiti."""
    
    def __init__(
        self,
        neo4j_storage: Neo4jStorage,
        graphiti_config: Optional[GraphitiConfig] = None,
        batch_size: int = 100
    ):
        self.neo4j_storage = neo4j_storage
        self.graphiti_storage = GraphitiEntityStorage(graphiti_config)
        self.batch_size = batch_size
        self.stats = MigrationStats()
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    async def migrate_all(
        self,
        migrate_entities: bool = True,
        migrate_relations: bool = True,
        auto_deduplicate: bool = True
    ) -> MigrationResult:
        """Migrate all data from Neo4j to Graphiti.
        
        Args:
            migrate_entities: Whether to migrate entities
            migrate_relations: Whether to migrate relations
            auto_deduplicate: Whether to deduplicate entities
            
        Returns:
            MigrationResult with operation details
        """
        self.stats = MigrationStats()
        self.stats.start_time = datetime.utcnow()
        self.errors = []
        self.warnings = []
        
        try:
            logger.info("Starting Neo4j to Graphiti migration")
            
            # Migrate entities first
            if migrate_entities:
                await self._migrate_entities(auto_deduplicate)
            
            # Then migrate relations
            if migrate_relations:
                await self._migrate_relations()
            
            self.stats.end_time = datetime.utcnow()
            
            logger.info(
                f"Migration completed in {self.stats.duration_seconds:.2f} seconds. "
                f"Entities: {self.stats.entities_migrated}/{self.stats.entities_processed}, "
                f"Relations: {self.stats.relations_migrated}/{self.stats.relations_processed}"
            )
            
            return MigrationResult(
                success=True,
                stats=self.stats,
                errors=self.errors,
                warnings=self.warnings
            )
            
        except Exception as e:
            self.stats.end_time = datetime.utcnow()
            error_msg = f"Migration failed: {str(e)}"
            self.errors.append(error_msg)
            logger.error(error_msg)
            
            return MigrationResult(
                success=False,
                stats=self.stats,
                errors=self.errors,
                warnings=self.warnings
            )
    
    async def _migrate_entities(self, auto_deduplicate: bool = True):
        """Migrate entities from Neo4j to Graphiti."""
        logger.info("Starting entity migration")
        
        try:
            # Get all entities from Neo4j in batches
            async for entity_batch in self._get_entities_batch():
                self.stats.entities_processed += len(entity_batch)
                
                # Store entities in Graphiti
                results = await self.graphiti_storage.store_entities_batch(
                    entity_batch, auto_deduplicate
                )
                
                # Process results
                for result in results:
                    if result.success:
                        self.stats.entities_migrated += 1
                        if result.deduplication_info:
                            self.stats.entities_deduplicated += 1
                    else:
                        self.stats.entities_failed += 1
                        self.errors.append(f"Entity {result.entity_id}: {result.error}")
                
                # Log progress
                logger.info(
                    f"Entity migration progress: {self.stats.entities_processed} processed, "
                    f"{self.stats.entities_migrated} migrated, "
                    f"{self.stats.entities_failed} failed"
                )
        
        except Exception as e:
            error_msg = f"Entity migration failed: {str(e)}"
            self.errors.append(error_msg)
            logger.error(error_msg)
    
    async def _migrate_relations(self):
        """Migrate relations from Neo4j to Graphiti."""
        logger.info("Starting relation migration")
        
        try:
            # Get all relations from Neo4j in batches
            async for relation_batch in self._get_relations_batch():
                self.stats.relations_processed += len(relation_batch)
                
                # Store relations in Graphiti
                results = await self.graphiti_storage.store_relations_batch(
                    relation_batch, ensure_entities_exist=True
                )
                
                # Process results
                for result in results:
                    if result.success:
                        self.stats.relations_migrated += 1
                    else:
                        self.stats.relations_failed += 1
                        if result.missing_entities:
                            self.warnings.append(
                                f"Relation {result.relation_id} has missing entities: "
                                f"{result.missing_entities}"
                            )
                        else:
                            self.errors.append(f"Relation {result.relation_id}: {result.error}")
                
                # Log progress
                logger.info(
                    f"Relation migration progress: {self.stats.relations_processed} processed, "
                    f"{self.stats.relations_migrated} migrated, "
                    f"{self.stats.relations_failed} failed"
                )
        
        except Exception as e:
            error_msg = f"Relation migration failed: {str(e)}"
            self.errors.append(error_msg)
            logger.error(error_msg)
    
    async def _get_entities_batch(self) -> AsyncGenerator[List[Entity], None]:
        """Get entities from Neo4j in batches.
        
        Yields:
            Batches of entities
        """
        try:
            # This is a simplified implementation
            # In practice, you'd use Neo4j's pagination capabilities
            offset = 0
            
            while True:
                # Get batch of entities from Neo4j
                entities = await self._fetch_entities_from_neo4j(offset, self.batch_size)
                
                if not entities:
                    break
                
                yield entities
                offset += len(entities)
                
                # Break if we got fewer entities than batch size (end of data)
                if len(entities) < self.batch_size:
                    break
        
        except Exception as e:
            logger.error(f"Error fetching entities from Neo4j: {e}")
            raise
    
    async def _get_relations_batch(self) -> AsyncGenerator[List[Relation], None]:
        """Get relations from Neo4j in batches.
        
        Yields:
            Batches of relations
        """
        try:
            offset = 0
            
            while True:
                # Get batch of relations from Neo4j
                relations = await self._fetch_relations_from_neo4j(offset, self.batch_size)
                
                if not relations:
                    break
                
                yield relations
                offset += len(relations)
                
                # Break if we got fewer relations than batch size (end of data)
                if len(relations) < self.batch_size:
                    break
        
        except Exception as e:
            logger.error(f"Error fetching relations from Neo4j: {e}")
            raise
    
    async def _fetch_entities_from_neo4j(self, offset: int, limit: int) -> List[Entity]:
        """Fetch entities from Neo4j with pagination.
        
        Args:
            offset: Number of entities to skip
            limit: Maximum number of entities to return
            
        Returns:
            List of entities
        """
        # This is a placeholder implementation
        # In practice, you'd use the actual Neo4j storage methods
        try:
            # Example query - adjust based on your Neo4j schema
            query = """
            MATCH (e:Entity)
            RETURN e
            SKIP $offset
            LIMIT $limit
            """
            
            # Execute query and convert results to Entity objects
            # This would depend on your actual Neo4j storage implementation
            entities = []
            
            # Placeholder - replace with actual Neo4j query execution
            logger.debug(f"Fetching entities with offset {offset}, limit {limit}")
            
            return entities
            
        except Exception as e:
            logger.error(f"Error fetching entities from Neo4j: {e}")
            return []
    
    async def _fetch_relations_from_neo4j(self, offset: int, limit: int) -> List[Relation]:
        """Fetch relations from Neo4j with pagination.
        
        Args:
            offset: Number of relations to skip
            limit: Maximum number of relations to return
            
        Returns:
            List of relations
        """
        # This is a placeholder implementation
        # In practice, you'd use the actual Neo4j storage methods
        try:
            # Example query - adjust based on your Neo4j schema
            query = """
            MATCH (source:Entity)-[r:RELATION]->(target:Entity)
            RETURN source, r, target
            SKIP $offset
            LIMIT $limit
            """
            
            # Execute query and convert results to Relation objects
            # This would depend on your actual Neo4j storage implementation
            relations = []
            
            # Placeholder - replace with actual Neo4j query execution
            logger.debug(f"Fetching relations with offset {offset}, limit {limit}")
            
            return relations
            
        except Exception as e:
            logger.error(f"Error fetching relations from Neo4j: {e}")
            return []
    
    def get_migration_summary(self) -> Dict[str, Any]:
        """Get a summary of the migration operation.
        
        Returns:
            Dictionary with migration summary
        """
        return {
            "stats": {
                "entities": {
                    "processed": self.stats.entities_processed,
                    "migrated": self.stats.entities_migrated,
                    "failed": self.stats.entities_failed,
                    "deduplicated": self.stats.entities_deduplicated,
                    "success_rate": self.stats.entity_success_rate
                },
                "relations": {
                    "processed": self.stats.relations_processed,
                    "migrated": self.stats.relations_migrated,
                    "failed": self.stats.relations_failed,
                    "success_rate": self.stats.relation_success_rate
                },
                "duration_seconds": self.stats.duration_seconds
            },
            "errors": self.errors,
            "warnings": self.warnings,
            "graphiti_storage_stats": self.graphiti_storage.get_storage_stats()
        }


def create_migrator(
    neo4j_storage: Neo4jStorage,
    graphiti_config: Optional[GraphitiConfig] = None,
    batch_size: int = 100
) -> Neo4jToGraphitiMigrator:
    """Create a Neo4jToGraphitiMigrator instance.
    
    Args:
        neo4j_storage: Neo4j storage instance
        graphiti_config: Optional Graphiti configuration
        batch_size: Batch size for migration
        
    Returns:
        Neo4jToGraphitiMigrator instance
    """
    return Neo4jToGraphitiMigrator(neo4j_storage, graphiti_config, batch_size)
