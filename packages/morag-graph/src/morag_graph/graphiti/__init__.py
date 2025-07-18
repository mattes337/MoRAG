"""Graphiti integration for MoRAG.

This module provides integration with Graphiti, a temporal knowledge graph system
that uses episodes to represent knowledge with built-in deduplication, temporal queries,
and hybrid search capabilities.
"""

from .config import GraphitiConfig, create_graphiti_instance
from .connection import GraphitiConnectionService
from .episode_mapper import DocumentEpisodeMapper, create_episode_mapper
from .search_service import GraphitiSearchService, SearchResult, SearchMetrics, SearchResultAdapter, create_search_service
from .search_integration import SearchInterface, GraphitiSearchAdapter, HybridSearchService, create_search_adapter, create_hybrid_search_service
from .temporal_service import GraphitiTemporalService, TemporalSnapshot, TemporalChange, TemporalQueryType, create_temporal_service
from .custom_schema import (
    MoragEntityType, MoragRelationType, BaseEntitySchema, BaseRelationSchema,
    PersonEntity, OrganizationEntity, TechnologyEntity, ConceptEntity, DocumentEntity,
    SemanticRelation, TemporalRelation, DocumentRelation, SchemaRegistry, schema_registry
)
from .schema_storage import SchemaAwareEntityStorage, SchemaAwareSearchService, create_schema_aware_storage, create_schema_aware_search

# Import adapter components
try:
    from .adapters import (
        BaseAdapter, BatchAdapter, AdapterRegistry, ConversionResult, ConversionDirection,
        AdapterError, ConversionError, ValidationError, adapter_registry,
        DocumentAdapter, DocumentChunkAdapter, EntityAdapter, RelationAdapter,
        ADAPTERS_AVAILABLE
    )
except ImportError:
    # Graceful degradation if adapter dependencies are missing
    BaseAdapter = None
    BatchAdapter = None
    AdapterRegistry = None
    ConversionResult = None
    ConversionDirection = None
    AdapterError = None
    ConversionError = None
    ValidationError = None
    adapter_registry = None
    DocumentAdapter = None
    DocumentChunkAdapter = None
    EntityAdapter = None
    RelationAdapter = None
    ADAPTERS_AVAILABLE = False

# Import entity storage components
try:
    from .entity_storage import (
        GraphitiEntityStorage, EntityStorageResult, RelationStorageResult,
        create_entity_storage
    )
    from .migration_utils import (
        Neo4jToGraphitiMigrator, MigrationStats, MigrationResult,
        create_migrator
    )
    ENTITY_STORAGE_AVAILABLE = True
except ImportError:
    # Graceful degradation if dependencies are missing
    GraphitiEntityStorage = None
    EntityStorageResult = None
    RelationStorageResult = None
    create_entity_storage = None
    Neo4jToGraphitiMigrator = None
    MigrationStats = None
    MigrationResult = None
    create_migrator = None
    ENTITY_STORAGE_AVAILABLE = False

# Import integration components
try:
    from .ingestion_service import GraphitiIngestionService, create_ingestion_service
    from .integration_service import (
        GraphitiIntegrationService, StorageBackend, IngestionResult,
        create_integration_service
    )
    INTEGRATION_AVAILABLE = True
except ImportError:
    # Graceful degradation if dependencies are missing
    GraphitiIngestionService = None
    create_ingestion_service = None
    GraphitiIntegrationService = None
    StorageBackend = None
    IngestionResult = None
    create_integration_service = None
    INTEGRATION_AVAILABLE = False

# Import chunk-entity relationship components
try:
    from .chunk_entity_service import (
        ChunkEntityRelationshipService, ChunkEntityMapping, EntityMentionResult,
        create_chunk_entity_service
    )
    CHUNK_ENTITY_AVAILABLE = True
except ImportError:
    # Graceful degradation if dependencies are missing
    ChunkEntityRelationshipService = None
    ChunkEntityMapping = None
    EntityMentionResult = None
    create_chunk_entity_service = None
    CHUNK_ENTITY_AVAILABLE = False

__all__ = [
    # Core configuration and connection
    "GraphitiConfig",
    "create_graphiti_instance",
    "GraphitiConnectionService",

    # Document mapping
    "DocumentEpisodeMapper",
    "create_episode_mapper",

    # Search services
    "GraphitiSearchService",
    "SearchResult",
    "SearchMetrics",
    "SearchResultAdapter",
    "create_search_service",

    # Search integration
    "SearchInterface",
    "GraphitiSearchAdapter",
    "HybridSearchService",
    "create_search_adapter",
    "create_hybrid_search_service",

    # Temporal queries
    "GraphitiTemporalService",
    "TemporalSnapshot",
    "TemporalChange",
    "TemporalQueryType",
    "create_temporal_service",

    # Custom schema
    "MoragEntityType",
    "MoragRelationType",
    "BaseEntitySchema",
    "BaseRelationSchema",
    "PersonEntity",
    "OrganizationEntity",
    "TechnologyEntity",
    "ConceptEntity",
    "DocumentEntity",
    "SemanticRelation",
    "TemporalRelation",
    "DocumentRelation",
    "SchemaRegistry",
    "schema_registry",

    # Schema-aware storage and search
    "SchemaAwareEntityStorage",
    "SchemaAwareSearchService",
    "create_schema_aware_storage",
    "create_schema_aware_search",

    # Adapter layer
    "BaseAdapter",
    "BatchAdapter",
    "AdapterRegistry",
    "ConversionResult",
    "ConversionDirection",
    "AdapterError",
    "ConversionError",
    "ValidationError",
    "adapter_registry",
    "DocumentAdapter",
    "DocumentChunkAdapter",
    "EntityAdapter",
    "RelationAdapter",
    "ADAPTERS_AVAILABLE",

    # Entity storage
    "GraphitiEntityStorage",
    "EntityStorageResult",
    "RelationStorageResult",
    "create_entity_storage",
    "Neo4jToGraphitiMigrator",
    "MigrationStats",
    "MigrationResult",
    "create_migrator",
    "ENTITY_STORAGE_AVAILABLE",

    # Integration services
    "GraphitiIngestionService",
    "create_ingestion_service",
    "GraphitiIntegrationService",
    "StorageBackend",
    "IngestionResult",
    "create_integration_service",
    "INTEGRATION_AVAILABLE",

    # Chunk-entity relationships
    "ChunkEntityRelationshipService",
    "ChunkEntityMapping",
    "EntityMentionResult",
    "create_chunk_entity_service",
    "CHUNK_ENTITY_AVAILABLE"
]
