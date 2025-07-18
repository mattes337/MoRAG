"""MORAG Graph Package.

A graph-based knowledge representation and storage system.
"""

__version__ = "0.1.0"

# Core models
from .models import Entity, Relation, Graph

# Extraction components
from .extraction import EntityExtractor, RelationExtractor

# Storage backends
from .storage import Neo4jStorage, QdrantStorage, Neo4jConfig, QdrantConfig

# Operations
from .operations import GraphCRUD, GraphTraversal, GraphPath, GraphAnalytics

# Builders
from .builders import GraphBuilder, GraphBuildResult, GraphBuildError

# Database configuration models
from .models.database_config import DatabaseType, DatabaseConfig, DatabaseServerConfig, DatabaseServerArray, DatabaseResult

# Graphiti integration (optional)
try:
    from .graphiti import (
        GraphitiConfig, create_graphiti_instance, GraphitiConnectionService,
        DocumentEpisodeMapper, create_episode_mapper,
        GraphitiSearchService, SearchResult, SearchMetrics, SearchResultAdapter, create_search_service,
        SearchInterface, GraphitiSearchAdapter, HybridSearchService, create_search_adapter, create_hybrid_search_service,
        GraphitiTemporalService, TemporalSnapshot, TemporalChange, TemporalQueryType, create_temporal_service,
        MoragEntityType, MoragRelationType, BaseEntitySchema, BaseRelationSchema,
        PersonEntity, OrganizationEntity, TechnologyEntity, ConceptEntity, DocumentEntity,
        SemanticRelation, TemporalRelation, DocumentRelation, SchemaRegistry, schema_registry,
        SchemaAwareEntityStorage, SchemaAwareSearchService, create_schema_aware_storage, create_schema_aware_search
    )
    GRAPHITI_AVAILABLE = True
except ImportError:
    GRAPHITI_AVAILABLE = False
    GraphitiConfig = None
    create_graphiti_instance = None
    GraphitiConnectionService = None
    DocumentEpisodeMapper = None
    create_episode_mapper = None
    GraphitiSearchService = None
    SearchResult = None
    SearchMetrics = None
    SearchResultAdapter = None
    create_search_service = None
    SearchInterface = None
    GraphitiSearchAdapter = None
    HybridSearchService = None
    create_search_adapter = None
    create_hybrid_search_service = None
    GraphitiTemporalService = None
    TemporalSnapshot = None
    TemporalChange = None
    TemporalQueryType = None
    create_temporal_service = None
    MoragEntityType = None
    MoragRelationType = None
    BaseEntitySchema = None
    BaseRelationSchema = None
    PersonEntity = None
    OrganizationEntity = None
    TechnologyEntity = None
    ConceptEntity = None
    DocumentEntity = None
    SemanticRelation = None
    TemporalRelation = None
    DocumentRelation = None
    SchemaRegistry = None
    schema_registry = None
    SchemaAwareEntityStorage = None
    SchemaAwareSearchService = None
    create_schema_aware_storage = None
    create_schema_aware_search = None

# Query processing
from .query import QueryEntityExtractor, QueryEntity, QueryAnalysis, QueryIntentAnalyzer

# Retrieval system
from .retrieval import (
    RetrievalResult, HybridRetrievalConfig, RetrievalError,
    HybridRetrievalCoordinator, ContextExpansionEngine, ExpandedContext,
    ResultFusionEngine, WeightedCombinationFusion, ReciprocalRankFusion
)

__all__ = [
    "Entity",
    "Relation",
    "Graph",
    "EntityExtractor",
    "RelationExtractor",
    "Neo4jStorage",
    "QdrantStorage",
    "Neo4jConfig",
    "QdrantConfig",
    "DatabaseType",
    "DatabaseConfig",
    "DatabaseServerConfig",
    "DatabaseServerArray",
    "DatabaseResult",
    "GraphCRUD",
    "GraphTraversal",
    "GraphPath",
    "GraphAnalytics",
    "GraphBuilder",
    "GraphBuildResult",
    "GraphBuildError",
    "QueryEntityExtractor",
    "QueryEntity",
    "QueryAnalysis",
    "QueryIntentAnalyzer",
    "RetrievalResult",
    "HybridRetrievalConfig",
    "RetrievalError",
    "HybridRetrievalCoordinator",
    "ContextExpansionEngine",
    "ExpandedContext",
    "ResultFusionEngine",
    "WeightedCombinationFusion",
    "ReciprocalRankFusion",
    # Graphiti integration (conditional)
    "GRAPHITI_AVAILABLE",
    "GraphitiConfig",
    "create_graphiti_instance",
    "GraphitiTemporalService",
    "TemporalSnapshot",
    "TemporalChange",
    "TemporalQueryType",
    "create_temporal_service",
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
    "SchemaAwareEntityStorage",
    "SchemaAwareSearchService",
    "create_schema_aware_storage",
    "create_schema_aware_search",
    "GraphitiConnectionService",
    "DocumentEpisodeMapper",
    "create_episode_mapper",
    "GraphitiSearchService",
    "SearchResult",
    "SearchMetrics",
    "SearchResultAdapter",
    "create_search_service",
    "SearchInterface",
    "GraphitiSearchAdapter",
    "HybridSearchService",
    "create_search_adapter",
    "create_hybrid_search_service",
]