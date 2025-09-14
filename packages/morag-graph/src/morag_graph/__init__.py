"""MORAG Graph Package.

A graph-based knowledge representation and storage system.
"""

__version__ = "0.1.0"

# Core models - import these first to avoid circular dependencies
try:
    from .models import Entity, Relation, Graph
    _MODELS_AVAILABLE = True
except ImportError as e:
    import warnings
    warnings.warn(f"Core models not available: {e}")
    _MODELS_AVAILABLE = False

    # Create placeholder classes
    class Entity:
        pass
    class Relation:
        pass
    class Graph:
        pass

# Extraction components - import with graceful fallback
try:
    from .extraction import EntityExtractor, RelationExtractor
    _EXTRACTION_AVAILABLE = True
except ImportError as e:
    import warnings
    warnings.warn(f"Extraction components not available: {e}")
    _EXTRACTION_AVAILABLE = False

    # Create placeholder classes
    class EntityExtractor:
        pass
    class RelationExtractor:
        pass

# Storage backends - import with graceful fallback
try:
    from .storage import Neo4jStorage, QdrantStorage, Neo4jConfig, QdrantConfig
    _STORAGE_AVAILABLE = True
except ImportError as e:
    import warnings
    warnings.warn(f"Storage backends not available: {e}")
    _STORAGE_AVAILABLE = False

    # Create placeholder classes
    class Neo4jStorage:
        pass
    class QdrantStorage:
        pass
    class Neo4jConfig:
        pass
    class QdrantConfig:
        pass

# Operations - import with graceful fallback
try:
    from .operations import GraphCRUD, GraphTraversal, GraphPath, GraphAnalytics
    _OPERATIONS_AVAILABLE = True
except ImportError as e:
    import warnings
    warnings.warn(f"Operations not available: {e}")
    _OPERATIONS_AVAILABLE = False

    # Create placeholder classes
    class GraphCRUD:
        pass
    class GraphTraversal:
        pass
    class GraphPath:
        pass
    class GraphAnalytics:
        pass

# Database configuration models - import with graceful fallback
try:
    from .models.database_config import DatabaseType, DatabaseConfig, DatabaseServerConfig, DatabaseServerArray, DatabaseResult
    _DATABASE_CONFIG_AVAILABLE = True
except ImportError as e:
    import warnings
    warnings.warn(f"Database config not available: {e}")
    _DATABASE_CONFIG_AVAILABLE = False

    # Create placeholder classes
    class DatabaseType:
        pass
    class DatabaseConfig:
        pass
    class DatabaseServerConfig:
        pass
    class DatabaseServerArray:
        pass
    class DatabaseResult:
        pass

# Query processing - import with graceful fallback
try:
    from .query import QueryEntityExtractor, QueryEntity, QueryAnalysis, QueryIntentAnalyzer
    _QUERY_AVAILABLE = True
except ImportError as e:
    import warnings
    warnings.warn(f"Query processing not available: {e}")
    _QUERY_AVAILABLE = False

    # Create placeholder classes
    class QueryEntityExtractor:
        pass
    class QueryEntity:
        pass
    class QueryAnalysis:
        pass
    class QueryIntentAnalyzer:
        pass

# Retrieval system - import with graceful fallback
try:
    from .retrieval import (
        RetrievalResult, HybridRetrievalConfig, RetrievalError,
        HybridRetrievalCoordinator, ContextExpansionEngine, ExpandedContext,
        ResultFusionEngine, WeightedCombinationFusion, ReciprocalRankFusion
    )
    _RETRIEVAL_AVAILABLE = True
except ImportError as e:
    import warnings
    warnings.warn(f"Retrieval system not available: {e}")
    _RETRIEVAL_AVAILABLE = False

    # Create placeholder classes
    class RetrievalResult:
        pass
    class HybridRetrievalConfig:
        pass
    class RetrievalError:
        pass
    class HybridRetrievalCoordinator:
        pass
    class ContextExpansionEngine:
        pass
    class ExpandedContext:
        pass
    class ResultFusionEngine:
        pass
    class WeightedCombinationFusion:
        pass
    class ReciprocalRankFusion:
        pass

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
]