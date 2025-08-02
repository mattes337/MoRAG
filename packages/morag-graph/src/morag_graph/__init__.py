"""MORAG Graph Package.

A graph-based knowledge representation and storage system.
"""

__version__ = "0.1.0"

# Import main components with graceful fallback
try:
    from .models import Entity, Relation
    from .storage import GraphStorage, InMemoryGraphStorage, DummyGraphStorage

    # Try to import extraction components
    try:
        from .extraction.base import BaseExtractor
    except ImportError:
        class BaseExtractor:
            """Placeholder BaseExtractor."""
            pass

    # Placeholder classes for compatibility
    class HybridRetrievalCoordinator:
        """Placeholder for hybrid retrieval coordinator."""
        pass

    class ContextExpansionEngine:
        """Placeholder for context expansion engine."""
        pass

    class QueryEntityExtractor:
        """Placeholder for query entity extractor."""
        pass

    __all__ = [
        "BaseExtractor",
        "Entity",
        "Relation",
        "GraphStorage",
        "InMemoryGraphStorage",
        "DummyGraphStorage",
        "HybridRetrievalCoordinator",
        "ContextExpansionEngine",
        "QueryEntityExtractor"
    ]

except ImportError as e:
    # If imports fail, create minimal placeholders
    import warnings
    warnings.warn(f"Graph processing components not fully available: {e}")

    class BaseExtractor:
        pass

    class Entity:
        pass

    class Relation:
        pass

    class GraphStorage:
        pass

    class InMemoryGraphStorage:
        pass

    class DummyGraphStorage:
        pass

    class HybridRetrievalCoordinator:
        pass

    class ContextExpansionEngine:
        pass

    class QueryEntityExtractor:
        pass

    __all__ = [
        "BaseExtractor",
        "Entity",
        "Relation",
        "GraphStorage",
        "InMemoryGraphStorage",
        "DummyGraphStorage",
        "HybridRetrievalCoordinator",
        "ContextExpansionEngine",
        "QueryEntityExtractor"
    ]

# Core models
from .models import Entity, Relation, Graph

# Extraction components
from .extraction import EntityExtractor, RelationExtractor

# OpenIE components (optional)
try:
    from .extractors import OpenIEExtractor
    _OPENIE_AVAILABLE = True
except ImportError:
    _OPENIE_AVAILABLE = False
    OpenIEExtractor = None

# Storage backends
from .storage import Neo4jStorage, QdrantStorage, Neo4jConfig, QdrantConfig

# Operations
from .operations import GraphCRUD, GraphTraversal, GraphPath, GraphAnalytics

# Builders
from .builders import GraphBuilder, GraphBuildResult, GraphBuildError

# Enhanced builders (optional)
try:
    from .builders import EnhancedGraphBuilder, EnhancedGraphBuildResult
    _ENHANCED_BUILDER_AVAILABLE = True
except ImportError:
    _ENHANCED_BUILDER_AVAILABLE = False
    EnhancedGraphBuilder = None
    EnhancedGraphBuildResult = None

# Database configuration models
from .models.database_config import DatabaseType, DatabaseConfig, DatabaseServerConfig, DatabaseServerArray, DatabaseResult

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
]