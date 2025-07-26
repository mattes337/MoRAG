"""MORAG Graph Package.

A graph-based knowledge representation and storage system.
"""

__version__ = "0.1.0"

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