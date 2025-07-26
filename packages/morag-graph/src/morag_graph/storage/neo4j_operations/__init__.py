"""Neo4j storage operations modules."""

from .connection_operations import ConnectionOperations
from .document_operations import DocumentOperations
from .entity_operations import EntityOperations
from .relation_operations import RelationOperations
from .graph_operations import GraphOperations
from .query_operations import QueryOperations

# OpenIE operations (optional)
try:
    from .openie_operations import OpenIEOperations
    _OPENIE_AVAILABLE = True
except ImportError:
    _OPENIE_AVAILABLE = False
    OpenIEOperations = None

__all__ = [
    "ConnectionOperations",
    "DocumentOperations",
    "EntityOperations",
    "RelationOperations",
    "GraphOperations",
    "QueryOperations"
]

if _OPENIE_AVAILABLE:
    __all__.append("OpenIEOperations")
