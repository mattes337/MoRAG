"""Adapter layer for MoRAG-Graphiti integration.

This module provides comprehensive adapters for converting between MoRAG models
and Graphiti episode formats, enabling seamless integration while maintaining
backward compatibility.
"""

from .core import (
    BaseAdapter,
    BatchAdapter,
    AdapterRegistry,
    ConversionResult,
    ConversionDirection,
    AdapterError,
    ConversionError,
    ValidationError,
    adapter_registry
)

try:
    from .document_adapter import DocumentAdapter, DocumentChunkAdapter
    from .entity_adapter import EntityAdapter, RelationAdapter
    ADAPTERS_AVAILABLE = True
except ImportError:
    # Graceful degradation if dependencies are missing
    DocumentAdapter = None
    DocumentChunkAdapter = None
    EntityAdapter = None
    RelationAdapter = None
    ADAPTERS_AVAILABLE = False

__all__ = [
    # Core adapter classes
    "BaseAdapter",
    "BatchAdapter", 
    "AdapterRegistry",
    "ConversionResult",
    "ConversionDirection",
    "AdapterError",
    "ConversionError",
    "ValidationError",
    "adapter_registry",
    # Specific adapters
    "DocumentAdapter",
    "DocumentChunkAdapter",
    "EntityAdapter",
    "RelationAdapter",
    # Status
    "ADAPTERS_AVAILABLE"
]
