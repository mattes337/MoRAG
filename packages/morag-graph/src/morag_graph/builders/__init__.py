"""Graph builders for constructing knowledge graphs from documents."""

from .graph_builder import GraphBuilder, GraphBuildResult, GraphBuildError

# Enhanced builder (optional)
try:
    from .enhanced_graph_builder import EnhancedGraphBuilder, EnhancedGraphBuildResult
    _ENHANCED_AVAILABLE = True
except ImportError:
    _ENHANCED_AVAILABLE = False
    EnhancedGraphBuilder = None
    EnhancedGraphBuildResult = None

__all__ = [
    "GraphBuilder",
    "GraphBuildResult",
    "GraphBuildError"
]

if _ENHANCED_AVAILABLE:
    __all__.extend(["EnhancedGraphBuilder", "EnhancedGraphBuildResult"])
