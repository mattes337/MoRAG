"""Build knowledge graphs from extracted facts.

REFACTORED: This module has been split for better maintainability.
The main functionality is now distributed across:
- graph_builder_interface.py: Main coordination and interface
- fact_graph_operations.py: Core operations and LLM interactions
- graph_utilities.py: Graph construction utilities

This file provides backward compatibility.
"""

# Export additional components if needed
from .fact_graph_operations import FactGraphOperations

# Re-export the refactored components for backward compatibility
from .graph_builder_interface import FactGraphBuilder
from .graph_utilities import GraphUtilities

__all__ = ["FactGraphBuilder", "FactGraphOperations", "GraphUtilities"]
