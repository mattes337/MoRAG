"""Markitdown-based document converter implementation.

DEPRECATED: This module has been split for better maintainability.
Use document_converters.MarkitdownConverter instead.
"""

# Re-export the split components for backward compatibility
from .base_converter import BaseMarkitdownConverter
from .document_converters import MarkitdownConverter
from .chunking_processor import ChunkingProcessor

# Maintain backward compatibility
__all__ = ["MarkitdownConverter", "BaseMarkitdownConverter", "ChunkingProcessor"]

# Note: The original implementation has been moved to split files:
# - base_converter.py: Abstract base converter interface
# - document_converters.py: Concrete converter implementations
# - chunking_processor.py: Document chunking functionality