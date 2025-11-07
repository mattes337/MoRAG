"""Markdown conversion stage implementation.

REFACTORED: This module has been split for better maintainability.
The main functionality is now distributed across:
- markdown_conversion_stage.py: Main stage interface and coordination
- converter_factory.py: Converter selection logic
- conversion_processors.py: Individual converter implementations

This file provides backward compatibility.
"""

# Re-export the split components for backward compatibility
from .markdown_conversion_stage import MarkdownConversionStage
from .converter_factory import ConverterFactory
from .conversion_processors import ConversionProcessors

# Maintain backward compatibility by exporting the main class
__all__ = ["MarkdownConversionStage", "ConverterFactory", "ConversionProcessors"]
