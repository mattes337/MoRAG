"""Processors package for MoRAG - Legacy compatibility layer"""

# Import from modular packages for backward compatibility
try:
    from morag_web import WebProcessor, WebScrapingConfig, WebContent, WebScrapingResult
except ImportError:
    # Fallback to local implementation if package not available
    from .web import WebProcessor, WebScrapingConfig, WebContent, WebScrapingResult

__all__ = [
    'WebProcessor',
    'WebScrapingConfig',
    'WebContent',
    'WebScrapingResult'
]
