"""Interface definitions for MoRAG components."""

from .processor import IContentProcessor, IServiceCoordinator, BaseProcessor, ProcessingResult, ProcessingConfig

__all__ = [
    'IContentProcessor',
    'IServiceCoordinator',
    'BaseProcessor',
    'ProcessingResult',
    'ProcessingConfig'
]
