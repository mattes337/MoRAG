"""Interface definitions for MoRAG components."""

from .processor import (
    BaseProcessor,
    IContentProcessor,
    IServiceCoordinator,
    ProcessingConfig,
    ProcessingResult,
)

__all__ = [
    "IContentProcessor",
    "IServiceCoordinator",
    "BaseProcessor",
    "ProcessingResult",
    "ProcessingConfig",
]
