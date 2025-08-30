"""Processor interfaces and registry for markdown conversion stage."""

from .interface import StageProcessor, ProcessorResult
from .registry import ProcessorRegistry, get_registry

__all__ = [
    "StageProcessor",
    "ProcessorResult",
    "ProcessorRegistry",
    "get_registry"
]
