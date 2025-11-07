"""MoRAG Services - Unified service layer for the MoRAG system.

This package integrates all specialized processing packages into a cohesive API,
making it easy to work with multiple content types through a single interface.
"""

from .contextual_retrieval import ContextualRetrievalService
from .data_file_writer import DataFileWriter
from .embedding import (
    EmbeddingResult,
    EmbeddingServiceFactory,
    GeminiEmbeddingService,
    GeminiService,
    SummaryResult,
)
from .pipeline import Pipeline, PipelineContext, PipelineStep, PipelineStepType
from .services import ContentType, MoRAGServices, ProcessingResult, ServiceConfig
from .storage import EmbeddingCache, QdrantService, QdrantVectorStorage

__version__ = "0.1.0"

__all__ = [
    "MoRAGServices",
    "ServiceConfig",
    "ProcessingResult",
    "ContentType",
    "Pipeline",
    "PipelineStep",
    "PipelineContext",
    "PipelineStepType",
    "QdrantVectorStorage",
    "QdrantService",
    "EmbeddingCache",
    "GeminiEmbeddingService",
    "EmbeddingServiceFactory",
    "EmbeddingResult",
    "SummaryResult",
    "GeminiService",
    "ContextualRetrievalService",
    "DataFileWriter",
]
