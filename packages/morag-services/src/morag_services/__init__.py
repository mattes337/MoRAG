"""MoRAG Services - Unified service layer for the MoRAG system.

This package integrates all specialized processing packages into a cohesive API,
making it easy to work with multiple content types through a single interface.
"""

from .services import MoRAGServices, ServiceConfig, ProcessingResult, ContentType
from .pipeline import Pipeline, PipelineStep, PipelineContext, PipelineStepType
from .storage import QdrantVectorStorage, QdrantService, EmbeddingCache
from .data_file_writer import DataFileWriter
from .embedding import (
    GeminiEmbeddingService,
    EmbeddingServiceFactory,
    EmbeddingResult,
    SummaryResult,
    GeminiService
)
from .contextual_retrieval import ContextualRetrievalService

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
