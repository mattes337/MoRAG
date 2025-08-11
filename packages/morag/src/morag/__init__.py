"""MoRAG - Modular Retrieval Augmented Generation System.

This is the main integration package that provides a unified interface
to all MoRAG components and services.
"""

from .api import MoRAGAPI
from morag.cli import main as cli_main
from morag.server import create_app, main as server_main
from morag.worker import main as worker_main
from morag.orchestrator import MoRAGOrchestrator

# Pipeline orchestration components
from .agents import MoRAGPipelineAgent, IngestionOptions, ResolutionOptions
from .pipeline import IntermediateFileManager, PipelineStateManager

# Re-export key components from sub-packages
from morag_core.models import Document, DocumentChunk, ProcessingResult
from morag_services import MoRAGServices, ServiceConfig, ContentType
from morag_web import WebProcessor, WebConverter
from morag_youtube import YouTubeProcessor

__version__ = "0.1.0"

__all__ = [
    "MoRAGAPI",
    "MoRAGOrchestrator", 
    "MoRAGServices",
    "ServiceConfig",
    "ContentType",
    "Document",
    "DocumentChunk",
    "ProcessingResult",
    "WebProcessor",
    "WebConverter",
    "YouTubeProcessor",
    "create_app",
    "cli_main",
    "server_main", 
    "worker_main",
]
