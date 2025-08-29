"""MoRAG - Modular Retrieval Augmented Generation System.

This is the main integration package that provides a unified interface
to all MoRAG components and services.
"""

# Load environment variables from .env file early
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    # Look for .env file in current directory and parent directories
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        # Try parent directories up to 3 levels
        for parent in list(Path.cwd().parents)[:3]:
            env_path = parent / ".env"
            if env_path.exists():
                break
    if env_path.exists():
        load_dotenv(env_path)
        # Only print in debug mode to avoid spam
        if os.getenv('MORAG_DEBUG', '').lower() in ('true', '1', 'yes'):
            print(f"[DEBUG] Loaded environment variables from: {env_path}")
except ImportError:
    # python-dotenv not available, continue without .env loading
    pass

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
