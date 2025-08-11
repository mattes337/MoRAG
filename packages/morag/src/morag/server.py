"""FastAPI server for MoRAG system."""

import asyncio
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json
import uuid

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import structlog

from morag.api import MoRAGAPI
from morag.api_models.models import (
    ProcessURLRequest, ProcessBatchRequest, SearchRequest, ProcessingResultResponse,
    IngestFileRequest, IngestURLRequest, IngestBatchRequest, IngestRemoteFileRequest,
    ProcessRemoteFileRequest, IngestResponse, BatchIngestResponse, TaskStatus
)
from morag.api_models.utils import (
    download_remote_file, normalize_content_type, normalize_processing_result,
    encode_thumbnails_to_base64
)
from morag_services import ServiceConfig
from morag_core.models import ProcessingResult, IngestionResponse, BatchIngestionResponse, TaskStatusResponse
from morag_graph.models.database_config import DatabaseType, DatabaseConfig
from morag.utils.file_upload import get_upload_handler, FileUploadError, validate_temp_directory_access
from morag.services.cleanup_service import start_cleanup_service, stop_cleanup_service, force_cleanup
from morag.worker import (
    process_file_task, process_url_task, process_web_page_task,
    process_youtube_video_task, process_batch_task, celery_app
)
from morag.ingest_tasks import ingest_file_task, ingest_url_task, ingest_batch_task
from morag.endpoints import remote_jobs_router
from morag.api_models.endpoints.processing import setup_processing_endpoints
from morag.api_models.endpoints.search import setup_search_endpoints
from morag.api_models.endpoints.ingestion import setup_ingestion_endpoints
from morag.api_models.endpoints.tasks import setup_task_endpoints, setup_task_management_endpoints
from morag.api_models.endpoints.admin import setup_admin_endpoints

logger = structlog.get_logger(__name__)


# Note: Utility functions have been moved to morag.api.utils module





# Note: Pydantic models have been moved to morag.api.models module


def create_app(config: Optional[ServiceConfig] = None) -> FastAPI:
    """Create FastAPI application."""

    # Initialize MoRAG API lazily to avoid settings validation at import time
    morag_api = None

    def get_morag_api() -> MoRAGAPI:
        """Get or create MoRAG API instance."""
        nonlocal morag_api
        if morag_api is None:
            morag_api = MoRAGAPI(config)
        return morag_api

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan context manager for startup and shutdown."""
        # Startup
        logger.info("MoRAG API server starting up")

        # Validate temp directory access early - fail fast if not accessible
        try:
            validate_temp_directory_access()
            logger.info("Temp directory validation passed")
        except RuntimeError as e:
            logger.error("STARTUP FAILURE: Temp directory validation failed", error=str(e))
            raise RuntimeError(f"Cannot start server: {str(e)}")

        # Start periodic cleanup service
        start_cleanup_service(
            cleanup_interval_hours=1,    # Run cleanup every hour
            max_file_age_hours=24,       # Files older than 24 hours are eligible for cleanup
            max_disk_usage_mb=10000       # Aggressive cleanup if temp files exceed 10GB
        )
        logger.info("Periodic cleanup service started")

        yield

        # Shutdown
        logger.info("MoRAG API server shutting down")
        stop_cleanup_service()
        logger.info("Periodic cleanup service stopped")
        await get_morag_api().cleanup()
        logger.info("MoRAG API server shut down")

    app = FastAPI(
        title="MoRAG API",
        description="""
        Modular Retrieval Augmented Generation System

        ## Features
        - **Processing Endpoints**: Process content and return results immediately
        - **Ingestion Endpoints**: Process content and store in vector database for retrieval
        - **Task Management**: Track processing status and manage background tasks
        - **Search**: Query stored content using vector similarity

        ## Endpoint Categories
        - `/process/*` - Immediate processing (no storage)
        - `/api/v1/ingest/*` - Background processing with vector storage
        - `/api/v1/status/*` - Task status and management
        - `/search` - Vector similarity search
        """,
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {"message": "MoRAG API", "version": "0.1.0"}

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        try:
            status = await get_morag_api().health_check()
            return JSONResponse(content=status)
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    # Setup modular endpoints
    processing_router = setup_processing_endpoints(get_morag_api)
    search_router = setup_search_endpoints(get_morag_api)
    ingestion_router = setup_ingestion_endpoints(get_morag_api)
    tasks_router = setup_task_endpoints(get_morag_api)
    task_mgmt_router = setup_task_management_endpoints(get_morag_api)
    admin_router = setup_admin_endpoints(get_morag_api)

    # Include routers
    app.include_router(processing_router)
    app.include_router(search_router)
    app.include_router(ingestion_router)
    app.include_router(tasks_router)
    app.include_router(task_mgmt_router)
    app.include_router(admin_router)


    # Include routers
    app.include_router(remote_jobs_router)

    # Include enhanced query router (v2 API)
    try:
        from morag.endpoints import enhanced_query_router
        app.include_router(enhanced_query_router)
        logger.info("Enhanced query API endpoints loaded")
    except ImportError as e:
        logger.warning("Enhanced query endpoints not available", error=str(e))

    # Include intelligent retrieval router (v2 API)
    try:
        from morag.endpoints.intelligent_retrieval import router as intelligent_retrieval_router
        app.include_router(intelligent_retrieval_router)
        logger.info("Intelligent retrieval API endpoints loaded")
    except ImportError as e:
        logger.warning("Intelligent retrieval endpoints not available", error=str(e))

    # Legacy router temporarily disabled
    # try:
    #     from morag.endpoints import legacy_router
    #     app.include_router(legacy_router)
    #     logger.info("Legacy API endpoints loaded")
    # except ImportError as e:
    #     logger.warning("Legacy endpoints not available", error=str(e))

    # Include reasoning router (multi-hop reasoning API)
    try:
        from morag.endpoints.reasoning import router as reasoning_router
        app.include_router(reasoning_router, prefix="/api/v2")
        logger.info("Multi-hop reasoning API endpoints loaded")
    except ImportError as e:
        logger.warning("Multi-hop reasoning endpoints not available", error=str(e))

    # Include recursive fact retrieval router (recursive fact retrieval API)
    try:
        from morag.endpoints.recursive_fact_retrieval import router as recursive_fact_retrieval_router
        app.include_router(recursive_fact_retrieval_router)
        logger.info("Recursive fact retrieval API endpoints loaded")
    except ImportError as e:
        logger.warning("Recursive fact retrieval endpoints not available", error=str(e))

    return app


def main():
    """Main entry point for the server."""
    import argparse

    parser = argparse.ArgumentParser(description="MoRAG API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--config", help="Configuration file path")

    args = parser.parse_args()

    # Load configuration if provided
    config = None
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            import json
            with open(config_path) as f:
                config_data = json.load(f)
                config = ServiceConfig(**config_data)

    # Create app
    app = create_app(config)

    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1
    )


if __name__ == "__main__":
    main()
