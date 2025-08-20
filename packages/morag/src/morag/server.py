"""FastAPI server for MoRAG Stage-Based Processing System."""

import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path
import sys
import os

# Load environment variables from .env file early
try:
    from dotenv import load_dotenv
    # Look for .env file in current directory and parent directories
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        # Try parent directories
        for parent in Path.cwd().parents:
            env_path = parent / ".env"
            if env_path.exists():
                break
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment variables from: {env_path}")
    else:
        print("No .env file found")
except ImportError:
    print("python-dotenv not available, skipping .env file loading")

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from contextlib import asynccontextmanager
import structlog

# Add path for stage imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "morag-stages" / "src"))

from morag.api_models.endpoints.stages import router as stages_router
from morag.api_models.endpoints.files import router as files_router
from morag.utils.file_upload import validate_temp_directory_access

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    logger.info("Starting MoRAG Stage-Based Processing Server...")

    try:
        # Validate temp directory access early
        validate_temp_directory_access()
        logger.info("Temp directory access validated")

        yield

    except Exception as e:
        logger.error("Failed to initialize MoRAG server", error=str(e))
        raise
    finally:
        logger.info("Shutting down MoRAG server...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    # Create FastAPI app with lifespan management
    app = FastAPI(
        title="MoRAG Stage-Based Processing API",
        description="""
        # Stage-Based Processing System

        MoRAG Stage-Based Processing API using canonical stage names:
        - **markdown-conversion**: Convert input files to unified markdown format
        - **markdown-optimizer**: LLM-based text improvement and error correction (optional)
        - **chunker**: Create summary, chunks, and contextual embeddings
        - **fact-generator**: Extract facts, entities, relations, and keywords
        - **ingestor**: Database ingestion and storage

        ## Key Features
        - Individual stage execution with canonical names
        - Stage chain execution for multi-step processing
        - File management and download capabilities
        - Webhook notifications for stage completion
        - Resume capability with existing output detection

        ## API Endpoints
        - `/api/v1/stages/` - List available stages
        - `/api/v1/stages/{stage-name}/execute` - Execute individual stages
        - `/api/v1/stages/chain` - Execute stage chains
        - `/api/v1/stages/status` - Check execution status
        - `/api/v1/stages/health` - Health check
        - `/api/v1/files/` - File management endpoints

        ## Stage Names (Canonical)
        All endpoints use these exact canonical stage names:
        - `markdown-conversion`
        - `markdown-optimizer`
        - `chunker`
        - `fact-generator`
        - `ingestor`

        **Note**: This API completely replaces all previous MoRAG endpoints.
        No backward compatibility is provided.
        """,
        version="1.0.0",
        lifespan=lifespan,
        openapi_tags=[
            {
                "name": "stages",
                "description": "Stage execution endpoints using canonical stage names"
            },
            {
                "name": "files",
                "description": "File management and download endpoints"
            }
        ]
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error("Unhandled exception",
                    path=request.url.path,
                    method=request.method,
                    error=str(exc),
                    exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error": str(exc)}
        )

    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "MoRAG Stage-Based Processing API",
            "version": "1.0.0",
            "description": "Stage-based processing system using canonical stage names",
            "available_stages": [
                "markdown-conversion",
                "markdown-optimizer",
                "chunker",
                "fact-generator",
                "ingestor"
            ],
            "docs_url": "/docs",
            "redoc_url": "/redoc",
            "health_url": "/api/v1/stages/health",
            "migration_notice": "This API completely replaces all previous MoRAG endpoints. No backward compatibility is provided."
        }

    # Legacy endpoint deprecation notice
    @app.get("/api/v1/process")
    @app.post("/api/v1/process")
    @app.get("/api/v1/ingest")
    @app.post("/api/v1/ingest")
    @app.get("/process")
    @app.post("/process")
    async def deprecated_endpoint():
        """Deprecated endpoint notice."""
        return JSONResponse(
            status_code=410,
            content={
                "error": "Endpoint Deprecated",
                "message": "This endpoint has been removed. Use the new stage-based API.",
                "migration_guide": {
                    "old_process_endpoints": "Use /api/v1/stages/chain with appropriate stages",
                    "old_ingest_endpoints": "Use /api/v1/stages/ingestor/execute",
                    "documentation": "/docs",
                    "available_stages": [
                        "markdown-conversion",
                        "markdown-optimizer",
                        "chunker",
                        "fact-generator",
                        "ingestor"
                    ]
                }
            }
        )

    # Include stage-based routers ONLY
    app.include_router(stages_router)
    app.include_router(files_router)

    # Custom OpenAPI schema
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema

        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )

        # Add custom info
        openapi_schema["info"]["x-logo"] = {
            "url": "https://example.com/logo.png"
        }

        # Add migration notice
        openapi_schema["info"]["x-migration-notice"] = {
            "message": "This API completely replaces all previous MoRAG endpoints",
            "backward_compatibility": False,
            "canonical_stage_names": [
                "markdown-conversion",
                "markdown-optimizer",
                "chunker",
                "fact-generator",
                "ingestor"
            ]
        }

        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi

    return app


def main():
    """Main entry point for running the server."""
    import os

    # Get configuration from environment with proper defaults
    host = os.getenv("MORAG_HOST", "0.0.0.0")
    port = int(os.getenv("MORAG_PORT", "8000"))
    reload = os.getenv("MORAG_RELOAD", "false").lower() == "true"
    log_level = os.getenv("MORAG_LOG_LEVEL", "info")

    # Create app
    app = create_app()

    # Run server
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
        access_log=True
    )


if __name__ == "__main__":
    main()