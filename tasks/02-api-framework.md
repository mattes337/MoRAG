# Task 02: FastAPI Framework Setup

## Overview
Set up the FastAPI application with async support, middleware, error handling, and basic API structure.

## Prerequisites
- Task 01: Project Setup completed
- Python virtual environment activated

## Dependencies
- Task 01: Project Setup

## Implementation Steps

### 1. Main Application Setup
Create `src/morag/api/main.py`:
```python
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import structlog
import time
from typing import Dict, Any

from morag.core.config import settings
from morag.core.exceptions import MoragException
from morag.api.routes import health, ingestion, status

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer() if settings.log_format == "json" else structlog.dev.ConsoleRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title="MoRAG Ingestion Pipeline",
        description="Multimodal RAG content ingestion and processing service",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # Configure appropriately for production
    )
    
    # Add request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.time()
        
        # Log request
        logger.info(
            "Request started",
            method=request.method,
            url=str(request.url),
            client_ip=request.client.host if request.client else None,
        )
        
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        logger.info(
            "Request completed",
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            process_time=round(process_time, 4),
        )
        
        return response
    
    # Add exception handlers
    @app.exception_handler(MoragException)
    async def morag_exception_handler(request: Request, exc: MoragException):
        logger.error(
            "MoRAG exception occurred",
            error=str(exc),
            error_type=type(exc).__name__,
            url=str(request.url),
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.message, "type": exc.error_type}
        )
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        logger.error(
            "HTTP exception occurred",
            error=str(exc.detail),
            status_code=exc.status_code,
            url=str(request.url),
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.detail}
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(
            "Unexpected exception occurred",
            error=str(exc),
            error_type=type(exc).__name__,
            url=str(request.url),
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        )
    
    # Include routers
    app.include_router(health.router, prefix="/health", tags=["health"])
    app.include_router(ingestion.router, prefix="/api/v1/ingest", tags=["ingestion"])
    app.include_router(status.router, prefix="/api/v1/status", tags=["status"])
    
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "morag.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )
```

### 2. Custom Exceptions
Create `src/morag/core/exceptions.py`:
```python
from typing import Optional

class MoragException(Exception):
    """Base exception for MoRAG application."""
    
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_type: str = "internal_error"
    ):
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        super().__init__(self.message)

class ValidationError(MoragException):
    """Raised when input validation fails."""
    
    def __init__(self, message: str):
        super().__init__(message, status_code=400, error_type="validation_error")

class ProcessingError(MoragException):
    """Raised when content processing fails."""
    
    def __init__(self, message: str):
        super().__init__(message, status_code=422, error_type="processing_error")

class StorageError(MoragException):
    """Raised when storage operations fail."""
    
    def __init__(self, message: str):
        super().__init__(message, status_code=500, error_type="storage_error")

class ExternalServiceError(MoragException):
    """Raised when external service calls fail."""
    
    def __init__(self, message: str, service: str):
        self.service = service
        super().__init__(
            f"{service} error: {message}",
            status_code=502,
            error_type="external_service_error"
        )

class AuthenticationError(MoragException):
    """Raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication required"):
        super().__init__(message, status_code=401, error_type="authentication_error")

class RateLimitError(MoragException):
    """Raised when rate limits are exceeded."""
    
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, status_code=429, error_type="rate_limit_error")
```

### 3. Health Check Router
Create `src/morag/api/routes/health.py`:
```python
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Dict, Any
import asyncio
import structlog

from morag.core.config import settings

logger = structlog.get_logger()
router = APIRouter()

class HealthResponse(BaseModel):
    status: str
    version: str
    services: Dict[str, str]

@router.get("/", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        services={}
    )

@router.get("/ready")
async def readiness_check():
    """Readiness check with service dependencies."""
    services = {}
    
    # Check Redis connection (will be implemented in task 04)
    try:
        # Placeholder for Redis check
        services["redis"] = "healthy"
    except Exception as e:
        logger.error("Redis health check failed", error=str(e))
        services["redis"] = "unhealthy"
    
    # Check Qdrant connection (will be implemented in task 03)
    try:
        # Placeholder for Qdrant check
        services["qdrant"] = "healthy"
    except Exception as e:
        logger.error("Qdrant health check failed", error=str(e))
        services["qdrant"] = "unhealthy"
    
    # Check Gemini API (will be implemented in task 14)
    try:
        # Placeholder for Gemini check
        services["gemini"] = "healthy"
    except Exception as e:
        logger.error("Gemini health check failed", error=str(e))
        services["gemini"] = "unhealthy"
    
    all_healthy = all(status == "healthy" for status in services.values())
    
    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        version="0.1.0",
        services=services
    )
```

### 4. Placeholder Route Files
Create `src/morag/api/routes/__init__.py`:
```python
# API routes package
```

Create `src/morag/api/routes/ingestion.py`:
```python
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

router = APIRouter()

class IngestionRequest(BaseModel):
    source_type: str
    webhook_url: Optional[str] = None

class IngestionResponse(BaseModel):
    task_id: str
    status: str
    message: str

@router.post("/", response_model=IngestionResponse)
async def create_ingestion_task(request: IngestionRequest):
    """Create a new ingestion task. (Placeholder - will be implemented in task 17)"""
    return IngestionResponse(
        task_id="placeholder-task-id",
        status="pending",
        message="Ingestion endpoint not yet implemented"
    )
```

Create `src/morag/api/routes/status.py`:
```python
from fastapi import APIRouter, Path
from pydantic import BaseModel
from typing import Optional, Dict, Any

router = APIRouter()

class StatusResponse(BaseModel):
    task_id: str
    status: str
    progress: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@router.get("/{task_id}", response_model=StatusResponse)
async def get_task_status(task_id: str = Path(..., description="Task ID to check")):
    """Get the status of an ingestion task. (Placeholder - will be implemented in task 18)"""
    return StatusResponse(
        task_id=task_id,
        status="pending",
        message="Status endpoint not yet implemented"
    )
```

### 5. Application Entry Point
Create `src/morag/__init__.py`:
```python
"""MoRAG - Multimodal RAG Ingestion Pipeline"""

__version__ = "0.1.0"
```

Create `src/morag/api/__init__.py`:
```python
"""API package for MoRAG"""
```

## Testing Instructions

### 1. Start the Application
```bash
# From project root
cd src
python -m morag.api.main
```

### 2. Test Endpoints
```bash
# Health check
curl http://localhost:8000/health/

# Readiness check
curl http://localhost:8000/health/ready

# API documentation
# Visit http://localhost:8000/docs in browser
```

## Mandatory Testing Requirements

### 1. API Framework Tests
Create `tests/test_02_api_framework.py`:
```python
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json
from morag.api.main import create_app
from morag.core.exceptions import MoragException, ValidationError

@pytest.fixture
def app():
    """Create test FastAPI application."""
    return create_app()

@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)

class TestAPIFramework:
    """Test FastAPI framework setup and configuration."""

    def test_app_creation(self, app):
        """Test that FastAPI app is created correctly."""
        assert app is not None
        assert app.title == "MoRAG Ingestion Pipeline"
        assert app.version == "0.1.0"

    def test_health_endpoint(self, client):
        """Test basic health check endpoint."""
        response = client.get("/health/")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"
        assert "services" in data

    def test_readiness_endpoint(self, client):
        """Test readiness check endpoint."""
        response = client.get("/health/ready")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "services" in data
        assert isinstance(data["services"], dict)

    def test_docs_endpoint(self, client):
        """Test API documentation endpoint."""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_redoc_endpoint(self, client):
        """Test ReDoc documentation endpoint."""
        response = client.get("/redoc")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

class TestMiddleware:
    """Test middleware functionality."""

    def test_cors_middleware(self, client):
        """Test CORS middleware configuration."""
        response = client.options("/health/")
        assert response.status_code == 200

        # Test preflight request
        headers = {
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "Content-Type"
        }
        response = client.options("/health/", headers=headers)
        assert response.status_code == 200

    def test_request_logging_middleware(self, client):
        """Test that request logging middleware is working."""
        with patch('structlog.get_logger') as mock_logger:
            mock_log = MagicMock()
            mock_logger.return_value = mock_log

            response = client.get("/health/")
            assert response.status_code == 200

            # Verify logging was called
            assert mock_log.info.call_count >= 2  # Request start and completion

class TestExceptionHandling:
    """Test exception handling."""

    def test_morag_exception_handler(self, app):
        """Test custom MoRAG exception handling."""
        from fastapi import Request
        from morag.api.main import app as main_app

        # Create a test route that raises MoragException
        @app.get("/test-morag-exception")
        async def test_route():
            raise ValidationError("Test validation error")

        client = TestClient(app)
        response = client.get("/test-morag-exception")

        assert response.status_code == 400
        data = response.json()
        assert data["error"] == "Test validation error"
        assert data["type"] == "validation_error"

    def test_http_exception_handler(self, client):
        """Test HTTP exception handling."""
        # Test 404 for non-existent endpoint
        response = client.get("/non-existent-endpoint")
        assert response.status_code == 404

        data = response.json()
        assert "error" in data

    def test_general_exception_handler(self, app):
        """Test general exception handling."""
        # Create a test route that raises a general exception
        @app.get("/test-general-exception")
        async def test_route():
            raise Exception("Test general error")

        client = TestClient(app)
        response = client.get("/test-general-exception")

        assert response.status_code == 500
        data = response.json()
        assert data["error"] == "Internal server error"

class TestRouterIntegration:
    """Test router integration."""

    def test_health_router_included(self, client):
        """Test that health router is properly included."""
        response = client.get("/health/")
        assert response.status_code == 200

    def test_ingestion_router_included(self, client):
        """Test that ingestion router is properly included."""
        # This should return method not allowed or similar, not 404
        response = client.post("/api/v1/ingest/")
        assert response.status_code != 404  # Router is included

    def test_status_router_included(self, client):
        """Test that status router is properly included."""
        # This should return some response, not 404
        response = client.get("/api/v1/status/test-id")
        assert response.status_code != 404  # Router is included
```

### 2. Exception Handling Tests
Create `tests/test_02_exceptions.py`:
```python
import pytest
from morag.core.exceptions import (
    MoragException, ValidationError, ProcessingError,
    StorageError, ExternalServiceError, AuthenticationError, RateLimitError
)

class TestExceptions:
    """Test custom exception classes."""

    def test_morag_exception_base(self):
        """Test base MoragException."""
        exc = MoragException("Test message", status_code=500, error_type="test_error")

        assert str(exc) == "Test message"
        assert exc.message == "Test message"
        assert exc.status_code == 500
        assert exc.error_type == "test_error"

    def test_validation_error(self):
        """Test ValidationError."""
        exc = ValidationError("Invalid input")

        assert exc.status_code == 400
        assert exc.error_type == "validation_error"
        assert exc.message == "Invalid input"

    def test_processing_error(self):
        """Test ProcessingError."""
        exc = ProcessingError("Processing failed")

        assert exc.status_code == 422
        assert exc.error_type == "processing_error"
        assert exc.message == "Processing failed"

    def test_storage_error(self):
        """Test StorageError."""
        exc = StorageError("Storage operation failed")

        assert exc.status_code == 500
        assert exc.error_type == "storage_error"
        assert exc.message == "Storage operation failed"

    def test_external_service_error(self):
        """Test ExternalServiceError."""
        exc = ExternalServiceError("API call failed", "gemini")

        assert exc.status_code == 502
        assert exc.error_type == "external_service_error"
        assert "gemini error: API call failed" in exc.message
        assert exc.service == "gemini"

    def test_authentication_error(self):
        """Test AuthenticationError."""
        exc = AuthenticationError("Invalid token")

        assert exc.status_code == 401
        assert exc.error_type == "authentication_error"
        assert exc.message == "Invalid token"

    def test_rate_limit_error(self):
        """Test RateLimitError."""
        exc = RateLimitError("Too many requests")

        assert exc.status_code == 429
        assert exc.error_type == "rate_limit_error"
        assert exc.message == "Too many requests"
```

### 3. Health Check Tests
Create `tests/test_02_health_checks.py`:
```python
import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from morag.api.main import create_app

@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)

class TestHealthChecks:
    """Test health check functionality."""

    def test_basic_health_check(self, client):
        """Test basic health endpoint returns correct structure."""
        response = client.get("/health/")
        assert response.status_code == 200

        data = response.json()
        required_fields = ["status", "version", "services"]
        for field in required_fields:
            assert field in data

        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"

    @patch('morag.api.routes.health.task_manager')
    @patch('redis.from_url')
    def test_readiness_check_all_healthy(self, mock_redis, mock_task_manager, client):
        """Test readiness check when all services are healthy."""
        # Mock Redis
        mock_redis_instance = AsyncMock()
        mock_redis_instance.ping.return_value = True
        mock_redis.return_value = mock_redis_instance

        # Mock task manager
        mock_task_manager.get_queue_stats.return_value = {"active_tasks": 0}

        response = client.get("/health/ready")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "services" in data

    @patch('redis.from_url')
    def test_readiness_check_redis_unhealthy(self, mock_redis, client):
        """Test readiness check when Redis is unhealthy."""
        # Mock Redis to fail
        mock_redis.side_effect = Exception("Redis connection failed")

        response = client.get("/health/ready")
        assert response.status_code == 200

        data = response.json()
        assert data["services"]["redis"] == "unhealthy"

    def test_health_check_response_format(self, client):
        """Test that health check response has correct format."""
        response = client.get("/health/")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"

        data = response.json()
        assert isinstance(data["services"], dict)
        assert isinstance(data["status"], str)
        assert isinstance(data["version"], str)
```

### 4. Test Execution Instructions
```bash
# Install test dependencies (if not already installed)
pip install pytest pytest-asyncio pytest-mock httpx

# Run API framework tests
pytest tests/test_02_api_framework.py -v

# Run exception tests
pytest tests/test_02_exceptions.py -v

# Run health check tests
pytest tests/test_02_health_checks.py -v

# Run all Task 02 tests with coverage
pytest tests/test_02_*.py -v --cov=src/morag/api --cov=src/morag/core/exceptions --cov-report=html

# Test API startup and shutdown
python -c "
from morag.api.main import create_app
app = create_app()
print('✅ API app created successfully')
"
```

## Success Criteria (MANDATORY - ALL MUST PASS)
- [ ] FastAPI application starts without errors
- [ ] All health endpoints return expected responses with correct structure
- [ ] API documentation is accessible at /docs and /redoc
- [ ] CORS middleware is properly configured and tested
- [ ] Request logging middleware captures all requests
- [ ] All custom exception handlers work correctly
- [ ] Exception hierarchy is properly implemented
- [ ] Router integration works for all planned endpoints
- [ ] Structured logging is functional and tested
- [ ] All unit tests pass with >95% coverage
- [ ] API can handle concurrent requests without errors
- [ ] Error responses follow consistent format
- [ ] Health checks properly report service status

## Advancement Blocker
**⚠️ CRITICAL: Cannot proceed to Task 03 until ALL tests pass and the following integration tests succeed:**

### Integration Test Requirements
```bash
# Start API server
python src/morag/api/main.py &
API_PID=$!

# Wait for startup
sleep 5

# Test all endpoints
curl -f http://localhost:8000/health/ || exit 1
curl -f http://localhost:8000/health/ready || exit 1
curl -f http://localhost:8000/docs || exit 1

# Test error handling
curl -X GET http://localhost:8000/non-existent-endpoint | grep -q "error" || exit 1

# Cleanup
kill $API_PID

echo "✅ All integration tests passed"
```

### Coverage Requirements
- API routes: >95% coverage
- Exception handling: 100% coverage
- Middleware: >90% coverage
- Health checks: 100% coverage

## Next Steps (Only after ALL tests pass)
- Task 03: Database Setup (Qdrant)
- Task 04: Task Queue Setup (Celery/Redis)
- Task 14: Gemini Integration
