# Task 34: Create MoRAG Integration Layer

## Overview

Create the main `morag` package that serves as the integration layer, providing a unified API, CLI interface, and orchestration for all MoRAG components. This package will depend on all other MoRAG packages and provide a single entry point for users.

## Current State

### Completed
- ✅ All processor packages implemented (audio, video, document, image, embedding)
- ✅ Core package with interfaces and utilities
- ✅ Services package framework in place

### Remaining Work
- [ ] Create main morag package structure
- [ ] Implement unified API interface
- [ ] Create CLI interface
- [ ] Implement orchestration layer
- [ ] Create configuration management
- [ ] Implement monitoring and logging
- [ ] Create comprehensive documentation

## Implementation Steps

### Step 1: Create Package Structure

**Directory Structure:**
```
packages/morag/
├── pyproject.toml
├── README.md
├── src/
│   └── morag/
│       ├── __init__.py
│       ├── api/
│       │   ├── __init__.py
│       │   ├── main.py
│       │   ├── routes/
│       │   └── models/
│       ├── cli/
│       │   ├── __init__.py
│       │   ├── main.py
│       │   └── commands/
│       ├── orchestration/
│       │   ├── __init__.py
│       │   ├── pipeline.py
│       │   └── scheduler.py
│       ├── config/
│       │   ├── __init__.py
│       │   └── settings.py
│       └── monitoring/
│           ├── __init__.py
│           ├── health.py
│           └── metrics.py
├── tests/
└── docs/
```

### Step 2: Implement Unified API Interface

**Files to create:**
- `packages/morag/src/morag/api/main.py` - FastAPI application
- `packages/morag/src/morag/api/routes/` - API route modules
- `packages/morag/src/morag/api/models/` - Pydantic models

**API Endpoints:**
1. **Processing Endpoints**
   - `POST /process/document` - Document processing
   - `POST /process/audio` - Audio processing
   - `POST /process/video` - Video processing
   - `POST /process/image` - Image processing
   - `POST /process/web` - Web content processing
   - `POST /process/youtube` - YouTube video processing

2. **Management Endpoints**
   - `GET /status/{task_id}` - Task status
   - `GET /health` - System health
   - `GET /metrics` - System metrics
   - `POST /webhook` - Webhook endpoints

3. **Storage Endpoints**
   - `GET /search` - Vector search
   - `GET /documents/{id}` - Document retrieval
   - `DELETE /documents/{id}` - Document deletion

### Step 3: Create CLI Interface

**Files to create:**
- `packages/morag/src/morag/cli/main.py` - CLI entry point
- `packages/morag/src/morag/cli/commands/` - Command modules

**CLI Commands:**
1. **Processing Commands**
   - `morag process document <file>` - Process document
   - `morag process audio <file>` - Process audio
   - `morag process video <file>` - Process video
   - `morag process image <file>` - Process image
   - `morag process web <url>` - Process web content
   - `morag process youtube <url>` - Process YouTube video

2. **Management Commands**
   - `morag status <task_id>` - Check task status
   - `morag health` - System health check
   - `morag config` - Configuration management
   - `morag worker start` - Start worker processes

3. **Storage Commands**
   - `morag search <query>` - Search documents
   - `morag list` - List documents
   - `morag delete <id>` - Delete document

### Step 4: Implement Orchestration Layer

**Files to create:**
- `packages/morag/src/morag/orchestration/pipeline.py`
- `packages/morag/src/morag/orchestration/scheduler.py`

**Features:**
1. **Pipeline Management**
   - Dynamic pipeline creation based on content type
   - Pipeline step orchestration and monitoring
   - Error handling and recovery
   - Progress tracking and reporting

2. **Task Scheduling**
   - Celery integration for async processing
   - Priority-based task scheduling
   - Resource allocation and load balancing
   - Retry logic and failure handling

### Step 5: Create Configuration Management

**Files to create:**
- `packages/morag/src/morag/config/settings.py`
- Configuration templates and examples

**Features:**
1. **Unified Configuration**
   - Environment-based configuration
   - Service discovery configuration
   - Processing pipeline configuration
   - Storage and AI service configuration

2. **Configuration Validation**
   - Pydantic-based validation
   - Configuration schema documentation
   - Environment-specific overrides
   - Runtime configuration updates

### Step 6: Implement Monitoring and Logging

**Files to create:**
- `packages/morag/src/morag/monitoring/health.py`
- `packages/morag/src/morag/monitoring/metrics.py`

**Features:**
1. **Health Monitoring**
   - Service health checks
   - Dependency health monitoring
   - System resource monitoring
   - Alert generation and notification

2. **Metrics Collection**
   - Processing metrics (throughput, latency)
   - Resource usage metrics
   - Error rate monitoring
   - Custom business metrics

## Dependencies

### Required Packages
- `morag-core>=0.1.0` - Core interfaces and utilities
- `morag-services>=0.1.0` - Unified service layer
- `morag-document>=0.1.0` - Document processing
- `morag-audio>=0.1.0` - Audio processing
- `morag-video>=0.1.0` - Video processing
- `morag-image>=0.1.0` - Image processing
- `morag-embedding>=0.1.0` - Embedding services
- `morag-web>=0.1.0` - Web processing
- `morag-youtube>=0.1.0` - YouTube processing
- `fastapi>=0.104.0` - API framework
- `uvicorn>=0.24.0` - ASGI server
- `celery>=5.3.0` - Task queue
- `click>=8.1.0` - CLI framework
- `structlog>=23.1.0` - Structured logging
- `pydantic>=2.0.0` - Data validation

### Optional Packages
- `prometheus-client>=0.17.0` - Metrics collection
- `sentry-sdk>=1.38.0` - Error tracking
- `redis>=5.0.0` - Caching and sessions

## API Design

### Request/Response Models
```python
class ProcessingRequest(BaseModel):
    file_path: Optional[str] = None
    url: Optional[str] = None
    content: Optional[str] = None
    options: Dict[str, Any] = {}
    webhook_url: Optional[str] = None

class ProcessingResponse(BaseModel):
    task_id: str
    status: str
    message: str
    estimated_completion: Optional[datetime] = None

class ProcessingResult(BaseModel):
    task_id: str
    status: str
    result: Dict[str, Any]
    metadata: Dict[str, Any]
    processing_time: float
    created_at: datetime
    completed_at: Optional[datetime] = None
```

### CLI Interface Design
```bash
# Process files
morag process document ./document.pdf --output ./output.md
morag process audio ./audio.mp3 --enhance --speakers
morag process video ./video.mp4 --extract-audio --keyframes

# Batch processing
morag batch process ./input_dir --type document --output ./output_dir

# System management
morag worker start --concurrency 4
morag health --detailed
morag config set gemini.api_key "your-key"

# Search and retrieval
morag search "machine learning" --limit 10
morag export --format json --output ./export.json
```

## Testing Requirements

### Unit Tests
- [ ] Test API endpoint functionality
- [ ] Test CLI command execution
- [ ] Test orchestration pipeline logic
- [ ] Test configuration management
- [ ] Test monitoring and health checks

### Integration Tests
- [ ] Test end-to-end processing workflows
- [ ] Test API and CLI integration
- [ ] Test with all processor packages
- [ ] Test error handling and recovery
- [ ] Test performance under load

### System Tests
- [ ] Test complete system deployment
- [ ] Test Docker container orchestration
- [ ] Test scaling and load balancing
- [ ] Test monitoring and alerting
- [ ] Test backup and recovery

## Success Criteria

1. **Unified Interface**: Single entry point for all MoRAG functionality
2. **API Completeness**: Full REST API covering all processing capabilities
3. **CLI Usability**: Intuitive command-line interface for all operations
4. **Orchestration**: Robust pipeline management and task scheduling
5. **Monitoring**: Comprehensive health monitoring and metrics collection
6. **Documentation**: Complete API and CLI documentation

## Validation Steps

1. **API Testing**: Verify all endpoints work correctly
2. **CLI Testing**: Test all commands and options
3. **Integration Testing**: Verify integration with all packages
4. **Performance Testing**: Ensure system meets performance requirements
5. **Documentation Review**: Verify completeness and accuracy

## Notes

- Design for both programmatic and interactive use
- Implement proper authentication and authorization
- Plan for API versioning and backward compatibility
- Consider GraphQL API for complex queries
- Implement proper rate limiting and throttling
- Plan for multi-tenant deployments
