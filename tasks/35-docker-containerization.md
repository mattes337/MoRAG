# Task 35: Docker Containerization for Modular MoRAG

## Overview

Create individual Docker containers for each MoRAG package component, enabling isolated deployment, scaling, and dependency management. This addresses the original dependency conflict issues while providing a microservices-ready architecture.

## Current State

### Completed
- ✅ Monolithic Docker containers (main app and worker)
- ✅ All packages separated and functional
- ✅ Package-specific dependencies defined

### Remaining Work
- [ ] Create individual Dockerfiles for each package
- [ ] Create specialized Docker Compose configurations
- [ ] Implement container orchestration
- [ ] Create deployment scripts and documentation
- [ ] Optimize container sizes and build times
- [ ] Implement health checks and monitoring

## Implementation Steps

### Step 1: Create Package-Specific Dockerfiles

**Containers to create:**
1. `morag-core` - Base container with core utilities
2. `morag-document` - Document processing container
3. `morag-audio` - Audio processing container
4. `morag-video` - Video processing container
5. `morag-image` - Image processing container
6. `morag-web` - Web scraping container
7. `morag-youtube` - YouTube processing container
8. `morag-services` - Services orchestration container
9. `morag-api` - Main API container
10. `morag-worker` - Generic worker container

### Step 2: Document Processing Container

**File:** `packages/morag-document/Dockerfile`

**Features:**
- Python 3.11 Alpine base
- Document processing dependencies (docling, unstructured)
- Office suite support (LibreOffice)
- PDF processing tools
- Optimized for document workloads

**Dependencies:**
- docling for PDF processing
- python-docx for Word documents
- openpyxl for Excel files
- python-pptx for PowerPoint
- BeautifulSoup for HTML

### Step 3: Audio Processing Container

**File:** `packages/morag-audio/Dockerfile`

**Features:**
- Python 3.11 with audio libraries
- FFmpeg for audio conversion
- Whisper models for transcription
- Speaker diarization tools
- GPU support (optional)

**Dependencies:**
- faster-whisper for transcription
- pyannote.audio for speaker diarization
- librosa for audio analysis
- soundfile for audio I/O
- FFmpeg system package

### Step 4: Video Processing Container

**File:** `packages/morag-video/Dockerfile`

**Features:**
- Python 3.11 with video libraries
- FFmpeg with full codec support
- OpenCV for computer vision
- Scene detection tools
- GPU support (optional)

**Dependencies:**
- opencv-python for video analysis
- scenedetect for scene detection
- FFmpeg with full codec support
- Pillow for image processing

### Step 5: Image Processing Container

**File:** `packages/morag-image/Dockerfile`

**Features:**
- Python 3.11 with image libraries
- Tesseract OCR engine
- Computer vision libraries
- GPU support for AI models

**Dependencies:**
- tesseract-ocr for text extraction
- easyocr for enhanced OCR
- Pillow for image processing
- opencv-python for computer vision

### Step 6: Web Processing Container

**File:** `packages/morag-web/Dockerfile`

**Features:**
- Python 3.11 with web libraries
- Chromium browser for dynamic content
- Playwright for browser automation
- Network tools and utilities

**Dependencies:**
- playwright for browser automation
- beautifulsoup4 for HTML parsing
- requests/httpx for HTTP requests
- lxml for XML processing

### Step 7: Services Container

**File:** `packages/morag-services/Dockerfile`

**Features:**
- Python 3.11 with service libraries
- Redis client for caching
- Qdrant client for vector storage
- Monitoring and health check tools

**Dependencies:**
- qdrant-client for vector storage
- redis for caching
- prometheus-client for metrics
- structlog for logging

### Step 8: Create Docker Compose Orchestration

**Files to create:**
- `docker-compose.yml` - Main orchestration
- `docker-compose.dev.yml` - Development environment
- `docker-compose.prod.yml` - Production environment
- `docker-compose.scale.yml` - Scaling configuration

**Services Configuration:**
```yaml
version: '3.8'
services:
  # Core Infrastructure
  redis:
    image: redis:7-alpine
    
  qdrant:
    image: qdrant/qdrant:latest
    
  # MoRAG Services
  morag-api:
    build: ./packages/morag
    depends_on: [redis, qdrant, morag-services]
    
  morag-services:
    build: ./packages/morag-services
    depends_on: [redis, qdrant]
    
  # Processing Workers
  document-worker:
    build: ./packages/morag-document
    depends_on: [redis, morag-services]
    
  audio-worker:
    build: ./packages/morag-audio
    depends_on: [redis, morag-services]
    
  video-worker:
    build: ./packages/morag-video
    depends_on: [redis, morag-services]
    
  image-worker:
    build: ./packages/morag-image
    depends_on: [redis, morag-services]
    
  web-worker:
    build: ./packages/morag-web
    depends_on: [redis, morag-services]
    
  youtube-worker:
    build: ./packages/morag-youtube
    depends_on: [redis, morag-services]
```

### Step 9: Implement Container Optimization

**Optimization Strategies:**
1. **Multi-stage builds** to reduce final image size
2. **Alpine Linux** base images for smaller footprint
3. **Dependency caching** for faster builds
4. **Layer optimization** to maximize cache hits
5. **Security scanning** for vulnerability management

**Example Multi-stage Dockerfile:**
```dockerfile
# Build stage
FROM python:3.11-alpine AS builder
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Runtime stage
FROM python:3.11-alpine
COPY --from=builder /root/.local /root/.local
COPY src/ /app/src/
WORKDIR /app
CMD ["python", "-m", "morag_package"]
```

### Step 10: Create Deployment Scripts

**Files to create:**
- `scripts/deploy-containers.sh` - Container deployment script
- `scripts/scale-workers.sh` - Worker scaling script
- `scripts/health-check.sh` - Health monitoring script
- `scripts/backup-containers.sh` - Backup and recovery script

## Container Architecture Benefits

### Isolated Dependencies
- Each converter has its own container with specific dependencies
- No more dependency conflicts between different processing types
- Easy to update individual components without affecting others

### Scalability
- Scale individual converters based on demand
- More document workers during heavy document processing
- Fewer video workers when video processing is light
- Independent resource allocation per container type

### Resource Optimization
- Allocate appropriate resources per converter type
- Video processing containers get more CPU/memory
- Document processing containers optimized for I/O
- Web scraping containers with network optimization

### Simplified Maintenance
- Update converters independently
- Roll back individual components if issues arise
- Test new versions in isolation
- Easier debugging and troubleshooting

## Testing Requirements

### Container Tests
- [ ] Test individual container builds
- [ ] Test container startup and health checks
- [ ] Test inter-container communication
- [ ] Test resource usage and limits
- [ ] Test scaling and load balancing

### Integration Tests
- [ ] Test complete Docker Compose deployment
- [ ] Test processing workflows across containers
- [ ] Test failure scenarios and recovery
- [ ] Test data persistence and volumes
- [ ] Test network connectivity and security

### Performance Tests
- [ ] Test container startup times
- [ ] Test processing performance vs monolithic
- [ ] Test resource utilization efficiency
- [ ] Test scaling response times
- [ ] Test memory and storage usage

## Success Criteria

1. **Functional Isolation**: Each container works independently
2. **Dependency Resolution**: No dependency conflicts between containers
3. **Scalability**: Easy horizontal scaling of individual components
4. **Performance**: No significant performance degradation
5. **Maintainability**: Easy updates and maintenance of individual containers
6. **Resource Efficiency**: Optimal resource utilization per container type

## Deployment Configurations

### Development Environment
- Single instance of each container
- Shared volumes for development
- Hot reloading for code changes
- Debug logging enabled

### Production Environment
- Multiple instances of high-demand containers
- Persistent volumes for data storage
- Load balancing and health checks
- Production logging and monitoring

### Scaling Configuration
- Auto-scaling based on queue length
- Resource limits and requests
- Priority-based scheduling
- Graceful shutdown handling

## Validation Steps

1. **Container Build**: Verify all containers build successfully
2. **Orchestration**: Test Docker Compose deployment
3. **Processing**: Verify all processing types work correctly
4. **Scaling**: Test horizontal scaling of workers
5. **Monitoring**: Verify health checks and metrics collection

## Notes

- Use consistent base images across containers
- Implement proper security practices (non-root users, minimal privileges)
- Plan for secrets management and configuration
- Consider using container registries for distribution
- Implement proper logging and monitoring integration
- Plan for backup and disaster recovery scenarios
