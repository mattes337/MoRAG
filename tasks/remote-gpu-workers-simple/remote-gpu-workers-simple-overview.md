# Remote GPU Workers - Simplified Implementation Plan

## Overview

This document outlines a simplified implementation plan for adding remote GPU worker support to the MoRAG system. This approach focuses on getting a working solution quickly with minimal complexity, using a simple `gpu` boolean flag in API endpoints to route tasks to appropriate workers.

## Current Architecture Analysis

### Existing System
- **Main Server**: CPU-only server running FastAPI, Redis, and Qdrant
- **Local Workers**: Celery workers running on the same machine as the server
- **Task Queue**: Redis-based Celery task queue with single queue ("celery")
- **File Handling**: Shared volume system (`/app/temp`) for file access between API and workers
- **Processing Types**: Audio (Whisper), Video (FFmpeg), Document (docling), Web, YouTube, Image

### Current API Endpoints
- **Processing**: `/process/*` - Immediate processing (no storage)
- **Ingestion**: `/api/v1/ingest/*` - Background processing with vector storage
- **Task Management**: `/api/v1/status/*` - Task status and management

## Simplified Implementation Strategy

### Core Concept
Add an optional `gpu` boolean parameter to existing API endpoints that routes tasks to GPU workers when `true`, while maintaining backward compatibility with CPU workers as the default.

### Key Simplifications vs Advanced Plan
1. **No Complex Priority Queues**: Use simple queue routing (gpu-queue vs default queue)
2. **No File Transfer Service**: Use shared network storage or simple HTTP file transfer
3. **No Worker Registration System**: Manual worker configuration
4. **No Advanced Monitoring**: Basic Celery monitoring only
5. **No Authentication**: Simple network-based security

### Implementation Phases

#### Phase 1: Basic Queue Routing (Tasks 1-2)
1. **Queue Architecture Setup** - Configure separate GPU and CPU queues
2. **API Parameter Addition** - Add `gpu` parameter to all relevant endpoints

#### Phase 2: Worker Configuration (Tasks 3-4)
3. **GPU Worker Configuration** - Create GPU worker startup scripts and configs
4. **Task Routing Logic** - Implement queue selection based on `gpu` parameter

#### Phase 3: Network Setup & Documentation (Tasks 5-6)
5. **Network Configuration** - Configure shared storage and network access
6. **Documentation & Testing** - Complete setup guides and test scripts

## Technical Implementation Details

### Queue Architecture
- **GPU Queue**: `gpu-tasks` - For GPU-intensive processing
- **CPU Queue**: `celery` (default) - For CPU processing and fallback

### API Changes
Add optional `gpu` parameter to these endpoints:
- `POST /process/file?gpu=true`
- `POST /process/url?gpu=true`
- `POST /api/v1/ingest/file` (form parameter `gpu`)
- `POST /api/v1/ingest/url` (JSON parameter `gpu`)
- `POST /api/v1/ingest/batch` (JSON parameter `gpu`)

### Worker Types
- **GPU Workers**: Remote workers with CUDA/GPU capabilities, consume from `gpu-tasks` queue
- **CPU Workers**: Local workers, consume from `celery` queue (default)

### File Sharing Strategy
**Option A (Recommended)**: Shared Network Storage
- Mount same storage volume on both server and GPU workers
- Use NFS, SMB, or cloud storage (S3, etc.)

**Option B**: Simple HTTP File Transfer
- GPU workers download files from server via HTTP
- GPU workers upload results back via HTTP

## Success Criteria

1. **Functionality**: GPU workers process audio/video tasks when `gpu=true` parameter is used
2. **Fallback**: System continues working with CPU workers when GPU workers unavailable
3. **Performance**: GPU tasks show measurable performance improvement
4. **Simplicity**: Setup process takes < 30 minutes for new GPU worker
5. **Documentation**: Complete setup guide with troubleshooting section

## Implementation Timeline

- **Day 1**: Phase 1 (Queue setup and API changes)
- **Day 2**: Phase 2 (Worker configuration and routing)
- **Day 3**: Phase 3 (Network setup and documentation)
- **Day 4**: Testing and refinement

## Dependencies

### External Dependencies
- Redis for queue management (already available)
- Shared storage solution (NFS/SMB/S3) OR HTTP file transfer capability
- GPU-enabled machine with CUDA drivers

### Internal Dependencies
- Existing Celery task system
- Current file upload/handling system
- MoRAG modular architecture (audio, video, document packages)

## Risk Mitigation

1. **GPU Worker Unavailable**: Tasks fallback to CPU workers automatically
2. **Network Issues**: Retry mechanisms and clear error messages
3. **File Access Issues**: Validation and clear error reporting
4. **Configuration Errors**: Comprehensive setup validation scripts

## Next Steps

1. Review and approve this simplified implementation plan
2. Begin with Task 1: Queue Architecture Setup
3. Set up test environment with one GPU worker
4. Validate file sharing approach

---

**Related Documents:**
- [Task 1: Queue Architecture Setup](./task-01-queue-architecture-setup.md)
- [Task 2: API Parameter Addition](./task-02-api-parameter-addition.md)
- [Task 3: GPU Worker Configuration](./task-03-gpu-worker-configuration.md)
- [Task 4: Task Routing Logic](./task-04-task-routing-logic.md)
- [Task 5: Network Configuration](./task-05-network-configuration.md)
- [Task 6: Documentation & Testing](./task-06-documentation-testing.md)
