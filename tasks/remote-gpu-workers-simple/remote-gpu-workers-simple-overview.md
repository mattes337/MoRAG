# Remote GPU Workers - Simplified Implementation Plan

## Overview

This document outlines a simplified implementation plan for adding remote GPU worker support to the MoRAG system. This approach focuses on getting a working solution quickly with minimal complexity, using API key-based user identification and HTTP file transfer for remote processing.

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
Use API key-based user identification to route tasks to user-specific GPU workers. Remote workers only perform heavy processing (GPU/CPU intensive tasks) and return markdown content to the server, which handles all external service connections (Qdrant, Gemini, etc.).

### Key Simplifications vs Advanced Plan
1. **API Key-Based User Identification**: Use API keys to match users with their dedicated workers
2. **HTTP File Transfer Only**: Remove NFS/shared storage, use HTTP downloads for file access
3. **Server-Side External Services**: Workers only do processing, server handles Qdrant/Gemini/etc.
4. **Direct URL Processing**: URLs (web, YouTube) passed directly to workers for processing
5. **Simple Worker Authentication**: API key-based worker authentication and task routing

### Implementation Phases

#### Phase 1: API Key Authentication & User Routing (Tasks 1-2)
1. **Queue Architecture Setup** - Configure user-specific queues with API key routing
2. **API Key Integration** - Add API key authentication and user-based task routing

#### Phase 2: Worker Configuration & HTTP Transfer (Tasks 3-4)
3. **GPU Worker Configuration** - Create GPU worker configs with HTTP file download
4. **Task Routing Logic** - Implement user-specific queue selection and HTTP file transfer

#### Phase 3: URL Processing & Documentation (Tasks 5-6)
5. **URL Processing Configuration** - Configure direct URL processing with cookie support
6. **Documentation & Testing** - Complete setup guides and test scripts

## Technical Implementation Details

### Queue Architecture
- **User-Specific GPU Queues**: `gpu-tasks-{user_id}` - For user's GPU-intensive processing
- **User-Specific CPU Queues**: `cpu-tasks-{user_id}` - For user's CPU processing and fallback
- **Default Queue**: `celery` - For anonymous/legacy processing

### API Changes
Add API key authentication to all endpoints:
- **Header**: `Authorization: Bearer {api_key}` or `X-API-Key: {api_key}`
- **User Identification**: API key maps to user_id for queue routing
- **Queue Selection**: Tasks routed to user-specific queues based on API key
- **Worker Matching**: Workers connect with API keys and only process their user's tasks

### Worker Types
- **Remote GPU Workers**: Connect with user API key, consume from user's GPU queue only
- **Remote CPU Workers**: Connect with user API key, consume from user's CPU queue only
- **Local Workers**: Continue using default queue for anonymous processing

### File Transfer Strategy
**HTTP File Transfer Only**:
- Workers download files from server via authenticated HTTP endpoints
- Server provides temporary download URLs with authentication tokens
- Workers process files locally and return markdown content only
- No shared storage required - workers are completely independent

### External Service Isolation
**Server-Side Only**:
- **Qdrant**: Vector storage handled only by main server
- **Gemini**: Embedding and LLM calls handled only by main server
- **Workers**: Only perform heavy processing (transcription, conversion, etc.)
- **Return Format**: Workers return structured markdown + metadata to server

## Success Criteria

1. **User Isolation**: Each user's tasks only processed by their dedicated workers
2. **API Key Authentication**: Workers authenticate with API keys and process only their user's tasks
3. **HTTP File Transfer**: Workers download files via HTTP without shared storage
4. **External Service Isolation**: Workers only do processing, server handles all external services
5. **URL Processing**: Workers can process URLs (web, YouTube) directly with cookie support
6. **Performance**: GPU workers show measurable performance improvement for heavy tasks
7. **Simplicity**: Setup process takes < 30 minutes for new GPU worker
8. **Documentation**: Complete setup guide with troubleshooting section

## Implementation Timeline

- **Day 1**: Phase 1 (API key authentication and user-specific queues)
- **Day 2**: Phase 2 (Worker configuration and HTTP file transfer)
- **Day 3**: Phase 3 (URL processing and cookie configuration)
- **Day 4**: Testing and refinement

## Dependencies

### External Dependencies
- Redis for queue management (already available)
- HTTP file transfer capability (no shared storage needed)
- GPU-enabled machine with CUDA drivers
- API key management system

### Internal Dependencies
- Existing Celery task system
- Current file upload/handling system
- MoRAG modular architecture (audio, video, document packages)
- User authentication and API key system

## Risk Mitigation

1. **User Worker Unavailable**: Tasks fallback to default queue for processing
2. **API Key Issues**: Clear authentication error messages and validation
3. **HTTP Transfer Failures**: Retry mechanisms with exponential backoff
4. **File Access Issues**: Temporary download URLs with proper authentication
5. **Configuration Errors**: Comprehensive setup validation scripts
6. **External Service Failures**: Workers isolated from external service outages

## Next Steps

1. Review and approve this simplified implementation plan
2. Begin with Task 1: Queue Architecture Setup with API key routing
3. Set up test environment with API key-based worker
4. Validate HTTP file transfer approach
5. Test URL processing with cookie configuration

---

**Related Documents:**
- [Task 1: Queue Architecture Setup](./task-01-queue-architecture-setup.md)
- [Task 2: API Parameter Addition](./task-02-api-parameter-addition.md)
- [Task 3: GPU Worker Configuration](./task-03-gpu-worker-configuration.md)
- [Task 4: Task Routing Logic](./task-04-task-routing-logic.md)
- [Task 5: Network Configuration](./task-05-network-configuration.md)
- [Task 6: Documentation & Testing](./task-06-documentation-testing.md)
