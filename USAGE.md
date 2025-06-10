# MoRAG Usage Guide

This guide provides comprehensive examples for using MoRAG in different ways: Python API, CLI, and REST API.

## Python API Usage

### Unified API (Recommended)

```python
import asyncio
from morag import MoRAGAPI

async def main():
    api = MoRAGAPI()
    
    # Process different content types
    web_result = await api.process_web_page("https://example.com")
    youtube_result = await api.process_youtube_video("https://youtube.com/watch?v=...")
    doc_result = await api.process_document("document.pdf")
    
    # Search for similar content
    results = await api.search("machine learning", limit=5)
    
    await api.cleanup()

asyncio.run(main())
```

### Individual Package Usage

#### morag-core
Core interfaces, models, and base classes used by all other packages.

```python
from morag_core.models import Document, DocumentChunk
from morag_core.interfaces.processor import BaseProcessor
```

#### morag-services  
Unified service layer with AI services (Gemini), vector storage (Qdrant), and processing pipelines.

```python
from morag_services import MoRAGServices, ServiceConfig
from morag_services.storage import QdrantVectorStorage
from morag_services.embedding import GeminiEmbeddingService
```

#### morag-web
Web content processing with scraping, conversion, and extraction capabilities.

```python
from morag_web import WebProcessor, WebConverter
result = await WebProcessor().process("https://example.com")
```

#### morag-youtube
YouTube video processing with download, metadata extraction, and transcription.

```python
from morag_youtube import YouTubeProcessor
result = await YouTubeProcessor().process("https://youtube.com/watch?v=...")
```

#### morag (Main Package)
Integration layer providing unified API, CLI, and orchestration across all packages.

```python
from morag import MoRAGAPI
api = MoRAGAPI()
result = await api.process_web_page("https://example.com")
```

## CLI Interface

### Basic Commands

```bash
# Process different content types
morag process-url https://example.com --format markdown
morag process-file document.pdf --output result.json

# Search content
morag search "artificial intelligence" --limit 10

# Health check
morag health

# Batch processing
morag batch input.json --output results.json
```

### Background Workers

```bash
# Start background workers
morag-worker --concurrency 4

# Or with custom configuration
morag-worker --broker redis://localhost:6379/1 --concurrency 2
```

## REST API Usage

### Starting the API Server

```bash
# Start the API server
morag-server --host 0.0.0.0 --port 8000
```

### API Endpoints

#### Process Content

```bash
# Process URL
curl -X POST "http://localhost:8000/process/url" \
     -H "Content-Type: application/json" \
     -d '{"url": "https://example.com"}'

# Process file upload
curl -X POST "http://localhost:8000/process/file" \
     -F "file=@document.pdf"

# Process YouTube video
curl -X POST "http://localhost:8000/process/youtube" \
     -H "Content-Type: application/json" \
     -d '{"url": "https://youtube.com/watch?v=VIDEO_ID"}'
```

#### Search Content

```bash
curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "machine learning", "limit": 5}'
```

#### Health Check

```bash
curl -X GET "http://localhost:8000/health"
```

## Remote Processing

### Overview

MoRAG supports remote processing for computationally intensive tasks (audio and video processing). This allows you to offload processing to remote workers with better hardware capabilities, including GPU support.

### Client Side Usage

#### REST API with Remote Processing

```bash
# Upload audio file with remote processing enabled
curl -X POST "http://localhost:8000/api/v1/ingest/file" \
     -F "file=@audio.mp3" \
     -F "remote=true" \
     -F "content_type=audio" \
     -F "webhook_url=http://my-app.com/webhook"

# Upload video file with remote processing
curl -X POST "http://localhost:8000/api/v1/ingest/file" \
     -F "file=@video.mp4" \
     -F "remote=true" \
     -F "content_type=video" \
     -F "fallback_to_local=true"
```

#### Python API with Remote Processing

```python
import asyncio
from morag import MoRAGAPI

async def main():
    api = MoRAGAPI()

    # Process audio with remote workers
    result = await api.ingest_file(
        "audio.mp3",
        content_type="audio",
        options={
            "remote": True,
            "webhook_url": "http://my-app.com/webhook",
            "fallback_to_local": True
        }
    )

    print(f"Task ID: {result.task_id}")

    # Monitor task progress
    status = await api.get_task_status(result.task_id)
    print(f"Status: {status}")

asyncio.run(main())
```

#### CLI with Remote Processing

```bash
# Process audio with remote workers
python tests/cli/test-audio.py my-audio.mp3 --ingest --remote

# Process video with remote workers and webhook
python tests/cli/test-video.py my-video.mp4 --ingest --remote --webhook-url https://my-app.com/webhook

# Test remote conversion system
python cli/test-remote-conversion.py --test all
```

### Remote Worker Setup

#### Environment Configuration

```bash
# Required environment variables for remote workers
export MORAG_WORKER_ID="gpu-worker-01"
export MORAG_API_BASE_URL="https://your-morag-server.com"
export MORAG_WORKER_CONTENT_TYPES="audio,video"
export MORAG_WORKER_POLL_INTERVAL="10"
export MORAG_WORKER_MAX_CONCURRENT_JOBS="2"
export MORAG_TEMP_DIR="/tmp/morag_remote"

# Optional API authentication
export MORAG_API_KEY="your-api-key"
```

#### Configuration File

Create `remote_converter_config.yaml`:

```yaml
worker_id: "gpu-worker-01"
api_base_url: "https://your-morag-server.com"
api_key: "your-api-key-here"
content_types: ["audio", "video"]
poll_interval: 10
max_concurrent_jobs: 2
log_level: "INFO"
temp_dir: "/tmp/morag_remote"
```

#### Starting Remote Worker

```bash
# Install remote converter dependencies
pip install -e packages/morag-core
pip install -e packages/morag-audio
pip install -e packages/morag-video
pip install requests pyyaml python-dotenv structlog

# Start remote worker
python tools/remote-converter/cli.py --config remote_converter_config.yaml

# Or with command line options
python tools/remote-converter/cli.py \
    --worker-id gpu-worker-01 \
    --api-url https://your-morag-server.com \
    --content-types audio,video \
    --max-jobs 2
```

### Remote Job Management

#### Check Job Status

```bash
# Get specific job status
curl "http://localhost:8000/api/v1/remote-jobs/{job_id}/status"

# Poll for available jobs (worker perspective)
curl "http://localhost:8000/api/v1/remote-jobs/poll?worker_id=worker-1&content_types=audio,video&max_jobs=1"
```

#### Monitor Remote Processing

```python
import requests

# Check remote job status
def check_remote_job_status(job_id):
    response = requests.get(f"http://localhost:8000/api/v1/remote-jobs/{job_id}/status")
    return response.json()

# List all remote jobs (admin endpoint)
def list_remote_jobs():
    response = requests.get("http://localhost:8000/api/v1/remote-jobs/")
    return response.json()
```

### Remote Processing Features

#### Automatic Fallback

```python
# Enable automatic fallback to local processing
result = await api.ingest_file(
    "audio.mp3",
    options={
        "remote": True,
        "fallback_to_local": True  # Falls back if no remote workers available
    }
)
```

#### Content Type Support

- **Supported for Remote Processing**: `audio`, `video`
- **Local Processing Only**: `document`, `image`, `web`, `youtube`

#### Job Timeouts

- **Audio jobs**: 30 minutes default
- **Video jobs**: 1 hour default
- **Configurable**: Set custom timeouts in task options

```python
result = await api.ingest_file(
    "long-video.mp4",
    options={
        "remote": True,
        "timeout_seconds": 7200  # 2 hours
    }
)
```

## Configuration

### Environment Variables

```bash
# Required
GEMINI_API_KEY=your-gemini-api-key

# Vector Database
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=your-qdrant-api-key  # For cloud

# Task Queue
REDIS_URL=redis://localhost:6379/0

# Optional
LOG_LEVEL=INFO
MAX_WORKERS=4
CHUNK_SIZE=1000

# Remote Processing (Optional)
MORAG_REMOTE_JOBS_DATA_DIR=/app/data/remote_jobs
MORAG_WORKER_ID=remote-worker-01
MORAG_API_BASE_URL=http://localhost:8000
MORAG_WORKER_CONTENT_TYPES=audio,video
MORAG_WORKER_POLL_INTERVAL=10
MORAG_WORKER_MAX_CONCURRENT_JOBS=2
```

### Configuration File

```json
{
  "gemini_api_key": "your-api-key",
  "qdrant_host": "localhost", 
  "qdrant_port": 6333,
  "redis_url": "redis://localhost:6379/0",
  "max_workers": 4,
  "chunk_size": 1000,
  "chunk_overlap": 200
}
```

## Testing Individual Components

### Audio Processing
```bash
python tests/cli/test-audio.py my-audio.mp3
python tests/cli/test-audio.py recording.wav
python tests/cli/test-audio.py video.mp4  # Extract audio from video
```

### Document Processing
```bash
python tests/cli/test-document.py my-document.pdf
python tests/cli/test-document.py presentation.pptx
python tests/cli/test-document.py spreadsheet.xlsx
```

### Video Processing
```bash
python tests/cli/test-video.py my-video.mp4
python tests/cli/test-video.py recording.avi
```

### Image Processing
```bash
python tests/cli/test-image.py my-image.jpg
python tests/cli/test-image.py screenshot.png
```

### Web Content Processing
```bash
python tests/cli/test-web.py https://example.com
python tests/cli/test-web.py https://en.wikipedia.org/wiki/Python
```

### YouTube Processing
```bash
python tests/cli/test-youtube.py https://www.youtube.com/watch?v=VIDEO_ID
python tests/cli/test-youtube.py https://youtu.be/VIDEO_ID
```

### Complete System Test
```bash
python tests/cli/test-simple.py  # Quick system validation (recommended)
python tests/cli/test-all.py     # Comprehensive system test with detailed report
```

## Docker Usage

### Complete System with Docker Compose

```bash
# Start all services (API, workers, Redis, Qdrant)
docker-compose up -d

# View logs
docker-compose logs -f morag-api

# Scale workers
docker-compose up -d --scale morag-worker=4

# Stop services
docker-compose down
```

### Individual Package Containers

```bash
# Build and run web processing service
cd packages/morag-web
docker build -t morag-web .
docker run -p 8001:8001 morag-web

# Build and run YouTube processing service  
cd packages/morag-youtube
docker build -t morag-youtube .
docker run -p 8002:8002 morag-youtube
```

## Supported Formats

- **Documents**: PDF, Word, PowerPoint, text files
- **Audio**: MP3, WAV, FLAC, M4A, OGG  
- **Video**: MP4, AVI, MKV, MOV, WebM
- **Web**: HTML pages, web scraping
- **YouTube**: Video downloads and processing
- **Images**: OCR text extraction

## Migration from Monolithic to Modular

### Update Import Statements

```python
# Old import
from morag.processors.web import WebProcessor

# New import  
from morag_web import WebProcessor
```

### Configuration Updates

1. Review `morag_config.json`
2. Update environment variables
3. Test with `morag health`

For more detailed information, see the main [README.md](README.md) and individual package documentation.
