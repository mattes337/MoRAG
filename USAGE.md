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
