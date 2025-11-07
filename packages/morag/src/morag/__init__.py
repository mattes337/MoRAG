"""MoRAG - Modular Retrieval Augmented Generation System.

MoRAG v2.0 is a comprehensive, modular system for processing and indexing various
content types (documents, audio, video, web content) using AI-powered pipelines
with stage-based processing architecture.

## Quick Start

### Installation
```bash
pip install packages/morag/
```

### Basic Usage
```python
from morag import MoRAGAPI, MoRAGOrchestrator

# Initialize the system
api = MoRAGAPI()
orchestrator = MoRAGOrchestrator()

# Process a document through the full pipeline
result = await orchestrator.process_file(
    file_path="document.pdf",
    stages=["markdown-conversion", "chunker", "fact-generator", "ingestor"]
)
```

### CLI Usage
```bash
# Start the API server
python -m morag server --port 8000

# Process files through CLI
python cli/morag-stages.py process document.pdf --optimize

# Start worker for background processing
python -m morag worker
```

## Key Components

### API Server
FastAPI-based REST API with stage-based endpoints:
```python
from morag import create_app

app = create_app()
# Provides endpoints like /api/v1/stages/{stage}/execute
```

### Orchestrator
High-level pipeline coordination:
```python
from morag import MoRAGOrchestrator

orchestrator = MoRAGOrchestrator()
result = await orchestrator.process_file("input.pdf")
```

### Pipeline Agent
AI-powered processing with structured outputs:
```python
from morag import MoRAGPipelineAgent, IngestionOptions

agent = MoRAGPipelineAgent()
options = IngestionOptions(
    extract_entities=True,
    generate_embeddings=True,
    store_in_graph=True
)

result = await agent.process_document("document.pdf", options)
```

## Content Processing

### Supported Formats
- **Documents**: PDF, Word, Excel, PowerPoint, text files
- **Audio**: MP3, WAV, FLAC (with transcription)
- **Video**: MP4, AVI, MOV (with transcription)
- **Images**: PNG, JPG, SVG (with OCR)
- **Web**: URLs, HTML pages
- **YouTube**: Video URLs with transcript extraction

### Processing Stages
1. **Markdown Conversion** - Universal format conversion
2. **Optimization** - LLM-powered content improvement
3. **Chunking** - Semantic content segmentation
4. **Fact Generation** - Entity and relationship extraction
5. **Ingestion** - Database storage and indexing

## Advanced Features

### Intelligent Retrieval
```python
from morag.endpoints import IntelligentRetrievalService

service = IntelligentRetrievalService()
results = await service.query(
    query="What are the key findings?",
    context_window=5,
    similarity_threshold=0.8
)
```

### Pipeline State Management
```python
from morag.pipeline import PipelineStateManager

state_manager = PipelineStateManager()
state = await state_manager.get_pipeline_state(file_id)
```

### Background Processing
```python
from morag import worker_main

# Start background worker
worker_main()  # Processes tasks via Celery
```

## Configuration

### Environment Variables
```bash
# API Keys
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key

# Database Configuration
QDRANT_URL=http://localhost:6333
NEO4J_URI=bolt://localhost:7687
REDIS_URL=redis://localhost:6379

# Processing Configuration
MORAG_CHUNK_SIZE=2000
MORAG_BATCH_SIZE=50
MORAG_MAX_WORKERS=4
```

### Service Configuration
```python
from morag import ServiceConfig, ContentType

config = ServiceConfig(
    embedding_model="gemini-embedding-001",
    chunk_size=1500,
    supported_types=[ContentType.PDF, ContentType.AUDIO]
)
```

## API Endpoints

### Stage-Based Processing
- `POST /api/v1/stages/{stage}/execute` - Execute single stage
- `POST /api/v1/stages/chain` - Execute stage chain
- `GET /api/v1/stages/list` - List available stages

### File Management
- `POST /api/v1/files/upload` - Upload files for processing
- `GET /api/v1/files/{file_id}/status` - Check processing status
- `GET /api/v1/files/{file_id}/download` - Download processed results

### Intelligent Retrieval
- `POST /api/v1/query/intelligent` - Smart query processing
- `POST /api/v1/query/recursive` - Multi-hop fact retrieval

## Integration Examples

### Web Content Processing
```python
from morag import WebProcessor

processor = WebProcessor()
result = await processor.process_url("https://example.com")
```

### YouTube Video Processing
```python
from morag import YouTubeProcessor

processor = YouTubeProcessor()
result = await processor.process_video("https://youtube.com/watch?v=...")
```

### Batch Processing
```python
from morag import MoRAGOrchestrator

orchestrator = MoRAGOrchestrator()
results = await orchestrator.process_batch([
    "document1.pdf",
    "audio1.mp3",
    "https://example.com"
])
```

## Architecture

MoRAG v2.0 uses a modular architecture with independent packages:
- `morag-core` - Core interfaces and models
- `morag-stages` - Stage-based processing system
- `morag-services` - AI and storage services
- `morag-document` - Document processing
- `morag-audio` - Audio processing with transcription
- `morag-video` - Video processing
- `morag-web` - Web content processing
- `morag-graph` - Knowledge graph management

## Migration from v1.x

**IMPORTANT**: v2.0 has NO backward compatibility with v1.x. See `MIGRATION.md` for
detailed migration instructions.

## Version

Current version: {__version__}
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
