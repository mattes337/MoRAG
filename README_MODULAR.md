# MoRAG - Modular Retrieval Augmented Generation

A comprehensive, modular system for processing and indexing various types of content for retrieval-augmented generation (RAG) applications.

## 🎉 Modular Architecture Complete!

MoRAG has been successfully transformed into a modular architecture with separate, independently deployable packages:

### Package Structure
```
packages/
├── morag-core/          # Core interfaces and models
├── morag-services/      # AI and storage services  
├── morag-web/          # Web content processing
├── morag-youtube/      # YouTube video processing
└── morag/              # Main integration package
```

## Features

- **Modular Design**: Independent packages for different content types
- **Multi-format Support**: Process PDFs, audio, video, web pages, YouTube videos, and more
- **Advanced AI Integration**: Gemini API for embeddings and summarization
- **Vector Storage**: Qdrant integration for similarity search
- **Background Processing**: Celery-based task queue for scalable processing
- **Docker Support**: Complete containerization with docker-compose
- **Multiple Interfaces**: REST API, CLI, and Python API
- **Migration Tools**: Scripts to transition from monolithic to modular

## Quick Start

### Option 1: Using the Main Package (Recommended)

```bash
# Install the main package (includes all components)
pip install packages/morag/

# Or install from source
cd packages/morag
pip install -e .

# Start services with Docker
docker-compose up -d

# Use the CLI
morag process-url https://example.com
morag health
```

### Option 2: Individual Packages

```bash
# Install only what you need
pip install packages/morag-core/
pip install packages/morag-web/
pip install packages/morag-youtube/
```

### Option 3: Migration from Existing Installation

```bash
# Run the migration script
python scripts/migrate_to_modular.py --source . --backup ./backup

# Follow the migration report instructions
cat migration_report.json
```

## Package Overview

### morag-core
Core interfaces, models, and base classes used by all other packages.

```python
from morag_core.models import Document, DocumentChunk
from morag_core.interfaces.processor import BaseProcessor
```

### morag-services  
Unified service layer with AI services (Gemini), vector storage (Qdrant), and processing pipelines.

```python
from morag_services import MoRAGServices, ServiceConfig
from morag_services.storage import QdrantVectorStorage
from morag_services.embedding import GeminiEmbeddingService
```

### morag-web
Web content processing with scraping, conversion, and extraction capabilities.

```python
from morag_web import WebProcessor, WebConverter
result = await WebProcessor().process("https://example.com")
```

### morag-youtube
YouTube video processing with download, metadata extraction, and transcription.

```python
from morag_youtube import YouTubeProcessor
result = await YouTubeProcessor().process("https://youtube.com/watch?v=...")
```

### morag (Main Package)
Integration layer providing unified API, CLI, and orchestration across all packages.

```python
from morag import MoRAGAPI
api = MoRAGAPI()
result = await api.process_web_page("https://example.com")
```

## Usage Examples

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

### CLI Interface

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

### REST API

```bash
# Start the API server
morag-server --host 0.0.0.0 --port 8000

# Process content
curl -X POST "http://localhost:8000/process/url" \
     -H "Content-Type: application/json" \
     -d '{"url": "https://example.com"}'

# Search
curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "machine learning", "limit": 5}'
```

### Background Workers

```bash
# Start background workers
morag-worker --concurrency 4

# Or with custom configuration
morag-worker --broker redis://localhost:6379/1 --concurrency 2
```

## Docker Deployment

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

## Architecture

### Modular Design Benefits

- **Independent Deployment**: Each package can be deployed separately
- **Scalability**: Scale individual components based on load
- **Maintainability**: Clear separation of concerns
- **Flexibility**: Use only the packages you need
- **Testing**: Isolated testing of individual components

## Migration Guide

### From Monolithic to Modular

1. **Backup your existing installation**:
   ```bash
   python scripts/migrate_to_modular.py --source . --backup ./backup
   ```

2. **Install new packages**:
   ```bash
   pip install packages/morag/
   ```

3. **Update your code**:
   ```python
   # Old import
   from morag.processors.web import WebProcessor
   
   # New import  
   from morag_web import WebProcessor
   ```

4. **Update configuration**:
   - Review `morag_config.json`
   - Update environment variables
   - Test with `morag health`

## Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/yourusername/morag.git
cd morag

# Install all packages in development mode
pip install -e packages/morag-core/
pip install -e packages/morag-services/
pip install -e packages/morag-web/
pip install -e packages/morag-youtube/
pip install -e packages/morag/

# Install development dependencies
pip install packages/morag/[dev]

# Run tests
pytest tests/
```

### Testing

```bash
# Run all tests
pytest

# Test specific package
pytest packages/morag-web/tests/

# Integration tests
pytest tests/test_modular_integration.py

# With coverage
pytest --cov=morag --cov-report=html
```

## Supported Formats

- **Documents**: PDF, Word, PowerPoint, text files
- **Audio**: MP3, WAV, FLAC, M4A, OGG  
- **Video**: MP4, AVI, MKV, MOV, WebM
- **Web**: HTML pages, web scraping
- **YouTube**: Video downloads and processing
- **Images**: OCR text extraction

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes in the appropriate package
4. Add tests for your changes
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [packages/morag/README.md](packages/morag/README.md)
- **Issues**: [GitHub Issues](https://github.com/yourusername/morag/issues)
- **Migration Help**: See [scripts/migrate_to_modular.py](scripts/migrate_to_modular.py)

## Acknowledgments

- [Qdrant](https://qdrant.tech/) for vector database
- [Google Gemini](https://ai.google.dev/) for AI services
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Celery](https://docs.celeryproject.org/) for task queue
