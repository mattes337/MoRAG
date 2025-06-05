# MoRAG - Modular Retrieval Augmented Generation System

MoRAG is a comprehensive, modular system for processing and indexing various types of content for retrieval-augmented generation (RAG) applications. This is the main integration package that provides a unified interface to all MoRAG components.

## Features

- **Unified API**: Single interface for processing multiple content types
- **Modular Architecture**: Separate packages for different content types
- **Multiple Interfaces**: CLI, REST API, and Python API
- **Background Processing**: Celery-based task queue for long-running operations
- **Vector Storage**: Qdrant integration for similarity search
- **AI Services**: Gemini API integration for embeddings and summarization

## Supported Content Types

- **Web Pages**: Scrape and process web content
- **YouTube Videos**: Download and process video metadata, audio, and subtitles
- **Documents**: PDF, Word, text files with OCR support
- **Audio/Video**: Transcription and processing
- **Images**: OCR and vision analysis

## Installation

```bash
pip install morag
```

Or install from source:

```bash
git clone https://github.com/yourusername/morag
cd morag/packages/morag
pip install -e .
```

## Quick Start

### Python API

```python
import asyncio
from morag import MoRAGAPI

async def main():
    api = MoRAGAPI()
    
    # Process a web page
    result = await api.process_web_page("https://example.com")
    print(f"Extracted content: {result.content[:100]}...")
    
    # Process a YouTube video
    result = await api.process_youtube_video("https://youtube.com/watch?v=...")
    print(f"Video title: {result.metadata['title']}")
    
    # Search for similar content
    results = await api.search("machine learning", limit=5)
    for result in results:
        print(f"Score: {result['score']:.3f} - {result['text'][:50]}...")
    
    await api.cleanup()

asyncio.run(main())
```

### Command Line Interface

```bash
# Process a web page
morag process-url https://example.com --format markdown

# Process a file
morag process-file document.pdf --output result.json

# Search for content
morag search "artificial intelligence" --limit 10

# Check system health
morag health
```

### REST API Server

```bash
# Start the API server
morag-server --host 0.0.0.0 --port 8000

# Or with custom configuration
morag-server --config config.json --workers 4
```

The API will be available at `http://localhost:8000` with interactive documentation at `/docs`.

### Background Worker

```bash
# Start a background worker
morag-worker --concurrency 4

# Or with custom broker
morag-worker --broker redis://localhost:6379/1 --concurrency 2
```

## Configuration

Create a configuration file (JSON format):

```json
{
  "gemini_api_key": "your-gemini-api-key",
  "qdrant_host": "localhost",
  "qdrant_port": 6333,
  "qdrant_collection_name": "morag_vectors",
  "redis_url": "redis://localhost:6379/0",
  "max_workers": 4,
  "chunk_size": 1000,
  "chunk_overlap": 200
}
```

## API Endpoints

### Process Content

- `POST /process/url` - Process content from URL
- `POST /process/file` - Process uploaded file
- `POST /process/web` - Process web page
- `POST /process/youtube` - Process YouTube video
- `POST /process/batch` - Process multiple items

### Search

- `POST /search` - Search for similar content

### System

- `GET /health` - Health check
- `GET /` - API information

## Architecture

MoRAG consists of several modular packages:

- **morag-core**: Core interfaces and models
- **morag-services**: Unified service layer with AI and storage services
- **morag-web**: Web content processing
- **morag-youtube**: YouTube video processing
- **morag**: Main integration package (this package)

## Development

### Setup Development Environment

```bash
git clone https://github.com/yourusername/morag
cd morag
pip install -e packages/morag[dev]
```

### Running Tests

```bash
pytest packages/morag/tests/
```

### Code Quality

```bash
black packages/morag/src/
isort packages/morag/src/
flake8 packages/morag/src/
mypy packages/morag/src/
```

## Docker Deployment

```bash
# Build the image
docker build -t morag:latest .

# Run the API server
docker run -p 8000:8000 morag:latest morag-server

# Run a worker
docker run morag:latest morag-worker
```

## Dependencies

### Required Services

- **Redis**: For task queue and caching
- **Qdrant**: For vector storage and similarity search

### Optional Services

- **PostgreSQL**: For metadata storage (if using database backend)

## Environment Variables

- `GEMINI_API_KEY`: Google Gemini API key
- `QDRANT_HOST`: Qdrant server host
- `QDRANT_PORT`: Qdrant server port
- `QDRANT_API_KEY`: Qdrant API key (if using cloud)
- `REDIS_URL`: Redis connection URL
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

- Documentation: https://morag.readthedocs.io
- Issues: https://github.com/yourusername/morag/issues
- Discussions: https://github.com/yourusername/morag/discussions
