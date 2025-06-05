# MoRAG - Multimodal RAG Ingestion Pipeline

A comprehensive, production-ready multimodal RAG (Retrieval Augmented Generation) ingestion pipeline that processes documents, audio, video, web content, and YouTube videos into a searchable vector database.

## Features

- **Universal Document Conversion**: Unified framework for converting any document format to structured markdown
- **Multimodal Processing**: Support for documents (PDF, DOCX, MD), audio, video, images, web content, and YouTube videos
- **Intelligent Chunking**: Page-based chunking for documents with configurable strategies (page, semantic, sentence, paragraph)
- **Quality Assessment**: Comprehensive quality scoring for conversion results with fallback mechanisms
- **Vector Storage**: Qdrant vector database integration for efficient similarity search
- **Async Processing**: Celery-based task queue for scalable background processing
- **API-First**: FastAPI-based REST API with comprehensive documentation
- **Monitoring**: Built-in progress tracking and webhook notifications
- **Production Ready**: Docker support, logging, monitoring, and deployment configurations
- **Modular Architecture**: Separate packages for each processing type with isolated dependencies

## Quick Start

### Prerequisites

- Python 3.9+
- Redis (for task queue)
- Qdrant (vector database)
- Gemini API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/morag.git
cd morag
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e .
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Start the services:
```bash
# Start Redis
redis-server

# Start Qdrant
docker run -d -p 6333:6333 --name morag-qdrant qdrant/qdrant:latest

# Start Celery worker (REQUIRED for task processing)
python scripts/start_worker.py
# OR alternatively:
# celery -A src.morag.core.celery_app worker --loglevel=info --concurrency=4

# Start the API (in a separate terminal)
uvicorn morag.api.main:app --reload
```

**Important**: The Celery worker is required for processing ingestion tasks. Without it, submitted tasks will remain in "pending" status and never complete.

## Docker Deployment

### Quick Docker Start

```bash
# Start all services with Docker Compose
docker-compose up -d

# Check service health
curl http://localhost:8000/health

# View logs
docker-compose logs -f
```

### Docker Deployment Options

1. **Monolithic (Development)**:
   ```bash
   docker-compose up -d
   ```

2. **Development with Hot-Reload**:
   ```bash
   docker-compose -f docker-compose.dev.yml up -d
   ```

3. **Microservices (Production)**:
   ```bash
   docker-compose -f docker-compose.microservices.yml up -d
   ```

For detailed Docker deployment instructions, see [Docker Deployment Guide](docs/DOCKER_DEPLOYMENT.md).

## Testing Individual Packages

MoRAG provides individual test scripts for each processing component:

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

## Architecture

The MoRAG pipeline consists of several key components:

- **API Layer**: FastAPI-based REST endpoints for content ingestion
- **Processing Layer**: Specialized processors for different content types
- **Embedding Layer**: Gemini API integration for text embeddings
- **Storage Layer**: Qdrant vector database for similarity search
- **Task Queue**: Celery for async processing and scalability
- **Modular Packages**: Independent packages for each content type

## Documentation

- [API Documentation](http://localhost:8000/docs) (when running locally)
- [Docker Deployment Guide](docs/DOCKER_DEPLOYMENT.md) - Complete Docker deployment instructions
- [Universal Document Conversion](docs/UNIVERSAL_DOCUMENT_CONVERSION.md) - Complete guide to the conversion framework
- [Architecture Guide](docs/ARCHITECTURE.md) - Detailed system architecture
- [Development Guide](docs/DEVELOPMENT_GUIDE.md) - Development setup and guidelines
- [Page-Based Chunking Guide](docs/page-based-chunking.md)
- [Task Implementation Guide](tasks/README.md)

## Development

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run specific test categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
pytest tests/manual/        # Manual tests

# Run with coverage
pytest --cov=src/morag --cov-report=html

# Test individual components
python tests/cli/test-all.py          # Complete system test
python tests/cli/test-audio.py sample.mp3
python tests/cli/test-document.py sample.pdf
```

### Code Quality

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint code
flake8 src/ tests/
mypy src/
```

### Package Development

Each MoRAG component is a separate package in the `packages/` directory:

```
packages/
├── morag-core/          # Core functionality and models
├── morag-audio/         # Audio processing
├── morag-document/      # Document processing
├── morag-video/         # Video processing
├── morag-image/         # Image processing
├── morag-web/           # Web scraping
├── morag-youtube/       # YouTube processing
├── morag-services/      # Shared services
└── morag-embedding/     # Embedding services
```

Install packages individually for development:
```bash
pip install -e packages/morag-core
pip install -e packages/morag-audio
# ... etc
```

## Troubleshooting

### Tasks Not Processing

If you submit tasks but they remain in "pending" status:

1. **Check if Celery worker is running:**
   ```bash
   # Check for running worker processes
   ps aux | grep celery
   ```

2. **Start the Celery worker:**
   ```bash
   # Using the provided script
   python scripts/start_worker.py

   # Or directly with celery command
   celery -A src.morag.core.celery_app worker --loglevel=info --concurrency=4
   ```

3. **Check Redis connection:**
   ```bash
   # Test Redis connectivity
   redis-cli ping
   # Should return "PONG"
   ```

4. **Check Qdrant connection:**
   ```bash
   # Test Qdrant connectivity
   curl http://localhost:6333/health
   # Should return health status
   ```

5. **Monitor worker logs:**
   ```bash
   # Run worker with debug logging
   celery -A src.morag.core.celery_app worker --loglevel=debug
   ```

### Common Issues

- **Windows Users**: The worker script uses "solo" pool for Windows compatibility
- **Port Conflicts**: Ensure ports 6379 (Redis) and 6333 (Qdrant) are available
- **Environment Variables**: Check that `.env` file is properly configured with API keys

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.
