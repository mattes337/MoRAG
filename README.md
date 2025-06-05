# MoRAG - Multimodal Retrieval Augmented Generation

A comprehensive, modular system for processing and indexing various types of content for retrieval-augmented generation (RAG) applications.

## 🎉 Modular Architecture Complete!

MoRAG has been successfully transformed into a modular architecture with separate, independently deployable packages:

### Package Structure
```
packages/
├── morag-core/          # Core interfaces and models
├── morag-services/      # AI and storage services
├── morag-audio/         # Audio processing
├── morag-document/      # Document processing
├── morag-video/         # Video processing
├── morag-image/         # Image processing
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
- **Universal Document Conversion**: Unified framework for converting any document format to structured markdown
- **Intelligent Chunking**: Page-based chunking for documents with configurable strategies (page, semantic, sentence, paragraph)
- **Quality Assessment**: Comprehensive quality scoring for conversion results with fallback mechanisms
- **Production Ready**: Docker support, logging, monitoring, and deployment configurations

## Quick Start

### Prerequisites

- Python 3.9+
- Redis (for task queue)
- Qdrant (vector database)
- Gemini API key

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

### Option 3: Traditional Installation

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

### Environment Setup

Create a `.env` file from the template:
```bash
cp .env.example .env
# Edit .env with your API keys
```

Required variables:
```bash
GOOGLE_API_KEY=your_google_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### Testing Docker Setup

Test infrastructure services:
```bash
# Test Redis
docker exec morag-redis-dev redis-cli ping

# Test Qdrant
curl http://localhost:6333/health

# Check all services
docker-compose -f docker-compose.dev.yml ps
```

### Troubleshooting Docker

- **Port conflicts**: Stop existing services on ports 6379 (Redis), 6333/6334 (Qdrant), 8000 (API)
- **Build issues**: Run `docker-compose build --no-cache` to rebuild images
- **Permission issues**: Ensure Docker has access to the project directory

### Recent Docker Fixes (January 2025)

✅ **Fixed Qdrant Health Checks**: Updated all docker-compose files to use the correct `/healthz` endpoint
✅ **Fixed Whisper Model Permissions**: Resolved permission errors when loading AI models in containers
- Proper home directory setup for the `morag` user
- Configured cache directories for Hugging Face and Whisper models
- Added environment variables: `HF_HOME`, `TRANSFORMERS_CACHE`, `WHISPER_CACHE_DIR`

Test the fixes:
```bash
# Test Docker health checks and permissions
python tests/cli/test-docker-fixes.py
```

For detailed Docker deployment instructions, see [Docker Deployment Guide](docs/DOCKER_DEPLOYMENT.md).

## Usage

For detailed usage examples including Python API, CLI commands, REST API endpoints, and configuration options, see [USAGE.md](USAGE.md).

### Quick Examples

```bash
# Test individual components
python tests/cli/test-audio.py my-audio.mp3
python tests/cli/test-document.py my-document.pdf
python tests/cli/test-web.py https://example.com

# Complete system test
python tests/cli/test-simple.py  # Quick validation (recommended)
python tests/cli/test-all.py     # Comprehensive test with detailed report
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

- [Usage Guide](USAGE.md) - Comprehensive usage examples and API reference
- [API Documentation](http://localhost:8000/docs) (when running locally)
- [Docker Deployment Guide](docs/DOCKER_DEPLOYMENT.md) - Complete Docker deployment instructions
- [Universal Document Conversion](docs/UNIVERSAL_DOCUMENT_CONVERSION.md) - Complete guide to the conversion framework
- [Architecture Guide](docs/ARCHITECTURE.md) - Detailed system architecture
- [Development Guide](docs/DEVELOPMENT_GUIDE.md) - Development setup and guidelines
- [Page-Based Chunking Guide](docs/page-based-chunking.md)

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
