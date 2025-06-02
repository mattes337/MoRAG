# MoRAG - Multimodal RAG Ingestion Pipeline

A comprehensive, production-ready multimodal RAG (Retrieval Augmented Generation) ingestion pipeline that processes documents, audio, video, web content, and YouTube videos into a searchable vector database.

## Features

- **Multimodal Processing**: Support for documents (PDF, DOCX, MD), audio, video, images, web content, and YouTube videos
- **Intelligent Chunking**: Semantic text chunking with spaCy for optimal retrieval
- **Vector Storage**: Qdrant vector database integration for efficient similarity search
- **Async Processing**: Celery-based task queue for scalable background processing
- **API-First**: FastAPI-based REST API with comprehensive documentation
- **Monitoring**: Built-in progress tracking and webhook notifications
- **Production Ready**: Docker support, logging, monitoring, and deployment configurations

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
docker run -p 6333:6333 qdrant/qdrant:latest

# Start the API
uvicorn morag.api.main:app --reload
```

## Architecture

The MoRAG pipeline consists of several key components:

- **API Layer**: FastAPI-based REST endpoints for content ingestion
- **Processing Layer**: Specialized processors for different content types
- **Embedding Layer**: Gemini API integration for text embeddings
- **Storage Layer**: Qdrant vector database for similarity search
- **Task Queue**: Celery for async processing and scalability

## Documentation

- [API Documentation](http://localhost:8000/docs) (when running locally)
- [Task Implementation Guide](tasks/README.md)
- [Deployment Guide](docs/deployment.md)

## Development

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=src/morag --cov-report=html
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

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.
