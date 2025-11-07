# MoRAG - Modular Retrieval Augmented Generation System

MoRAG is a comprehensive, modular system for processing and indexing various types of content for retrieval-augmented generation (RAG) applications. This is the main integration package that provides a unified interface to all MoRAG components.

## Features

- **Unified API**: Single interface for processing multiple content types
- **Modular Architecture**: Separate packages for different content types
- **Multiple Interfaces**: CLI, REST API, and Python API
- **Background Processing**: Celery-based task queue for long-running operations
- **Vector Storage**: Qdrant integration for similarity search
- **AI Services**: Gemini API integration for embeddings and summarization

### Enhanced Query & Retrieval (NEW in v2)
- **Graph-Augmented RAG**: Combines vector search with knowledge graph traversal
- **Multiple Query Types**: Simple, entity-focused, relation-focused, multi-hop, and analytical queries
- **Flexible Expansion Strategies**: Direct neighbors, breadth-first, shortest path, and adaptive expansion
- **Result Fusion**: Multiple strategies for combining vector and graph results (weighted, RRF, adaptive)
- **Entity & Relationship Queries**: Direct exploration of knowledge graph entities and relationships
- **Graph Analytics**: Statistics and insights about the knowledge graph structure
- **Streaming Support**: Real-time result streaming for large queries
- **Multi-hop Reasoning**: Complex reasoning across multiple entities and relationships
- **Graceful Degradation**: Fallback to vector-only search when graph components unavailable

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

### Enhanced Query API (v2)

```python
import requests

# Basic enhanced query
response = requests.post("http://localhost:8000/api/v2/query", json={
    "query": "How does machine learning relate to artificial intelligence?",
    "query_type": "entity_focused",
    "max_results": 10,
    "include_graph_context": True
})

result = response.json()
print(f"Found {len(result['results'])} results")
print(f"Graph entities: {len(result['graph_context']['entities'])}")

# Multi-hop reasoning query
response = requests.post("http://localhost:8000/api/v2/query", json={
    "query": "What are the applications of deep learning in computer vision?",
    "query_type": "multi_hop",
    "expansion_strategy": "adaptive",
    "enable_multi_hop": True,
    "include_reasoning_path": True
})

# Entity exploration
response = requests.post("http://localhost:8000/api/v2/entity/query", json={
    "entity_name": "neural networks",
    "include_relations": True,
    "relation_depth": 2
})

# Graph traversal
response = requests.post("http://localhost:8000/api/v2/graph/traverse", json={
    "start_entity": "machine_learning",
    "end_entity": "computer_vision",
    "traversal_type": "shortest_path"
})
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
  "qdrant_collection_name": "morag_documents",
  "redis_url": "redis://localhost:6379/0",
  "max_workers": 4,
  "chunk_size": 1000,
  "chunk_overlap": 200
}
```

## API Endpoints

### Enhanced Query API (v2) - Graph-Augmented RAG

- `POST /api/v2/query` - Enhanced query with graph-augmented retrieval
- `POST /api/v2/query/stream` - Streaming enhanced query for real-time results
- `POST /api/v2/entity/query` - Query specific entities and relationships
- `POST /api/v2/graph/traverse` - Perform graph traversal between entities
- `GET /api/v2/graph/analytics` - Get graph analytics and statistics

### Legacy Query API (v1) - Deprecated

- `POST /api/v1/query` - Legacy query endpoint (deprecated, use v2)
- `GET /api/v1/migration-guide` - Migration guidance from v1 to v2
- `GET /api/v1/health` - Legacy health check with deprecation notice

### Unified Processing Endpoint (Recommended)

**`POST /api/v1/process`** - Single endpoint for all processing needs

**Modes:**
- `mode=convert` - Fast markdown conversion for UI preview
- `mode=process` - Full processing with immediate results
- `mode=ingest` - Full processing + vector storage (background)

**Source Types:**
- `source_type=file` - Upload file via multipart form
- `source_type=url` - Process content from URL
- `source_type=batch` - Process multiple items

**Example Usage:**
```bash
# Convert file to markdown
curl -X POST "/api/v1/process" \
  -F "file=@document.pdf" \
  -F 'request_data={"mode":"convert","source_type":"file"}'

# Process URL with full pipeline
curl -X POST "/api/v1/process" \
  -F 'request_data={"mode":"process","source_type":"url","url":"https://example.com"}'

# Ingest with webhook notifications
curl -X POST "/api/v1/process" \
  -F "file=@document.pdf" \
  -F 'request_data={"mode":"ingest","source_type":"file","webhook_config":{"url":"https://webhook.example.com"}}'
```

### Legacy Endpoints (Backward Compatibility)

**Processing (Immediate Results):**
- `POST /process/url` - Process content from URL (returns results immediately)
- `POST /process/file` - Process uploaded file (returns results immediately)
- `POST /process/web` - Process web page (returns results immediately)
- `POST /process/youtube` - Process YouTube video (returns results immediately)
- `POST /process/batch` - Process multiple items (returns results immediately)

**Ingestion (Background Processing + Vector Storage):**
- `POST /api/v1/ingest/file` - Ingest file (background processing, stores in vector DB)
- `POST /api/v1/ingest/url` - Ingest URL content (background processing, stores in vector DB)
- `POST /api/v1/ingest/batch` - Ingest multiple items (background processing, stores in vector DB)

**Conversion:**
- `POST /api/convert/markdown` - Fast markdown conversion
- `POST /api/convert/process-ingest` - Full processing + ingestion with webhooks

### Task Management

- `GET /api/v1/status/{task_id}` - Get task status
- `GET /api/v1/status/` - List active tasks
- `GET /api/v1/status/stats/queues` - Get queue statistics
- `DELETE /api/v1/ingest/{task_id}` - Cancel task

### Search

- `POST /search` - Search for similar content in vector database

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

### Optional Services (for Enhanced Features)

- **Neo4j**: For knowledge graph storage and graph-augmented retrieval
- **PostgreSQL**: For metadata storage (if using database backend)

### Service Setup

```bash
# Start required services with Docker
docker run -d --name redis -p 6379:6379 redis:alpine
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant

# Start Neo4j for graph features (optional)
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
```

## Environment Variables

### Core Services
- `GEMINI_API_KEY`: Google Gemini API key
- `QDRANT_HOST`: Qdrant server host
- `QDRANT_PORT`: Qdrant server port
- `QDRANT_API_KEY`: Qdrant API key (if using cloud)
- `REDIS_URL`: Redis connection URL
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

### Graph Features (Optional)
- `NEO4J_URI`: Neo4j database URI (default: neo4j://localhost:7687)
- `NEO4J_USERNAME`: Neo4j username (default: neo4j)
- `NEO4J_PASSWORD`: Neo4j password (default: password)
- `NEO4J_DATABASE`: Neo4j database name (default: neo4j)

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
