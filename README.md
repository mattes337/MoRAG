# MoRAG - Multimodal Retrieval Augmented Generation

A comprehensive, modular system for processing and indexing various types of content for retrieval-augmented generation (RAG) applications.

## 🚀 PydanticAI Integration Complete!

MoRAG has been enhanced with **PydanticAI integration** for superior AI-powered processing:

### ✅ Completed Enhancements

- **🤖 PydanticAI Foundation**: Complete migration to PydanticAI for all LLM interactions
- **🧠 Enhanced Entity Extraction**: Hybrid AI + pattern matching for 20% better accuracy
- **📝 Semantic Chunking**: Intelligent content segmentation across all content types
- **🔗 Relation Extraction**: Advanced relationship detection with confidence scoring
- **⚡ Structured Outputs**: Type-safe, validated responses from all AI agents
- **🛡️ Error Handling**: Robust retry logic and circuit breaker patterns
- **📊 Pattern Matching**: Curated knowledge bases for technology, organizations, locations, dates, and more

## 🎉 Modular Architecture Complete!

MoRAG features a modular architecture with separate, independently deployable packages:

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

### 🤖 AI-Powered Processing
- **PydanticAI Integration**: Type-safe, structured AI interactions with validation
- **Hybrid Entity Extraction**: AI + pattern matching for superior accuracy
- **Semantic Chunking**: Intelligent content segmentation based on meaning
- **Advanced Relation Extraction**: Context-aware relationship detection
- **Query Analysis**: Intent detection and entity extraction from user queries
- **Content Summarization**: Structured summaries with key points and metadata

### 🏗️ Architecture & Design
- **Modular Design**: Independent packages for different content types
- **Multi-format Support**: Process PDFs, audio, video, web pages, YouTube videos, and more
- **Vector Storage**: Qdrant integration for similarity search
- **Background Processing**: Celery-based task queue for scalable processing
- **Docker Support**: Complete containerization with docker-compose
- **Multiple Interfaces**: REST API, CLI, and Python API

### 📊 Processing Capabilities
- **Universal Document Conversion**: Unified framework for converting any document format to structured markdown
- **Intelligent Chunking**: Multiple strategies (semantic, page, sentence, paragraph) with AI-powered boundary detection
- **Quality Assessment**: Comprehensive quality scoring for conversion results with fallback mechanisms
- **Batch Embedding**: Optimized batch processing using Gemini's native batch API for 4x faster embeddings
- **Remote Processing**: Offload computationally intensive tasks (audio/video) to remote workers with GPU support
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

## 🤖 PydanticAI Features

### Enhanced Entity Extraction

```python
from morag_graph.extraction import HybridEntityExtractor

# Create hybrid extractor (AI + Pattern Matching)
extractor = HybridEntityExtractor(
    min_confidence=0.7,
    enable_pattern_matching=True,
    pattern_confidence_boost=0.1
)

# Extract entities with enhanced accuracy
text = "I'm using Python and React to build applications for Microsoft."
entities = await extractor.extract(text)

for entity in entities:
    print(f"{entity.name} ({entity.type}): {entity.confidence:.2f}")
    print(f"  Method: {entity.attributes.get('extraction_method')}")
```

### Semantic Chunking

```python
from morag_core.chunking import SemanticChunker, ChunkingConfig

# Configure semantic chunking
config = ChunkingConfig.for_documents(
    strategy="semantic",
    max_chunk_size=4000,
    min_chunk_size=500
)

# Create semantic chunker
chunker = SemanticChunker(config)

# Chunk text intelligently
chunks = await chunker.chunk_text(
    text="Your long document text here...",
    config=config
)
```

### Query Analysis

```python
from morag_core.ai import QueryAnalysisAgent

# Create query analysis agent
agent = QueryAnalysisAgent()

# Analyze user query
result = await agent.analyze_query(
    query="Find documents about Python machine learning frameworks",
    context="User is looking for technical documentation"
)

print(f"Intent: {result.intent}")
print(f"Entities: {[e.name for e in result.entities]}")
print(f"Keywords: {result.keywords}")
```

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
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  # Optional
ANTHROPIC_API_KEY=your_anthropic_api_key_here  # Optional
```

**Note**: Use `GEMINI_API_KEY` for consistency. The deprecated `GOOGLE_API_KEY` is still supported for backward compatibility.

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
- **NNPACK issues**: If running on proxmox, add this to the vm config: args: -cpu host,+kvm_pv_eoi,+kvm_pv_unhalt,host-cache-info=on,topoext=on
-- See https://github.com/Maratyszcza/NNPACK/issues/221#issuecomment-2899754029

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

### CPU Compatibility (NEW)

✅ **Fixed CPU Compatibility Issues**: MoRAG now includes comprehensive CPU compatibility fixes to prevent crashes on systems with limited instruction set support.

**What was fixed:**
- **SIGILL crashes**: Workers no longer crash with "Illegal Instruction" errors
- **PyTorch compatibility**: CPU-only PyTorch installation with safety settings
- **Docling fallback**: Automatic fallback to pypdf when docling/PyTorch fails
- **Environment safety**: Automatic CPU safety configuration

**Features:**
- Automatic CPU feature detection and compatibility checking
- Safe startup scripts that configure environment for maximum compatibility
- Fallback mechanisms for ML libraries that require advanced CPU features
- Comprehensive logging and debugging for CPU-related issues

For detailed information, see [CPU Compatibility Guide](docs/CPU_COMPATIBILITY.md).

## Remote Processing (NEW)

✅ **Remote Conversion System**: MoRAG now supports offloading computationally intensive tasks to remote workers with GPU support.

**Key Features:**
- **Horizontal Scaling**: Distribute audio/video processing across multiple machines
- **GPU Support**: Leverage remote GPUs for faster processing
- **Automatic Fallback**: Falls back to local processing if remote workers are unavailable
- **Secure File Transfer**: Safe file download and result submission
- **Job Management**: Complete job lifecycle tracking and monitoring

**Quick Start:**
```bash
# Enable remote processing for audio/video files
python tests/cli/test-audio.py my-audio.mp3 --ingest --remote
python tests/cli/test-video.py my-video.mp4 --ingest --remote

# Test remote conversion system
python cli/test-remote-conversion.py --test all
```

**Remote Worker Setup:**
```bash
# Set up remote worker (on GPU machine)
export MORAG_WORKER_ID="gpu-worker-01"
export MORAG_API_BASE_URL="https://your-morag-server.com"
export MORAG_WORKER_CONTENT_TYPES="audio,video"

# Start remote worker
python tools/remote-converter/cli.py
```

For complete documentation, see [Remote Conversion System Guide](docs/remote-conversion-system.md).

## Usage

For detailed usage examples including Python API, CLI commands, REST API endpoints, and configuration options, see [USAGE.md](USAGE.md).

### Quick Examples

**✨ NEW: All CLI scripts now support both processing (immediate results) and ingestion (background + storage) modes!**

```bash
# System validation
python tests/cli/test-simple.py  # Quick validation (recommended)
python tests/cli/test-all.py     # Comprehensive test with detailed report

# Processing Mode (immediate results)
python tests/cli/test-audio.py my-audio.mp3
python tests/cli/test-document.py my-document.pdf
python tests/cli/test-web.py https://example.com

# Ingestion Mode (background processing + vector storage)
python tests/cli/test-audio.py my-audio.mp3 --ingest
python tests/cli/test-document.py my-document.pdf --ingest --metadata '{"category": "research"}'
python tests/cli/test-web.py https://example.com --ingest --webhook-url https://my-app.com/webhook
```

For comprehensive CLI documentation, see [CLI.md](CLI.md).

## Architecture

The MoRAG pipeline consists of several key components:

- **API Layer**: FastAPI-based REST endpoints for content ingestion
- **Processing Layer**: Specialized processors for different content types
- **Embedding Layer**: Gemini API integration for text embeddings
- **Storage Layer**: Qdrant vector database for similarity search
- **Task Queue**: Celery for async processing and scalability
- **Modular Packages**: Independent packages for each content type

## Documentation

- [CLI Documentation](CLI.md) - **NEW**: Comprehensive CLI commands for both processing and ingestion modes
- [Usage Guide](USAGE.md) - Comprehensive usage examples and API reference
- [API Documentation](http://localhost:8000/docs) (when running locally)
- [Docker Deployment Guide](docs/DOCKER_DEPLOYMENT.md) - Complete Docker deployment instructions
- [Universal Document Conversion](docs/UNIVERSAL_DOCUMENT_CONVERSION.md) - Complete guide to the conversion framework
- [Batch Embedding Guide](docs/batch-embedding.md) - **NEW**: Optimized batch embedding with Gemini API for 4x performance improvement
- [Remote Conversion System](docs/remote-conversion-system.md) - **NEW**: Remote processing with GPU workers for scalable audio/video processing
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

# Test individual components (NEW: dual-mode support)
python tests/cli/test-all.py          # Complete system test
python tests/cli/test-audio.py sample.mp3                    # Processing mode
python tests/cli/test-audio.py sample.mp3 --ingest           # Ingestion mode
python tests/cli/test-document.py sample.pdf                 # Processing mode
python tests/cli/test-document.py sample.pdf --ingest        # Ingestion mode
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

## Local Development (Without Docker)

For detailed local development setup without Docker, see [LOCAL_DEVELOPMENT.md](LOCAL_DEVELOPMENT.md).

### Quick Local Setup

```bash
# 1. Install packages in development mode
pip install -e packages/morag-core/
pip install -e packages/morag-services/
pip install -e packages/morag/

# 2. Start infrastructure (Redis + Qdrant)
docker run -d --name morag-redis -p 6379:6379 redis:alpine
docker run -d --name morag-qdrant -p 6333:6333 qdrant/qdrant:latest

# 3. Start services
python scripts/start_worker.py  # Terminal 1 (REQUIRED)
uvicorn morag.api.main:app --reload  # Terminal 2

# 4. Test setup
python debug_morag.py  # Comprehensive debugging
python test_qdrant_fix.py  # Quick validation
```

## Troubleshooting

### Fixed: Abstract Class Errors

✅ **FIXED**: Both abstract class instantiation errors have been resolved:

1. **QdrantVectorStorage Error**: "Can't instantiate abstract class QdrantVectorStorage"
2. **GeminiEmbeddingService Error**: "Can't instantiate abstract class GeminiEmbeddingService"

**What was fixed:**

**QdrantVectorStorage** - Implemented all missing abstract methods:
- `initialize()`, `shutdown()`, `health_check()`
- `put_object()`, `get_object()`, `delete_object()`, `list_objects()`, `get_object_metadata()`, `object_exists()`
- `add_vectors()`, `search_vectors()`, `delete_vectors()`, `update_vector_metadata()`

**GeminiEmbeddingService** - Implemented all missing abstract methods:
- `health_check()`, `generate_embeddings()`
- `get_embedding_dimension()`, `get_supported_models()`, `get_max_tokens()`
- Fixed method signatures to match interface requirements

**Test the fixes:**
```bash
python tests/cli/test-qdrant-fix.py        # Test QdrantVectorStorage
python tests/cli/test-embedding-fix.py     # Test GeminiEmbeddingService
python tests/cli/test-ingest-workflow.py   # Test complete workflow
```

### Debugging Issues

If you encounter issues:

1. **Run comprehensive debug script:**
   ```bash
   python debug_morag.py
   ```

2. **Check logs:** The debug script creates detailed log files with timestamps

3. **Test individual components:**
   ```bash
   python tests/cli/test-simple.py  # Quick validation
   python tests/cli/test-document.py sample.pdf  # Test document processing
   ```

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
