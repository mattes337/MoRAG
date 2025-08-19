# MoRAG - Multimodal Retrieval Augmented Generation

A comprehensive, modular system for processing and indexing various types of content for retrieval-augmented generation (RAG) applications.

## 🚀 Latest Enhancements Complete!

MoRAG has been enhanced with **PydanticAI integration** and **unified citation formats** for superior AI-powered processing:

### ✅ Completed Enhancements

- **🤖 PydanticAI Foundation**: Complete migration to PydanticAI for all LLM interactions
- **🧠 Structured Fact Extraction**: Advanced AI-powered extraction of actionable facts with detailed source attribution
- **📝 Semantic Chunking**: Intelligent content segmentation across all content types
- **🔗 Fact Relationships**: Automatic detection of relationships between extracted facts
- **⚡ Structured Outputs**: Type-safe, validated responses from all AI agents
- **🛡️ Error Handling**: Robust retry logic and circuit breaker patterns
- **📊 Pattern Matching**: Curated knowledge bases for technology, organizations, locations, dates, and more
- **📋 Unified Citation Format**: Standardized structured citations across all content types
- **🔧 Graph Tool Controller**: Gemini function calling interface for graph operations

## 🎉 Stage-Based Processing Architecture Complete!

MoRAG has been completely refactored to use a **stage-based processing architecture** with canonical stage names for better modularity, reusability, and control over the processing pipeline.

### 🚨 Breaking Changes

**This is a complete rewrite with NO backward compatibility.**

- All previous API endpoints have been removed
- All previous CLI commands have been replaced
- New canonical stage names are used throughout
- New file naming conventions
- New configuration structure

### Stage-Based Architecture

#### Canonical Stage Names

The system uses these exact canonical stage names:

1. **`markdown-conversion`** - Convert input files to unified markdown format
2. **`markdown-optimizer`** - LLM-based text improvement and error correction (optional)
3. **`chunker`** - Create summary, chunks, and contextual embeddings
4. **`fact-generator`** - Extract facts, entities, relations, and keywords
5. **`ingestor`** - Database ingestion and storage

#### Configuration System

MoRAG uses a unified environment variable configuration system with CLI override support:

- **Environment Variables**: All configuration uses `MORAG_` prefixed environment variables
- **CLI Overrides**: Command-line arguments can override any environment variable
- **Fallback Chain**: Stage-specific → Global LLM → Legacy → Defaults
- **LLM Model Fallback**: Always falls back to `MORAG_GEMINI_MODEL` for compatibility

#### Stage Flow

```
Input File → markdown-conversion → [markdown-optimizer] → chunker → fact-generator → ingestor
```

The `markdown-optimizer` stage is optional and can be skipped.

### Package Structure
```
packages/
├── morag-core/          # Core interfaces and models
├── morag-services/      # AI and storage services
├── morag-stages/        # Stage-based processing system
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
- **Structured Fact Extraction**: AI-powered extraction of actionable facts with subject-object-approach-solution patterns
- **Semantic Chunking**: Intelligent content segmentation based on meaning
- **Fact Relationship Detection**: Context-aware relationship detection between facts
- **Query Analysis**: Intent detection and fact-based query processing
- **Content Summarization**: Structured summaries with key points and metadata

### 🏗️ Architecture & Design
- **Modular Design**: Independent packages for different content types
- **Multi-format Support**: Process PDFs, audio, video, web pages, YouTube videos, and more
- **Vector Storage**: Qdrant integration for similarity search
- **Background Processing**: Celery-based task queue for scalable processing
- **Docker Support**: Complete containerization with docker-compose
- **Multiple Interfaces**: REST API, CLI, and Python API

### 📊 Processing Capabilities
- **Universal Document Conversion**: Unified framework powered by markitdown for converting 47+ file formats to structured markdown
- **Specialized Converters**: Domain-specific converters organized by file type:
  - **Documents**: PDF, Word, Excel, PowerPoint, Text, Archives (in `morag-document`)
  - **Images**: JPG, PNG, GIF, BMP, TIFF, WEBP, SVG with OCR (in `morag-image`)
  - **Audio**: MP3, WAV, M4A, FLAC, AAC, OGG with transcription (in `morag-audio`)
  - **Video**: MP4, AVI, MOV, MKV, WEBM with transcription (in `morag-video`)
- **Intelligent Chunking**: Multiple strategies (semantic, page, sentence, paragraph) with AI-powered boundary detection
- **Quality Assessment**: Comprehensive quality scoring for conversion results with fallback mechanisms
- **Batch Embedding**: Optimized batch processing using Gemini's native batch API for 4x faster embeddings
- **Remote Processing**: Offload computationally intensive tasks (audio/video) to remote workers with GPU support
- **Production Ready**: Docker support, logging, monitoring, and deployment configurations

### 📋 Citation System
- **Structured Citations**: Unified citation format across all content types: `[source_type:source_name:source_index:metadata]`
- **Source Traceability**: Complete provenance tracking from original sources to extracted facts
- **Multi-format Support**: Citations for documents, audio, video, web content, and extracted facts
- **Metadata Preservation**: Rich metadata including page numbers, timestamps, sections, and fact IDs

### 🔧 Graph Operations
- **Gemini Function Calling**: Secure interface for graph operations with policy enforcement
- **Entity Operations**: Extract, match, and traverse entities in the knowledge graph
- **Fact Extraction**: AI-powered structured fact extraction with confidence scoring
- **Neighbor Expansion**: Graph traversal with configurable depth and relationship filtering
- **Content Retrieval**: Fetch entity-associated content chunks with relevance scoring
- **Action Tracing**: Complete audit trail of all graph operations for monitoring and debugging

## Quick Start

### Stage-Based Processing

#### CLI Usage

```bash
# List available stages
python cli/morag-stages.py list

# Execute a single stage
python cli/morag-stages.py stage markdown-conversion input.pdf --output-dir ./output

# Execute a chain of stages
python cli/morag-stages.py stages "markdown-conversion,chunker,fact-generator" input.pdf

# Execute full pipeline
python cli/morag-stages.py process input.pdf --optimize --output-dir ./output

# Override LLM configuration via CLI
python cli/morag-stages.py stage markdown-optimizer input.md --llm-model gemini-1.5-pro --llm-temperature 0.2

# Override stage-specific parameters
python cli/morag-stages.py stage chunker input.md --chunk-size 2000
python cli/morag-stages.py stage fact-generator input.md --domain medical

# Chain with LLM overrides
python cli/morag-stages.py stages "markdown-optimizer,chunker" input.pdf --llm-model gemini-1.5-flash
```

#### REST API Usage

```bash
# Start the server
python -m morag.server

# List available stages
curl http://localhost:8000/api/v1/stages/

# Execute a single stage
curl -X POST http://localhost:8000/api/v1/stages/markdown-conversion/execute \
  -F "file=@input.pdf" \
  -F "request={\"output_dir\": \"./output\"}"

# Execute stage chain
curl -X POST http://localhost:8000/api/v1/stages/chain \
  -F "file=@input.pdf" \
  -F "request={\"stages\": [\"markdown-conversion\", \"chunker\", \"fact-generator\"]}"
```

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

### Structured Fact Extraction

```python
from morag_graph.extraction.fact_extractor import FactExtractor

# Create fact extractor with domain specialization
extractor = FactExtractor(
    model_id="gemini-2.0-flash",
    api_key="your-api-key",
    domain="technology",
    min_confidence=0.7,
    max_facts_per_chunk=10
)

# Extract structured facts
text = "Python and React are used together to build full-stack applications. This approach enables rapid development with strong type safety."
facts = await extractor.extract_facts(
    chunk_text=text,
    chunk_id="chunk_1",
    document_id="doc_1",
    context={'domain': 'technology', 'language': 'en'}
)

for fact in facts:
    print(f"Subject: {fact.subject}")
    print(f"Object: {fact.object}")
    print(f"Approach: {fact.approach}")
    print(f"Solution: {fact.solution}")
    print(f"Confidence: {fact.extraction_confidence:.2f}")
    print(f"Citation: {fact.get_citation()}")
    print("---")
```

**Note**: Facts provide structured, actionable knowledge with complete source attribution, making them more useful than generic entities for downstream applications.

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

### Stage-Based Configuration

#### File Naming Conventions

Each stage produces files with standardized naming:

- **markdown-conversion**: `filename.md`
- **markdown-optimizer**: `filename.opt.md`
- **chunker**: `filename.chunks.json`
- **fact-generator**: `filename.facts.json`
- **ingestor**: `filename.ingestion.json`

#### Stage-Specific Configuration

```json
{
  "markdown-conversion": {
    "include_timestamps": true,
    "speaker_diarization": true,
    "topic_segmentation": true
  },
  "markdown-optimizer": {
    "fix_transcription_errors": true,
    "improve_readability": true,
    "preserve_timestamps": true
  },
  "chunker": {
    "chunk_strategy": "semantic",
    "chunk_size": 2000,
    "generate_summary": true
  },
  "fact-generator": {
    "extract_entities": true,
    "extract_relations": true,
    "domain": "general"
  },
  "ingestor": {
    "databases": ["qdrant", "neo4j"],
    "collection_name": "my_collection"
  }
}
```

#### Chunking Strategies

- **`semantic`** - Intelligent semantic boundaries (default for documents)
- **`page-level`** - Split by pages (default for PDFs)
- **`topic-based`** - Split by topics with timestamps (default for audio/video)

### Database Setup

MoRAG automatically creates Qdrant collections and Neo4j databases when needed. However, for Neo4j Community Edition or when you want to pre-create databases, use the database creation utility:

```bash
# Create both Neo4j database and Qdrant collection
python cli/create-databases.py --neo4j-database smartcard --qdrant-collection smartcard_docs

# Create only Neo4j database
python cli/create-databases.py --neo4j-database my_database

# Create only Qdrant collection
python cli/create-databases.py --qdrant-collection my_collection

# List existing databases and collections
python cli/create-databases.py --list-existing
```

**Important Notes:**
- **Neo4j Enterprise**: Supports automatic database creation
- **Neo4j Community**: Requires manual database creation or using the utility script
- **Qdrant**: Always supports automatic collection creation
- **Collection/Database Names**: Use `--qdrant-collection` and `--neo4j-database` arguments in test scripts

### Stage-Based API Endpoints

#### Stage Execution

- `GET /api/v1/stages/` - List available stages
- `POST /api/v1/stages/{stage-name}/execute` - Execute single stage
- `POST /api/v1/stages/chain` - Execute stage chain
- `GET /api/v1/stages/status` - Check execution status
- `GET /api/v1/stages/health` - Health check

#### File Management

- `GET /api/v1/files/list` - List output files
- `GET /api/v1/files/download/{file_path}` - Download file
- `GET /api/v1/files/info/{file_path}` - Get file metadata
- `DELETE /api/v1/files/delete/{file_path}` - Delete file
- `DELETE /api/v1/files/cleanup` - Cleanup old files

### Webhook Notifications

Configure webhooks to receive notifications when stages complete:

```json
{
  "webhook_config": {
    "url": "https://your-webhook-url.com/notifications",
    "auth_token": "your-token",
    "headers": {"Custom-Header": "value"},
    "retry_count": 3,
    "timeout": 30
  }
}
```

Webhook payload example:
```json
{
  "event_type": "stage_completed",
  "timestamp": "2024-01-01T12:00:00Z",
  "stage": {
    "name": "chunker",
    "status": "completed",
    "execution_time": 15.5
  },
  "source_file": "/path/to/input.pdf",
  "output_files": [
    {
      "filename": "input.chunks.json",
      "file_id": "abc123",
      "download_url": "/api/v1/files/download/abc123"
    }
  ]
}
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

### Maintenance: Keyword Hierarchization (Standalone)

- Build image:
```bash
docker build -f Dockerfile.maintenance -t morag-maintenance:latest .
```
- Run (dry-run by default):
```bash
docker run --rm \
  -e NEO4J_URI -e NEO4J_USERNAME -e NEO4J_PASSWORD -e NEO4J_DATABASE \
  morag-maintenance:latest
```
- Apply with detachment:
```bash
docker run --rm \
  -e NEO4J_URI -e NEO4J_USERNAME -e NEO4J_PASSWORD -e NEO4J_DATABASE \
  -e MORAG_KWH_APPLY=true -e MORAG_KWH_DETACH_MOVED=true \
  morag-maintenance:latest
```
- CLI (no Docker):
```bash
python -m morag_graph.maintenance.keyword_hierarchization --threshold 50 --limit-keywords 5 --apply --detach-moved
```


#### Generalized maintenance runner (multi-job)
- The maintenance image uses a generic runner. Select jobs via MORAG_MAINT_JOBS (comma-separated). Currently supported: keyword_hierarchization.
```bash
# Run keyword hierarchization (dry-run by default)
docker run --rm \
  -e NEO4J_URI -e NEO4J_USERNAME -e NEO4J_PASSWORD -e NEO4J_DATABASE \
  -e MORAG_MAINT_JOBS=keyword_hierarchization \
  morag-maintenance:latest

# Apply with common tuning overrides
# If proposals are empty, lower the share and/or min-new thresholds
# (See maintenance/KEYWORD_HIERARCHIZATION.md for details)
docker run --rm \
  -e NEO4J_URI -e NEO4J_USERNAME -e NEO4J_PASSWORD -e NEO4J_DATABASE \
  -e MORAG_MAINT_JOBS=keyword_hierarchization \
  -e MORAG_KWH_APPLY=true \
  -e MORAG_KWH_DETACH_MOVED=true \
  -e MORAG_KWH_SHARE=0.08 \
  -e MORAG_KWH_MIN_NEW=2 \
  morag-maintenance:latest
```

Environment overrides for keyword hierarchization:
- MORAG_KWH_THRESHOLD: minimum facts on a keyword (default 50)
- MORAG_KWH_MIN_NEW / MORAG_KWH_MAX_NEW: min/max number of proposed keywords (defaults 3/6)
- MORAG_KWH_MIN_PER: minimum facts per proposed keyword (default 5)
- MORAG_KWH_MAX_MOVE_RATIO: cap on fraction of facts to move (default 0.8)
- MORAG_KWH_SHARE: minimum co-occurrence share for a proposal (default 0.18)
- MORAG_KWH_BATCH_SIZE: batch size for writes (default 200)
- MORAG_KWH_DETACH_MOVED: true to delete the original edges after reattach
- MORAG_KWH_APPLY: true to apply (false = dry-run)
- MORAG_KWH_JOB_TAG: optional tag stored on created relationships
- MORAG_KWH_LIMIT_KEYWORDS: process up to N keywords per run (default 5)


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

### Resume Capability

The system automatically detects existing output files and skips stages that have already been completed:

```bash
# First run - executes all stages
python cli/morag-stages.py stages "markdown-conversion,chunker,fact-generator" input.pdf

# Second run - skips completed stages
python cli/morag-stages.py stages "markdown-conversion,chunker,fact-generator" input.pdf
```

### Migration from Previous Version

#### CLI Migration

| Old Command | New Command |
|-------------|-------------|
| `morag process file.pdf` | `python cli/morag-stages.py process file.pdf` |
| `morag ingest file.pdf` | `python cli/morag-stages.py stages "markdown-conversion,chunker,fact-generator,ingestor" file.pdf` |

#### API Migration

| Old Endpoint | New Endpoint |
|--------------|--------------|
| `POST /api/v1/process` | `POST /api/v1/stages/chain` |
| `POST /api/v1/ingest` | `POST /api/v1/stages/ingestor/execute` |
| `GET /health` | `GET /api/v1/stages/health` |

#### Configuration Migration

Old configuration files need to be restructured to use canonical stage names as top-level keys.

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
