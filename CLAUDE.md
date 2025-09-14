# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MoRAG is a comprehensive, modular Multimodal Retrieval Augmented Generation system for processing and indexing various content types (documents, audio, video, web content) using AI-powered processing pipelines. The project has undergone a complete rewrite with NO backward compatibility to use a stage-based processing architecture with PydanticAI integration.

## Development Commands

### Code Quality and Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
pytest tests/manual/        # Manual tests

# Test with coverage
pytest --cov=src/morag --cov-report=html

# Comprehensive syntax checking (recommended before commits)
python check_syntax.py --verbose

# Auto-fix import sorting issues
python check_syntax.py --fix

# Code formatting
black src/ tests/
isort src/ tests/ --profile black

# Linting
flake8 src/ tests/
mypy src/

# Pre-commit hooks (runs static analysis, import checks, build checks)
pre-commit run --all-files
```

### Development Setup

```bash
# Install development dependencies (if using traditional setup)
pip install -e ".[dev]"

# Install main package with all components (recommended)
pip install packages/morag/

# Install individual packages for development
pip install -e packages/morag-core
pip install -e packages/morag-services
pip install -e packages/morag-audio
pip install -e packages/morag-document

# Start infrastructure services
docker run -d --name morag-redis -p 6379:6379 redis:alpine
docker run -d --name morag-qdrant -p 6333:6333 qdrant/qdrant:latest

# Start Celery worker (REQUIRED for task processing)
python scripts/start_worker.py

# Start API server
uvicorn morag.api.main:app --reload
```

### Testing and Validation

```bash
# Quick system validation (recommended)
python tests/cli/test-simple.py

# Comprehensive system test with detailed report
python tests/cli/test-all.py

# Test individual components with stage-based CLI
python cli/morag-stages.py stage markdown-conversion sample.pdf
python cli/morag-stages.py stage chunker output/sample.md
python cli/morag-stages.py stage fact-generator output/sample.chunks.json

# Test complete pipeline
python cli/morag-stages.py stages "markdown-conversion,chunker,fact-generator" sample.pdf

# Test individual components (legacy dual-mode support)
python tests/cli/test-audio.py sample.mp3                    # Processing mode
python tests/cli/test-audio.py sample.mp3 --ingest           # Ingestion mode
python tests/cli/test-document.py sample.pdf                 # Processing mode
python tests/cli/test-document.py sample.pdf --ingest        # Ingestion mode
python tests/cli/test-web.py https://example.com --ingest

# Debug system issues
python debug_morag.py

# Test specific fixes
python tests/cli/test-qdrant-fix.py        # QdrantVectorStorage
python tests/cli/test-embedding-fix.py     # GeminiEmbeddingService
python tests/cli/test-ingest-workflow.py   # Complete workflow
```

### Stage-Based Processing

The system uses canonical stage names for modular processing:

```bash
# List available stages
python cli/morag-stages.py list

# Execute single stage
python cli/morag-stages.py stage markdown-conversion input.pdf --output-dir ./output

# Execute stage chain
python cli/morag-stages.py stages "markdown-conversion,chunker,fact-generator" input.pdf

# Execute full pipeline
python cli/morag-stages.py process input.pdf --optimize --output-dir ./output

# Override LLM configuration
python cli/morag-stages.py stage fact-generator input.md --llm-model gemini-1.5-flash --fact-extraction-agent-model gemini-2.0-flash
```

### Docker Operations

```bash
# Start all services
docker-compose up -d

# Development with hot-reload
docker-compose -f docker-compose.dev.yml up -d

# Production microservices
docker-compose -f docker-compose.microservices.yml up -d

# Check service health
curl http://localhost:8000/health

# View logs
docker-compose logs -f

# Test Docker setup
python tests/cli/test-docker-fixes.py
```

## High-Level Architecture

### Core Components

- **API Layer**: FastAPI-based REST endpoints (`/api/v1/stages/`, `/api/v1/files/`)
- **Processing Layer**: Stage-based pipeline with canonical stages:
  1. `markdown-conversion` - Convert input files to unified markdown
  2. `markdown-optimizer` - LLM-based text improvement (optional)
  3. `chunker` - Create summary, chunks, and contextual embeddings
  4. `fact-generator` - Extract facts, entities, relations, and keywords
  5. `ingestor` - Database ingestion and storage
- **Task Queue**: Celery with Redis for async processing
- **Vector Storage**: Qdrant for similarity search
- **Graph Storage**: Neo4j for knowledge graph
- **Embedding Layer**: Gemini API integration with batch processing (4x faster)

### Package Structure

```
packages/
├── morag-core/          # Core interfaces, models, chunking, utilities
├── morag-services/      # AI and storage services, embedding services
├── morag-stages/        # Stage-based processing system
├── morag-audio/         # Audio processing with Whisper transcription
├── morag-document/      # Document processing (PDF, Word, Excel, etc.)
├── morag-video/         # Video processing with transcription
├── morag-image/         # Image processing with OCR
├── morag-web/           # Web content processing with Playwright
├── morag-youtube/       # YouTube video processing
├── morag-graph/         # Knowledge graph extraction and management
├── morag-embedding/     # Embedding services and batch processing
└── morag/               # Main integration package
```

### Stage-Based Configuration

Each stage produces standardized output files:
- `markdown-conversion`: `filename.md`
- `markdown-optimizer`: `filename.opt.md`
- `chunker`: `filename.chunks.json`
- `fact-generator`: `filename.facts.json`
- `ingestor`: `filename.ingestion.json`

### AI-Powered Features

- **PydanticAI Integration**: Type-safe, structured AI interactions with validation
- **Structured Fact Extraction**: Subject-object-approach-solution patterns with confidence scoring
- **Semantic Chunking**: Intelligent content segmentation based on meaning and context
- **Fact Relationship Detection**: Context-aware relationship detection between extracted facts
- **Query Analysis**: Intent detection and fact-based query processing with entity extraction

### Environment Configuration

The system uses `MORAG_` prefixed environment variables with CLI override support:

```bash
# Required API keys
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key  # Optional
ANTHROPIC_API_KEY=your_anthropic_api_key  # Optional

# LLM configuration
MORAG_GEMINI_MODEL=gemini-1.5-pro  # Global fallback
MORAG_FACT_EXTRACTION_AGENT_MODEL=gemini-2.0-flash  # Agent-specific

# Database configuration
QDRANT_URL=http://localhost:6333
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
REDIS_URL=redis://localhost:6379

# Processing configuration
MORAG_CHUNK_SIZE=2000
MORAG_BATCH_SIZE=50
MORAG_MAX_WORKERS=4
```

### Key Architectural Principles

1. **Modular Design**: Independent packages for different content types with clear interfaces
2. **Stage-Based Processing**: Canonical stage names for predictable pipeline execution
3. **Type Safety**: PydanticAI for structured, validated AI interactions
4. **Resume Capability**: Automatic detection of existing output files to skip completed stages
5. **Scalable Processing**: Celery task queue with remote worker support for GPU processing
6. **Citation System**: Unified structured citations across all content types
7. **Quality Assessment**: Comprehensive quality scoring with fallback mechanisms

### Known Issues and Fixes

- **Fixed**: Abstract class instantiation errors (QdrantVectorStorage, GeminiEmbeddingService)
- **Fixed**: CPU compatibility issues with SIGILL crashes and PyTorch conflicts
- **Fixed**: Qdrant health checks and Whisper model permissions in Docker
- **Current Status**: All 574 Python files compile successfully with no syntax errors

### Important Notes for Development

1. **Celery Worker Required**: The Celery worker must be running for task processing. Without it, tasks remain in "pending" status.
2. **No Package.json**: This is a Python project with no Node.js dependencies. Use `requirements.txt` and Python package management.
3. **Breaking Changes**: This is a complete rewrite with NO backward compatibility from previous versions.
4. **Stage Output**: Always check stage output files for resume capability and debugging.
5. **Remote Processing**: GPU-intensive audio/video processing can be offloaded to remote workers.