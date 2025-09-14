# MoRAG CLI Documentation

This document provides comprehensive documentation for all MoRAG CLI commands and scripts, covering both **stage-based processing** and legacy **ingestion/processing** operations.

## 🚨 Breaking Changes - Stage-Based Architecture

**MoRAG has been completely refactored to use a stage-based processing architecture with NO backward compatibility.**

- All previous API endpoints have been removed
- All previous CLI commands have been replaced
- New canonical stage names are used throughout
- New file naming conventions
- New configuration structure

## 🎯 Stage-Based Processing Commands

### Configuration System

MoRAG uses a unified configuration system based on environment variables with CLI override support:

#### Environment Variables
All configuration is done through environment variables with the `MORAG_` prefix:

```bash
# Global LLM Configuration (fallback for all stages)
MORAG_LLM_PROVIDER=gemini
MORAG_LLM_MODEL=gemini-1.5-flash
MORAG_LLM_TEMPERATURE=0.1
MORAG_LLM_MAX_TOKENS=4000
MORAG_LLM_MAX_RETRIES=5

# Stage-specific overrides
MORAG_MARKDOWN_OPTIMIZER_MODEL=gemini-1.5-flash
MORAG_MARKDOWN_OPTIMIZER_MAX_CHUNK_SIZE=50000
MORAG_FACT_GENERATOR_DOMAIN=general
MORAG_CHUNKER_CHUNK_SIZE=4000
```

#### Fallback Chain
The system uses this fallback chain for LLM configuration:
1. CLI arguments (highest priority)
2. Stage-specific environment variables (e.g., `MORAG_MARKDOWN_OPTIMIZER_MODEL`)
3. Global LLM environment variables (e.g., `MORAG_LLM_MODEL`)
4. `MORAG_GEMINI_MODEL` (legacy fallback)
5. Default values (lowest priority)

### Core Stage Commands

#### `morag-stages.py` - Main Stage Controller

**Purpose**: Execute individual stages or stage chains with the new canonical stage names

**Available Stages**:
1. **`markdown-conversion`** - Convert input files to unified markdown format
2. **`markdown-optimizer`** - LLM-based text improvement and error correction (optional)
3. **`chunker`** - Create summary, chunks, and contextual embeddings
4. **`fact-generator`** - Extract facts, entities, relations, and keywords
5. **`ingestor`** - Database ingestion and storage

**Basic Usage**:
```bash
# List available stages
python cli/morag-stages.py list

# Execute a single stage
python cli/morag-stages.py stage markdown-conversion input.pdf --output-dir ./output

# Execute a chain of stages
python cli/morag-stages.py stages "markdown-conversion,chunker,fact-generator" input.pdf

# Execute full pipeline
python cli/morag-stages.py process input.pdf --optimize --output-dir ./output
```

**LLM Configuration Overrides**:
```bash
# Override LLM model for a single stage
python cli/morag-stages.py stage markdown-optimizer input.md --llm-model gemini-1.5-flash

# Override multiple LLM parameters
python cli/morag-stages.py stage fact-generator input.md --llm-model gemini-1.5-pro --llm-temperature 0.2 --llm-max-tokens 8000

# Override LLM for entire stage chain
python cli/morag-stages.py stages "markdown-optimizer,chunker,fact-generator" input.pdf --llm-model gemini-1.5-pro

# Stage-specific parameter overrides
python cli/morag-stages.py stage chunker input.md --chunk-size 2000
python cli/morag-stages.py stage markdown-optimizer input.md --max-chunk-size 25000
python cli/morag-stages.py stage fact-generator input.md --domain medical
```

**Advanced Options**:
```bash
# Skip the optional markdown-optimizer stage
python cli/morag-stages.py stages "markdown-conversion,chunker,fact-generator" input.pdf

# Include optimization stage
python cli/morag-stages.py stages "markdown-conversion,markdown-optimizer,chunker" input.pdf

# Custom output directory
python cli/morag-stages.py process input.pdf --output-dir /custom/path

# Resume from existing files (automatic skip of completed stages)
python cli/morag-stages.py stages "markdown-conversion,chunker,fact-generator" input.pdf
```

#### `stage-status.py` - Stage Status Checker

**Purpose**: Check the status of stage processing and output files

```bash
# Check stage status for output directory
python cli/stage-status.py --output-dir ./output

# Check specific file processing status
python cli/stage-status.py --file input.pdf --output-dir ./output
```

### File Naming Conventions

Each stage produces files with standardized naming:

- **markdown-conversion**: `filename.md`
- **markdown-optimizer**: `filename.opt.md`
- **chunker**: `filename.chunks.json`
- **fact-generator**: `filename.facts.json`
- **ingestor**: `filename.ingestion.json`

## 🔄 Operation Modes

MoRAG supports multiple distinct operation modes:

### **Stage-Based Processing Mode** (New Default)
- **Purpose**: Execute processing pipeline using canonical stages with resume capability
- **Use Case**: Modular processing, reusable intermediate files, better error recovery
- **Output**: Standardized files for each stage that can be reused across runs
- **Storage**: Configurable (Qdrant, Neo4j, or file-only)
- **Resume**: Automatically skips completed stages
- **Reusability**: Output files can be used as input for other databases/systems

### **Processing Mode** (Legacy - Immediate Results)
- **Purpose**: Get immediate processing results without storage
- **Use Case**: One-time analysis, testing, development
- **Output**: Direct results returned immediately
- **Storage**: No vector database storage
- **Searchable**: No

### **Ingestion Mode** (Background Processing + Storage)
- **Purpose**: Process content and store in vector database for later retrieval
- **Use Case**: Building searchable knowledge base
- **Output**: Task ID for monitoring progress
- **Storage**: Results stored in Qdrant vector database
- **Searchable**: Yes, via `/search` endpoint

### **Remote Processing Mode** (Offload to Remote Workers)
- **Purpose**: Offload computationally intensive tasks to remote workers with GPU support
- **Use Case**: Scalable processing, GPU acceleration, distributed workloads
- **Supported Content**: Audio and video files only
- **Requirements**: Remote workers must be running and accessible
- **Fallback**: Can automatically fall back to local processing if configured

## 📁 Available CLI Scripts

All CLI scripts are located in `cli/` and support both operation modes with enhanced graph extraction capabilities.

### Database Setup Scripts

#### `create-databases.py`
**Purpose**: Create Neo4j databases and Qdrant collections before ingestion
```bash
# Create both database and collection
python cli/create-databases.py --neo4j-database smartcard --qdrant-collection smartcard_docs

# Create only Neo4j database
python cli/create-databases.py --neo4j-database my_database

# Create only Qdrant collection
python cli/create-databases.py --qdrant-collection my_collection

# List existing databases and collections
python cli/create-databases.py --list-existing
```

**Use Cases**:
- Neo4j Community Edition (doesn't support automatic database creation)
- Pre-creating databases/collections for better organization
- Troubleshooting database connection issues

### Stage-Based Testing Scripts

#### `test-simple.py`
**Purpose**: Quick system validation and health check
```bash
python cli/test-simple.py
```

#### `test-all.py`
**Purpose**: Comprehensive system testing with detailed reporting
```bash
python cli/test-all.py
```

### Stage Testing Framework

#### Running Stage Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run all stage tests
cd packages/morag-stages
pytest tests/

# Run specific stage test
pytest tests/test_markdown_conversion.py

# Run with coverage
pytest --cov=morag_stages tests/
```

#### Test Framework Usage

```python
from tests.test_framework import StageTestFramework

async def test_my_stage():
    framework = StageTestFramework()
    await framework.setup()

    # Create test file
    test_file = framework.create_test_markdown()

    # Execute stage
    result = await framework.execute_stage_test(
        StageType.CHUNKER,
        [test_file]
    )

    # Assert success
    framework.assert_stage_success(result)

    await framework.teardown()
```

### Component Testing Scripts

#### `test-audio.py`
**Purpose**: Audio processing and transcription testing

**Processing Mode** (immediate results):
```bash
python cli/test-audio.py my-audio.mp3
python cli/test-audio.py recording.wav
python cli/test-audio.py video.mp4  # Extract audio from video
```

**Ingestion Mode** (background processing + storage):
```bash
python cli/test-audio.py my-audio.mp3 --ingest --qdrant
python cli/test-audio.py recording.wav --ingest --qdrant --webhook-url https://my-app.com/webhook
```

**Graph Extraction Mode** (entities/relations + dual database storage):
```bash
python cli/test-audio.py my-audio.mp3 --ingest --qdrant --neo4j
python cli/test-audio.py recording.wav --ingest --neo4j --webhook-url https://my-app.com/webhook
```

**Remote Processing Mode** (offload to remote workers):
```bash
python cli/test-audio.py my-audio.mp3 --ingest --remote --qdrant
python cli/test-audio.py recording.wav --ingest --remote --qdrant --webhook-url https://my-app.com/webhook
```

**Features**:
- Audio transcription with Whisper
- Speaker diarization (when enabled)
- Topic segmentation (when enabled)
- Remote processing support (GPU workers)
- Supports MP3, WAV, M4A, and video files (audio extraction)

#### `test-document.py`
**Purpose**: Document processing and text extraction testing

**Processing Mode** (immediate results):
```bash
python cli/test-document.py my-document.pdf
python cli/test-document.py presentation.pptx
python cli/test-document.py spreadsheet.xlsx
python cli/test-document.py document.docx
```

**Ingestion Mode** (background processing + storage):
```bash
python cli/test-document.py my-document.pdf --ingest --qdrant
python cli/test-document.py document.docx --ingest --qdrant --metadata '{"category": "research", "priority": 1}'
```

**Graph Extraction Mode** (entities/relations + dual database storage):
```bash
python cli/test-document.py my-document.pdf --ingest --qdrant --neo4j
python cli/test-document.py document.docx --ingest --neo4j --metadata '{"category": "research", "priority": 1}'
```

**Features**:
- Supports PDF, DOCX, PPTX, XLSX, TXT, MD formats
- Page-level chunking (configurable)
- Metadata extraction
- Chapter detection for PDFs

#### `test-video.py`
**Purpose**: Video processing with audio extraction and analysis

**Processing Mode** (immediate results):
```bash
python cli/test-video.py my-video.mp4
python cli/test-video.py recording.avi
python cli/test-video.py presentation.mov
```

**Ingestion Mode** (background processing + storage):
```bash
python cli/test-video.py my-video.mp4 --ingest --qdrant
python cli/test-video.py recording.avi --ingest --qdrant --thumbnails
```

**Graph Extraction Mode** (entities/relations + dual database storage):
```bash
python cli/test-video.py my-video.mp4 --ingest --qdrant --neo4j
python cli/test-video.py recording.avi --ingest --neo4j --thumbnails
```

**Remote Processing Mode** (offload to remote workers):
```bash
python cli/test-video.py my-video.mp4 --ingest --remote --qdrant
python cli/test-video.py recording.avi --ingest --remote --qdrant --thumbnails
```

**Features**:
- Audio extraction and transcription
- Optional thumbnail generation (opt-in)
- Video metadata extraction
- OCR on video frames (when enabled)
- Remote processing support (GPU workers)

#### `test-image.py`
**Purpose**: Image processing with OCR and AI description

**Processing Mode** (immediate results):
```bash
python cli/test-image.py image.jpg
python cli/test-image.py screenshot.png
python cli/test-image.py diagram.gif
```

**Ingestion Mode** (background processing + storage):
```bash
python cli/test-image.py image.jpg --ingest --qdrant
python cli/test-image.py screenshot.png --ingest --qdrant --metadata '{"source": "screenshot", "date": "2025-01-01"}'
```

**Graph Extraction Mode** (entities/relations + dual database storage):
```bash
python cli/test-image.py image.jpg --ingest --qdrant --neo4j
python cli/test-image.py screenshot.png --ingest --neo4j --metadata '{"source": "screenshot", "date": "2025-01-01"}'
```

**Features**:
- OCR text extraction
- AI-powered image descriptions
- Supports JPG, PNG, GIF, BMP, WebP, TIFF, SVG

#### `test-folder.py`
**Purpose**: Batch processing of all files in a folder

**Processing Mode** (immediate results):
```bash
python cli/test-folder.py /path/to/documents
python cli/test-folder.py /path/to/documents --dry-run
python cli/test-folder.py /path/to/documents --no-recursive
```

**Ingestion Mode** (background processing + storage):
```bash
python cli/test-folder.py /path/to/documents --ingest --qdrant
python cli/test-folder.py /path/to/documents --ingest --qdrant --neo4j
python cli/test-folder.py /path/to/documents --ingest --qdrant --metadata '{"category": "research"}'
```

**Advanced Options**:
```bash
# Force reprocess files even if already processed
python cli/test-folder.py /path/to/documents --ingest --qdrant --force-reprocess

# Limit concurrent processing
python cli/test-folder.py /path/to/documents --ingest --qdrant --max-concurrent 1

# Process with custom language
python cli/test-folder.py /path/to/documents --ingest --qdrant --language en
```

**Features**:
- Automatic file type detection (documents, images, audio, video)
- Recursive folder processing (configurable)
- Skip already processed files (based on ingest_result.json existence)
- Resume from existing ingest_data.json files
- Configurable concurrency (default: 3 files at once)
- Dry run mode to preview what would be processed
- Force reprocess option to override skip logic
- Support for custom metadata and language specification
- Comprehensive error handling and progress reporting
- Supports all file types: PDF, DOCX, TXT, MD, MP3, MP4, JPG, PNG, etc.

#### `test-web.py`
**Purpose**: Web content extraction and processing

**Processing Mode** (immediate results):
```bash
python cli/test-web.py https://example.com
python cli/test-web.py https://en.wikipedia.org/wiki/Python
python cli/test-web.py https://github.com/your-repo
```

**Ingestion Mode** (background processing + storage):
```bash
python cli/test-web.py https://example.com --ingest --qdrant
python cli/test-web.py https://news-site.com/article --ingest --qdrant --metadata '{"category": "news"}'
```

**Graph Extraction Mode** (entities/relations + dual database storage):
```bash
python cli/test-web.py https://example.com --ingest --qdrant --neo4j
python cli/test-web.py https://news-site.com/article --ingest --neo4j --metadata '{"category": "news"}'
```

**Features**:
- Content extraction and cleaning
- Link and image detection
- Markdown conversion
- Supports various website structures

#### `test-youtube.py`
**Purpose**: YouTube video processing with transcription

**Processing Mode** (immediate results):
```bash
python cli/test-youtube.py https://www.youtube.com/watch?v=VIDEO_ID
python cli/test-youtube.py https://youtu.be/VIDEO_ID
```

**Ingestion Mode** (background processing + storage):
```bash
python cli/test-youtube.py https://www.youtube.com/watch?v=VIDEO_ID --ingest --qdrant
python cli/test-youtube.py https://youtu.be/VIDEO_ID --ingest --qdrant --webhook-url https://my-app.com/webhook
```

**Graph Extraction Mode** (entities/relations + dual database storage):
```bash
python cli/test-youtube.py https://www.youtube.com/watch?v=VIDEO_ID --ingest --qdrant --neo4j
python cli/test-youtube.py https://youtu.be/VIDEO_ID --ingest --neo4j --webhook-url https://my-app.com/webhook
```

**Features**:
- Video download and processing
- Audio transcription
- Metadata extraction (title, description, duration)
- Supports standard YouTube URL formats

### Remote Conversion Testing Scripts

#### `test-remote-conversion.py`
**Purpose**: Test the remote conversion system functionality

**Basic Usage**:
```bash
python cli/test-remote-conversion.py --test all
python cli/test-remote-conversion.py --test lifecycle
python cli/test-remote-conversion.py --test api
```

**Features**:
- Tests complete remote job lifecycle
- Validates API endpoints
- Simulates worker behavior
- Comprehensive error testing

## 🔧 Command Line Options

### Stage-Based Options

#### Core Stage Options

| Option | Description | Example |
|--------|-------------|---------|
| `--output-dir DIR` | Output directory for stage files | `--output-dir ./output` |
| `--optimize` | Include markdown-optimizer stage | `--optimize` |
| `--config FILE` | Stage configuration file | `--config stages.json` |
| `--force` | Force re-execution of completed stages | `--force` |
| `--dry-run` | Show what would be executed without running | `--dry-run` |

#### Stage Configuration File

Create a `stages.json` file for advanced configuration:

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

### Legacy Options (All Scripts)

| Option | Description | Example |
|--------|-------------|---------|
| `--ingest` | Enable ingestion mode (background processing + storage) | `--ingest` |
| `--qdrant` | Store vectors in Qdrant database (requires `--ingest`) | `--qdrant` |
| `--qdrant-collection NAME` | Qdrant collection name (auto-created if not exists) | `--qdrant-collection my_docs` |
| `--neo4j` | Store graph entities/relations in Neo4j (requires `--ingest`) | `--neo4j` |
| `--neo4j-database NAME` | Neo4j database name (auto-created if not exists) | `--neo4j-database my_graph` |
| `--remote` | Enable remote processing (requires `--ingest`) | `--remote` |
| `--webhook-url URL` | Webhook URL for completion notifications | `--webhook-url https://my-app.com/webhook` |
| `--metadata JSON` | Additional metadata as JSON string | `--metadata '{"category": "research"}'` |
| `--help` | Show help message | `--help` |

### Audio-Specific Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model-size SIZE` | Whisper model size (tiny, base, small, medium, large) | `base` |
| `--enable-diarization` | Enable speaker diarization | `false` |
| `--enable-topics` | Enable topic segmentation | `false` |

### Video-Specific Options

| Option | Description | Default |
|--------|-------------|---------|
| `--thumbnails` | Generate thumbnails | `false` |
| `--thumbnail-count N` | Number of thumbnails to generate | `3` |
| `--enable-ocr` | Enable OCR on video frames | `false` |

### Document-Specific Options

| Option | Description | Default |
|--------|-------------|---------|
| `--chunking-strategy STRATEGY` | Chunking strategy (paragraph, sentence, page, chapter) | `paragraph` |
| `--chunk-size SIZE` | Maximum chunk size in characters | `1000` |
| `--chunk-overlap SIZE` | Overlap between chunks in characters | `200` |

## 📊 Output Files

### Stage-Based Output (New Default)
Each stage produces standardized files that can be reused:

- **markdown-conversion**: `{filename}.md` - Unified markdown format
- **markdown-optimizer**: `{filename}.opt.md` - Optimized markdown (optional)
- **chunker**: `{filename}.chunks.json` - Chunks with embeddings and summary
- **fact-generator**: `{filename}.facts.json` - Extracted facts, entities, relations
- **ingestor**: `{filename}.ingestion.json` - Database ingestion results

### Resume Capability
The system automatically detects existing output files and skips completed stages:

```bash
# First run - executes all stages
python cli/morag-stages.py stages "markdown-conversion,chunker,fact-generator" input.pdf

# Second run - skips completed stages automatically
python cli/morag-stages.py stages "markdown-conversion,chunker,fact-generator" input.pdf
```

### Legacy Processing Mode Output
All legacy processing mode scripts generate output files in the same directory as the input file:

- `{filename}_test_result.json` - Detailed processing results
- `{filename}_converted.md` - Markdown-formatted content (when applicable)

### Legacy Ingestion Mode Output
Legacy ingestion mode scripts generate:

- `{filename}_ingest_result.json` - Task ID and ingestion status
- Background processing stores results in Qdrant vector database
- Results become searchable via `/search` endpoint

### Special Cases
- **Web content**: `web_{safe_url}_*` format
- **YouTube videos**: `youtube_{video_id}_*` format

## 🔍 Task Monitoring

When using ingestion mode, monitor task progress:

```bash
# Check specific task status
curl "http://localhost:8000/api/v1/status/TASK_ID"

# List all active tasks
curl "http://localhost:8000/api/v1/status/"

# Search ingested content
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "your search terms", "limit": 5}'
```

## ⚙️ Prerequisites

### Required Packages
```bash
pip install -e packages/morag-core
pip install -e packages/morag-audio
pip install -e packages/morag-document
pip install -e packages/morag-video
pip install -e packages/morag-image
pip install -e packages/morag-web
pip install -e packages/morag-youtube
pip install -e packages/morag-services
```

### Optional Dependencies
```bash
# Speaker diarization
pip install pyannote.audio

# Advanced topic segmentation
pip install sentence-transformers

# Dynamic web content
pip install playwright

# Advanced web extraction
pip install trafilatura readability newspaper3k
```

### Environment Configuration
Create a `.env` file with required API keys:
```bash
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key  # Optional
ANTHROPIC_API_KEY=your_anthropic_api_key  # Optional
```

## 🚀 Quick Start Examples

### Stage-Based Processing (Recommended)

#### Test System Health
```bash
python cli/test-simple.py
```

#### Process a Document with Stages
```bash
# Execute full pipeline
python cli/morag-stages.py process document.pdf --output-dir ./output

# Execute specific stages
python cli/morag-stages.py stages "markdown-conversion,chunker" document.pdf

# Include optimization
python cli/morag-stages.py process document.pdf --optimize --output-dir ./output
```

#### Process Audio/Video with Stages
```bash
# Audio processing with stages
python cli/morag-stages.py process audio.mp3 --output-dir ./output

# Video processing with optimization
python cli/morag-stages.py process video.mp4 --optimize --output-dir ./output
```

#### Resume Processing
```bash
# First run - processes all stages
python cli/morag-stages.py stages "markdown-conversion,chunker,fact-generator" document.pdf

# Second run - automatically skips completed stages
python cli/morag-stages.py stages "markdown-conversion,chunker,fact-generator" document.pdf
```

### Legacy Examples

#### Process a Document (Immediate Results)
```bash
python tests/cli/test-document.py uploads/document.pdf
```

#### Ingest a Document (Background + Storage)
```bash
python tests/cli/test-document.py uploads/document.pdf --ingest
```

#### Process Audio with Advanced Features
```bash
python tests/cli/test-audio.py uploads/audio.mp3 --enable-diarization --enable-topics
```

#### Ingest Web Content with Metadata
```bash
python tests/cli/test-web.py https://example.com --ingest --metadata '{"category": "reference"}'
```

### Remote Processing Examples
```bash
# Process audio with remote GPU workers
python tests/cli/test-audio.py uploads/audio.mp3 --ingest --remote

# Process video with remote workers and webhook
python tests/cli/test-video.py uploads/video.mp4 --ingest --remote --webhook-url https://my-app.com/webhook

# Test remote conversion system
python cli/test-remote-conversion.py --test all
```

## 🔄 Remote Processing Setup

### Remote Worker Configuration

#### Environment Setup
```bash
# Set up environment variables for remote worker
export MORAG_WORKER_ID="gpu-worker-01"
export MORAG_API_BASE_URL="https://your-morag-server.com"
export MORAG_WORKER_CONTENT_TYPES="audio,video"
export MORAG_WORKER_POLL_INTERVAL="10"
export MORAG_WORKER_MAX_CONCURRENT_JOBS="2"
export MORAG_TEMP_DIR="/tmp/morag_remote"
```

#### Configuration File
Create `remote_converter_config.yaml`:
```yaml
worker_id: "gpu-worker-01"
api_base_url: "https://your-morag-server.com"
content_types: ["audio", "video"]
poll_interval: 10
max_concurrent_jobs: 2
log_level: "INFO"
temp_dir: "/tmp/morag_remote"
```

#### Starting Remote Worker
```bash
# Install dependencies
pip install -e packages/morag-core packages/morag-audio packages/morag-video
pip install requests pyyaml python-dotenv structlog

# Start remote worker
python tools/remote-converter/cli.py --config remote_converter_config.yaml

# Or with command line options
python tools/remote-converter/cli.py \
    --worker-id gpu-worker-01 \
    --api-url https://your-morag-server.com \
    --content-types audio,video
```

### Remote Processing Examples

#### Audio Processing with Remote Workers
```bash
# Process audio file using remote workers
python tests/cli/test-audio.py uploads/audio.mp3 --ingest --remote

# With webhook notification
python tests/cli/test-audio.py uploads/audio.mp3 --ingest --remote \
    --webhook-url https://my-app.com/webhook

# With metadata and advanced options
python tests/cli/test-audio.py uploads/audio.mp3 --ingest --remote \
    --metadata '{"priority": "high", "category": "meeting"}' \
    --enable-diarization --enable-topics
```

#### Video Processing with Remote Workers
```bash
# Process video file using remote workers
python tests/cli/test-video.py uploads/video.mp4 --ingest --remote

# With thumbnail generation
python tests/cli/test-video.py uploads/video.mp4 --ingest --remote --thumbnails

# With custom settings
python tests/cli/test-video.py uploads/video.mp4 --ingest --remote \
    --thumbnail-count 5 --enable-ocr
```

#### Testing Remote Conversion System
```bash
# Test complete remote conversion system
python cli/test-remote-conversion.py --test all

# Test only job lifecycle
python cli/test-remote-conversion.py --test lifecycle

# Test only API endpoints
python cli/test-remote-conversion.py --test api
```

### Remote Job Monitoring

#### Check Job Status
```bash
# Monitor remote job progress
curl "http://localhost:8000/api/v1/remote-jobs/{job_id}/status"

# List all remote jobs
curl "http://localhost:8000/api/v1/remote-jobs/"
```

#### Worker Health Check
```bash
# Test worker connection to API
python tools/remote-converter/cli.py --test-connection

# Check worker capabilities
python tools/remote-converter/cli.py --worker-id test-worker --content-types audio,video --test-connection
```

## 🔧 Troubleshooting

### Remote Processing Issues

#### No Remote Workers Available
```bash
# Check if remote workers are running
curl "http://localhost:8000/api/v1/remote-jobs/poll?worker_id=test&content_types=audio"

# Enable fallback to local processing
python tests/cli/test-audio.py audio.mp3 --ingest --remote --fallback-local
```

#### Remote Job Timeouts
```bash
# Check job status for timeout issues
curl "http://localhost:8000/api/v1/remote-jobs/{job_id}/status"

# Increase timeout for large files
python tests/cli/test-video.py large-video.mp4 --ingest --remote \
    --metadata '{"timeout_seconds": 7200}'
```

#### Worker Connection Issues
```bash
# Test API connectivity from worker machine
curl "http://your-morag-server.com/health"

# Check worker logs
python tools/remote-converter/cli.py --log-level DEBUG
```

### Import Errors
```bash
pip install -e .
```

### Missing Dependencies
```bash
pip install pyannote.audio sentence-transformers playwright trafilatura
```

### File Not Found
```bash
ls -la uploads/  # Check available files
```

### Processing Failures
Check the generated JSON files for detailed error information and logs.

## 🧪 Integration with Testing

These CLI scripts complement the main test suite:
- **Unit Tests**: `pytest tests/unit/`
- **Integration Tests**: `pytest tests/integration/`
- **Manual Tests**: `python tests/manual/test_*.py`
- **CLI Tests**: `python tests/cli/test-*.py` (this documentation)

For more information, see the main project documentation in `/docs/`.
