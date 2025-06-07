# MoRAG CLI Documentation

This document provides comprehensive documentation for all MoRAG CLI commands and scripts, covering both **ingestion** and **processing** operations.

## 🔄 Operation Modes

MoRAG supports two distinct operation modes:

### **Processing Mode** (Immediate Results)
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

## 📁 Available CLI Scripts

All CLI scripts are located in `tests/cli/` and support both operation modes.

### System Validation Scripts

#### `test-simple.py`
**Purpose**: Quick system validation and health check
```bash
python tests/cli/test-simple.py
```

#### `test-all.py`
**Purpose**: Comprehensive system testing with detailed reporting
```bash
python tests/cli/test-all.py
```

### Component Testing Scripts

#### `test-audio.py`
**Purpose**: Audio processing and transcription testing

**Processing Mode** (immediate results):
```bash
python tests/cli/test-audio.py my-audio.mp3
python tests/cli/test-audio.py recording.wav
python tests/cli/test-audio.py video.mp4  # Extract audio from video
```

**Ingestion Mode** (background processing + storage):
```bash
python tests/cli/test-audio.py my-audio.mp3 --ingest
python tests/cli/test-audio.py recording.wav --ingest --webhook-url https://my-app.com/webhook
```

**Features**:
- Audio transcription with Whisper
- Speaker diarization (when enabled)
- Topic segmentation (when enabled)
- Supports MP3, WAV, M4A, and video files (audio extraction)

#### `test-document.py`
**Purpose**: Document processing and text extraction testing

**Processing Mode** (immediate results):
```bash
python tests/cli/test-document.py my-document.pdf
python tests/cli/test-document.py presentation.pptx
python tests/cli/test-document.py spreadsheet.xlsx
python tests/cli/test-document.py document.docx
```

**Ingestion Mode** (background processing + storage):
```bash
python tests/cli/test-document.py my-document.pdf --ingest
python tests/cli/test-document.py document.docx --ingest --metadata '{"category": "research", "priority": 1}'
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
python tests/cli/test-video.py my-video.mp4
python tests/cli/test-video.py recording.avi
python tests/cli/test-video.py presentation.mov
```

**Ingestion Mode** (background processing + storage):
```bash
python tests/cli/test-video.py my-video.mp4 --ingest
python tests/cli/test-video.py recording.avi --ingest --thumbnails
```

**Features**:
- Audio extraction and transcription
- Optional thumbnail generation (opt-in)
- Video metadata extraction
- OCR on video frames (when enabled)

#### `test-image.py`
**Purpose**: Image processing with OCR and AI description

**Processing Mode** (immediate results):
```bash
python tests/cli/test-image.py image.jpg
python tests/cli/test-image.py screenshot.png
python tests/cli/test-image.py diagram.gif
```

**Ingestion Mode** (background processing + storage):
```bash
python tests/cli/test-image.py image.jpg --ingest
python tests/cli/test-image.py screenshot.png --ingest --metadata '{"source": "screenshot", "date": "2025-01-01"}'
```

**Features**:
- OCR text extraction
- AI-powered image descriptions
- Supports JPG, PNG, GIF, BMP, WebP, TIFF, SVG

#### `test-web.py`
**Purpose**: Web content extraction and processing

**Processing Mode** (immediate results):
```bash
python tests/cli/test-web.py https://example.com
python tests/cli/test-web.py https://en.wikipedia.org/wiki/Python
python tests/cli/test-web.py https://github.com/your-repo
```

**Ingestion Mode** (background processing + storage):
```bash
python tests/cli/test-web.py https://example.com --ingest
python tests/cli/test-web.py https://news-site.com/article --ingest --metadata '{"category": "news"}'
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
python tests/cli/test-youtube.py https://www.youtube.com/watch?v=VIDEO_ID
python tests/cli/test-youtube.py https://youtu.be/VIDEO_ID
```

**Ingestion Mode** (background processing + storage):
```bash
python tests/cli/test-youtube.py https://www.youtube.com/watch?v=VIDEO_ID --ingest
python tests/cli/test-youtube.py https://youtu.be/VIDEO_ID --ingest --webhook-url https://my-app.com/webhook
```

**Features**:
- Video download and processing
- Audio transcription
- Metadata extraction (title, description, duration)
- Supports standard YouTube URL formats

## 🔧 Command Line Options

### Common Options (All Scripts)

| Option | Description | Example |
|--------|-------------|---------|
| `--ingest` | Enable ingestion mode (background processing + storage) | `--ingest` |
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

### Processing Mode Output
All processing mode scripts generate output files in the same directory as the input file:

- `{filename}_test_result.json` - Detailed processing results
- `{filename}_converted.md` - Markdown-formatted content (when applicable)

### Ingestion Mode Output
Ingestion mode scripts generate:

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

### Test System Health
```bash
python tests/cli/test-simple.py
```

### Process a Document (Immediate Results)
```bash
python tests/cli/test-document.py uploads/document.pdf
```

### Ingest a Document (Background + Storage)
```bash
python tests/cli/test-document.py uploads/document.pdf --ingest
```

### Process Audio with Advanced Features
```bash
python tests/cli/test-audio.py uploads/audio.mp3 --enable-diarization --enable-topics
```

### Ingest Web Content with Metadata
```bash
python tests/cli/test-web.py https://example.com --ingest --metadata '{"category": "reference"}'
```

## 🔧 Troubleshooting

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
