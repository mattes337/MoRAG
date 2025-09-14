# MoRAG Installation Guide

This guide provides comprehensive instructions for installing MoRAG with different dependency configurations.

## Quick Start

### Option 1: Minimal Installation (Recommended for testing)
```bash
pip install -r requirements-minimal.txt
```

### Option 2: Full Installation (All features)
```bash
pip install -r requirements-full.txt
```

### Option 3: Development Installation
```bash
pip install -r requirements-dev.txt
```

### Option 4: Standard Installation
```bash
pip install -r requirements.txt
```

## Installation Options Explained

### 1. Minimal Installation (`requirements-minimal.txt`)
**Best for:** Testing, basic document processing, API development

**Includes:**
- Core web framework (FastAPI, Uvicorn)
- Basic document processing (markitdown)
- Vector database (Qdrant)
- LLM services (Google Gemini)
- Entity extraction (LangExtract)
- Graph database (Neo4j)

**Size:** ~500MB

### 2. Full Installation (`requirements-full.txt`)
**Best for:** Production use with all media types

**Includes everything from minimal plus:**
- Audio processing (Whisper, PyAnnote)
- Video processing (OpenCV, MoviePy)
- Image processing (OCR, EasyOCR)
- Advanced document processing (PDF, Office)
- Web scraping (Playwright, Trafilatura)
- Scientific computing (NumPy, Pandas)

**Size:** ~3-5GB (includes PyTorch, audio models)

### 3. Development Installation (`requirements-dev.txt`)
**Best for:** Contributors and developers

**Includes everything from full plus:**
- Testing framework (pytest)
- Code quality tools (black, flake8, mypy)
- Documentation tools (Sphinx)
- Development utilities (Jupyter, IPython)

**Size:** ~5-7GB

### 4. Standard Installation (`requirements.txt`)
**Best for:** Production deployment with all dependencies

**Includes:** All dependencies in a single file for production deployment

## Selective Installation

You can also install specific feature sets using pyproject.toml:

```bash
# Audio processing only
pip install -e ".[audio]"

# Video processing only  
pip install -e ".[video]"

# Image processing only
pip install -e ".[image]"

# Web scraping only
pip install -e ".[web]"

# Office documents only
pip install -e ".[office]"

# All optional dependencies
pip install -e ".[all-extras]"

# Development dependencies
pip install -e ".[dev]"
```

## System Requirements

### Minimum Requirements
- Python 3.9+
- 4GB RAM
- 2GB disk space

### Recommended Requirements
- Python 3.11
- 16GB RAM
- 10GB disk space
- GPU (optional, for faster audio/video processing)

## Platform-Specific Notes

### Windows
- Install Visual Studio Build Tools for C++ compilation
- FFmpeg must be installed and in PATH for audio/video processing

### Linux
- Install system dependencies:
  ```bash
  sudo apt-get update
  sudo apt-get install ffmpeg tesseract-ocr
  ```

### macOS
- Install Homebrew dependencies:
  ```bash
  brew install ffmpeg tesseract
  ```

## Troubleshooting

### Common Issues

1. **gRPC/Protobuf conflicts:**
   ```bash
   pip install --upgrade protobuf>=6.30.0,<7.0.0
   ```

2. **PyTorch installation issues:**
   ```bash
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Audio processing dependencies:**
   ```bash
   pip install asteroid-filterbanks pytorch-lightning pyannote-metrics
   ```

### Verification

Test your installation:
```bash
python -c "import morag; print('MoRAG installed successfully')"
```

Run basic functionality test:
```bash
python cli/test-simple.py
```

## Docker Installation

For containerized deployment:
```bash
docker-compose up -d
```

See `DOCKER_DEPLOYMENT.md` for detailed Docker instructions.
