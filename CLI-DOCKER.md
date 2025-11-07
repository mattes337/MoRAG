# MoRAG CLI Docker Setup

This document explains how to build and use the MoRAG CLI Docker container for running both **stage-based processing** and legacy CLI scripts.

## 🚨 Breaking Changes - Stage-Based Architecture

**MoRAG has been completely refactored to use a stage-based processing architecture with NO backward compatibility.**

- All previous API endpoints have been removed
- All previous CLI commands have been replaced
- New canonical stage names are used throughout
- New file naming conventions
- New configuration structure

## Quick Start

### 1. Build the CLI Docker Image

```bash
docker build -f Dockerfile.cli -t morag-cli .
```

### 2. Prepare Environment Variables

Copy the example environment file and configure it:

```bash
cp .env.example .env
# Edit .env with your actual configuration values
```

### 3. Run CLI Scripts

#### Stage-Based Processing (Recommended)

```bash
# Show available CLI scripts
docker run -it --rm --env-file .env morag-cli

# List available stages
docker run -it --rm --env-file .env morag-cli morag-stages.py list

# Execute a single stage
docker run -it --rm --env-file .env \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  morag-cli morag-stages.py stage markdown-conversion /app/input/document.pdf --output-dir /app/output

# Execute full pipeline
docker run -it --rm --env-file .env \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  morag-cli morag-stages.py process /app/input/document.pdf --output-dir /app/output

# Execute stage chain
docker run -it --rm --env-file .env \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  morag-cli morag-stages.py stages "markdown-conversion,chunker,fact-generator" /app/input/document.pdf
```

#### Legacy Script Usage

```bash
# Run a specific legacy script
docker run -it --rm --env-file .env morag-cli webdav-processor.py --help

# Run webdav-processor with arguments
docker run -it --rm --env-file .env morag-cli webdav-processor.py \
  --url https://webdav.example.com \
  --username user \
  --password pass \
  --folder /path/to/folder \
  --extension mp4
```

#### Using Docker Compose (Recommended)

```bash
# Build and run with docker-compose
docker-compose -f docker-compose.cli.yml build

# Show available scripts
docker-compose -f docker-compose.cli.yml run --rm morag-cli

# Stage-based processing with Docker Compose
docker-compose -f docker-compose.cli.yml run --rm morag-cli morag-stages.py list

# Execute full pipeline
docker-compose -f docker-compose.cli.yml run --rm morag-cli \
  morag-stages.py process /app/input/document.pdf --output-dir /app/output

# Execute specific stages
docker-compose -f docker-compose.cli.yml run --rm morag-cli \
  morag-stages.py stages "markdown-conversion,chunker" /app/input/document.pdf

# Legacy webdav-processor
docker-compose -f docker-compose.cli.yml run --rm morag-cli webdav-processor.py \
  --url https://webdav.example.com \
  --username user \
  --password pass \
  --folder /path/to/folder \
  --extension mp4
```

## Available CLI Scripts

### Stage-Based Processing Scripts (New)

- `morag-stages.py` - Main stage controller for canonical stage processing
- `stage-status.py` - Check status of stage processing and output files
- `batch-process-stages.py` - Batch processing with stages

### Legacy CLI Scripts

- `webdav-processor.py` - Process video files from WebDAV server
- `ingest-markdown-folder.py` - Ingest markdown files from a folder
- `fact-extraction.py` - Extract facts from documents
- `graph_extraction.py` - Extract graph data from documents
- `test-*.py` - Various testing scripts
- And more...

## Environment Variables

The CLI container uses the same environment variables as the main MoRAG application. Key variables for CLI scripts:

### Required
- `GEMINI_API_KEY` - Google AI API key for LLM operations
- `QDRANT_HOST` - Qdrant vector database host
- `QDRANT_PORT` - Qdrant vector database port
- `REDIS_URL` - Redis connection URL

### Optional
- `MORAG_PREFERRED_DEVICE` - Device preference (auto, cpu, cuda)
- `MORAG_FORCE_CPU` - Force CPU usage (true/false)
- `MORAG_DEBUG` - Enable debug mode (true/false)
- `MORAG_LOG_LEVEL` - Logging level (INFO, DEBUG, WARNING, ERROR)

## Volume Mounts

The Docker Compose setup automatically mounts the following directories:

- `./temp_storage:/app/temp_storage` - Temporary file storage
- `./uploads:/app/uploads` - File uploads
- `./data:/app/data` - Application data
- `./logs:/app/logs` - Log files
- `./.env:/app/.env:ro` - Environment configuration (read-only)

## Examples

### Stage-Based Processing Examples

#### Document Processing with Stages

```bash
# Process a PDF document through all stages
docker run -it --rm --env-file .env \
  -v $(pwd)/documents:/app/input \
  -v $(pwd)/output:/app/output \
  morag-cli morag-stages.py process /app/input/document.pdf --output-dir /app/output

# Process with optimization stage
docker run -it --rm --env-file .env \
  -v $(pwd)/documents:/app/input \
  -v $(pwd)/output:/app/output \
  morag-cli morag-stages.py process /app/input/document.pdf --optimize --output-dir /app/output

# Execute specific stage chain
docker run -it --rm --env-file .env \
  -v $(pwd)/documents:/app/input \
  -v $(pwd)/output:/app/output \
  morag-cli morag-stages.py stages "markdown-conversion,chunker,fact-generator" /app/input/document.pdf
```

#### Audio/Video Processing with Stages

```bash
# Process audio file
docker run -it --rm --env-file .env \
  -v $(pwd)/media:/app/input \
  -v $(pwd)/output:/app/output \
  morag-cli morag-stages.py process /app/input/audio.mp3 --output-dir /app/output

# Process video file with optimization
docker run -it --rm --env-file .env \
  -v $(pwd)/media:/app/input \
  -v $(pwd)/output:/app/output \
  morag-cli morag-stages.py process /app/input/video.mp4 --optimize --output-dir /app/output
```

#### Check Stage Status

```bash
# Check processing status
docker run -it --rm --env-file .env \
  -v $(pwd)/output:/app/output \
  morag-cli stage-status.py --output-dir /app/output
```

### Legacy Examples

#### WebDAV Video Processing

```bash
# Process MP4 files from WebDAV server
docker run -it --rm --env-file .env morag-cli webdav-processor.py \
  --url https://your-webdav-server.com \
  --username your-username \
  --password your-password \
  --folder "/Videos/ToProcess" \
  --extension mp4 \
  --output-folder "/Videos/Processed"
```

### Markdown Folder Ingestion

```bash
# Ingest markdown files from a local folder
docker run -it --rm --env-file .env \
  -v /path/to/markdown/files:/app/input \
  morag-cli ingest-markdown-folder.py /app/input
```

### Testing Scripts

#### Stage-Based Testing

```bash
# Run system status check
docker run -it --rm --env-file .env morag-cli check-system-status.py

# Test stage processing
docker run -it --rm --env-file .env \
  -v $(pwd)/test_files:/app/input \
  -v $(pwd)/output:/app/output \
  morag-cli morag-stages.py process /app/input/test.pdf --output-dir /app/output

# Check stage status
docker run -it --rm --env-file .env \
  -v $(pwd)/output:/app/output \
  morag-cli stage-status.py --output-dir /app/output
```

#### Legacy Testing

```bash
# Test document processing (legacy)
docker run -it --rm --env-file .env morag-cli test-document.py

# Test video processing (legacy)
docker run -it --rm --env-file .env morag-cli test-video.py
```

## Troubleshooting

### Common Issues

1. **Missing .env file**: The container will copy `.env.example` to `.env` if no `.env` file is found.

2. **Permission errors**: Ensure the mounted directories have proper permissions:
   ```bash
   chmod -R 755 temp_storage uploads data logs
   ```

3. **Memory issues**: For large file processing, increase Docker memory limits:
   ```bash
   docker run -it --rm --memory=4g --env-file .env morag-cli script.py
   ```

4. **GPU support**: The CLI container uses CPU-only PyTorch for compatibility. For GPU support, modify the Dockerfile to install CUDA-enabled PyTorch.

### Debugging

Enable debug mode by setting environment variables:

```bash
# Enable debug logging
echo "MORAG_DEBUG=true" >> .env
echo "MORAG_LOG_LEVEL=DEBUG" >> .env

# Run with debug output
docker run -it --rm --env-file .env morag-cli webdav-processor.py --help
```

### Interactive Shell

To debug issues or explore the container:

```bash
# Start interactive shell
docker run -it --rm --env-file .env --entrypoint /bin/bash morag-cli

# Inside container, you can run scripts directly:
python cli/webdav-processor.py --help
```

## Building for Production

For production use, consider:

1. **Multi-stage builds**: The Dockerfile already uses optimization techniques
2. **Smaller base image**: Consider using `python:3.11-alpine` for smaller images
3. **Security**: Run as non-root user
4. **Caching**: Use Docker BuildKit for better caching

```bash
# Build with BuildKit for better caching
DOCKER_BUILDKIT=1 docker build -f Dockerfile.cli -t morag-cli:latest .

# Tag for registry
docker tag morag-cli:latest your-registry/morag-cli:latest
docker push your-registry/morag-cli:latest
```

## Integration with CI/CD

Example GitHub Actions workflow:

```yaml
name: Run MoRAG CLI
on:
  workflow_dispatch:
    inputs:
      script:
        description: 'CLI script to run'
        required: true
        default: 'check-system-status.py'

jobs:
  run-cli:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build CLI image
        run: docker build -f Dockerfile.cli -t morag-cli .

      - name: Run CLI script
        env:
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
          QDRANT_HOST: ${{ secrets.QDRANT_HOST }}
        run: |
          docker run --rm \
            -e GEMINI_API_KEY \
            -e QDRANT_HOST \
            morag-cli ${{ github.event.inputs.script }}
```
