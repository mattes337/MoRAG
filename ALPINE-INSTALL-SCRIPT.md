# Alpine Linux Installation Script for MoRAG

This document describes the `alpine-install.sh` script that automates the installation of MoRAG on Alpine Linux without Qdrant.

## Overview

The script `alpine-install.sh` provides a complete automated installation of MoRAG on Alpine Linux with the following characteristics:

- **CPU-only processing** (no GPU dependencies)
- **No local vector database** (uses external Qdrant server)
- **External Qdrant configuration** for remote vector storage
- **Alpine Linux compatibility** with musl libc
- **Comprehensive dependency management**

## Features

### What the Script Installs

1. **System Dependencies**
   - Build tools (gcc, g++, make, cmake, etc.)
   - Core system libraries (glib, cairo, pango, etc.)
   - Media processing (FFmpeg, audio libraries)
   - OCR and image processing (Tesseract, image libraries)
   - Web scraping dependencies (Chromium, libxml2, etc.)

2. **Python Environment**
   - Python 3.11+ with development tools
   - Virtual environment setup
   - Essential Python packages

3. **Services**
   - Redis (for task queue)
   - Configuration for external Qdrant server

4. **MoRAG Application**
   - All MoRAG dependencies (CPU-only versions)
   - Feature-specific packages (docling, audio, image, office, web)
   - Alpine-specific compatibility fixes

### What the Script Excludes

- **Local vector database** (uses external Qdrant server)
- **GPU dependencies** (CUDA, etc.)
- **Playwright** (Alpine compatibility issues)
- **Dynamic web scraping** (falls back to static methods)

## Usage

### Prerequisites

1. **Alpine Linux 3.18+** (recommended: 3.19+)
2. **Minimum 4GB RAM** (recommended: 8GB+)
3. **10GB+ free disk space**
4. **Internet connection**
5. **Root access** (can be run as root or with sudo)

### Running the Script

1. **Download the script** to your Alpine Linux system:
   ```bash
   wget https://raw.githubusercontent.com/yourusername/morag/main/alpine-install.sh
   # or
   curl -O https://raw.githubusercontent.com/yourusername/morag/main/alpine-install.sh
   ```

2. **Make the script executable**:
   ```bash
   chmod +x alpine-install.sh
   ```

3. **Run the script**:
   ```bash
   # As root (recommended for Alpine)
   ./alpine-install.sh

   # Or as regular user (requires sudo to be installed)
   ./alpine-install.sh
   ```

### What Happens During Installation

The script performs these steps in order:

1. **System Checks**
   - Verifies Alpine Linux
   - Detects root/sudo usage
   - Updates system packages
   - Enables community repository for additional packages

2. **Dependency Installation**
   - Installs build tools and system libraries
   - Sets up Python environment
   - Installs Redis

3. **Application Setup**
   - Clones MoRAG repository
   - Creates Python virtual environment
   - Installs MoRAG dependencies
   - Handles Alpine-specific compatibility issues

4. **Configuration**
   - Creates environment configuration for external Qdrant
   - Sets up directories
   - Configures Redis service

## Post-Installation Steps

After the script completes successfully:

### 1. Configure Environment

Edit the `.env` file and configure your external services:
```bash
cd morag
nano .env
# Add/Update:
# GEMINI_API_KEY=your_actual_api_key_here
# QDRANT_HOST=your_qdrant_server_ip_or_hostname
# QDRANT_API_KEY=your_qdrant_api_key_if_needed
```

### 2. Verify Services

Check that services are running:
```bash
# Test Redis/Valkey (local)
redis-cli ping  # Should return "PONG"
# or if Valkey is installed:
valkey-cli ping  # Should return "PONG"

# Test external Qdrant server
curl http://your_qdrant_server:6333/health
```

### 3. Initialize Database

```bash
cd morag
source venv/bin/activate
python scripts/init_db.py
```

### 4. Start MoRAG Services

In separate terminals:

**Terminal 1 - Start Celery Worker:**
```bash
cd morag
source venv/bin/activate
python scripts/start_worker.py
```

**Terminal 2 - Start API Server:**
```bash
cd morag
source venv/bin/activate
uvicorn src.morag.api.main:app --host 0.0.0.0 --port 8000
```

### 5. Test Installation

```bash
# Test API health
curl http://localhost:8000/health/

# Test document processing
echo "Hello, MoRAG!" > test.txt
curl -X POST "http://localhost:8000/api/v1/ingestion/document" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test.txt"
```

## Configuration Details

### Environment Variables

The script creates a `.env` file with Alpine-specific settings:

```bash
# Device configuration
PREFERRED_DEVICE=cpu
FORCE_CPU=true

# External Qdrant server configuration
QDRANT_HOST=your_qdrant_server_ip_here
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=morag_documents
QDRANT_API_KEY=your_qdrant_api_key_if_needed

# Alpine compatibility
ENABLE_DYNAMIC_WEB_SCRAPING=false
WEB_SCRAPING_FALLBACK_ONLY=true

# Resource limits
MAX_CONCURRENT_TASKS=2
CELERY_WORKER_CONCURRENCY=1
MAX_FILE_SIZE=50MB
WHISPER_MODEL_SIZE=base
```

### Service Ports

- **Redis/Valkey**: localhost:6379
- **External Qdrant**: your_qdrant_server:6333
- **MoRAG API**: localhost:8000 (when started)

## Troubleshooting

### Common Issues

1. **Permission Errors**
   - Run as root for simplest installation
   - If running as user, ensure sudo is installed: `apk add sudo`

2. **Package Build Failures**
   - The script includes Alpine-specific fixes
   - Some packages are rebuilt from source for musl compatibility
   - FFmpeg and Chromium may not be available in all Alpine versions

3. **Missing FFmpeg**
   - Script automatically enables community repository
   - Falls back to alternative media libraries if FFmpeg unavailable
   - Audio processing will still work with fallback libraries

4. **Missing Tesseract OCR**
   - Package names changed in Alpine 3.21+ (tesseract vs tesseract-ocr)
   - Script tries both new and legacy package names
   - OCR functionality will be limited if Tesseract unavailable
   - Can be installed manually later if needed

5. **Python Package Changes**
   - Alpine 3.21+ changed Python package names (python3-pip vs py3-pip)
   - Script tries both new and legacy package names
   - Falls back to ensurepip if pip package unavailable
   - Missing packages will be installed via pip in virtual environment

6. **Redis Replaced with Valkey**
   - Alpine 3.21+ replaced Redis with Valkey due to licensing changes
   - Script tries Valkey first, then Redis6, then Redis
   - Both use same Redis protocol and commands
   - Existing Redis configurations will work with Valkey

7. **Memory Issues**
   - Ensure sufficient RAM (4GB minimum)
   - Consider adding swap space if needed

8. **Service Startup Issues**
   - Check service logs: `tail -f /var/log/messages` (as root) or `sudo tail -f /var/log/messages`
   - Check specific service status: `service redis status` or `service valkey status` (as root) or with sudo
   - Restart services: `service redis restart` or `service valkey restart` (as root) or with sudo

### Getting Help

If you encounter issues:

1. Check the installation logs for error messages
2. Verify system requirements are met
3. Ensure all prerequisites are installed
4. Check service status and logs

## Differences from Standard Installation

This Alpine installation differs from the standard Docker/Ubuntu installation:

### Excluded Components
- **Local vector database** (uses external Qdrant server)
- **Playwright** (Alpine compatibility issues)
- **GPU support** (CPU-only processing)

### Alpine-Specific Adaptations
- **musl libc compatibility** fixes
- **Package rebuilding** from source where needed
- **Conservative resource limits**
- **Static web scraping** only

### External Dependencies
- **External Qdrant server** for vector storage
- **Static content extraction** instead of dynamic web scraping
- **CPU-only** AI processing

## Security Considerations

- Services run under dedicated users (redis)
- File permissions are properly set
- Script can be run as root (standard for Alpine) or with sudo
- API keys should be kept secure in `.env` file
- External Qdrant server should be properly secured and accessible

## Performance Notes

This installation is optimized for:
- **Limited resources** (conservative limits)
- **CPU-only processing** (no GPU acceleration)
- **Alpine Linux efficiency** (minimal footprint)

For production use, consider:
- Increasing resource limits based on available hardware
- Setting up proper monitoring and logging
- Implementing backup strategies for data
- Configuring firewall and access controls
