# MoRAG Installation Comparison: Alpine vs Ubuntu

This document compares the two installation scripts for MoRAG on different Linux distributions.

## Overview

| Feature | Alpine Linux | Ubuntu Linux |
|---------|--------------|--------------|
| **Target Use Case** | Lightweight containers, production | Development, full-featured systems |
| **Package Manager** | apk | apt |
| **Python Version** | 3.11 (fixed) | 3.8+ (version-adaptive) |
| **GPU Support** | CPU-only | Full CUDA support |
| **Container Size** | Minimal (~200MB) | Larger (~1GB+) |
| **Whisper Support** | Multiple alternatives | All backends including faster-whisper |
| **Qdrant** | External only | Local + External options |

## Installation Scripts

### Alpine Linux (`alpine-install.sh`)
- **Purpose**: Lightweight, container-optimized installation
- **Target**: Production deployments, resource-constrained environments
- **Dependencies**: Minimal system packages, CPU-only processing
- **Whisper**: Alternative backends (Vosk, WhisperCpp, etc.)
- **Vector DB**: External Qdrant server required

### Ubuntu Linux (`ubuntu-install.sh`)
- **Purpose**: Full-featured development and production installation
- **Target**: Development machines, GPU-enabled servers
- **Dependencies**: Complete system packages, GPU support
- **Whisper**: All backends including faster-whisper
- **Vector DB**: Optional local Qdrant with Docker

## Detailed Comparison

### System Dependencies

#### Alpine Linux
```bash
# Minimal packages for container environments
apk add --no-cache \
    build-base gcc g++ make cmake \
    python3 python3-dev py3-pip \
    ffmpeg tesseract-ocr chromium \
    redis
```

#### Ubuntu Linux
```bash
# Full development environment
apt install -y \
    build-essential gcc g++ make cmake \
    python3.11 python3.11-dev python3-pip \
    ffmpeg tesseract-ocr chromium-browser \
    redis-server \
    cuda-toolkit-12-2  # GPU support
```

### Whisper Backend Support

#### Alpine Linux
- **Primary**: Vosk (lightweight, Alpine-friendly)
- **Alternatives**: WhisperCpp, PyWhisperCpp, whisper-cpp-python
- **Excluded**: faster-whisper (compilation issues)
- **Fallback**: OpenAI whisper (dependency conflicts possible)

#### Ubuntu Linux
- **Primary**: faster-whisper (best performance)
- **Alternatives**: All Alpine options plus faster-whisper
- **GPU Support**: Full CUDA acceleration for whisper models
- **Fallback**: Comprehensive fallback chain

### GPU Support

#### Alpine Linux
```bash
# CPU-only configuration
PREFERRED_DEVICE=cpu
FORCE_CPU=true
```

#### Ubuntu Linux
```bash
# Auto-detection with GPU support
PREFERRED_DEVICE=auto  # or gpu if available
FORCE_CPU=false
# Includes CUDA toolkit installation
```

### Vector Database Options

#### Alpine Linux
```bash
# External Qdrant only
QDRANT_HOST=your_qdrant_server_ip_here
QDRANT_PORT=6333
# No local installation
```

#### Ubuntu Linux
```bash
# Interactive choice
read -p "Do you want to install Qdrant locally? (y/N): "
# Includes Docker and docker-compose setup
# Creates docker-compose.qdrant.yml
```

### Resource Configuration

#### Alpine Linux
```bash
# Conservative limits
MAX_CONCURRENT_TASKS=2
CELERY_WORKER_CONCURRENCY=1
MAX_FILE_SIZE=50MB
WHISPER_MODEL_SIZE=base
```

#### Ubuntu Linux
```bash
# Higher limits
MAX_CONCURRENT_TASKS=4
CELERY_WORKER_CONCURRENCY=2
MAX_FILE_SIZE=100MB
WHISPER_MODEL_SIZE=base  # Can use larger models
```

## Use Case Recommendations

### Choose Alpine Linux When:
- ✅ **Container Deployments**: Docker/Kubernetes production environments
- ✅ **Resource Constraints**: Limited CPU/memory/storage
- ✅ **Security Focus**: Minimal attack surface
- ✅ **CPU-Only Processing**: No GPU requirements
- ✅ **External Services**: Using external Qdrant/Redis
- ✅ **Microservices**: Part of larger distributed system

### Choose Ubuntu Linux When:
- ✅ **Development Environment**: Local development and testing
- ✅ **GPU Processing**: CUDA-enabled AI acceleration
- ✅ **Full Features**: Need all MoRAG capabilities
- ✅ **Local Services**: Want local Qdrant/Redis installation
- ✅ **Performance**: Maximum processing performance
- ✅ **Flexibility**: Need multiple whisper backend options

## Installation Commands

### Alpine Linux
```bash
# Clone repository
git clone https://github.com/yourusername/morag.git
cd morag

# Run Alpine installation
chmod +x alpine-install.sh
./alpine-install.sh

# Configure external Qdrant
nano .env  # Update QDRANT_HOST
```

### Ubuntu Linux
```bash
# Clone repository
git clone https://github.com/yourusername/morag.git
cd morag

# Run Ubuntu installation
chmod +x ubuntu-install.sh
./ubuntu-install.sh

# Follow prompts for Qdrant installation
# GPU drivers will be detected automatically
```

## Post-Installation

### Common Steps (Both Distributions)
```bash
# Activate environment
source venv/bin/activate

# Configure API keys
nano .env  # Add GEMINI_API_KEY

# Initialize database
python scripts/init_db.py

# Start worker
python scripts/start_worker.py &

# Start API server
uvicorn src.morag.api.main:app --host 0.0.0.0 --port 8000
```

### Testing Installation
```bash
# Test whisper backends
python tests/manual/test_whisper_backends.py

# Test Ubuntu-specific features
python scripts/test_ubuntu_install.py

# Test Alpine-specific features  
python scripts/test_alpine_whisper_fix.py

# Health check
curl http://localhost:8000/health/
```

## Migration Between Distributions

### Alpine to Ubuntu
1. Export data from Alpine deployment
2. Install Ubuntu version with GPU support
3. Import data and update configuration
4. Benefit from faster-whisper and GPU acceleration

### Ubuntu to Alpine
1. Export data from Ubuntu installation
2. Set up external Qdrant server
3. Install Alpine version in container
4. Configure external service connections

## Performance Comparison

| Metric | Alpine Linux | Ubuntu Linux |
|--------|--------------|--------------|
| **Container Size** | ~200MB | ~1GB+ |
| **Memory Usage** | ~512MB | ~1GB+ |
| **Startup Time** | ~10s | ~30s |
| **Whisper Speed** | Moderate (CPU) | Fast (GPU) |
| **Build Time** | ~5min | ~15min |
| **Security** | High (minimal) | Good (standard) |

## Troubleshooting

### Alpine-Specific Issues
- **Whisper compilation**: Use alternative backends
- **musl libc**: Some Python packages may not work
- **Limited packages**: Some features may be unavailable

### Ubuntu-Specific Issues
- **GPU drivers**: May need manual NVIDIA driver installation
- **Docker permissions**: User may need to be in docker group
- **Resource usage**: Higher memory and disk requirements

## Conclusion

Both installation scripts provide robust MoRAG deployments optimized for their target environments:

- **Alpine**: Perfect for production containers and resource-constrained deployments
- **Ubuntu**: Ideal for development and GPU-accelerated processing

Choose based on your specific requirements for performance, resources, and deployment environment.
