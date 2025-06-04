# MoRAG Ubuntu Linux Installation Guide

This guide provides instructions for installing MoRAG on Ubuntu Linux with full feature support including GPU acceleration and local Qdrant installation.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/morag.git
cd morag

# Make the script executable
chmod +x ubuntu-install.sh

# Run the installation script
./ubuntu-install.sh
```

## Features

The Ubuntu installation script provides:

### ✅ **Full Feature Support**
- Complete system dependencies for all MoRAG features
- GPU acceleration with automatic CUDA installation
- All whisper backends including faster-whisper
- Local Qdrant installation option with Docker
- Enhanced resource limits for development/production

### ✅ **GPU Support**
- Automatic NVIDIA GPU detection
- CUDA toolkit installation (12.2)
- GPU driver setup assistance
- Automatic fallback to CPU when GPU unavailable

### ✅ **Whisper Backend Options**
- **faster-whisper** (preferred for performance)
- **Vosk** (lightweight alternative)
- **whispercpp** (C++ implementation)
- **pywhispercpp** (alternative C++ bindings)
- **whisper-cpp-python** (another C++ option)
- **openai-whisper** (original implementation)

### ✅ **Vector Database Options**
- **Local Qdrant**: Automatic Docker installation and setup
- **External Qdrant**: Configure existing server
- Interactive choice during installation

## System Requirements

### Supported Ubuntu Versions
- Ubuntu 18.04 LTS or later
- Ubuntu 20.04 LTS (recommended)
- Ubuntu 22.04 LTS (latest)

### Hardware Requirements
- **Minimum**: 4GB RAM, 10GB disk space
- **Recommended**: 8GB+ RAM, 20GB+ disk space
- **GPU (Optional)**: NVIDIA GPU with CUDA support

### Prerequisites
- User with sudo privileges
- Internet connection for package downloads

## Installation Process

### 1. System Update and Dependencies
```bash
# Updates system packages
sudo apt update && sudo apt upgrade -y

# Installs build tools
sudo apt install -y build-essential gcc g++ make cmake

# Installs system libraries
sudo apt install -y libffi-dev libssl-dev zlib1g-dev
```

### 2. Media Processing Dependencies
```bash
# Audio/Video processing
sudo apt install -y ffmpeg libavcodec-dev

# Image processing
sudo apt install -y imagemagick libmagickwand-dev

# OCR support
sudo apt install -y tesseract-ocr tesseract-ocr-eng
```

### 3. Python Environment
```bash
# Python installation (version-adaptive)
sudo apt install -y python3.11 python3.11-dev python3-pip

# Virtual environment creation
python3 -m venv venv
source venv/bin/activate
```

### 4. GPU Support (Optional)
```bash
# NVIDIA GPU detection
lspci | grep -i nvidia

# CUDA toolkit installation (if GPU detected)
sudo apt install -y cuda-toolkit-12-2
```

### 5. Services Installation
```bash
# Redis for task queue
sudo apt install -y redis-server
sudo systemctl enable redis-server

# Docker for local Qdrant (optional)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

### 6. MoRAG Installation
```bash
# Install dependencies
pip install -r requirements_ubuntu.txt

# Install MoRAG in development mode
pip install -e .

# Install whisper alternatives
# (automatic fallback chain)
```

## Configuration

### Environment Variables
The script creates a `.env` file with Ubuntu-optimized settings:

```bash
# Device configuration (auto-detected)
PREFERRED_DEVICE=auto  # or cpu if no GPU
FORCE_CPU=false        # or true if no GPU

# Qdrant configuration
QDRANT_HOST=localhost           # or external server
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=morag_documents

# Resource limits (higher than Alpine)
MAX_CONCURRENT_TASKS=4
CELERY_WORKER_CONCURRENCY=2
MAX_FILE_SIZE=100MB
WHISPER_MODEL_SIZE=base

# Whisper backend (auto-selection)
WHISPER_BACKEND=  # empty for auto-selection
```

### Required Configuration
After installation, edit `.env` to add:

```bash
# Required: Add your Gemini API key
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: External Qdrant server (if not using local)
QDRANT_HOST=your_qdrant_server_ip
QDRANT_API_KEY=your_qdrant_api_key
```

## Starting Services

### 1. Initialize Database
```bash
source venv/bin/activate
python scripts/init_db.py
```

### 2. Start Celery Worker
```bash
source venv/bin/activate
python scripts/start_worker.py &
```

### 3. Start API Server
```bash
source venv/bin/activate
uvicorn src.morag.api.main:app --host 0.0.0.0 --port 8000
```

### 4. Verify Installation
```bash
# Health check
curl http://localhost:8000/health/

# Redis check
redis-cli ping

# Qdrant check (if local)
curl http://localhost:6333/health
```

## Testing

### Test Whisper Backends
```bash
python tests/manual/test_whisper_backends.py
```

### Test Ubuntu Installation
```bash
python scripts/test_ubuntu_install.py
```

### Test GPU Support
```bash
# Check NVIDIA GPU
nvidia-smi

# Test CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

## Troubleshooting

### Common Issues

#### GPU Not Detected
```bash
# Install NVIDIA drivers
sudo ubuntu-drivers autoinstall
sudo reboot

# Verify installation
nvidia-smi
```

#### Docker Permission Issues
```bash
# Add user to docker group
sudo usermod -aG docker $USER
# Log out and back in
```

#### Whisper Installation Fails
```bash
# Try alternative backends
pip install vosk
pip install whispercpp
```

#### Redis Connection Issues
```bash
# Check Redis status
sudo systemctl status redis-server

# Restart Redis
sudo systemctl restart redis-server
```

### Performance Optimization

#### For GPU Systems
- Use `WHISPER_MODEL_SIZE=large-v3` for best accuracy
- Set `PREFERRED_DEVICE=auto` for automatic GPU usage
- Monitor GPU memory with `nvidia-smi`

#### For CPU-Only Systems
- Use `WHISPER_MODEL_SIZE=base` for speed
- Set `FORCE_CPU=true` to skip GPU detection
- Consider using Vosk for lightweight processing

## Comparison with Alpine

| Feature | Ubuntu | Alpine |
|---------|--------|--------|
| **Target** | Development/Production | Container/Production |
| **Size** | ~1GB+ | ~200MB |
| **GPU** | Full CUDA support | CPU-only |
| **Whisper** | All backends | Compatible alternatives |
| **Qdrant** | Local + External | External only |
| **Resources** | Higher limits | Conservative |

## Support

### Documentation
- [Installation Comparison](docs/installation-comparison.md)
- [Whisper Backends](docs/whisper-backends.md)
- [GPU/CPU Fallback](docs/gpu-cpu-fallback.md)

### Testing Scripts
- `scripts/test_ubuntu_install.py` - Validate installation
- `tests/manual/test_whisper_backends.py` - Test whisper functionality
- `scripts/test_alpine_whisper_fix.py` - Cross-platform whisper testing

### Getting Help
1. Check the logs: `tail -f logs/morag.log`
2. Run health checks: `curl http://localhost:8000/health/`
3. Test individual components with manual test scripts
4. Review the installation comparison documentation

## Next Steps

After successful installation:

1. **Configure API Keys**: Add your Gemini API key to `.env`
2. **Test Features**: Run the test scripts to verify functionality
3. **Upload Content**: Use the API to upload and process documents
4. **Monitor Performance**: Check GPU usage and processing times
5. **Scale Up**: Adjust resource limits based on your hardware

The Ubuntu installation provides a full-featured MoRAG deployment suitable for both development and production use with optimal performance on GPU-enabled systems.
