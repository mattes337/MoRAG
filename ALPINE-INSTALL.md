# MoRAG Alpine Linux Installation Guide (CPU Only)

This guide provides step-by-step instructions for installing MoRAG on Alpine Linux without Docker, using CPU-only processing.

## System Requirements

- **Alpine Linux**: 3.18+ (recommended: 3.19+)
- **RAM**: Minimum 4GB, recommended 8GB+
- **CPU**: 2+ cores recommended
- **Storage**: 10GB+ free space
- **Network**: Internet connection for package installation

## 1. System Preparation

### Update System
```bash
# Update package index
sudo apk update

# Upgrade system packages
sudo apk upgrade
```

### Install Essential Build Tools
```bash
# Install build dependencies
sudo apk add --no-cache \
    build-base \
    linux-headers \
    musl-dev \
    gcc \
    g++ \
    make \
    cmake \
    pkgconfig \
    git \
    curl \
    wget \
    unzip
```

## 2. Install System Dependencies

### Core System Libraries
```bash
# Install core system libraries
sudo apk add --no-cache \
    glib-dev \
    cairo-dev \
    pango-dev \
    gdk-pixbuf-dev \
    libffi-dev \
    openssl-dev \
    zlib-dev \
    bzip2-dev \
    xz-dev \
    sqlite-dev \
    readline-dev
```

### Media Processing Dependencies
```bash
# Install FFmpeg and audio/video libraries
sudo apk add --no-cache \
    ffmpeg \
    ffmpeg-dev \
    libsndfile \
    libsndfile-dev \
    portaudio \
    portaudio-dev \
    alsa-lib \
    alsa-lib-dev
```

### OCR and Image Processing
```bash
# Install Tesseract OCR
sudo apk add --no-cache \
    tesseract-ocr \
    tesseract-ocr-data-eng \
    tesseract-ocr-data-deu \
    tesseract-ocr-dev

# Install image processing libraries
sudo apk add --no-cache \
    jpeg-dev \
    png-dev \
    tiff-dev \
    freetype-dev \
    lcms2-dev \
    openjpeg-dev \
    libwebp-dev \
    zlib-dev
```

### Web Scraping Dependencies
```bash
# Install web scraping dependencies
sudo apk add --no-cache \
    libxml2-dev \
    libxslt-dev \
    chromium \
    chromium-chromedriver
```

## 3. Install Python 3.9+

### Install Python and Development Tools
```bash
# Install Python 3.11 (recommended)
sudo apk add --no-cache \
    python3 \
    python3-dev \
    py3-pip \
    py3-virtualenv \
    py3-wheel \
    py3-setuptools

# Verify Python version
python3 --version  # Should be 3.9+
```

### Install Additional Python Dependencies
```bash
# Install Python build dependencies
sudo apk add --no-cache \
    py3-numpy \
    py3-scipy \
    py3-pillow \
    py3-lxml \
    py3-cryptography \
    py3-cffi
```

## 4. Install Redis

### Install and Configure Redis
```bash
# Install Redis
sudo apk add --no-cache redis

# Enable Redis service
sudo rc-update add redis default

# Start Redis
sudo service redis start

# Verify Redis is running
redis-cli ping  # Should return "PONG"
```

### Configure Redis (Optional)
```bash
# Edit Redis configuration if needed
sudo vi /etc/redis.conf

# Key settings for MoRAG:
# maxmemory 512mb
# maxmemory-policy allkeys-lru
# appendonly yes

# Restart Redis after configuration changes
sudo service redis restart
```

## 5. Install Qdrant Vector Database

### Download and Install Qdrant
```bash
# Create qdrant user
sudo adduser -D -s /bin/sh qdrant

# Create directories
sudo mkdir -p /opt/qdrant /var/lib/qdrant /var/log/qdrant
sudo chown qdrant:qdrant /opt/qdrant /var/lib/qdrant /var/log/qdrant

# Download Qdrant binary
cd /tmp
wget https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-musl.tar.gz
tar -xzf qdrant-x86_64-unknown-linux-musl.tar.gz
sudo mv qdrant /opt/qdrant/

# Make executable
sudo chmod +x /opt/qdrant/qdrant
sudo chown qdrant:qdrant /opt/qdrant/qdrant
```

### Create Qdrant Service
```bash
# Create OpenRC service file
sudo tee /etc/init.d/qdrant << 'EOF'
#!/sbin/openrc-run

name="qdrant"
description="Qdrant Vector Database"
command="/opt/qdrant/qdrant"
command_user="qdrant"
command_background="yes"
pidfile="/run/qdrant.pid"
command_args="--config-path /etc/qdrant/config.yaml"

depend() {
    need net
    after firewall
}

start_pre() {
    checkpath --directory --owner qdrant:qdrant --mode 0755 /var/lib/qdrant
    checkpath --directory --owner qdrant:qdrant --mode 0755 /var/log/qdrant
}
EOF

# Make service executable
sudo chmod +x /etc/init.d/qdrant
```

### Configure Qdrant
```bash
# Create config directory
sudo mkdir -p /etc/qdrant

# Create basic configuration
sudo tee /etc/qdrant/config.yaml << 'EOF'
service:
  host: 0.0.0.0
  http_port: 6333
  grpc_port: 6334

storage:
  storage_path: /var/lib/qdrant

log_level: INFO
EOF

# Set permissions
sudo chown -R qdrant:qdrant /etc/qdrant

# Enable and start Qdrant
sudo rc-update add qdrant default
sudo service qdrant start

# Verify Qdrant is running
curl http://localhost:6333/health
```

## 6. Install MoRAG Application

### Clone Repository
```bash
# Clone the MoRAG repository
git clone https://github.com/yourusername/morag.git
cd morag
```

### Create Virtual Environment
```bash
# Create Python virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip wheel setuptools
```

### Install MoRAG Dependencies
```bash
# Install base dependencies (CPU-only)
pip install -e .

# Install additional feature sets (CPU-only versions)
pip install -e ".[docling,audio,image,office,web]"

# Force CPU-only versions for key packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install tensorflow-cpu

# Download spaCy language model
python -m spacy download en_core_web_sm
```

## 7. Configuration

### Create Environment File
```bash
# Copy environment template
cp .env.example .env

# Edit configuration
vi .env
```

### Essential Environment Variables
```bash
# Required settings for Alpine Linux
GEMINI_API_KEY=your_gemini_api_key_here
REDIS_URL=redis://localhost:6379/0
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Force CPU usage
PREFERRED_DEVICE=cpu
FORCE_CPU=true

# File paths
UPLOAD_DIR=./uploads
TEMP_DIR=./temp
LOG_FILE=./logs/morag.log

# API settings
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
```

### Create Required Directories
```bash
# Create application directories
mkdir -p uploads temp logs

# Set permissions
chmod 755 uploads temp logs
```

## 8. Initialize Database

### Run Database Initialization
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Initialize Qdrant collections
python scripts/init_db.py
```

## 9. Start Services

### Start Celery Worker
```bash
# In one terminal, start Celery worker
source venv/bin/activate
python scripts/start_worker.py

# Or manually:
# celery -A src.morag.core.celery_app worker --loglevel=info --pool=solo
```

### Start API Server
```bash
# In another terminal, start the API server
source venv/bin/activate
uvicorn src.morag.api.main:app --host 0.0.0.0 --port 8000 --reload
```

## 10. Verification

### Test System Health
```bash
# Test API health
curl http://localhost:8000/health/

# Test Redis connection
redis-cli ping

# Test Qdrant connection
curl http://localhost:6333/health

# Check queue status
curl http://localhost:8000/api/v1/status/stats/queues
```

### Test Document Processing
```bash
# Test with a simple text file
echo "Hello, MoRAG!" > test.txt
curl -X POST "http://localhost:8000/api/v1/ingestion/document" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test.txt"
```

## 11. Service Management

### Create System Services (Optional)

#### MoRAG Worker Service
```bash
sudo tee /etc/init.d/morag-worker << 'EOF'
#!/sbin/openrc-run

name="morag-worker"
description="MoRAG Celery Worker"
command="/path/to/morag/venv/bin/celery"
command_args="-A src.morag.core.celery_app worker --loglevel=info --pool=solo"
command_user="morag"
command_background="yes"
pidfile="/run/morag-worker.pid"
directory="/path/to/morag"

depend() {
    need redis qdrant
}
EOF

sudo chmod +x /etc/init.d/morag-worker
```

#### MoRAG API Service
```bash
sudo tee /etc/init.d/morag-api << 'EOF'
#!/sbin/openrc-run

name="morag-api"
description="MoRAG API Server"
command="/path/to/morag/venv/bin/uvicorn"
command_args="src.morag.api.main:app --host 0.0.0.0 --port 8000"
command_user="morag"
command_background="yes"
pidfile="/run/morag-api.pid"
directory="/path/to/morag"

depend() {
    need redis qdrant morag-worker
}
EOF

sudo chmod +x /etc/init.d/morag-api
```

## 12. Troubleshooting

### Common Alpine-Specific Issues

#### 1. musl libc Compatibility
```bash
# If you encounter glibc compatibility issues:
sudo apk add --no-cache gcompat

# For Python packages requiring glibc:
pip install --force-reinstall --no-binary=:all: package_name
```

#### 2. Missing Shared Libraries
```bash
# Check for missing libraries:
ldd /path/to/binary

# Install missing libraries:
sudo apk add --no-cache library-name-dev
```

#### 3. Python Package Build Failures
```bash
# Install additional build dependencies:
sudo apk add --no-cache \
    rust \
    cargo \
    libffi-dev \
    openssl-dev

# For packages requiring Rust:
export CARGO_NET_GIT_FETCH_WITH_CLI=true
pip install package_name
```

#### 4. Memory Issues
```bash
# Monitor memory usage:
free -h
top

# Adjust worker concurrency:
# Edit .env file:
MAX_CONCURRENT_TASKS=2
CELERY_WORKER_CONCURRENCY=1
```

#### 5. Service Startup Issues
```bash
# Check service logs:
sudo tail -f /var/log/messages

# Check specific service status:
sudo service redis status
sudo service qdrant status

# Restart services:
sudo service redis restart
sudo service qdrant restart
```

### Performance Optimization

#### 1. Optimize for Limited Resources
```bash
# In .env file, set conservative limits:
MAX_CONCURRENT_TASKS=2
MAX_FILE_SIZE=50MB
WHISPER_MODEL_SIZE=base  # Use smaller model
```

#### 2. Enable Swap (if needed)
```bash
# Create swap file for additional memory:
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent:
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## Next Steps

1. **Configure Monitoring**: Set up log rotation and monitoring
2. **Security**: Configure firewall and access controls
3. **Backup**: Set up regular backups of Qdrant data
4. **Scaling**: Consider horizontal scaling for production use

For more information, see:
- [Main README](README.md)
- [API Documentation](http://localhost:8000/docs)
- [Deployment Guide](DEPLOYMENT.md)
