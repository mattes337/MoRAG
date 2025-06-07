# Task 3: GPU Worker Configuration

## Objective
Create configuration files and startup scripts for GPU workers that can run on remote machines with GPU capabilities.

## Background
GPU workers need specific configuration to:
1. Connect to the same Redis instance as the main server
2. Consume only from the 'gpu-tasks' queue
3. Access shared files (via network storage or HTTP)
4. Use GPU-accelerated processing libraries

## Implementation Steps

### 3.1 Create GPU Worker Configuration

**File**: `configs/gpu-worker.env`

```bash
# GPU Worker Configuration
# Copy this file to your GPU machine and modify as needed

# Redis Connection (must match main server)
REDIS_URL=redis://YOUR_MAIN_SERVER_IP:6379/0

# Worker Configuration
WORKER_TYPE=gpu
WORKER_NAME=gpu-worker-01
WORKER_QUEUES=gpu-tasks
WORKER_CONCURRENCY=2

# Celery Configuration
CELERY_SOFT_TIME_LIMIT=7200  # 2 hours
CELERY_TIME_LIMIT=7800       # 2 hours 10 minutes

# File Access Configuration
# Option A: Shared Network Storage
TEMP_DIR=/mnt/shared/morag-temp
UPLOAD_DIR=/mnt/shared/morag-uploads

# Option B: HTTP File Transfer (if shared storage not available)
# MAIN_SERVER_URL=http://YOUR_MAIN_SERVER_IP:8000
# FILE_TRANSFER_MODE=http

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
WHISPER_MODEL_SIZE=large-v3
ENABLE_GPU_ACCELERATION=true

# Qdrant Configuration (must match main server)
QDRANT_URL=http://YOUR_MAIN_SERVER_IP:6333
QDRANT_COLLECTION_NAME=morag_vectors

# Gemini API Configuration
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-flash
GEMINI_VISION_MODEL=gemini-1.5-flash
```

### 3.2 Create GPU Worker Startup Script

**File**: `scripts/start-gpu-worker.sh`

```bash
#!/bin/bash
# GPU Worker Startup Script

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${1:-$PROJECT_ROOT/configs/gpu-worker.env}"

echo "🚀 Starting MoRAG GPU Worker"
echo "================================"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Configuration file not found: $CONFIG_FILE"
    echo "Please copy configs/gpu-worker.env and configure it for your environment"
    exit 1
fi

# Load configuration
echo "📋 Loading configuration from: $CONFIG_FILE"
source "$CONFIG_FILE"

# Validate required environment variables
required_vars=(
    "REDIS_URL"
    "QDRANT_URL" 
    "QDRANT_COLLECTION_NAME"
    "GEMINI_API_KEY"
)

for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "❌ Required environment variable not set: $var"
        exit 1
    fi
done

# Check GPU availability
echo "🔍 Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
    echo "✅ GPU detected"
else
    echo "⚠️  nvidia-smi not found. GPU acceleration may not be available."
fi

# Check Redis connectivity
echo "🔍 Checking Redis connectivity..."
if command -v redis-cli &> /dev/null; then
    if redis-cli -u "$REDIS_URL" ping > /dev/null 2>&1; then
        echo "✅ Redis connection successful"
    else
        echo "❌ Cannot connect to Redis at: $REDIS_URL"
        exit 1
    fi
else
    echo "⚠️  redis-cli not found. Cannot test Redis connectivity."
fi

# Check file access
echo "🔍 Checking file access..."
if [ -n "$TEMP_DIR" ]; then
    if [ -d "$TEMP_DIR" ] && [ -w "$TEMP_DIR" ]; then
        echo "✅ Temp directory accessible: $TEMP_DIR"
    else
        echo "❌ Temp directory not accessible: $TEMP_DIR"
        echo "Please ensure shared storage is mounted or create the directory"
        exit 1
    fi
fi

# Set default values
export WORKER_QUEUES="${WORKER_QUEUES:-gpu-tasks}"
export WORKER_CONCURRENCY="${WORKER_CONCURRENCY:-2}"
export WORKER_NAME="${WORKER_NAME:-gpu-worker-$(hostname)}"

# Change to project directory
cd "$PROJECT_ROOT"

# Install/update dependencies if needed
if [ "$INSTALL_DEPS" = "true" ]; then
    echo "📦 Installing dependencies..."
    pip install -e packages/morag_core
    pip install -e packages/morag_services  
    pip install -e packages/morag_audio
    pip install -e packages/morag_video
    pip install -e packages/morag_document
    pip install -e packages/morag_image
    pip install -e packages/morag_web
    pip install -e packages/morag_youtube
    pip install -e packages/morag
fi

# Start the GPU worker
echo "🎯 Starting Celery worker..."
echo "Worker Name: $WORKER_NAME"
echo "Queues: $WORKER_QUEUES"
echo "Concurrency: $WORKER_CONCURRENCY"
echo "Redis URL: $REDIS_URL"

exec celery -A morag.worker worker \
    --hostname="$WORKER_NAME@%h" \
    --queues="$WORKER_QUEUES" \
    --concurrency="$WORKER_CONCURRENCY" \
    --loglevel=info \
    --time-limit="${CELERY_TIME_LIMIT:-7800}" \
    --soft-time-limit="${CELERY_SOFT_TIME_LIMIT:-7200}"
```

### 3.3 Create Windows GPU Worker Script

**File**: `scripts/start-gpu-worker.bat`

```batch
@echo off
REM GPU Worker Startup Script for Windows

echo 🚀 Starting MoRAG GPU Worker
echo ================================

REM Configuration
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
set CONFIG_FILE=%1
if "%CONFIG_FILE%"=="" set CONFIG_FILE=%PROJECT_ROOT%\configs\gpu-worker.env

REM Check if config file exists
if not exist "%CONFIG_FILE%" (
    echo ❌ Configuration file not found: %CONFIG_FILE%
    echo Please copy configs\gpu-worker.env and configure it for your environment
    exit /b 1
)

REM Load configuration (simplified - user should set environment variables)
echo 📋 Please ensure environment variables are set from: %CONFIG_FILE%
echo See the .env file for required variables

REM Check GPU availability
echo 🔍 Checking GPU availability...
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
if %errorlevel% equ 0 (
    echo ✅ GPU detected
) else (
    echo ⚠️  nvidia-smi failed. GPU acceleration may not be available.
)

REM Set default values
if "%WORKER_QUEUES%"=="" set WORKER_QUEUES=gpu-tasks
if "%WORKER_CONCURRENCY%"=="" set WORKER_CONCURRENCY=2
if "%WORKER_NAME%"=="" set WORKER_NAME=gpu-worker-%COMPUTERNAME%

REM Change to project directory
cd /d "%PROJECT_ROOT%"

REM Start the GPU worker
echo 🎯 Starting Celery worker...
echo Worker Name: %WORKER_NAME%
echo Queues: %WORKER_QUEUES%
echo Concurrency: %WORKER_CONCURRENCY%

celery -A morag.worker worker ^
    --hostname="%WORKER_NAME%@%%h" ^
    --queues="%WORKER_QUEUES%" ^
    --concurrency="%WORKER_CONCURRENCY%" ^
    --loglevel=info ^
    --time-limit=7800 ^
    --soft-time-limit=7200
```

### 3.4 Create Docker GPU Worker Configuration

**File**: `docker/gpu-worker/Dockerfile`

```dockerfile
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    ffmpeg \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy project files
COPY packages/ ./packages/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

# Install Python dependencies
RUN pip3 install -e packages/morag_core && \
    pip3 install -e packages/morag_services && \
    pip3 install -e packages/morag_audio && \
    pip3 install -e packages/morag_video && \
    pip3 install -e packages/morag_document && \
    pip3 install -e packages/morag_image && \
    pip3 install -e packages/morag_web && \
    pip3 install -e packages/morag_youtube && \
    pip3 install -e packages/morag

# Create temp directory
RUN mkdir -p /app/temp && chmod 777 /app/temp

# Set environment variables
ENV PYTHONPATH=/app
ENV TEMP_DIR=/app/temp

# Default command
CMD ["./scripts/start-gpu-worker.sh", "/app/configs/gpu-worker.env"]
```

**File**: `docker/gpu-worker/docker-compose.yml`

```yaml
version: '3.8'

services:
  gpu-worker:
    build: .
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - REDIS_URL=redis://host.docker.internal:6379/0
      - QDRANT_URL=http://host.docker.internal:6333
      - QDRANT_COLLECTION_NAME=morag_vectors
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - WORKER_QUEUES=gpu-tasks
      - WORKER_CONCURRENCY=2
      - TEMP_DIR=/app/temp
    volumes:
      - /path/to/shared/storage:/app/temp  # Adjust path as needed
    depends_on:
      - redis
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    restart: unless-stopped
```

## Testing

### 3.1 Test GPU Worker Startup
```bash
# Test configuration validation
./scripts/start-gpu-worker.sh configs/gpu-worker.env

# Check worker registration
celery -A morag.worker inspect active_queues

# Check GPU worker status
celery -A morag.worker inspect stats
```

### 3.2 Test GPU Task Processing
```bash
# Submit a GPU task and monitor processing
python -c "
from morag.worker import process_file_task_gpu
task = process_file_task_gpu.delay('/path/to/test.mp3', 'audio')
print(f'Task ID: {task.id}')
print(f'Task result: {task.get(timeout=300)}')
"
```

## Acceptance Criteria

- [ ] GPU worker configuration file created with all required settings
- [ ] Startup scripts work on Linux and Windows
- [ ] Docker configuration supports GPU workers
- [ ] GPU workers connect to Redis and consume from 'gpu-tasks' queue
- [ ] File access validation works (shared storage or HTTP)
- [ ] GPU availability detection works
- [ ] Worker registration and monitoring works
- [ ] Configuration validation prevents startup with missing settings

## Files Created

- `configs/gpu-worker.env`
- `scripts/start-gpu-worker.sh`
- `scripts/start-gpu-worker.bat`
- `docker/gpu-worker/Dockerfile`
- `docker/gpu-worker/docker-compose.yml`

## Next Steps

After completing this task:
1. Proceed to Task 4: Task Routing Logic
2. Test GPU worker startup on actual GPU machine
3. Validate file sharing configuration

## Notes

- Configuration supports both shared storage and HTTP file transfer
- Scripts include comprehensive validation and error checking
- Docker configuration uses NVIDIA runtime for GPU access
- Windows batch script provided for Windows GPU machines
- All sensitive configuration externalized to environment variables
