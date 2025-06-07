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
    pip install -e packages/morag-core
    pip install -e packages/morag-services  
    pip install -e packages/morag-audio
    pip install -e packages/morag-video
    pip install -e packages/morag-document
    pip install -e packages/morag-image
    pip install -e packages/morag-web
    pip install -e packages/morag-youtube
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
