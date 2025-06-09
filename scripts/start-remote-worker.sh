#!/bin/bash
# Remote Worker Startup Script

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${1:-$PROJECT_ROOT/configs/remote-worker.env}"

echo "🚀 Starting MoRAG Remote Worker"
echo "================================"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Configuration file not found: $CONFIG_FILE"
    echo "Please copy configs/gpu-worker.env.example to configs/remote-worker.env and configure it"
    exit 1
fi

# Load configuration
echo "📋 Loading configuration from: $CONFIG_FILE"
source "$CONFIG_FILE"

# Validate required environment variables
required_vars=(
    "MORAG_API_KEY"
    "USER_ID"
    "REDIS_URL"
    "MAIN_SERVER_URL"
)

for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "❌ Required environment variable not set: $var"
        exit 1
    fi
done

# Validate API key with server
echo "🔍 Validating API key with server..."
if command -v curl &> /dev/null; then
    response=$(curl -s -w "%{http_code}" -H "Authorization: Bearer $MORAG_API_KEY" "$MAIN_SERVER_URL/health" -o /dev/null)
    if [ "$response" = "200" ] || [ "$response" = "401" ]; then
        echo "✅ Server connection successful"
        # Test API key validation
        auth_response=$(curl -s -H "Authorization: Bearer $MORAG_API_KEY" "$MAIN_SERVER_URL/api/v1/auth/queue-info")
        if echo "$auth_response" | grep -q "error\|unauthorized" 2>/dev/null; then
            echo "❌ API key validation failed"
            echo "Please check your MORAG_API_KEY"
            exit 1
        else
            echo "✅ API key validated successfully"
        fi
    else
        echo "❌ Cannot connect to server at: $MAIN_SERVER_URL"
        exit 1
    fi
else
    echo "⚠️  curl not found. Cannot test server connectivity."
fi

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

# Create and check temp directory
echo "🔍 Setting up temp directory..."
if [ -n "$TEMP_DIR" ]; then
    mkdir -p "$TEMP_DIR"
    if [ -d "$TEMP_DIR" ] && [ -w "$TEMP_DIR" ]; then
        echo "✅ Temp directory ready: $TEMP_DIR"
    else
        echo "❌ Cannot create or write to temp directory: $TEMP_DIR"
        exit 1
    fi
fi

# Calculate user-specific queue name
USER_QUEUE="gpu-tasks-${USER_ID}"

# Set default values
export WORKER_QUEUES="${USER_QUEUE}"
export WORKER_CONCURRENCY="${WORKER_CONCURRENCY:-2}"
export WORKER_NAME="${WORKER_NAME:-remote-worker-${USER_ID}-$(hostname)}"

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

# Start the remote worker
echo "🎯 Starting Remote Celery Worker..."
echo "User ID: $USER_ID"
echo "Worker Name: $WORKER_NAME"
echo "Queue: $WORKER_QUEUES"
echo "Concurrency: $WORKER_CONCURRENCY"
echo "Redis URL: $REDIS_URL"
echo "Server URL: $MAIN_SERVER_URL"
echo ""
echo "⚠️  IMPORTANT: This worker will ONLY process tasks for user: $USER_ID"
echo "⚠️  External services (Qdrant, Gemini) are handled by the main server"
echo ""

exec celery -A morag.worker worker \
    --hostname="$WORKER_NAME@%h" \
    --queues="$WORKER_QUEUES" \
    --concurrency="$WORKER_CONCURRENCY" \
    --loglevel=info \
    --time-limit="${CELERY_TIME_LIMIT:-7800}" \
    --soft-time-limit="${CELERY_SOFT_TIME_LIMIT:-7200}"
