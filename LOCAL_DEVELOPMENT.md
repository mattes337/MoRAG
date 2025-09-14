# Local Development Setup (Without Docker)

This guide explains how to run MoRAG locally for development and debugging without Docker.

## Prerequisites

- Python 3.9+ (3.11 recommended)
- Redis server
- Qdrant vector database
- Git

## Quick Setup

### 1. Install Dependencies

```bash
# Clone the repository
git clone https://github.com/mattes337/MoRAG.git
cd MoRAG

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install MoRAG packages in development mode
pip install -e packages/morag-core/
pip install -e packages/morag-services/
pip install -e packages/morag-audio/
pip install -e packages/morag-document/
pip install -e packages/morag-video/
pip install -e packages/morag-image/
pip install -e packages/morag-web/
pip install -e packages/morag-youtube/
pip install -e packages/morag/

# Install additional dependencies
pip install redis celery uvicorn structlog python-dotenv
```

### 2. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
# Required:
GEMINI_API_KEY=your_gemini_api_key_here
REDIS_URL=redis://localhost:6379
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Optional:
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### 3. Start Infrastructure Services

#### Option A: Using Docker for Infrastructure Only
```bash
# Start only Redis and Qdrant with Docker
docker run -d --name morag-redis -p 6379:6379 redis:alpine
docker run -d --name morag-qdrant -p 6333:6333 qdrant/qdrant:latest

# Verify services
redis-cli ping  # Should return PONG
curl http://localhost:6333/health  # Should return health status
```

#### Option B: Native Installation
```bash
# Install Redis (Ubuntu/Debian)
sudo apt update
sudo apt install redis-server
sudo systemctl start redis-server

# Install Qdrant
# Download from https://github.com/qdrant/qdrant/releases
# Or use Docker as shown in Option A
```

### 4. Start MoRAG Services

**Important**: You need to start both the API server AND the Celery worker for full functionality.

```bash
# Terminal 1: Start Celery Worker (REQUIRED for task processing)
cd MoRAG
source venv/bin/activate
python scripts/start_worker.py

# Terminal 2: Start API Server
cd MoRAG
source venv/bin/activate
uvicorn morag.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Verify Setup

```bash
# Test the fix we just applied
python test_qdrant_fix.py

# Test API health
curl http://localhost:8000/health

# Test a simple processing task
python tests/cli/test-simple.py
```

## Debugging Tips

### 1. Enable Debug Logging

Add to your `.env` file:
```bash
LOG_LEVEL=DEBUG
STRUCTLOG_LEVEL=DEBUG
```

### 2. Check Service Status

```bash
# Check Redis
redis-cli ping

# Check Qdrant
curl http://localhost:6333/health

# Check if Celery worker is running
ps aux | grep celery
```

### 3. Monitor Logs

```bash
# Worker logs (run worker with debug)
celery -A morag.worker.celery_app worker --loglevel=debug

# API logs (uvicorn with debug)
uvicorn morag.api.main:app --reload --log-level debug
```

### 4. Test Individual Components

```bash
# Test document processing
python tests/cli/test-document.py sample.pdf

# Test web processing
python tests/cli/test-web.py https://example.com

# Test audio processing (if you have audio files)
python tests/cli/test-audio.py sample.mp3
```

## Common Issues and Solutions

### Issue: Tasks Stay in "PENDING" Status
**Solution**: Make sure Celery worker is running
```bash
python scripts/start_worker.py
```

### Issue: "Can't instantiate abstract class QdrantVectorStorage"
**Solution**: This is fixed! Run the test to verify:
```bash
python test_qdrant_fix.py
```

### Issue: Connection Refused to Qdrant/Redis
**Solution**: Make sure services are running and ports are correct
```bash
# Check if ports are in use
netstat -an | grep 6379  # Redis
netstat -an | grep 6333  # Qdrant
```

### Issue: Import Errors
**Solution**: Make sure all packages are installed in development mode
```bash
pip install -e packages/morag-core/
pip install -e packages/morag-services/
# ... etc for all packages
```

## Development Workflow

1. **Make code changes** in the relevant package
2. **Test changes** using CLI test scripts
3. **Check logs** for any errors
4. **Run integration tests** to ensure everything works together

## Performance Tips

- Use `--reload` with uvicorn for API development
- Use `--loglevel=info` for normal operation, `--loglevel=debug` for debugging
- Monitor memory usage with large files
- Use Redis for caching to improve performance

## Next Steps

Once you have the local setup working:
1. Test with your specific use case
2. Add extensive logging where needed
3. Use the CLI test scripts to reproduce issues
4. Check the worker logs for detailed error information
