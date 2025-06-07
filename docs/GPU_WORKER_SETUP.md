# Remote GPU Workers Setup Guide

## Overview
This guide walks you through setting up remote GPU workers for MoRAG to accelerate audio, video, and image processing tasks.

GPU workers provide significant performance improvements for:
- Audio transcription (5-10x faster with GPU-accelerated Whisper)
- Video processing (faster FFmpeg with GPU acceleration)
- Large model inference

## Prerequisites

### Main Server Requirements
- MoRAG server running with Redis and Qdrant
- Network connectivity to GPU workers
- Shared storage (NFS) OR HTTP file transfer capability

### GPU Worker Requirements
- NVIDIA GPU with CUDA support
- CUDA drivers installed (version 12.1+)
- Python 3.11+ with pip
- Network access to main server (Redis, Qdrant, HTTP API)
- Sufficient disk space for temporary files

## Quick Start

### 1. Prepare Main Server

```bash
# Clone MoRAG repository on main server (if not already done)
git clone https://github.com/your-org/morag.git
cd morag

# Setup shared storage (Option A - Recommended)
./scripts/setup-nfs-server.sh

# OR configure for HTTP file transfer (Option B)
# No additional setup needed - HTTP endpoints are built-in
```

### 2. Setup GPU Worker Machine

```bash
# Clone MoRAG repository on GPU worker
git clone https://github.com/your-org/morag.git
cd morag

# Install dependencies
pip install -e packages/morag-core
pip install -e packages/morag-services
pip install -e packages/morag-audio
pip install -e packages/morag-video
pip install -e packages/morag-document
pip install -e packages/morag-image
pip install -e packages/morag-web
pip install -e packages/morag-youtube
pip install -e packages/morag

# Setup shared storage (if using NFS)
./scripts/setup-nfs-client.sh YOUR_MAIN_SERVER_IP

# Copy and configure GPU worker settings
cp configs/gpu-worker.env configs/my-gpu-worker.env
# Edit configs/my-gpu-worker.env with your settings
```

### 3. Configure GPU Worker

Edit `configs/my-gpu-worker.env`:

```bash
# Required Settings
REDIS_URL=redis://YOUR_MAIN_SERVER_IP:6379/0
QDRANT_URL=http://YOUR_MAIN_SERVER_IP:6333
QDRANT_COLLECTION_NAME=morag_documents
GEMINI_API_KEY=your_gemini_api_key_here

# File Access (choose one)
# Option A: Shared Storage
TEMP_DIR=/mnt/morag-shared/temp
UPLOAD_DIR=/mnt/morag-shared/uploads

# Option B: HTTP Transfer
# MAIN_SERVER_URL=http://YOUR_MAIN_SERVER_IP:8000
# FILE_TRANSFER_MODE=http

# GPU Settings
CUDA_VISIBLE_DEVICES=0
WHISPER_MODEL_SIZE=large-v3
ENABLE_GPU_ACCELERATION=true
```

### 4. Start GPU Worker

```bash
# Start GPU worker
./scripts/start-gpu-worker.sh configs/my-gpu-worker.env

# Verify worker is running
celery -A morag.worker inspect active_queues
```

### 5. Test GPU Processing

```bash
# Test GPU processing from main server
curl -X POST "http://YOUR_MAIN_SERVER_IP:8000/api/v1/ingest/file" \
  -F "file=@test-audio.mp3" \
  -F "gpu=true"

# Check worker status
curl http://YOUR_MAIN_SERVER_IP:8000/api/v1/status/workers
```

## File Sharing Options

### Option A: Shared Network Storage (Recommended)

Mount the same storage on both main server and GPU worker:

```bash
# Example: NFS mount
sudo mount -t nfs 192.168.1.100:/shared/morag-temp /mnt/shared/morag-temp

# Set in configuration
TEMP_DIR=/mnt/shared/morag-temp
```

### Option B: HTTP File Transfer

If shared storage is not available, files can be transferred via HTTP:

```bash
# Set in configuration
MAIN_SERVER_URL=http://192.168.1.100:8000
FILE_TRANSFER_MODE=http
```

## Docker Deployment

### Build and Run

```bash
cd docker/gpu-worker

# Set your API key
export GEMINI_API_KEY=your_api_key_here

# Build and start
docker-compose up --build
```

### Custom Configuration

Edit `docker-compose.yml` to match your environment:

```yaml
environment:
  - REDIS_URL=redis://your-server-ip:6379/0
  - QDRANT_URL=http://your-server-ip:6333
  - GEMINI_API_KEY=${GEMINI_API_KEY}
volumes:
  - /your/shared/path:/app/temp
```

## Testing

### Test Configuration

```bash
# Load your configuration
source configs/my-gpu-worker.env

# Run configuration test
python scripts/test-gpu-worker-config.py
```

### Test GPU Task Processing

```bash
# Submit a test task with GPU flag
curl -X POST "http://your-server:8000/api/v1/ingest/file" \
  -F "file=@test-audio.mp3" \
  -F "gpu=true"

# Monitor task processing
celery -A morag.worker inspect active
```

## Monitoring

### Check Worker Status

```bash
# List active workers
celery -A morag.worker inspect active_queues

# Check worker statistics
celery -A morag.worker inspect stats

# Monitor task processing
celery -A morag.worker events
```

### Queue Monitoring

```bash
# Check queue lengths
redis-cli -u $REDIS_URL llen gpu-tasks
redis-cli -u $REDIS_URL llen celery
```

## Detailed Configuration

### Network Configuration

Required ports between main server and GPU workers:
- **6379/tcp**: Redis (task queue)
- **6333/tcp**: Qdrant (vector database)
- **8000/tcp**: HTTP API (file transfer, if using HTTP mode)
- **2049/tcp**: NFS (if using shared storage)

### Firewall Setup

On main server:
```bash
# Allow access from GPU worker
sudo ufw allow from GPU_WORKER_IP to any port 6379
sudo ufw allow from GPU_WORKER_IP to any port 6333
sudo ufw allow from GPU_WORKER_IP to any port 8000
sudo ufw allow from GPU_WORKER_IP to any port 2049  # If using NFS
```

On GPU worker:
```bash
# Allow outbound connections
sudo ufw allow out to MAIN_SERVER_IP port 6379
sudo ufw allow out to MAIN_SERVER_IP port 6333
sudo ufw allow out to MAIN_SERVER_IP port 8000
sudo ufw allow out to MAIN_SERVER_IP port 2049  # If using NFS
```

### Performance Tuning

GPU worker configuration for optimal performance:
```bash
# In configs/my-gpu-worker.env
WORKER_CONCURRENCY=2          # Adjust based on GPU memory
CELERY_SOFT_TIME_LIMIT=7200   # 2 hours
CELERY_TIME_LIMIT=7800        # 2 hours 10 minutes
WHISPER_MODEL_SIZE=large-v3   # Best quality, requires more GPU memory
```

## Troubleshooting

### Common Issues

#### GPU Worker Not Connecting
```bash
# Check Redis connectivity
redis-cli -h MAIN_SERVER_IP -p 6379 ping

# Check Qdrant connectivity
curl http://MAIN_SERVER_IP:6333/collections

# Check firewall rules
sudo ufw status
```

#### File Access Issues
```bash
# For NFS: Check mount status
mountpoint /mnt/morag-shared
ls -la /mnt/morag-shared

# For HTTP: Check API connectivity
curl http://MAIN_SERVER_IP:8000/health
```

#### GPU Not Detected
```bash
# Check NVIDIA drivers
nvidia-smi

# Check CUDA installation
nvcc --version

# Check GPU visibility
echo $CUDA_VISIBLE_DEVICES
```

#### Tasks Not Routing to GPU Worker
```bash
# Check worker registration
celery -A morag.worker inspect active_queues

# Check queue status
curl http://MAIN_SERVER_IP:8000/api/v1/status/workers

# Check task routing logs
tail -f /var/log/morag/worker.log
```

### Performance Issues

#### Slow Processing
- Increase `WORKER_CONCURRENCY` if GPU memory allows
- Use larger Whisper model (`large-v3`) for better quality
- Ensure GPU worker has sufficient CPU and RAM

#### High Network Usage
- Use NFS instead of HTTP file transfer
- Reduce `WORKER_CONCURRENCY` to limit parallel transfers
- Consider local caching strategies

## Monitoring

### Worker Status
```bash
# Check all workers
curl http://MAIN_SERVER_IP:8000/api/v1/status/workers

# Check queue lengths
curl http://MAIN_SERVER_IP:8000/api/v1/status/stats/queues

# Monitor active tasks
curl http://MAIN_SERVER_IP:8000/api/v1/status/
```

### Performance Metrics
```bash
# GPU utilization
nvidia-smi -l 1

# System resources
htop

# Network usage
iftop
```

## Performance Tips

1. **Concurrency**: Start with 2 workers per GPU, adjust based on memory usage
2. **Model Size**: Use `large-v3` Whisper model for best quality/speed balance
3. **Batch Size**: Increase batch sizes for better GPU utilization
4. **Memory**: Monitor GPU memory usage and adjust concurrency accordingly

## Security Considerations

1. **Network**: Use VPN or private networks for Redis/Qdrant connections
2. **API Keys**: Store API keys securely, never commit to version control
3. **File Access**: Ensure shared storage has appropriate permissions
4. **Firewall**: Only open required ports (Redis: 6379, Qdrant: 6333)
