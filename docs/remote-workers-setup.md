# User-Specific Remote Workers Setup Guide

## Overview
This guide walks you through setting up user-specific remote workers for MoRAG that authenticate with API keys and process only their user's tasks.

## Prerequisites

### Main Server Requirements
- MoRAG server running with Redis
- API key management system configured
- Network connectivity to remote workers
- HTTP file transfer capability

### Remote Worker Requirements
- NVIDIA GPU with CUDA support (recommended)
- CUDA drivers installed (version 12.1+)
- Python 3.11+ with pip
- Network access to main server (Redis and HTTP API only)
- Sufficient disk space for temporary files
- Valid API key for authentication

## Quick Start

### 1. Prepare Main Server

```bash
# Clone MoRAG repository on main server (if not already done)
git clone https://github.com/your-org/morag.git
cd morag

# Start MoRAG server with Redis
docker-compose up -d

# Create API key for remote worker user
curl -X POST "http://localhost:8000/api/v1/auth/create-key" \
  -F "user_id=gpu_user_01" \
  -F "description=GPU Worker for User 01" \
  -F "expires_days=365"

# Note the returned API key for worker configuration
```

### 2. Setup GPU Worker Machine

```bash
# Clone MoRAG repository on GPU worker
git clone https://github.com/your-org/morag.git
cd morag

# Install dependencies
pip install -e packages/morag_core
pip install -e packages/morag_services
pip install -e packages/morag_audio
pip install -e packages/morag_video
pip install -e packages/morag_document
pip install -e packages/morag_image
pip install -e packages/morag_web
pip install -e packages/morag_youtube
pip install -e packages/morag

# Copy and configure remote worker settings
cp configs/gpu-worker.env.example configs/remote-worker.env
# Edit configs/remote-worker.env with your settings
```

### 3. Configure Remote Worker

Edit `configs/remote-worker.env`:

```bash
# Required Settings
MORAG_API_KEY=your_api_key_from_step_1
USER_ID=gpu_user_01
REDIS_URL=redis://YOUR_MAIN_SERVER_IP:6379/0
MAIN_SERVER_URL=http://YOUR_MAIN_SERVER_IP:8000

# Worker Configuration
WORKER_NAME=remote-worker-01
WORKER_CONCURRENCY=2
TEMP_DIR=/tmp/morag-remote-worker

# GPU Settings
CUDA_VISIBLE_DEVICES=0
WHISPER_MODEL_SIZE=large-v3
ENABLE_GPU_ACCELERATION=true

# Processing Configuration
ENABLE_DIARIZATION=true
ENABLE_TOPIC_SEGMENTATION=true
USE_DOCLING=true
ENABLE_OCR=true

# YouTube Configuration (optional)
YOUTUBE_COOKIES_FILE=/path/to/youtube_cookies.txt
```

### 4. Start Remote Worker

```bash
# Start remote worker
./scripts/start-remote-worker.sh configs/remote-worker.env

# Verify worker is running
curl http://YOUR_MAIN_SERVER_IP:8000/api/v1/status/workers
```

### 5. Test GPU Processing

```bash
# Test GPU processing from main server
curl -X POST "http://YOUR_MAIN_SERVER_IP:8000/process/file" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "file=@test-audio.mp3" \
  -F "gpu=true"

# Check worker status
curl -H "Authorization: Bearer YOUR_API_KEY" \
  http://YOUR_MAIN_SERVER_IP:8000/api/v1/auth/queue-info
```

## Detailed Configuration

### Network Configuration

Required ports between main server and GPU workers:
- **6379/tcp**: Redis (task queue)
- **8000/tcp**: HTTP API (file transfer and authentication)

### Firewall Setup

On main server:
```bash
# Allow access from GPU worker
sudo ufw allow from GPU_WORKER_IP to any port 6379
sudo ufw allow from GPU_WORKER_IP to any port 8000
```

On GPU worker:
```bash
# Allow outbound connections
sudo ufw allow out to MAIN_SERVER_IP port 6379
sudo ufw allow out to MAIN_SERVER_IP port 8000
```

### Performance Tuning

GPU worker configuration for optimal performance:
```bash
# In configs/remote-worker.env
WORKER_CONCURRENCY=2          # Adjust based on GPU memory
CELERY_SOFT_TIME_LIMIT=7200   # 2 hours
CELERY_TIME_LIMIT=7800        # 2 hours 10 minutes
WHISPER_MODEL_SIZE=large-v3   # Best quality, requires more GPU memory
```

## Troubleshooting

### Common Issues

#### Remote Worker Not Connecting
```bash
# Check Redis connectivity
redis-cli -h MAIN_SERVER_IP -p 6379 ping

# Check API connectivity
curl http://MAIN_SERVER_IP:8000/health

# Check API key validation
curl -H "Authorization: Bearer YOUR_API_KEY" \
  http://MAIN_SERVER_IP:8000/api/v1/auth/queue-info
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
curl http://MAIN_SERVER_IP:8000/api/v1/status/workers

# Check user queue info
curl -H "Authorization: Bearer YOUR_API_KEY" \
  http://MAIN_SERVER_IP:8000/api/v1/auth/queue-info

# Check worker logs
tail -f /var/log/morag/worker.log
```

### Performance Issues

#### Slow Processing
- Increase `WORKER_CONCURRENCY` if GPU memory allows
- Use larger Whisper model (`large-v3`) for better quality
- Ensure GPU worker has sufficient CPU and RAM

#### High Network Usage
- Consider local caching strategies
- Reduce `WORKER_CONCURRENCY` to limit parallel transfers

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

## Security Considerations

### Network Security
- Use VPN or private network for worker communication
- Configure firewall rules to restrict access
- Consider using Redis AUTH for additional security

### API Key Security
- Keep API keys secure and don't share them
- Use unique API keys for each worker
- Regularly rotate API keys
- Monitor API key usage

## Scaling

### Multiple GPU Workers
1. Repeat GPU worker setup on additional machines
2. Use unique `WORKER_NAME` for each worker
3. All workers can use the same `USER_ID` and `MORAG_API_KEY`

### Load Balancing
- MoRAG automatically distributes tasks across available GPU workers for the same user
- Monitor queue lengths and worker utilization
- Add more workers during peak usage periods

## Next Steps

After successful setup:
1. Monitor performance and adjust configuration as needed
2. Set up automated monitoring and alerting
3. Plan for scaling based on usage patterns
4. Consider implementing worker auto-scaling
