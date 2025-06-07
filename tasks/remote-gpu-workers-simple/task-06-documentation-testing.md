# Task 6: Documentation & Testing

## Objective
Create comprehensive documentation and testing scripts for the simplified remote GPU workers implementation.

## Background
Users need complete setup guides, troubleshooting documentation, and test scripts to successfully deploy and validate remote GPU workers.

## Implementation Steps

### 6.1 Complete Setup Guide

**File**: `docs/remote-gpu-workers-setup.md`

```markdown
# Remote GPU Workers Setup Guide

## Overview
This guide walks you through setting up remote GPU workers for MoRAG to accelerate audio, video, and image processing tasks.

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
pip install -e packages/morag_core
pip install -e packages/morag_services
pip install -e packages/morag_audio
pip install -e packages/morag_video
pip install -e packages/morag_document
pip install -e packages/morag_image
pip install -e packages/morag_web
pip install -e packages/morag_youtube
pip install -e packages/morag

# Setup shared storage (if using NFS)
./scripts/setup-nfs-client.sh YOUR_MAIN_SERVER_IP

# Copy and configure GPU worker settings
cp configs/gpu-worker.env.example configs/gpu-worker.env
# Edit configs/gpu-worker.env with your settings
```

### 3. Configure GPU Worker

Edit `configs/gpu-worker.env`:

```bash
# Required Settings
REDIS_URL=redis://YOUR_MAIN_SERVER_IP:6379/0
QDRANT_URL=http://YOUR_MAIN_SERVER_IP:6333
QDRANT_COLLECTION_NAME=morag_vectors
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
./scripts/start-gpu-worker.sh configs/gpu-worker.env

# Verify worker is running
celery -A morag.worker inspect active_queues
```

### 5. Test GPU Processing

```bash
# Test GPU processing from main server
curl -X POST "http://YOUR_MAIN_SERVER_IP:8000/process/file" \
  -F "file=@test-audio.mp3" \
  -F "gpu=true"

# Check worker status
curl http://YOUR_MAIN_SERVER_IP:8000/api/v1/status/workers
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
# In configs/gpu-worker.env
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

## Security Considerations

### Network Security
- Use VPN or private network for worker communication
- Configure firewall rules to restrict access
- Consider using Redis AUTH for additional security

### File Security
- Ensure shared storage has appropriate permissions
- Regular cleanup of temporary files
- Monitor file access logs

## Scaling

### Multiple GPU Workers
1. Repeat GPU worker setup on additional machines
2. Use unique `WORKER_NAME` for each worker
3. All workers can use the same configuration otherwise

### Load Balancing
- MoRAG automatically distributes tasks across available GPU workers
- Monitor queue lengths and worker utilization
- Add more workers during peak usage periods

## Next Steps

After successful setup:
1. Monitor performance and adjust configuration as needed
2. Set up automated monitoring and alerting
3. Plan for scaling based on usage patterns
4. Consider implementing worker auto-scaling
```

### 6.2 Test Scripts

**File**: `tests/test-gpu-workers.py`

```python
#!/usr/bin/env python3
"""Test script for GPU workers functionality."""

import os
import sys
import time
import requests
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "packages" / "morag" / "src"))

def test_worker_status(server_url):
    """Test worker status endpoint."""
    print("🔍 Testing worker status...")
    
    try:
        response = requests.get(f"{server_url}/api/v1/status/workers")
        response.raise_for_status()
        
        data = response.json()
        print(f"✅ Total workers: {data['workers']['total']}")
        print(f"✅ GPU workers: {data['workers']['gpu']}")
        print(f"✅ CPU workers: {data['workers']['cpu']}")
        print(f"✅ GPU available: {data['gpu_available']}")
        
        return data['workers']['gpu'] > 0
        
    except Exception as e:
        print(f"❌ Worker status test failed: {e}")
        return False

def test_gpu_audio_processing(server_url, audio_file):
    """Test GPU audio processing."""
    print("🎵 Testing GPU audio processing...")
    
    try:
        with open(audio_file, 'rb') as f:
            files = {'file': f}
            data = {'gpu': 'true'}
            
            response = requests.post(
                f"{server_url}/process/file",
                files=files,
                data=data,
                timeout=300
            )
            response.raise_for_status()
        
        result = response.json()
        if result.get('success'):
            print("✅ GPU audio processing successful")
            print(f"   Content type: {result.get('content_type')}")
            print(f"   Processing time: {result.get('processing_time', 'N/A')}s")
            return True
        else:
            print(f"❌ GPU audio processing failed: {result.get('error_message')}")
            return False
            
    except Exception as e:
        print(f"❌ GPU audio processing test failed: {e}")
        return False

def test_cpu_fallback(server_url, audio_file):
    """Test CPU fallback when GPU requested but unavailable."""
    print("🔄 Testing CPU fallback...")
    
    try:
        with open(audio_file, 'rb') as f:
            files = {'file': f}
            data = {'gpu': 'true'}  # Request GPU but should fallback to CPU
            
            response = requests.post(
                f"{server_url}/process/file",
                files=files,
                data=data,
                timeout=300
            )
            response.raise_for_status()
        
        result = response.json()
        if result.get('success'):
            print("✅ CPU fallback successful")
            return True
        else:
            print(f"❌ CPU fallback failed: {result.get('error_message')}")
            return False
            
    except Exception as e:
        print(f"❌ CPU fallback test failed: {e}")
        return False

def create_test_audio_file():
    """Create a simple test audio file."""
    try:
        import numpy as np
        import wave
        
        # Generate 5 seconds of sine wave at 440 Hz
        sample_rate = 44100
        duration = 5
        frequency = 440
        
        t = np.linspace(0, duration, sample_rate * duration, False)
        audio_data = np.sin(2 * np.pi * frequency * t)
        
        # Convert to 16-bit integers
        audio_data = (audio_data * 32767).astype(np.int16)
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        
        with wave.open(temp_file.name, 'w') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        return temp_file.name
        
    except ImportError:
        print("⚠️  numpy/wave not available, using dummy file")
        # Create a dummy file
        temp_file = tempfile.NamedTemporaryFile(suffix='.txt', delete=False)
        temp_file.write(b"dummy audio content for testing")
        temp_file.close()
        return temp_file.name

def main():
    """Run all GPU worker tests."""
    print("🧪 MoRAG GPU Workers Test Suite")
    print("=" * 50)
    
    # Configuration
    server_url = os.getenv('MORAG_SERVER_URL', 'http://localhost:8000')
    audio_file = os.getenv('TEST_AUDIO_FILE')
    
    if not audio_file:
        print("📁 Creating test audio file...")
        audio_file = create_test_audio_file()
        print(f"   Created: {audio_file}")
    
    print(f"🌐 Server URL: {server_url}")
    print(f"🎵 Audio file: {audio_file}")
    print()
    
    # Run tests
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Worker status
    if test_worker_status(server_url):
        tests_passed += 1
    print()
    
    # Test 2: GPU processing (if GPU workers available)
    if test_gpu_audio_processing(server_url, audio_file):
        tests_passed += 1
    print()
    
    # Test 3: CPU fallback
    if test_cpu_fallback(server_url, audio_file):
        tests_passed += 1
    print()
    
    # Cleanup
    if not os.getenv('TEST_AUDIO_FILE'):
        os.unlink(audio_file)
        print(f"🗑️  Cleaned up test file: {audio_file}")
    
    # Results
    print("📊 Test Results")
    print("=" * 20)
    print(f"Passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

**File**: `tests/test-network-connectivity.sh`

```bash
#!/bin/bash
# Network connectivity test script for GPU workers

set -e

MAIN_SERVER_IP="${1:-localhost}"
echo "🌐 Testing network connectivity to MoRAG server: $MAIN_SERVER_IP"
echo "=" * 60

# Test Redis connectivity
echo "🔍 Testing Redis connectivity (port 6379)..."
if command -v redis-cli &> /dev/null; then
    if redis-cli -h "$MAIN_SERVER_IP" -p 6379 ping > /dev/null 2>&1; then
        echo "✅ Redis connection successful"
    else
        echo "❌ Redis connection failed"
        exit 1
    fi
else
    echo "⚠️  redis-cli not found, testing with nc..."
    if nc -z "$MAIN_SERVER_IP" 6379; then
        echo "✅ Redis port accessible"
    else
        echo "❌ Redis port not accessible"
        exit 1
    fi
fi

# Test Qdrant connectivity
echo "🔍 Testing Qdrant connectivity (port 6333)..."
if curl -s "http://$MAIN_SERVER_IP:6333/collections" > /dev/null; then
    echo "✅ Qdrant connection successful"
else
    echo "❌ Qdrant connection failed"
    exit 1
fi

# Test HTTP API connectivity
echo "🔍 Testing HTTP API connectivity (port 8000)..."
if curl -s "http://$MAIN_SERVER_IP:8000/health" > /dev/null; then
    echo "✅ HTTP API connection successful"
else
    echo "❌ HTTP API connection failed"
    exit 1
fi

# Test NFS connectivity (if applicable)
echo "🔍 Testing NFS connectivity (port 2049)..."
if nc -z "$MAIN_SERVER_IP" 2049 2>/dev/null; then
    echo "✅ NFS port accessible"
    
    # Test NFS exports
    if command -v showmount &> /dev/null; then
        if showmount -e "$MAIN_SERVER_IP" 2>/dev/null | grep -q "/mnt/morag-shared"; then
            echo "✅ NFS export found"
        else
            echo "⚠️  NFS export not found (may not be configured)"
        fi
    fi
else
    echo "⚠️  NFS port not accessible (may not be configured)"
fi

echo ""
echo "✅ Network connectivity tests completed successfully!"
```

## Testing

### 6.1 Test Documentation Completeness
```bash
# Check all required documentation exists
ls docs/remote-gpu-workers-setup.md
ls docs/network-requirements.md

# Validate markdown syntax
markdownlint docs/remote-gpu-workers-setup.md
```

### 6.2 Test Setup Scripts
```bash
# Test GPU worker test script
python tests/test-gpu-workers.py

# Test network connectivity script
./tests/test-network-connectivity.sh localhost
```

### 6.3 Test Complete Setup Process
```bash
# Follow complete setup guide on fresh machines
# Verify each step works as documented
# Test troubleshooting scenarios
```

## Acceptance Criteria

- [ ] Complete setup guide covers all installation steps
- [ ] Network requirements clearly documented with specific ports
- [ ] Troubleshooting section addresses common issues
- [ ] Test scripts validate GPU worker functionality
- [ ] Network connectivity test script works
- [ ] Performance tuning guidance provided
- [ ] Security considerations documented
- [ ] Scaling instructions included
- [ ] All scripts are executable and well-tested
- [ ] Documentation is clear and easy to follow

## Files Created

- `docs/remote-gpu-workers-setup.md`
- `tests/test-gpu-workers.py`
- `tests/test-network-connectivity.sh`

## Next Steps

After completing this task:
1. Test complete setup process on actual hardware
2. Gather feedback from users following the documentation
3. Refine documentation based on real-world usage
4. Consider creating video tutorials for complex setup steps

## Notes

- Documentation assumes basic Linux/networking knowledge
- Test scripts provide automated validation of setup
- Troubleshooting section based on common deployment issues
- Setup guide supports both NFS and HTTP file transfer modes
- Security section provides baseline recommendations
