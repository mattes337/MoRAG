# GPU/CPU Fallback System

MoRAG includes a comprehensive GPU/CPU fallback system that ensures all processing works seamlessly regardless of hardware availability. The system automatically detects available hardware and gracefully falls back to CPU processing when GPU is not available.

## Features

### Automatic Device Detection
- **Auto-detection**: Automatically detects the best available device (GPU/CPU)
- **Safe fallback**: Always falls back to CPU if GPU is not available or fails
- **Configuration options**: Allows manual device preference or forcing CPU usage

### Supported Components
All AI/ML components include GPU fallback:
- **Audio Processing**: Whisper models with CUDA/CPU fallback (auto-detects by default)
- **Image Processing**: EasyOCR with GPU/CPU fallback
- **Speaker Diarization**: PyAnnote.audio with device fallback
- **Topic Segmentation**: SentenceTransformer with device fallback
- **Video Processing**: Auto-detects best available device (GPU/CPU)

## Configuration

### Environment Variables
```bash
# Preferred device for AI processing (auto, cpu, cuda)
PREFERRED_DEVICE=auto

# Force CPU usage even if GPU is available
FORCE_CPU=false
```

### Programmatic Configuration
```python
from morag.core.config import settings

# Check current device configuration
device = settings.get_device()

# Force CPU usage
settings.force_cpu = True

# Set preferred device
settings.preferred_device = "cuda"  # or "cpu" or "auto"
```

## Device Detection Logic

### 1. Force CPU Check
If `force_cpu=True`, always returns "cpu" regardless of GPU availability.

### 2. Preferred Device Handling
- **"auto"**: Automatically detects best available device
- **"cpu"**: Always uses CPU
- **"cuda"/"gpu"**: Attempts GPU, falls back to CPU if unavailable

### 3. GPU Availability Check
```python
def detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass  # PyTorch not available
    except Exception:
        pass  # GPU detection failed
    
    return "cpu"  # Safe fallback
```

## Component-Specific Fallbacks

### Audio Processing (Whisper)
```python
# Automatic device fallback in AudioConfig
config = AudioConfig(device="cuda")  # Will use CPU if CUDA unavailable

# Whisper model loading with fallback
try:
    model = WhisperModel(model_size, device="cuda")
except Exception:
    model = WhisperModel(model_size, device="cpu")  # CPU fallback
```

### Image Processing (EasyOCR)
```python
# EasyOCR initialization with GPU fallback
try:
    reader = easyocr.Reader(['en'], gpu=True)
except Exception:
    reader = easyocr.Reader(['en'], gpu=False)  # CPU fallback
```

### Speaker Diarization (PyAnnote)
```python
# Pipeline initialization with device management
pipeline = Pipeline.from_pretrained(model_name)
try:
    pipeline.to("cuda")
except Exception:
    pipeline.to("cpu")  # CPU fallback
```

### Topic Segmentation (SentenceTransformer)
```python
# SentenceTransformer with device fallback
try:
    model = SentenceTransformer(model_name, device="cuda")
except Exception:
    model = SentenceTransformer(model_name, device="cpu")  # CPU fallback
```

## Error Handling

### GPU-Related Error Detection
The system automatically detects GPU-related errors and triggers fallbacks:
- CUDA out of memory errors
- GPU device not found
- Driver compatibility issues
- PyTorch/CUDA installation problems

### Error Patterns Detected
```python
gpu_error_patterns = [
    'cuda', 'gpu', 'device', 'out of memory', 'memory error',
    'torch', 'nvidia', 'cudnn', 'cublas', 'curand'
]
```

### Automatic Retry Logic
When GPU errors occur:
1. Log warning about GPU failure
2. Automatically retry with CPU
3. Continue processing without interruption
4. Update health monitoring

## Monitoring and Logging

### Device Selection Logging
```
INFO: GPU (CUDA) detected and available device=cuda
INFO: Whisper model loaded successfully device=cuda
WARN: GPU model loading failed, trying CPU fallback error=CUDA out of memory
INFO: Whisper model loaded on CPU fallback
```

### Health Monitoring
The system tracks device usage and failures:
```python
from morag.core.ai_error_handlers import get_ai_service_health

# Check service health including device status
health = get_ai_service_health("whisper")
```

## Best Practices

### 1. Use Auto-Detection
Let the system automatically detect the best device:
```python
# Recommended
settings.preferred_device = "auto"
```

### 2. Handle Mixed Environments
For deployments with mixed GPU/CPU nodes:
```python
# Force CPU for consistent behavior
settings.force_cpu = True
```

### 3. Monitor Resource Usage
Track GPU memory usage and fallback frequency:
```python
# Check if fallbacks are occurring frequently
health_status = get_ai_service_health()
```

### 4. Test Both Modes
Always test your application with both GPU and CPU:
```bash
# Test with GPU
PREFERRED_DEVICE=cuda python your_script.py

# Test with CPU fallback
FORCE_CPU=true python your_script.py
```

## Troubleshooting

### Common Issues

#### 1. PyTorch Not Installed
```
WARNING: PyTorch not available, using CPU
```
**Solution**: Install PyTorch with CUDA support if needed.

#### 2. CUDA Version Mismatch
```
WARNING: GPU detection failed, falling back to CPU
```
**Solution**: Check CUDA driver and PyTorch compatibility.

#### 3. Out of Memory Errors
```
WARNING: GPU model loading failed, trying CPU fallback error=CUDA out of memory
```
**Solution**: Use smaller models or force CPU usage.

### Debugging Device Issues
```python
from morag.core.config import detect_device, get_safe_device

# Test device detection
print(f"Detected device: {detect_device()}")
print(f"Safe device (cuda): {get_safe_device('cuda')}")
print(f"Safe device (auto): {get_safe_device('auto')}")
```

## Performance Considerations

### GPU vs CPU Performance
- **GPU**: Faster for large models and batch processing
- **CPU**: More predictable, lower memory usage, better for small tasks

### Memory Management
- GPU memory is limited and shared
- CPU fallback uses system RAM
- Monitor memory usage in production

### Batch Size Optimization
- Reduce batch sizes for GPU processing
- CPU can handle larger batches with more RAM

## Deployment Recommendations

### Docker Environments
```dockerfile
# Optional GPU support
FROM nvidia/cuda:11.8-runtime-ubuntu20.04
# Fallback works without GPU runtime
```

### Kubernetes
```yaml
# Optional GPU resources
resources:
  limits:
    nvidia.com/gpu: 1  # Optional
  requests:
    cpu: 2
    memory: 4Gi  # Always required
```

### Cloud Deployments
- Use CPU instances for cost optimization
- GPU instances for performance when needed
- Auto-scaling works with both configurations
