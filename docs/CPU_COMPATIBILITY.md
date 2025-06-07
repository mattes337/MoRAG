# CPU Compatibility Guide for MoRAG

This document explains the CPU compatibility fixes implemented in MoRAG to prevent crashes related to PyTorch and ML library instruction set incompatibilities.

## Problem Description

MoRAG workers were experiencing crashes with `SIGILL` (Illegal Instruction) errors when using docling for PDF processing. This typically occurs when:

1. PyTorch is compiled with advanced CPU instruction sets (AVX, AVX2, AVX-512)
2. The target CPU doesn't support these instruction sets
3. ML libraries try to use optimized code paths that aren't available

## Error Symptoms

- Worker processes crash with `signal 4 (SIGILL)`
- Error messages like "Could not initialize NNPACK! Reason: Unsupported hardware"
- Workers exit prematurely during document processing
- Docling/PyTorch operations fail with low-level CPU errors

## Solutions Implemented

### 1. Environment Variables for CPU Safety

The following environment variables are automatically set to ensure safe CPU operation:

```bash
# CPU Threading Limits
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1
NUMEXPR_NUM_THREADS=1

# PyTorch CPU Compatibility
PYTORCH_DISABLE_NNPACK=1
PYTORCH_DISABLE_AVX=1
PYTORCH_DISABLE_AVX2=1

# MoRAG CPU Mode
MORAG_FORCE_CPU=true
MORAG_PREFERRED_DEVICE=cpu
```

### 2. PyTorch CPU-Only Installation

The Dockerfile now explicitly installs PyTorch CPU-only version:

```dockerfile
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3. Docling Fallback Mechanism

Enhanced PDF converter with better fallback handling:

- Checks `MORAG_FORCE_CPU` and `MORAG_DISABLE_DOCLING` environment variables
- Disables docling when CPU compatibility issues are detected
- Falls back to pypdf for PDF processing
- Uses safer docling configuration in CPU mode

### 4. CPU Compatibility Check Script

`scripts/check_cpu_compatibility.py` performs startup checks:

- Detects available CPU instruction sets
- Tests PyTorch compatibility
- Sets appropriate environment variables
- Disables problematic components if needed

### 5. Safe Worker Startup

`scripts/start_worker_safe.sh` ensures workers start safely:

- Runs compatibility checks before starting Celery workers
- Configures environment for maximum compatibility
- Provides clear logging of compatibility status

## Configuration Options

### Force CPU Mode

To explicitly force CPU-only operation:

```bash
export MORAG_FORCE_CPU=true
export MORAG_PREFERRED_DEVICE=cpu
```

### Disable Docling

To disable docling completely:

```bash
export MORAG_DISABLE_DOCLING=true
```

### Custom PyTorch Settings

For advanced users, additional PyTorch environment variables:

```bash
export PYTORCH_DISABLE_NNPACK=1
export PYTORCH_DISABLE_AVX=1
export PYTORCH_DISABLE_AVX2=1
export PYTORCH_DISABLE_AVX512=1
```

## Troubleshooting

### Check CPU Features

Run the compatibility check manually:

```bash
python scripts/check_cpu_compatibility.py
```

### Test PyTorch

Test PyTorch compatibility:

```python
import torch
x = torch.tensor([1.0, 2.0, 3.0])
y = x + 1
print("PyTorch working:", y)
```

### Check Environment Variables

Verify environment variables are set:

```bash
env | grep -E "(PYTORCH|MORAG|OMP|MKL)"
```

### Monitor Worker Logs

Check worker logs for compatibility warnings:

```bash
docker logs morag-worker-1
docker logs morag-worker-2
```

## Performance Impact

The CPU compatibility fixes may impact performance:

- **Docling disabled**: PDF processing uses pypdf instead of docling
- **Single-threaded operations**: Reduced parallelism for stability
- **CPU-only PyTorch**: No GPU acceleration for ML operations
- **Disabled optimizations**: Safer but slower CPU operations

## Recommendations

1. **For Production**: Use these compatibility settings for maximum stability
2. **For Development**: Test with both modes to ensure fallback works
3. **For GPU Systems**: Consider separate GPU workers for ML-intensive tasks
4. **For High Performance**: Use dedicated ML hardware with proper CPU support

## Future Improvements

- Automatic CPU feature detection and selective optimization
- Dynamic fallback based on actual CPU capabilities
- Performance benchmarking for different CPU configurations
- GPU worker separation for ML-intensive operations
