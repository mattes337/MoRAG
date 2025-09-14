"""Device utilities for MoRAG."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_safe_device(prefer_gpu: bool = True) -> str:
    """Get a safe device for processing.
    
    Args:
        prefer_gpu: Whether to prefer GPU if available
        
    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    if not prefer_gpu:
        return "cpu"
    
    try:
        import torch
        
        # Check for CUDA
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            return device
        
        # Check for MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            logger.info("Using MPS device (Apple Silicon)")
            return device
            
    except ImportError:
        logger.warning("PyTorch not available, falling back to CPU")
    except Exception as e:
        logger.warning(f"Error checking GPU availability: {e}, falling back to CPU")
    
    logger.info("Using CPU device")
    return "cpu"


def get_device_info() -> dict:
    """Get information about available devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        "cpu": True,
        "cuda": False,
        "mps": False,
        "cuda_devices": [],
        "memory_info": {}
    }
    
    try:
        import torch
        
        # CUDA info
        if torch.cuda.is_available():
            info["cuda"] = True
            info["cuda_devices"] = [
                {
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_total": torch.cuda.get_device_properties(i).total_memory,
                    "memory_allocated": torch.cuda.memory_allocated(i),
                    "memory_cached": torch.cuda.memory_reserved(i),
                }
                for i in range(torch.cuda.device_count())
            ]
        
        # MPS info
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            info["mps"] = True
            
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Error getting device info: {e}")
    
    return info


def clear_gpu_memory():
    """Clear GPU memory cache if available."""
    try:
        import torch
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA memory cache")
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
            logger.info("Cleared MPS memory cache")
            
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Error clearing GPU memory: {e}")


def get_optimal_batch_size(base_batch_size: int = 32, device: Optional[str] = None) -> int:
    """Get optimal batch size based on available memory.
    
    Args:
        base_batch_size: Base batch size to start with
        device: Device to check (auto-detected if None)
        
    Returns:
        Optimal batch size
    """
    if device is None:
        device = get_safe_device()
    
    if device == "cpu":
        return base_batch_size
    
    try:
        import torch
        
        if device == "cuda" and torch.cuda.is_available():
            # Get available memory
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated(0)
            available_memory = total_memory - allocated_memory
            
            # Estimate batch size based on available memory
            # This is a rough heuristic - adjust based on your model
            memory_per_sample = 50 * 1024 * 1024  # 50MB per sample (rough estimate)
            max_batch_size = int(available_memory * 0.8 / memory_per_sample)
            
            return min(max_batch_size, base_batch_size * 2)
        
    except Exception as e:
        logger.warning(f"Error calculating optimal batch size: {e}")
    
    return base_batch_size
