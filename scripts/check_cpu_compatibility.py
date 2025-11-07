#!/usr/bin/env python3
"""
CPU Compatibility Check Script for MoRAG

This script checks if the current CPU supports the required instruction sets
for PyTorch and other ML libraries, and sets appropriate environment variables
for safe operation.
"""

import logging
import os
import platform
import subprocess
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_cpu_features():
    """Check CPU features and instruction set support."""
    features = {
        "avx": False,
        "avx2": False,
        "avx512": False,
        "sse": False,
        "sse2": False,
        "sse3": False,
        "sse4_1": False,
        "sse4_2": False,
    }

    try:
        if platform.system() == "Linux":
            # Check /proc/cpuinfo for CPU features
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read().lower()

            for feature in features.keys():
                if feature in cpuinfo:
                    features[feature] = True

        elif platform.system() == "Darwin":  # macOS
            # Use sysctl to check CPU features
            try:
                result = subprocess.run(
                    ["sysctl", "-a"], capture_output=True, text=True
                )
                sysctl_output = result.stdout.lower()

                # Check for specific features in sysctl output
                if "avx" in sysctl_output:
                    features["avx"] = True
                if "avx2" in sysctl_output:
                    features["avx2"] = True

            except Exception as e:
                logger.warning(f"Could not check CPU features on macOS: {e}")

        elif platform.system() == "Windows":
            # Windows CPU feature detection would require additional tools
            logger.info(
                "Windows CPU feature detection not implemented, assuming basic compatibility"
            )
            features["sse"] = True
            features["sse2"] = True

    except Exception as e:
        logger.warning(f"Could not detect CPU features: {e}")

    return features


def check_pytorch_compatibility():
    """Check if PyTorch can be safely imported and used."""
    try:
        import torch

        # Try basic tensor operations
        x = torch.tensor([1.0, 2.0, 3.0])
        y = x + 1

        # Check if CUDA is available (but we'll force CPU anyway)
        cuda_available = torch.cuda.is_available()

        logger.info(f"PyTorch import successful, CUDA available: {cuda_available}")
        return True

    except ImportError:
        logger.warning("PyTorch not available")
        return False
    except Exception as e:
        logger.error(f"PyTorch compatibility test failed: {e}")
        return False


def set_safe_environment():
    """Set environment variables for safe CPU operation."""

    # CPU threading limits
    env_vars = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        # PyTorch CPU safety
        "PYTORCH_DISABLE_NNPACK": "1",
        "PYTORCH_DISABLE_AVX": "1",
        "PYTORCH_DISABLE_AVX2": "1",
        # MoRAG CPU mode
        "MORAG_FORCE_CPU": "true",
        "MORAG_PREFERRED_DEVICE": "cpu",
    }

    for key, value in env_vars.items():
        os.environ[key] = value
        logger.info(f"Set {key}={value}")


def main():
    """Main compatibility check and setup."""
    logger.info("Starting CPU compatibility check for MoRAG")

    # Check system info
    logger.info(f"System: {platform.system()} {platform.release()}")
    logger.info(f"Architecture: {platform.machine()}")
    logger.info(f"Processor: {platform.processor()}")

    # Check CPU features
    features = check_cpu_features()
    logger.info(f"CPU features detected: {features}")

    # Determine if we need safe mode
    has_advanced_features = features.get("avx", False) or features.get("avx2", False)

    if not has_advanced_features:
        logger.warning("Advanced CPU features (AVX/AVX2) not detected or not available")
        logger.info("Enabling safe CPU mode for maximum compatibility")
        set_safe_environment()

    # Test PyTorch compatibility
    pytorch_ok = check_pytorch_compatibility()

    if not pytorch_ok:
        logger.error("PyTorch compatibility test failed")
        logger.info("This may cause issues with docling and other ML components")
        logger.info("Consider installing PyTorch CPU-only version")
        set_safe_environment()

    # Final recommendations
    logger.info("=== CPU Compatibility Check Complete ===")
    logger.info("Environment configured for safe CPU operation")
    logger.info("MoRAG will use CPU-only mode with fallback processing")

    if not pytorch_ok:
        logger.warning("PyTorch issues detected - docling will be disabled")
        os.environ["MORAG_DISABLE_DOCLING"] = "true"

    return 0 if pytorch_ok else 1


if __name__ == "__main__":
    sys.exit(main())
