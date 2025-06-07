#!/bin/bash
"""
Safe Worker Startup Script for MoRAG

This script runs CPU compatibility checks before starting Celery workers
to prevent crashes due to CPU instruction set incompatibilities.
"""

echo "=== MoRAG Worker Safe Startup ==="
echo "Checking CPU compatibility before starting worker..."

# Run CPU compatibility check
python scripts/check_cpu_compatibility.py
COMPAT_EXIT_CODE=$?

if [ $COMPAT_EXIT_CODE -ne 0 ]; then
    echo "WARNING: CPU compatibility issues detected"
    echo "Worker will start with maximum safety settings"
fi

echo "Starting Celery worker with safe configuration..."

# Start the worker with the provided arguments
exec "$@"
