#!/usr/bin/env python3
"""Start Celery worker."""

import sys
import os
from pathlib import Path

# Add packages to path for modular architecture
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "morag" / "src"))

from morag.worker_app import celery_app

if __name__ == "__main__":
    # Use solo pool for Windows compatibility
    pool_type = "solo" if os.name == "nt" else "prefork"

    celery_app.start([
        "worker",
        "--loglevel=info",
        f"--pool={pool_type}",
        "--concurrency=4"
    ])
