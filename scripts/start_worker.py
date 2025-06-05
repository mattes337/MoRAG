#!/usr/bin/env python3
"""Start Celery worker."""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morag_services.celery_app import celery_app

if __name__ == "__main__":
    # Use solo pool for Windows compatibility
    pool_type = "solo" if os.name == "nt" else "prefork"

    celery_app.start([
        "worker",
        "--loglevel=info",
        f"--pool={pool_type}",
        "--concurrency=4"
    ])
