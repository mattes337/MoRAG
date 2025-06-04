#!/usr/bin/env python3
"""Debug script to check configuration values and test connections."""

import os
import sys
import socket
import asyncio
from urllib.parse import urlparse
sys.path.insert(0, 'src')

from morag.core.config import settings

def main():
    """Run all debug checks."""
    print("=== MoRAG Configuration Debug ===")
    print(f"QDRANT_HOST: {settings.qdrant_host}")
    print(f"QDRANT_PORT: {settings.qdrant_port}")
    print(f"QDRANT_COLLECTION: {settings.qdrant_collection_name}")
    print(f"QDRANT_API_KEY: {'***' if settings.qdrant_api_key else 'NOT SET'}")
    print(f"GEMINI_API_KEY: {'***' if settings.gemini_api_key else 'NOT SET'}")
    print(f"REDIS_URL: {settings.redis_url}")

    # Test Qdrant connection
    print("\n=== Qdrant Connection Test ===")
    test_qdrant_connection()

    # Test dependencies
    print("\n=== Dependency Check ===")
    test_dependencies()

def test_qdrant_connection():
    """Test Qdrant connection."""
    try:
        from qdrant_client import QdrantClient

        if settings.qdrant_host.startswith(('http://', 'https://')):
            parsed = urlparse(settings.qdrant_host)
            hostname = parsed.hostname
            port = parsed.port or (443 if parsed.scheme == 'https' else 6333)
            use_https = parsed.scheme == 'https'

            print(f"Connecting to: {hostname}:{port} (HTTPS: {use_https})")

            # Test DNS resolution
            try:
                ip_address = socket.gethostbyname(hostname)
                print(f"DNS resolution: {hostname} -> {ip_address}")
            except Exception as e:
                print(f"DNS resolution failed: {e}")
                return

            client = QdrantClient(
                host=hostname,
                port=port,
                https=use_https,
                api_key=settings.qdrant_api_key,
                timeout=10
            )
        else:
            print(f"Connecting to: {settings.qdrant_host}:{settings.qdrant_port}")
            client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                api_key=settings.qdrant_api_key,
                timeout=10
            )

        # Test connection
        collections = client.get_collections()
        print(f"✓ Connection successful! Found {len(collections.collections)} collections")
        for col in collections.collections:
            print(f"  - {col.name}")

    except Exception as e:
        print(f"✗ Qdrant connection failed: {e}")

def test_dependencies():
    """Test key dependencies."""
    dependencies = [
        ("faster_whisper", "Audio processing"),
        ("qdrant_client", "Vector database"),
        ("google.genai", "Gemini API"),
        ("redis", "Task queue"),
        ("celery", "Task processing"),
        ("structlog", "Logging"),
    ]

    for module, description in dependencies:
        try:
            __import__(module)
            print(f"✓ {module} - {description}")
        except ImportError:
            print(f"✗ {module} - {description} (MISSING)")

if __name__ == "__main__":
    main()
