#!/usr/bin/env python3
"""
Test script to verify Qdrant connection with HTTPS/TLS configuration.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morag.core.config import settings
from morag.services.storage import qdrant_service
import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

async def test_qdrant_connection():
    """Test Qdrant connection and basic operations."""
    
    print("=== Qdrant Connection Test ===")
    print(f"Host: {settings.qdrant_host}")
    print(f"Port: {settings.qdrant_port}")
    print(f"API Key: {'***' if settings.qdrant_api_key else 'None'}")
    print(f"Collection: {settings.qdrant_collection_name}")
    print()
    
    try:
        # Test connection
        print("1. Testing connection...")
        await qdrant_service.connect()
        print("✅ Connection successful!")
        
        # Test getting collections
        print("\n2. Testing get collections...")
        collections = await qdrant_service.list_collections()
        print(f"✅ Found {len(collections)} collections:")
        for collection in collections:
            print(f"   - {collection.get('name', 'Unknown')}: {collection.get('points_count', 0)} points")
        
        # Test collection info (if collection exists)
        print(f"\n3. Testing collection info for '{settings.qdrant_collection_name}'...")
        try:
            collection_info = await qdrant_service.get_collection_info()
            print(f"✅ Collection info retrieved:")
            print(f"   - Points: {collection_info.get('points_count', 0)}")
            print(f"   - Vectors: {collection_info.get('vectors_count', 0)}")
            print(f"   - Status: {collection_info.get('status', 'Unknown')}")
        except Exception as e:
            print(f"⚠️  Collection '{settings.qdrant_collection_name}' not found or error: {e}")
            
            # Try to create the collection
            print(f"\n4. Creating collection '{settings.qdrant_collection_name}'...")
            await qdrant_service.create_collection(vector_size=768)
            print("✅ Collection created successfully!")
        
        # Test disconnection
        print("\n5. Testing disconnection...")
        await qdrant_service.disconnect()
        print("✅ Disconnection successful!")
        
        print("\n🎉 All tests passed! Qdrant connection is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Connection test failed: {e}")
        logger.error("Qdrant connection test failed", error=str(e), exc_info=True)
        return False

def test_url_parsing():
    """Test URL parsing logic."""
    print("=== URL Parsing Test ===")
    
    test_cases = [
        ("https://qdrant.drydev.de/", 443),
        ("https://qdrant.drydev.de", 443),
        ("http://localhost", 6333),
        ("localhost", 6333),
    ]
    
    for host, port in test_cases:
        print(f"\nTesting: host='{host}', port={port}")
        
        if host.startswith(('http://', 'https://')):
            qdrant_url = host
            if not qdrant_url.endswith('/'):
                qdrant_url += '/'
            # Remove trailing slash and add port if not in URL
            if ':' not in qdrant_url.split('://', 1)[1].split('/')[0] and port != 443:
                qdrant_url = qdrant_url.rstrip('/') + f':{port}'
            print(f"  → URL connection: {qdrant_url}")
        else:
            print(f"  → Host/Port connection: host={host}, port={port}")

if __name__ == "__main__":
    print("Qdrant Connection Test Script")
    print("=" * 50)
    
    # Test URL parsing logic
    test_url_parsing()
    print()
    
    # Test actual connection
    success = asyncio.run(test_qdrant_connection())
    
    if success:
        print("\n✅ Qdrant is properly configured and accessible!")
        sys.exit(0)
    else:
        print("\n❌ Qdrant connection issues detected. Check configuration and network connectivity.")
        sys.exit(1)
