#!/usr/bin/env python3
"""
Test Qdrant authentication and API key configuration.
"""

import asyncio
import sys
import os
import requests
import httpx
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morag_core.config import settings
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException

def test_requests_with_auth(url, api_key):
    """Test requests library with API key authentication."""
    print(f"Testing requests with API key authentication...")
    
    headers = {
        'api-key': api_key
    }
    
    endpoints = ['health', 'collections']
    
    for endpoint in endpoints:
        test_url = f"{url.rstrip('/')}/{endpoint}"
        print(f"  Testing {endpoint} endpoint: {test_url}")
        
        try:
            response = requests.get(test_url, headers=headers, timeout=10)
            print(f"    ✅ Status: {response.status_code}")
            if response.headers.get('content-type', '').startswith('application/json'):
                try:
                    json_data = response.json()
                    print(f"    📄 Response: {json_data}")
                except:
                    print(f"    📄 Response: {response.text[:200]}")
            else:
                print(f"    📄 Response: {response.text[:200]}")
        except Exception as e:
            print(f"    ❌ Failed: {e}")

async def test_httpx_with_auth(url, api_key):
    """Test httpx library with API key authentication."""
    print(f"Testing httpx with API key authentication...")
    
    headers = {
        'api-key': api_key
    }
    
    endpoints = ['health', 'collections']
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        for endpoint in endpoints:
            test_url = f"{url.rstrip('/')}/{endpoint}"
            print(f"  Testing {endpoint} endpoint: {test_url}")
            
            try:
                response = await client.get(test_url, headers=headers)
                print(f"    ✅ Status: {response.status_code}")
                if response.headers.get('content-type', '').startswith('application/json'):
                    try:
                        json_data = response.json()
                        print(f"    📄 Response: {json_data}")
                    except:
                        print(f"    📄 Response: {response.text[:200]}")
                else:
                    print(f"    📄 Response: {response.text[:200]}")
            except Exception as e:
                print(f"    ❌ Failed: {e}")

def test_qdrant_client_direct(url, api_key):
    """Test QdrantClient directly with different configurations."""
    print(f"Testing QdrantClient directly...")
    
    # Test 1: URL-based connection
    print(f"  Test 1: URL-based connection")
    try:
        client = QdrantClient(
            url=url,
            api_key=api_key,
            timeout=30
        )
        collections = client.get_collections()
        print(f"    ✅ Success! Found {len(collections.collections)} collections")
        for collection in collections.collections:
            print(f"      - {collection.name}")
        client.close()
        return True
    except Exception as e:
        print(f"    ❌ Failed: {e}")
    
    # Test 2: Host/port connection (fallback)
    print(f"  Test 2: Host/port connection")
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        
        client = QdrantClient(
            host=parsed.hostname,
            port=parsed.port or 443,
            https=True,
            api_key=api_key,
            timeout=30
        )
        collections = client.get_collections()
        print(f"    ✅ Success! Found {len(collections.collections)} collections")
        for collection in collections.collections:
            print(f"      - {collection.name}")
        client.close()
        return True
    except Exception as e:
        print(f"    ❌ Failed: {e}")
    
    return False

async def test_qdrant_client_async(url, api_key):
    """Test QdrantClient in async mode."""
    print(f"Testing QdrantClient in async mode...")
    
    try:
        from qdrant_client import AsyncQdrantClient
        
        client = AsyncQdrantClient(
            url=url,
            api_key=api_key,
            timeout=30
        )
        
        collections = await client.get_collections()
        print(f"    ✅ Success! Found {len(collections.collections)} collections")
        for collection in collections.collections:
            print(f"      - {collection.name}")
        
        await client.close()
        return True
    except Exception as e:
        print(f"    ❌ Failed: {e}")
        return False

async def main():
    """Main test function."""
    print("Qdrant Authentication Test Script")
    print("=" * 50)
    
    url = settings.qdrant_host
    api_key = settings.qdrant_api_key
    
    print(f"Configuration:")
    print(f"  URL: {url}")
    print(f"  API Key: {api_key[:10]}...{api_key[-10:] if api_key else 'None'}")
    print()
    
    if not api_key:
        print("❌ No API key configured!")
        return False
    
    # Test 1: Raw HTTP requests with authentication
    print("1. Raw HTTP Requests Test")
    print("-" * 30)
    test_requests_with_auth(url, api_key)
    print()
    
    # Test 2: HTTPX requests with authentication
    print("2. HTTPX Requests Test")
    print("-" * 30)
    await test_httpx_with_auth(url, api_key)
    print()
    
    # Test 3: QdrantClient direct test
    print("3. QdrantClient Direct Test")
    print("-" * 30)
    client_ok = test_qdrant_client_direct(url, api_key)
    print()
    
    # Test 4: AsyncQdrantClient test
    print("4. AsyncQdrantClient Test")
    print("-" * 30)
    async_ok = await test_qdrant_client_async(url, api_key)
    print()
    
    # Summary
    print("Summary")
    print("=" * 50)
    if client_ok or async_ok:
        print("✅ Qdrant client authentication working!")
        if not client_ok:
            print("   Note: Sync client failed, but async client worked")
        if not async_ok:
            print("   Note: Async client failed, but sync client worked")
    else:
        print("❌ Qdrant client authentication failed!")
        print("   Possible issues:")
        print("   - Invalid API key")
        print("   - API key format issues")
        print("   - Server configuration issues")
    
    return client_ok or async_ok

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
