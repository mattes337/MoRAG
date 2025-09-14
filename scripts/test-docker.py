#!/usr/bin/env python3
"""
Test script to verify Docker setup is working correctly.
"""

import sys
import time
import requests
import redis
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse


def test_redis():
    """Test Redis connection."""
    print("Testing Redis connection...")
    try:
        r = redis.Redis(host='localhost', port=6379, db=0)
        result = r.ping()
        if result:
            print("✅ Redis: Connection successful")
            return True
        else:
            print("❌ Redis: Ping failed")
            return False
    except Exception as e:
        print(f"❌ Redis: Connection failed - {e}")
        return False


def test_qdrant():
    """Test Qdrant connection."""
    print("Testing Qdrant connection...")
    try:
        client = QdrantClient(host='localhost', port=6333)
        # Try to get collections (should work even if empty)
        collections = client.get_collections()
        print("✅ Qdrant: Connection successful")
        print(f"   Collections: {len(collections.collections)}")
        return True
    except UnexpectedResponse as e:
        if "404" in str(e):
            print("✅ Qdrant: Connection successful (no collections yet)")
            return True
        else:
            print(f"❌ Qdrant: Connection failed - {e}")
            return False
    except Exception as e:
        print(f"❌ Qdrant: Connection failed - {e}")
        return False


def test_qdrant_http():
    """Test Qdrant HTTP endpoint."""
    print("Testing Qdrant HTTP endpoint...")
    try:
        response = requests.get('http://localhost:6333/', timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Qdrant HTTP: {data.get('title', 'Unknown')} v{data.get('version', 'Unknown')}")
            return True
        else:
            print(f"❌ Qdrant HTTP: Status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Qdrant HTTP: Connection failed - {e}")
        return False


def main():
    """Run all tests."""
    print("🐳 Testing Docker Infrastructure Setup")
    print("=" * 50)
    
    tests = [
        test_redis,
        test_qdrant_http,
        test_qdrant,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
        print()
    
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 All tests passed! ({passed}/{total})")
        print("✅ Docker infrastructure is ready for MoRAG")
        return 0
    else:
        print(f"⚠️  Some tests failed ({passed}/{total})")
        print("❌ Please check Docker services and try again")
        return 1


if __name__ == "__main__":
    sys.exit(main())
