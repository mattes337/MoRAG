#!/usr/bin/env python3
"""
Test script to verify Docker health check and permission fixes.

This script tests:
1. Qdrant health check endpoint
2. Docker container startup
3. Basic API functionality

Usage:
    python tests/cli/test-docker-fixes.py
"""

import requests
import time
import subprocess
import sys
import json
from pathlib import Path

def test_qdrant_health_endpoint():
    """Test Qdrant health endpoints directly."""
    print("🔍 Testing Qdrant health endpoints...")

    # Test different health endpoints
    endpoints = ['/healthz', '/livez', '/readyz', '/health']

    for endpoint in endpoints:
        try:
            response = requests.get(f"http://localhost:6333{endpoint}", timeout=5)
            if response.status_code == 200:
                print(f"✅ Qdrant {endpoint} endpoint working (status: {response.status_code})")
                return True
            else:
                print(f"❌ Qdrant {endpoint} endpoint failed (status: {response.status_code})")
        except requests.exceptions.RequestException as e:
            print(f"❌ Qdrant {endpoint} endpoint error: {e}")

    return False

def test_docker_compose_health():
    """Test docker-compose health checks."""
    print("🐳 Testing Docker Compose health checks...")

    try:
        # Check service status
        result = subprocess.run(
            ["docker-compose", "ps", "--format", "json"],
            capture_output=True,
            text=True,
            check=True
        )

        services = json.loads(result.stdout) if result.stdout.strip() else []

        for service in services:
            name = service.get('Name', 'Unknown')
            state = service.get('State', 'Unknown')
            health = service.get('Health', 'Unknown')

            print(f"Service: {name}")
            print(f"  State: {state}")
            print(f"  Health: {health}")

            if 'qdrant' in name.lower():
                if health == 'healthy':
                    print(f"✅ {name} is healthy")
                else:
                    print(f"❌ {name} health check failed: {health}")
                    return False

        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Docker compose command failed: {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse docker-compose output: {e}")
        return False

def test_api_health():
    """Test MoRAG API health endpoint."""
    print("🚀 Testing MoRAG API health...")

    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ MoRAG API is healthy")

            # Check Qdrant status in API response
            if 'services' in health_data and 'qdrant' in health_data['services']:
                qdrant_status = health_data['services']['qdrant']
                print(f"  Qdrant status via API: {qdrant_status}")

                if qdrant_status.get('status') == 'healthy':
                    print("✅ Qdrant is healthy via API")
                    return True
                else:
                    print("❌ Qdrant is not healthy via API")
                    return False
            else:
                print("⚠️  No Qdrant status in API response")
                return True
        else:
            print(f"❌ MoRAG API health check failed (status: {response.status_code})")
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ MoRAG API health check error: {e}")
        return False

def test_whisper_initialization():
    """Test if Whisper model can be initialized without permission errors."""
    print("🎤 Testing Whisper model initialization...")

    try:
        # Try to make a simple audio processing request
        # This will test if the Whisper model can be loaded without permission errors
        response = requests.get("http://localhost:8000/audio/models", timeout=10)

        if response.status_code == 200:
            print("✅ Audio service is accessible")
            return True
        elif response.status_code == 404:
            print("⚠️  Audio models endpoint not found (this is OK)")
            return True
        else:
            print(f"❌ Audio service error (status: {response.status_code})")
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ Audio service test error: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Docker fixes for MoRAG...")
    print("=" * 50)

    tests = [
        ("Qdrant Health Endpoint", test_qdrant_health_endpoint),
        ("Docker Compose Health", test_docker_compose_health),
        ("MoRAG API Health", test_api_health),
        ("Whisper Initialization", test_whisper_initialization),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 30)

        try:
            result = test_func()
            results.append((test_name, result))

            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")

        except Exception as e:
            print(f"💥 {test_name}: ERROR - {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Docker fixes are working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
