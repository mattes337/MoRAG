#!/usr/bin/env python3
"""
Network diagnostic script for Qdrant connection issues.
"""

import asyncio
import sys
import os
import socket
import ssl
import requests
import httpx
from pathlib import Path
from urllib.parse import urlparse

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morag_core.config import settings

def test_dns_resolution(hostname):
    """Test DNS resolution for the hostname."""
    print(f"Testing DNS resolution for {hostname}...")
    try:
        ip_addresses = socket.gethostbyname_ex(hostname)
        print(f"✅ DNS resolution successful:")
        print(f"   Hostname: {ip_addresses[0]}")
        print(f"   Aliases: {ip_addresses[1]}")
        print(f"   IP addresses: {ip_addresses[2]}")
        return True
    except socket.gaierror as e:
        print(f"❌ DNS resolution failed: {e}")
        return False

def test_tcp_connection(hostname, port, timeout=10):
    """Test TCP connection to hostname:port."""
    print(f"Testing TCP connection to {hostname}:{port}...")
    try:
        sock = socket.create_connection((hostname, port), timeout=timeout)
        sock.close()
        print(f"✅ TCP connection successful")
        return True
    except Exception as e:
        print(f"❌ TCP connection failed: {e}")
        return False

def test_ssl_connection(hostname, port, timeout=10):
    """Test SSL/TLS connection to hostname:port."""
    print(f"Testing SSL/TLS connection to {hostname}:{port}...")
    try:
        context = ssl.create_default_context()
        with socket.create_connection((hostname, port), timeout=timeout) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                print(f"✅ SSL/TLS connection successful")
                print(f"   SSL version: {ssock.version()}")
                print(f"   Cipher: {ssock.cipher()}")
                return True
    except Exception as e:
        print(f"❌ SSL/TLS connection failed: {e}")
        return False

def test_http_request(url, timeout=30):
    """Test HTTP request to the URL."""
    print(f"Testing HTTP request to {url}...")
    try:
        response = requests.get(url, timeout=timeout, verify=True)
        print(f"✅ HTTP request successful")
        print(f"   Status code: {response.status_code}")
        print(f"   Headers: {dict(response.headers)}")
        if response.text:
            print(f"   Response body (first 200 chars): {response.text[:200]}")
        return True
    except Exception as e:
        print(f"❌ HTTP request failed: {e}")
        return False

async def test_httpx_request(url, timeout=30):
    """Test HTTP request using httpx (same library as Qdrant client)."""
    print(f"Testing HTTPX request to {url}...")
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            print(f"✅ HTTPX request successful")
            print(f"   Status code: {response.status_code}")
            print(f"   Headers: {dict(response.headers)}")
            if response.text:
                print(f"   Response body (first 200 chars): {response.text[:200]}")
            return True
    except Exception as e:
        print(f"❌ HTTPX request failed: {e}")
        return False

def test_qdrant_endpoints(base_url):
    """Test common Qdrant endpoints."""
    endpoints = [
        "",  # Root
        "health",  # Health check
        "collections",  # Collections endpoint
    ]
    
    print(f"Testing Qdrant endpoints on {base_url}...")
    for endpoint in endpoints:
        url = f"{base_url.rstrip('/')}/{endpoint}" if endpoint else base_url
        print(f"\n  Testing endpoint: {url}")
        try:
            response = requests.get(url, timeout=10, verify=True)
            print(f"    ✅ Status: {response.status_code}")
            if response.headers.get('content-type', '').startswith('application/json'):
                try:
                    json_data = response.json()
                    print(f"    📄 JSON response: {json_data}")
                except:
                    print(f"    📄 Response: {response.text[:100]}")
            else:
                print(f"    📄 Response: {response.text[:100]}")
        except Exception as e:
            print(f"    ❌ Failed: {e}")

async def main():
    """Main diagnostic function."""
    print("Qdrant Network Diagnostic Script")
    print("=" * 50)
    
    # Parse the Qdrant URL
    qdrant_url = settings.qdrant_host
    parsed = urlparse(qdrant_url)
    hostname = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == 'https' else 6333)
    
    print(f"Configuration:")
    print(f"  URL: {qdrant_url}")
    print(f"  Hostname: {hostname}")
    print(f"  Port: {port}")
    print(f"  Scheme: {parsed.scheme}")
    print(f"  API Key: {'***' if settings.qdrant_api_key else 'None'}")
    print()
    
    # Test 1: DNS Resolution
    print("1. DNS Resolution Test")
    print("-" * 30)
    dns_ok = test_dns_resolution(hostname)
    print()
    
    if not dns_ok:
        print("❌ DNS resolution failed. Cannot proceed with further tests.")
        return False
    
    # Test 2: TCP Connection
    print("2. TCP Connection Test")
    print("-" * 30)
    tcp_ok = test_tcp_connection(hostname, port)
    print()
    
    if not tcp_ok:
        print("❌ TCP connection failed. Cannot proceed with SSL/HTTP tests.")
        return False
    
    # Test 3: SSL/TLS Connection (if HTTPS)
    if parsed.scheme == 'https':
        print("3. SSL/TLS Connection Test")
        print("-" * 30)
        ssl_ok = test_ssl_connection(hostname, port)
        print()
        
        if not ssl_ok:
            print("❌ SSL/TLS connection failed. HTTPS requests will fail.")
            return False
    
    # Test 4: Basic HTTP Request
    print("4. Basic HTTP Request Test")
    print("-" * 30)
    http_ok = test_http_request(qdrant_url)
    print()
    
    # Test 5: HTTPX Request (same as Qdrant client)
    print("5. HTTPX Request Test")
    print("-" * 30)
    httpx_ok = await test_httpx_request(qdrant_url)
    print()
    
    # Test 6: Qdrant-specific endpoints
    print("6. Qdrant Endpoints Test")
    print("-" * 30)
    test_qdrant_endpoints(qdrant_url)
    print()
    
    # Summary
    print("Summary")
    print("=" * 50)
    if dns_ok and tcp_ok and (parsed.scheme != 'https' or ssl_ok):
        print("✅ Network connectivity looks good!")
        print("   The issue might be:")
        print("   - Qdrant server not running")
        print("   - Firewall blocking the connection")
        print("   - Authentication issues")
        print("   - Wrong port or URL")
    else:
        print("❌ Network connectivity issues detected:")
        if not dns_ok:
            print("   - DNS resolution failed")
        if not tcp_ok:
            print("   - TCP connection failed")
        if parsed.scheme == 'https' and not ssl_ok:
            print("   - SSL/TLS connection failed")
    
    return dns_ok and tcp_ok

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
