#!/bin/bash
# Network connectivity test script for GPU workers

set -e

MAIN_SERVER_IP="${1:-localhost}"
echo "🌐 Testing network connectivity to MoRAG server: $MAIN_SERVER_IP"
echo "================================================================"

# Test Redis connectivity
echo "🔍 Testing Redis connectivity (port 6379)..."
if command -v redis-cli &> /dev/null; then
    if redis-cli -h "$MAIN_SERVER_IP" -p 6379 ping > /dev/null 2>&1; then
        echo "✅ Redis connection successful"
    else
        echo "❌ Redis connection failed"
        exit 1
    fi
else
    echo "⚠️  redis-cli not found, testing with telnet..."
    if command -v telnet &> /dev/null; then
        if timeout 5 telnet "$MAIN_SERVER_IP" 6379 < /dev/null > /dev/null 2>&1; then
            echo "✅ Redis port accessible"
        else
            echo "❌ Redis port not accessible"
            exit 1
        fi
    else
        echo "⚠️  Cannot test Redis connectivity (no redis-cli or telnet)"
    fi
fi

# Test Qdrant connectivity
echo "🔍 Testing Qdrant connectivity (port 6333)..."
if command -v curl &> /dev/null; then
    if curl -s "http://$MAIN_SERVER_IP:6333/collections" > /dev/null 2>&1; then
        echo "✅ Qdrant connection successful"
    else
        echo "❌ Qdrant connection failed"
        exit 1
    fi
else
    echo "⚠️  curl not found, testing with telnet..."
    if command -v telnet &> /dev/null; then
        if timeout 5 telnet "$MAIN_SERVER_IP" 6333 < /dev/null > /dev/null 2>&1; then
            echo "✅ Qdrant port accessible"
        else
            echo "❌ Qdrant port not accessible"
            exit 1
        fi
    else
        echo "⚠️  Cannot test Qdrant connectivity (no curl or telnet)"
    fi
fi

# Test HTTP API connectivity
echo "🔍 Testing HTTP API connectivity (port 8000)..."
if command -v curl &> /dev/null; then
    if curl -s "http://$MAIN_SERVER_IP:8000/health" > /dev/null 2>&1; then
        echo "✅ HTTP API connection successful"
    else
        echo "❌ HTTP API connection failed"
        exit 1
    fi
else
    echo "⚠️  curl not found, testing with telnet..."
    if command -v telnet &> /dev/null; then
        if timeout 5 telnet "$MAIN_SERVER_IP" 8000 < /dev/null > /dev/null 2>&1; then
            echo "✅ HTTP API port accessible"
        else
            echo "❌ HTTP API port not accessible"
            exit 1
        fi
    else
        echo "⚠️  Cannot test HTTP API connectivity (no curl or telnet)"
    fi
fi

# Test NFS connectivity (if applicable)
echo "🔍 Testing NFS connectivity (port 2049)..."
if command -v showmount &> /dev/null; then
    if showmount -e "$MAIN_SERVER_IP" > /dev/null 2>&1; then
        echo "✅ NFS server accessible"
    else
        echo "⚠️  NFS server not accessible (may not be configured)"
    fi
else
    echo "⚠️  showmount not found, testing with telnet..."
    if command -v telnet &> /dev/null; then
        if timeout 5 telnet "$MAIN_SERVER_IP" 2049 < /dev/null > /dev/null 2>&1; then
            echo "✅ NFS port accessible"
        else
            echo "⚠️  NFS port not accessible (may not be configured)"
        fi
    else
        echo "⚠️  Cannot test NFS connectivity (no showmount or telnet)"
    fi
fi

# Test basic network connectivity
echo "🔍 Testing basic network connectivity..."
if ping -c 1 "$MAIN_SERVER_IP" > /dev/null 2>&1; then
    echo "✅ Basic network connectivity successful"
else
    echo "❌ Basic network connectivity failed"
    exit 1
fi

echo ""
echo "✅ All network connectivity tests passed!"
echo "🎯 GPU worker should be able to connect to the main server"
