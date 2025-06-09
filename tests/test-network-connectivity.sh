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
        echo "   Check if Redis is running on the main server"
        echo "   Check firewall rules allow port 6379"
        exit 1
    fi
else
    echo "⚠️  redis-cli not found, testing with nc..."
    if nc -z "$MAIN_SERVER_IP" 6379 2>/dev/null; then
        echo "✅ Redis port accessible"
    else
        echo "❌ Redis port not accessible"
        echo "   Install redis-cli for better testing: apt-get install redis-tools"
        exit 1
    fi
fi

# Test HTTP API connectivity
echo "🔍 Testing HTTP API connectivity (port 8000)..."
if curl -s "http://$MAIN_SERVER_IP:8000/health" > /dev/null; then
    echo "✅ HTTP API connection successful"
    
    # Test API response
    response=$(curl -s "http://$MAIN_SERVER_IP:8000/health")
    if echo "$response" | grep -q "status\|message" 2>/dev/null; then
        echo "✅ API health check response valid"
    else
        echo "⚠️  API health check response unexpected: $response"
    fi
else
    echo "❌ HTTP API connection failed"
    echo "   Check if MoRAG server is running on port 8000"
    echo "   Check firewall rules allow port 8000"
    exit 1
fi

# Test API key endpoints
echo "🔍 Testing API key endpoints..."
if curl -s "http://$MAIN_SERVER_IP:8000/api/v1/auth/queue-info" > /dev/null; then
    echo "✅ API key endpoints accessible"
else
    echo "❌ API key endpoints not accessible"
    echo "   Check if authentication middleware is configured"
    exit 1
fi

# Test worker status endpoint
echo "🔍 Testing worker status endpoint..."
if curl -s "http://$MAIN_SERVER_IP:8000/api/v1/status/workers" > /dev/null; then
    echo "✅ Worker status endpoint accessible"
else
    echo "❌ Worker status endpoint not accessible"
    echo "   Check if worker status endpoint is implemented"
    exit 1
fi

# Test file transfer endpoints
echo "🔍 Testing file transfer endpoints..."
if curl -s -X POST "http://$MAIN_SERVER_IP:8000/api/v1/files/download" \
   -H "Content-Type: application/json" \
   -d '{"file_path": "/nonexistent"}' > /dev/null 2>&1; then
    echo "✅ File transfer endpoints accessible"
else
    echo "⚠️  File transfer endpoints may not be accessible (expected for unauthenticated requests)"
fi

# Test network latency
echo "🔍 Testing network latency..."
if command -v ping &> /dev/null; then
    latency=$(ping -c 3 "$MAIN_SERVER_IP" 2>/dev/null | tail -1 | awk -F '/' '{print $5}' 2>/dev/null || echo "unknown")
    if [ "$latency" != "unknown" ] && [ -n "$latency" ]; then
        echo "✅ Average latency: ${latency}ms"
        
        # Check if latency is reasonable for remote workers
        latency_num=$(echo "$latency" | cut -d'.' -f1)
        if [ "$latency_num" -gt 100 ]; then
            echo "⚠️  High latency detected - may impact performance"
        fi
    else
        echo "⚠️  Could not measure latency"
    fi
else
    echo "⚠️  ping command not available"
fi

# Test bandwidth (simple)
echo "🔍 Testing basic bandwidth..."
start_time=$(date +%s.%N)
if curl -s "http://$MAIN_SERVER_IP:8000/health" > /dev/null; then
    end_time=$(date +%s.%N)
    duration=$(echo "$end_time - $start_time" | bc 2>/dev/null || echo "unknown")
    if [ "$duration" != "unknown" ]; then
        echo "✅ API response time: ${duration}s"
    fi
fi

# Check for common network issues
echo "🔍 Checking for common network issues..."

# Check if we can resolve the hostname
if ! nslookup "$MAIN_SERVER_IP" > /dev/null 2>&1 && ! host "$MAIN_SERVER_IP" > /dev/null 2>&1; then
    if [[ "$MAIN_SERVER_IP" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "✅ Using IP address - DNS resolution not needed"
    else
        echo "⚠️  DNS resolution may be an issue for hostname: $MAIN_SERVER_IP"
    fi
fi

# Check if firewall might be blocking
if command -v iptables &> /dev/null; then
    if iptables -L 2>/dev/null | grep -q "DROP\|REJECT" 2>/dev/null; then
        echo "⚠️  Firewall rules detected - ensure ports 6379 and 8000 are allowed"
    fi
fi

# Summary
echo ""
echo "📋 Network Connectivity Summary"
echo "==============================="
echo "Server: $MAIN_SERVER_IP"
echo "Redis (6379): ✅ Accessible"
echo "HTTP API (8000): ✅ Accessible"
echo "API Endpoints: ✅ Accessible"
echo ""
echo "✅ Network connectivity tests completed successfully!"
echo ""
echo "📝 Next Steps:"
echo "1. Configure your remote worker with these connection details"
echo "2. Create an API key: curl -X POST 'http://$MAIN_SERVER_IP:8000/api/v1/auth/create-key' -F 'user_id=your_user'"
echo "3. Start your remote worker with the API key"
echo "4. Test processing: python tests/test-gpu-workers.py"
