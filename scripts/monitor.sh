#!/bin/bash

echo "📊 MoRAG System Status"
echo "====================="

# Service status
echo "🔧 Service Status:"
docker-compose -f docker-compose.prod.yml ps

echo ""
echo "💾 Resource Usage:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"

echo ""
echo "📈 Queue Status:"
curl -s http://localhost/api/v1/status/stats/queues | jq '.' || echo "Queue stats not available"

echo ""
echo "🏥 Health Status:"
curl -s http://localhost/health/ready | jq '.' || echo "Health check not available"

echo ""
echo "📝 Recent Logs (last 50 lines):"
docker-compose -f docker-compose.prod.yml logs --tail=50
