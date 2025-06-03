#!/bin/bash
set -e

echo "🚀 Starting MoRAG deployment..."

# Check if .env.prod exists
if [ ! -f .env.prod ]; then
    echo "❌ .env.prod file not found. Please copy .env.prod.example and configure it."
    exit 1
fi

# Check if required environment variables are set
source .env.prod
if [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ GEMINI_API_KEY not set in .env.prod"
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p uploads temp logs logs/nginx

# Build and start services
echo "🔨 Building and starting services..."
docker-compose -f docker-compose.prod.yml build
docker-compose -f docker-compose.prod.yml up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
sleep 30

# Check service health
echo "🔍 Checking service health..."
docker-compose -f docker-compose.prod.yml ps

# Initialize database (if init script exists)
if [ -f scripts/init_db.py ]; then
    echo "🗄️ Initializing database..."
    docker-compose -f docker-compose.prod.yml exec api python scripts/init_db.py
fi

# Run health checks
echo "🏥 Running health checks..."
curl -f http://localhost/health/ || echo "❌ API health check failed"
curl -f http://localhost/health/ready || echo "❌ API readiness check failed"

echo "✅ Deployment completed!"
echo "📊 Monitor services:"
echo "  - API: http://localhost"
echo "  - Flower (Celery): http://localhost:5555"
echo "  - Logs: docker-compose -f docker-compose.prod.yml logs -f"
