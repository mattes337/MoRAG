#!/bin/bash
set -e

BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "📦 Creating backup in $BACKUP_DIR..."

# Backup Qdrant data
echo "🗄️ Backing up Qdrant data..."
docker-compose -f docker-compose.prod.yml exec qdrant tar czf - /qdrant/storage > "$BACKUP_DIR/qdrant_data.tar.gz"

# Backup Redis data
echo "💾 Backing up Redis data..."
docker-compose -f docker-compose.prod.yml exec redis redis-cli BGSAVE
sleep 5
docker-compose -f docker-compose.prod.yml exec redis tar czf - /data > "$BACKUP_DIR/redis_data.tar.gz"

# Backup uploaded files
echo "📁 Backing up uploaded files..."
if [ -d "uploads" ]; then
    tar czf "$BACKUP_DIR/uploads.tar.gz" uploads/
fi

# Backup configuration
echo "⚙️ Backing up configuration..."
cp .env.prod "$BACKUP_DIR/"
cp docker-compose.prod.yml "$BACKUP_DIR/"

echo "✅ Backup completed: $BACKUP_DIR"
