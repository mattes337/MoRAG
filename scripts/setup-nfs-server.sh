#!/bin/bash
# Setup NFS server on main server for file sharing

set -e

echo "🗂️  Setting up NFS server for MoRAG file sharing"
echo "================================================"

# Install NFS server
if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y nfs-kernel-server
elif command -v yum &> /dev/null; then
    sudo yum install -y nfs-utils
else
    echo "❌ Unsupported package manager. Please install NFS server manually."
    exit 1
fi

# Create shared directories
SHARED_DIR="/mnt/morag-shared"
sudo mkdir -p "$SHARED_DIR/temp"
sudo mkdir -p "$SHARED_DIR/uploads"

# Set permissions
sudo chown -R $(whoami):$(whoami) "$SHARED_DIR"
sudo chmod -R 755 "$SHARED_DIR"

# Configure NFS exports
EXPORTS_FILE="/etc/exports"
BACKUP_FILE="/etc/exports.backup.$(date +%Y%m%d_%H%M%S)"

# Backup existing exports
if [ -f "$EXPORTS_FILE" ]; then
    sudo cp "$EXPORTS_FILE" "$BACKUP_FILE"
    echo "📋 Backed up existing exports to: $BACKUP_FILE"
fi

# Add MoRAG exports
echo "📝 Configuring NFS exports..."
echo "# MoRAG shared storage" | sudo tee -a "$EXPORTS_FILE"
echo "$SHARED_DIR *(rw,sync,no_subtree_check,no_root_squash)" | sudo tee -a "$EXPORTS_FILE"

# Restart NFS services
echo "🔄 Restarting NFS services..."
sudo systemctl restart nfs-kernel-server
sudo systemctl enable nfs-kernel-server

# Export the filesystems
sudo exportfs -ra

# Show current exports
echo "✅ NFS server configured. Current exports:"
sudo exportfs -v

echo ""
echo "📋 Next steps:"
echo "1. Configure firewall to allow NFS traffic from GPU workers"
echo "2. On GPU workers, mount the shared storage:"
echo "   sudo mount -t nfs MAIN_SERVER_IP:$SHARED_DIR /mnt/morag-shared"
echo "3. Update GPU worker configuration to use shared storage"
