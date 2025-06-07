#!/bin/bash
# Setup NFS client on GPU worker

set -e

MAIN_SERVER_IP="${1:-}"
if [ -z "$MAIN_SERVER_IP" ]; then
    echo "Usage: $0 <MAIN_SERVER_IP>"
    echo "Example: $0 192.168.1.100"
    exit 1
fi

echo "🗂️  Setting up NFS client for MoRAG file sharing"
echo "================================================"
echo "Main Server IP: $MAIN_SERVER_IP"

# Install NFS client
if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y nfs-common
elif command -v yum &> /dev/null; then
    sudo yum install -y nfs-utils
else
    echo "❌ Unsupported package manager. Please install NFS client manually."
    exit 1
fi

# Create mount point
MOUNT_POINT="/mnt/morag-shared"
sudo mkdir -p "$MOUNT_POINT"

# Test NFS connection
echo "🔍 Testing NFS connection..."
if showmount -e "$MAIN_SERVER_IP" | grep -q "/mnt/morag-shared"; then
    echo "✅ NFS export found on server"
else
    echo "❌ NFS export not found. Please check server configuration."
    exit 1
fi

# Mount the shared storage
echo "📁 Mounting shared storage..."
sudo mount -t nfs "$MAIN_SERVER_IP:/mnt/morag-shared" "$MOUNT_POINT"

# Verify mount
if mountpoint -q "$MOUNT_POINT"; then
    echo "✅ Shared storage mounted successfully"
    ls -la "$MOUNT_POINT"
else
    echo "❌ Failed to mount shared storage"
    exit 1
fi

# Add to fstab for persistent mounting
echo "📝 Adding to /etc/fstab for persistent mounting..."
FSTAB_ENTRY="$MAIN_SERVER_IP:/mnt/morag-shared $MOUNT_POINT nfs defaults 0 0"
if ! grep -q "$FSTAB_ENTRY" /etc/fstab; then
    echo "$FSTAB_ENTRY" | sudo tee -a /etc/fstab
    echo "✅ Added to /etc/fstab"
else
    echo "ℹ️  Entry already exists in /etc/fstab"
fi

echo ""
echo "📋 Next steps:"
echo "1. Update GPU worker configuration:"
echo "   TEMP_DIR=$MOUNT_POINT/temp"
echo "   UPLOAD_DIR=$MOUNT_POINT/uploads"
echo "2. Test file access from GPU worker"
echo "3. Start GPU worker with shared storage configuration"
