# Task 5: URL Processing Configuration

## Objective
Configure remote workers to process URLs directly (web pages, YouTube videos) with proper cookie support and authentication.

## Background
Remote workers need to handle URL processing independently:
1. Direct web page crawling and content extraction
2. YouTube video downloading with cookie authentication
3. Proper user-agent and header configuration
4. Cookie file management for authenticated content
5. No external service dependencies (Qdrant, Gemini handled by server)

## Implementation Steps

### 5.1 YouTube Cookie Configuration

**File**: `packages/morag/src/morag/services/youtube_config.py`

```python
"""YouTube configuration service for remote workers."""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
import structlog

logger = structlog.get_logger(__name__)

class YouTubeCookieManager:
    """Manage YouTube cookies for authenticated downloads."""

    def __init__(self, cookie_file_path: Optional[str] = None):
        self.cookie_file_path = cookie_file_path or os.getenv('YOUTUBE_COOKIES_FILE')

    def get_yt_dlp_options(self) -> Dict[str, Any]:
        """Get yt-dlp options with cookie configuration."""
        options = {
            # Basic options
            'format': 'best[height<=720]/best',
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en', 'en-US'],

            # Bot detection avoidance
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'referer': 'https://www.youtube.com/',
            'headers': {
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-us,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            },

            # Retry configuration
            'retries': 3,
            'fragment_retries': 3,
            'force_ipv4': True,
        }

        # Add cookies if available
        if self.cookie_file_path and Path(self.cookie_file_path).exists():
            options['cookiefile'] = self.cookie_file_path
            logger.info("Using YouTube cookies", cookie_file=self.cookie_file_path)
        else:
            logger.warning("No YouTube cookies configured - some videos may be inaccessible")

        return options

    def validate_cookie_file(self) -> bool:
        """Validate that cookie file exists and is readable."""
        if not self.cookie_file_path:
            return False

        cookie_path = Path(self.cookie_file_path)
        if not cookie_path.exists():
            logger.error("Cookie file not found", path=self.cookie_file_path)
            return False

        try:
            # Try to read the file
            with open(cookie_path, 'r') as f:
                content = f.read(100)  # Read first 100 chars
                if not content.strip():
                    logger.error("Cookie file is empty", path=self.cookie_file_path)
                    return False

            logger.info("Cookie file validated", path=self.cookie_file_path)
            return True

        except Exception as e:
            logger.error("Failed to read cookie file", path=self.cookie_file_path, error=str(e))
            return False
```

### 5.2 Web Processing Configuration

**File**: `packages/morag/src/morag/services/web_config.py`

```python
"""Web processing configuration for remote workers."""

import os
from typing import Dict, Any, Optional
import structlog

logger = structlog.get_logger(__name__)

class WebProcessingConfig:
    """Configuration for web content processing on remote workers."""

    def __init__(self):
        self.user_agent = os.getenv(
            'WEB_USER_AGENT',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        )
        self.timeout = int(os.getenv('WEB_TIMEOUT', '30'))
        self.max_retries = int(os.getenv('WEB_MAX_RETRIES', '3'))

    def get_requests_config(self) -> Dict[str, Any]:
        """Get requests configuration for web scraping."""
        return {
            'headers': {
                'User-Agent': self.user_agent,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            },
            'timeout': self.timeout,
            'allow_redirects': True,
            'verify': True,
        }

    def get_selenium_config(self) -> Dict[str, Any]:
        """Get Selenium configuration for dynamic content."""
        return {
            'user_agent': self.user_agent,
            'timeout': self.timeout,
            'headless': True,
            'disable_images': True,
            'disable_javascript': False,  # Some sites need JS
        }
```

### 5.2 Shared Storage Configuration

**File**: `scripts/setup-nfs-server.sh`

```bash
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
```

**File**: `scripts/setup-nfs-client.sh`

```bash
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
```

### 5.3 HTTP File Transfer Implementation

**File**: `packages/morag/src/morag/services/file_transfer.py`

```python
"""HTTP-based file transfer service for remote workers."""

import os
import hashlib
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import aiofiles
import httpx
from fastapi import HTTPException
import structlog

logger = structlog.get_logger(__name__)


class FileTransferService:
    """HTTP-based file transfer for remote workers."""
    
    def __init__(self, server_url: str, temp_dir: str = "/tmp"):
        self.server_url = server_url.rstrip('/')
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        
    async def download_file(self, file_path: str) -> str:
        """Download file from main server to local temp directory."""
        try:
            # Generate local temp file path
            file_hash = hashlib.md5(file_path.encode()).hexdigest()
            local_path = self.temp_dir / f"download_{file_hash}_{Path(file_path).name}"
            
            # Download file
            download_url = f"{self.server_url}/api/v1/files/download"
            
            async with httpx.AsyncClient(timeout=300) as client:
                response = await client.post(
                    download_url,
                    json={"file_path": file_path}
                )
                response.raise_for_status()
                
                # Save to local file
                async with aiofiles.open(local_path, 'wb') as f:
                    async for chunk in response.aiter_bytes():
                        await f.write(chunk)
            
            logger.info("File downloaded successfully",
                       remote_path=file_path,
                       local_path=str(local_path),
                       size=local_path.stat().st_size)
            
            return str(local_path)
            
        except Exception as e:
            logger.error("Failed to download file",
                        file_path=file_path,
                        error=str(e))
            raise
    
    async def upload_result(self, local_path: str, result_data: Dict[str, Any]) -> bool:
        """Upload processing result back to main server."""
        try:
            upload_url = f"{self.server_url}/api/v1/files/upload-result"
            
            # Prepare upload data
            files = {}
            if os.path.exists(local_path):
                files['file'] = open(local_path, 'rb')
            
            async with httpx.AsyncClient(timeout=300) as client:
                response = await client.post(
                    upload_url,
                    files=files,
                    data={'result_data': json.dumps(result_data)}
                )
                response.raise_for_status()
            
            # Cleanup local file
            if local_path and os.path.exists(local_path):
                os.unlink(local_path)
            
            logger.info("Result uploaded successfully", local_path=local_path)
            return True
            
        except Exception as e:
            logger.error("Failed to upload result",
                        local_path=local_path,
                        error=str(e))
            return False
        finally:
            # Close file handles
            for f in files.values():
                if hasattr(f, 'close'):
                    f.close()
    
    def cleanup_temp_files(self, max_age_hours: int = 24):
        """Clean up old temporary files."""
        try:
            import time
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            for file_path in self.temp_dir.glob("download_*"):
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > max_age_seconds:
                        file_path.unlink()
                        logger.debug("Cleaned up old temp file", file_path=str(file_path))
                        
        except Exception as e:
            logger.error("Failed to cleanup temp files", error=str(e))
```

### 5.4 Add File Transfer Endpoints

**File**: `packages/morag/src/morag/server.py`

Add file transfer endpoints for HTTP mode:

```python
from fastapi.responses import FileResponse
from morag.services.file_transfer import FileTransferService

@app.post("/api/v1/files/download", tags=["File Transfer"])
async def download_file(request: Dict[str, str]):
    """Download file for remote worker processing."""
    try:
        file_path = request.get('file_path')
        if not file_path:
            raise HTTPException(status_code=400, detail="file_path required")
        
        # Validate file exists and is accessible
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        # Security check - ensure file is in allowed directories
        allowed_dirs = ['/app/temp', '/app/uploads', '/tmp']
        if not any(file_path.startswith(d) for d in allowed_dirs):
            raise HTTPException(status_code=403, detail="File access denied")
        
        return FileResponse(
            path=file_path,
            filename=os.path.basename(file_path),
            media_type='application/octet-stream'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("File download failed", file_path=file_path, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/files/upload-result", tags=["File Transfer"])
async def upload_result(
    file: Optional[UploadFile] = File(None),
    result_data: str = Form(...)
):
    """Upload processing result from remote worker."""
    try:
        import json
        result = json.loads(result_data)
        
        # Handle file upload if provided
        if file:
            # Save uploaded file to temp directory
            temp_path = f"/tmp/result_{uuid.uuid4().hex}_{file.filename}"
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            result['uploaded_file'] = temp_path
        
        # Process result (store in database, trigger webhooks, etc.)
        logger.info("Result uploaded from remote worker", result=result)
        
        return {"status": "success", "message": "Result uploaded successfully"}
        
    except Exception as e:
        logger.error("Result upload failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
```

## Testing

### 5.1 Test Network Connectivity
```bash
# Test Redis connectivity from GPU worker
redis-cli -h MAIN_SERVER_IP -p 6379 ping

# Test Qdrant connectivity from GPU worker
curl http://MAIN_SERVER_IP:6333/collections

# Test HTTP API connectivity from GPU worker
curl http://MAIN_SERVER_IP:8000/health
```

### 5.2 Test File Sharing
```bash
# Test NFS mount
sudo mount -t nfs MAIN_SERVER_IP:/mnt/morag-shared /mnt/morag-shared
echo "test" > /mnt/morag-shared/test.txt
cat /mnt/morag-shared/test.txt

# Test HTTP file transfer
curl -X POST "http://MAIN_SERVER_IP:8000/api/v1/files/download" \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/app/temp/test.txt"}'
```

### 5.3 Test End-to-End Processing
```bash
# Submit task from main server, process on GPU worker
curl -X POST "http://localhost:8000/process/file" \
  -F "file=@test.mp3" \
  -F "gpu=true"

# Monitor worker logs and file access
```

## Acceptance Criteria

- [ ] Network requirements documented with specific ports and protocols
- [ ] Firewall configuration scripts provided for both server and workers
- [ ] NFS server setup script works correctly
- [ ] NFS client setup script works correctly
- [ ] HTTP file transfer service implemented and tested
- [ ] File transfer endpoints added to main server API
- [ ] Security validation prevents unauthorized file access
- [ ] Both shared storage and HTTP transfer modes work
- [ ] Network connectivity tests pass
- [ ] End-to-end file sharing works between server and workers

## Files Created/Modified

- `docs/network-requirements.md` (new)
- `scripts/setup-nfs-server.sh` (new)
- `scripts/setup-nfs-client.sh` (new)
- `packages/morag/src/morag/services/file_transfer.py` (new)
- `packages/morag/src/morag/server.py` (modified)

## Next Steps

After completing this task:
1. Proceed to Task 6: Documentation & Testing
2. Test network configuration with actual remote GPU worker
3. Validate file sharing performance and reliability

## Notes

- NFS provides better performance for high-frequency file access
- HTTP transfer is simpler to set up but has higher overhead
- Security measures prevent unauthorized file access
- Both options support the same worker functionality
- Network configuration is environment-specific and may need customization
