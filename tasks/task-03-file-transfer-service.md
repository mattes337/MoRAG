# Task 3: File Transfer Service

## Objective
Implement a secure file transfer system that enables the main server to send files to remote workers and receive processed results back, with encryption, authentication, and automatic cleanup.

## Current State Analysis

### Existing File Handling
- Files stored in shared volumes (`/app/temp`) accessible to local workers
- No mechanism for transferring files to remote workers
- File cleanup handled by periodic cleanup service
- Upload handling via `FileUploadHandler` class

### Remote Worker Requirements
- Remote workers cannot access shared volumes
- Need secure file transfer for input files and results
- Must handle large files (up to 5GB for video)
- Require authentication and authorization
- Need automatic cleanup of transferred files

## Implementation Plan

### Step 1: File Transfer Models

#### 1.1 Create File Transfer Models
**File**: `packages/morag-core/src/morag_core/models/file_transfer.py`

```python
"""File transfer models for remote worker communication."""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
import uuid

class TransferDirection(str, Enum):
    UPLOAD = "upload"      # Server → Worker
    DOWNLOAD = "download"  # Worker → Server

class TransferStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"

class FileTransferRequest(BaseModel):
    """Request to transfer a file."""
    transfer_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    worker_id: str
    direction: TransferDirection
    file_path: str
    original_filename: str
    content_type: Optional[str] = None
    file_size: int
    checksum: str  # SHA256 hash
    expires_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)

class FileTransferResponse(BaseModel):
    """Response to file transfer request."""
    transfer_id: str
    status: TransferStatus
    download_url: Optional[str] = None
    upload_url: Optional[str] = None
    auth_token: Optional[str] = None
    expires_at: datetime
    message: Optional[str] = None

class FileTransferStatus(BaseModel):
    """Status of a file transfer."""
    transfer_id: str
    status: TransferStatus
    progress_percent: float = 0.0
    bytes_transferred: int = 0
    total_bytes: int = 0
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class FileTransferResult(BaseModel):
    """Result of completed file transfer."""
    transfer_id: str
    local_path: str
    original_filename: str
    file_size: int
    checksum: str
    transferred_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

### Step 2: File Transfer Service

#### 2.1 Create File Transfer Service
**File**: `packages/morag/src/morag/services/file_transfer.py`

```python
"""File transfer service for remote worker communication."""

import asyncio
import hashlib
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, BinaryIO
import aiofiles
import structlog
from redis import Redis

from morag_core.models.file_transfer import (
    FileTransferRequest, FileTransferResponse, FileTransferStatus,
    FileTransferResult, TransferDirection, TransferStatus
)
from morag.utils.encryption import FileEncryption
from morag.utils.auth import generate_transfer_token, verify_transfer_token

logger = structlog.get_logger(__name__)

class FileTransferService:
    """Service for handling file transfers to/from remote workers."""
    
    def __init__(self, redis_client: Redis, transfer_dir: Path):
        self.redis = redis_client
        self.transfer_dir = transfer_dir
        self.transfer_dir.mkdir(parents=True, exist_ok=True)
        self.encryption = FileEncryption()
        self.active_transfers: Dict[str, FileTransferStatus] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start the file transfer service."""
        logger.info("Starting file transfer service", transfer_dir=str(self.transfer_dir))
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_transfers())
    
    async def stop(self):
        """Stop the file transfer service."""
        logger.info("Stopping file transfer service")
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
    
    async def prepare_upload(self, worker_id: str, file_path: str, 
                           original_filename: str, metadata: Dict = None) -> FileTransferResponse:
        """Prepare a file for upload to a remote worker."""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Calculate file info
            file_size = file_path.stat().st_size
            checksum = await self._calculate_checksum(file_path)
            
            # Create transfer request
            request = FileTransferRequest(
                worker_id=worker_id,
                direction=TransferDirection.UPLOAD,
                file_path=str(file_path),
                original_filename=original_filename,
                file_size=file_size,
                checksum=checksum,
                expires_at=datetime.utcnow() + timedelta(hours=1),
                metadata=metadata or {}
            )
            
            # Encrypt and store file
            encrypted_path = await self._encrypt_and_store_file(file_path, request.transfer_id)
            
            # Generate auth token
            auth_token = generate_transfer_token(request.transfer_id, worker_id)
            
            # Store transfer info in Redis
            await self._store_transfer_info(request)
            
            # Create download URL for worker
            download_url = f"/api/v1/transfers/{request.transfer_id}/download"
            
            response = FileTransferResponse(
                transfer_id=request.transfer_id,
                status=TransferStatus.PENDING,
                download_url=download_url,
                auth_token=auth_token,
                expires_at=request.expires_at
            )
            
            logger.info("File upload prepared",
                       transfer_id=request.transfer_id,
                       worker_id=worker_id,
                       filename=original_filename,
                       file_size=file_size)
            
            return response
            
        except Exception as e:
            logger.error("Failed to prepare file upload",
                        worker_id=worker_id,
                        file_path=str(file_path),
                        error=str(e))
            raise
    
    async def prepare_download(self, worker_id: str, original_filename: str,
                             expected_size: int, metadata: Dict = None) -> FileTransferResponse:
        """Prepare to receive a file from a remote worker."""
        try:
            # Create transfer request
            request = FileTransferRequest(
                worker_id=worker_id,
                direction=TransferDirection.DOWNLOAD,
                file_path="",  # Will be set when file is received
                original_filename=original_filename,
                file_size=expected_size,
                checksum="",  # Will be calculated when file is received
                expires_at=datetime.utcnow() + timedelta(hours=1),
                metadata=metadata or {}
            )
            
            # Generate auth token
            auth_token = generate_transfer_token(request.transfer_id, worker_id)
            
            # Store transfer info in Redis
            await self._store_transfer_info(request)
            
            # Create upload URL for worker
            upload_url = f"/api/v1/transfers/{request.transfer_id}/upload"
            
            response = FileTransferResponse(
                transfer_id=request.transfer_id,
                status=TransferStatus.PENDING,
                upload_url=upload_url,
                auth_token=auth_token,
                expires_at=request.expires_at
            )
            
            logger.info("File download prepared",
                       transfer_id=request.transfer_id,
                       worker_id=worker_id,
                       filename=original_filename)
            
            return response
            
        except Exception as e:
            logger.error("Failed to prepare file download",
                        worker_id=worker_id,
                        filename=original_filename,
                        error=str(e))
            raise
    
    async def handle_download_request(self, transfer_id: str, 
                                    auth_token: str) -> Optional[Path]:
        """Handle a download request from a worker."""
        try:
            # Verify auth token
            if not verify_transfer_token(auth_token, transfer_id):
                logger.warning("Invalid auth token for download",
                              transfer_id=transfer_id)
                return None
            
            # Get transfer info
            transfer_info = await self._get_transfer_info(transfer_id)
            if not transfer_info:
                logger.warning("Transfer not found", transfer_id=transfer_id)
                return None
            
            if transfer_info.direction != TransferDirection.UPLOAD:
                logger.warning("Invalid transfer direction for download",
                              transfer_id=transfer_id)
                return None
            
            # Check expiration
            if datetime.utcnow() > transfer_info.expires_at:
                logger.warning("Transfer expired", transfer_id=transfer_id)
                return None
            
            # Get encrypted file path
            encrypted_path = self.transfer_dir / f"{transfer_id}.enc"
            if not encrypted_path.exists():
                logger.error("Encrypted file not found",
                           transfer_id=transfer_id,
                           path=str(encrypted_path))
                return None
            
            # Update status
            await self._update_transfer_status(transfer_id, TransferStatus.IN_PROGRESS)
            
            logger.info("File download started",
                       transfer_id=transfer_id,
                       filename=transfer_info.original_filename)
            
            return encrypted_path
            
        except Exception as e:
            logger.error("Failed to handle download request",
                        transfer_id=transfer_id,
                        error=str(e))
            return None
    
    async def handle_upload_request(self, transfer_id: str, auth_token: str,
                                  file_data: BinaryIO) -> Optional[FileTransferResult]:
        """Handle an upload request from a worker."""
        try:
            # Verify auth token
            if not verify_transfer_token(auth_token, transfer_id):
                logger.warning("Invalid auth token for upload",
                              transfer_id=transfer_id)
                return None
            
            # Get transfer info
            transfer_info = await self._get_transfer_info(transfer_id)
            if not transfer_info:
                logger.warning("Transfer not found", transfer_id=transfer_id)
                return None
            
            if transfer_info.direction != TransferDirection.DOWNLOAD:
                logger.warning("Invalid transfer direction for upload",
                              transfer_id=transfer_id)
                return None
            
            # Check expiration
            if datetime.utcnow() > transfer_info.expires_at:
                logger.warning("Transfer expired", transfer_id=transfer_id)
                return None
            
            # Update status
            await self._update_transfer_status(transfer_id, TransferStatus.IN_PROGRESS)
            
            # Save uploaded file
            temp_path = self.transfer_dir / f"{transfer_id}.tmp"
            file_size = 0
            
            async with aiofiles.open(temp_path, 'wb') as f:
                while True:
                    chunk = file_data.read(8192)
                    if not chunk:
                        break
                    await f.write(chunk)
                    file_size += len(chunk)
            
            # Calculate checksum
            checksum = await self._calculate_checksum(temp_path)
            
            # Decrypt file
            decrypted_path = self.transfer_dir / f"{transfer_id}.dec"
            await self.encryption.decrypt_file(temp_path, decrypted_path)
            
            # Clean up temp file
            temp_path.unlink()
            
            # Update transfer info
            transfer_info.file_path = str(decrypted_path)
            transfer_info.file_size = file_size
            transfer_info.checksum = checksum
            await self._store_transfer_info(transfer_info)
            
            # Update status
            await self._update_transfer_status(transfer_id, TransferStatus.COMPLETED)
            
            result = FileTransferResult(
                transfer_id=transfer_id,
                local_path=str(decrypted_path),
                original_filename=transfer_info.original_filename,
                file_size=file_size,
                checksum=checksum,
                transferred_at=datetime.utcnow(),
                metadata=transfer_info.metadata
            )
            
            logger.info("File upload completed",
                       transfer_id=transfer_id,
                       filename=transfer_info.original_filename,
                       file_size=file_size)
            
            return result
            
        except Exception as e:
            logger.error("Failed to handle upload request",
                        transfer_id=transfer_id,
                        error=str(e))
            await self._update_transfer_status(transfer_id, TransferStatus.FAILED)
            return None
    
    async def get_transfer_status(self, transfer_id: str) -> Optional[FileTransferStatus]:
        """Get the status of a file transfer."""
        return self.active_transfers.get(transfer_id)
    
    async def cleanup_transfer(self, transfer_id: str):
        """Clean up files and data for a completed transfer."""
        try:
            # Remove files
            for suffix in ['.enc', '.tmp', '.dec']:
                file_path = self.transfer_dir / f"{transfer_id}{suffix}"
                if file_path.exists():
                    file_path.unlink()
            
            # Remove from active transfers
            if transfer_id in self.active_transfers:
                del self.active_transfers[transfer_id]
            
            # Remove from Redis
            await self._remove_transfer_info(transfer_id)
            
            logger.debug("Transfer cleaned up", transfer_id=transfer_id)
            
        except Exception as e:
            logger.error("Failed to cleanup transfer",
                        transfer_id=transfer_id,
                        error=str(e))
    
    async def _encrypt_and_store_file(self, source_path: Path, transfer_id: str) -> Path:
        """Encrypt and store a file for transfer."""
        encrypted_path = self.transfer_dir / f"{transfer_id}.enc"
        await self.encryption.encrypt_file(source_path, encrypted_path)
        return encrypted_path
    
    async def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        hash_sha256 = hashlib.sha256()
        async with aiofiles.open(file_path, 'rb') as f:
            while True:
                chunk = await f.read(8192)
                if not chunk:
                    break
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    async def _store_transfer_info(self, transfer_info: FileTransferRequest):
        """Store transfer information in Redis."""
        key = f"transfer:{transfer_info.transfer_id}"
        data = transfer_info.model_dump_json()
        # Store with expiration based on transfer expiration
        ttl = int((transfer_info.expires_at - datetime.utcnow()).total_seconds())
        self.redis.setex(key, max(ttl, 60), data)  # Minimum 1 minute TTL
    
    async def _get_transfer_info(self, transfer_id: str) -> Optional[FileTransferRequest]:
        """Get transfer information from Redis."""
        key = f"transfer:{transfer_id}"
        data = self.redis.get(key)
        if data:
            return FileTransferRequest.model_validate_json(data)
        return None
    
    async def _remove_transfer_info(self, transfer_id: str):
        """Remove transfer information from Redis."""
        key = f"transfer:{transfer_id}"
        self.redis.delete(key)
    
    async def _update_transfer_status(self, transfer_id: str, status: TransferStatus):
        """Update transfer status."""
        if transfer_id not in self.active_transfers:
            self.active_transfers[transfer_id] = FileTransferStatus(
                transfer_id=transfer_id,
                status=status
            )
        else:
            self.active_transfers[transfer_id].status = status
            
        if status == TransferStatus.IN_PROGRESS:
            self.active_transfers[transfer_id].started_at = datetime.utcnow()
        elif status in [TransferStatus.COMPLETED, TransferStatus.FAILED]:
            self.active_transfers[transfer_id].completed_at = datetime.utcnow()
    
    async def _cleanup_expired_transfers(self):
        """Periodically clean up expired transfers."""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                current_time = datetime.utcnow()
                expired_transfers = []
                
                # Check Redis for expired transfers
                pattern = "transfer:*"
                for key in self.redis.scan_iter(match=pattern):
                    data = self.redis.get(key)
                    if data:
                        try:
                            transfer_info = FileTransferRequest.model_validate_json(data)
                            if current_time > transfer_info.expires_at:
                                expired_transfers.append(transfer_info.transfer_id)
                        except Exception:
                            # Invalid data, mark for cleanup
                            transfer_id = key.decode().split(':')[1]
                            expired_transfers.append(transfer_id)
                
                # Clean up expired transfers
                for transfer_id in expired_transfers:
                    await self.cleanup_transfer(transfer_id)
                
                if expired_transfers:
                    logger.info("Cleaned up expired transfers", count=len(expired_transfers))
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in transfer cleanup task", error=str(e))
```

### Step 3: File Transfer API Endpoints

#### 3.1 Add File Transfer Endpoints
**File**: `packages/morag/src/morag/server.py`

Add new endpoints for file transfer:

```python
from morag.services.file_transfer import FileTransferService
from morag_core.models.file_transfer import FileTransferStatus

# Initialize file transfer service
transfer_service = FileTransferService(redis_client, Path("./transfers"))

@app.get("/api/v1/transfers/{transfer_id}/download")
async def download_file(transfer_id: str, token: str = Query(...)):
    """Download a file for a worker."""
    try:
        file_path = await transfer_service.handle_download_request(transfer_id, token)
        
        if not file_path:
            raise HTTPException(status_code=404, detail="Transfer not found or invalid")
        
        # Stream the encrypted file
        return FileResponse(
            path=str(file_path),
            media_type='application/octet-stream',
            filename=f"{transfer_id}.enc"
        )
        
    except Exception as e:
        logger.error("File download failed", transfer_id=transfer_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/transfers/{transfer_id}/upload")
async def upload_file(transfer_id: str, token: str = Query(...), 
                     file: UploadFile = File(...)):
    """Upload a file from a worker."""
    try:
        result = await transfer_service.handle_upload_request(
            transfer_id, token, file.file
        )
        
        if not result:
            raise HTTPException(status_code=400, detail="Upload failed")
        
        return {
            "success": True,
            "transfer_id": transfer_id,
            "file_size": result.file_size,
            "checksum": result.checksum
        }
        
    except Exception as e:
        logger.error("File upload failed", transfer_id=transfer_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/transfers/{transfer_id}/status", response_model=FileTransferStatus)
async def get_transfer_status(transfer_id: str):
    """Get the status of a file transfer."""
    status = await transfer_service.get_transfer_status(transfer_id)
    if not status:
        raise HTTPException(status_code=404, detail="Transfer not found")
    return status

@app.delete("/api/v1/transfers/{transfer_id}")
async def cleanup_transfer(transfer_id: str):
    """Clean up a completed transfer."""
    try:
        await transfer_service.cleanup_transfer(transfer_id)
        return {"success": True, "message": "Transfer cleaned up"}
    except Exception as e:
        logger.error("Transfer cleanup failed", transfer_id=transfer_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
```

## Testing Requirements

### Unit Tests
1. **File Transfer Service Tests**
   - Test file encryption/decryption
   - Test upload/download preparation
   - Test transfer status tracking
   - Test cleanup mechanisms

2. **File Transfer API Tests**
   - Test download endpoint
   - Test upload endpoint
   - Test status endpoint
   - Test authentication

### Integration Tests
1. **End-to-End Transfer Tests**
   - Test complete upload workflow (server → worker)
   - Test complete download workflow (worker → server)
   - Test large file transfers
   - Test transfer expiration and cleanup

### Test Files to Create
- `tests/test_file_transfer_service.py`
- `tests/test_file_transfer_api.py`
- `tests/integration/test_file_transfer_e2e.py`

### Step 4: Encryption and Authentication Utilities

#### 4.1 Create Encryption Utility
**File**: `packages/morag/src/morag/utils/encryption.py`

```python
"""File encryption utilities for secure transfer."""

import os
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import structlog

logger = structlog.get_logger(__name__)

class FileEncryption:
    """Handles file encryption/decryption for secure transfers."""

    def __init__(self, password: str = None):
        self.password = password or os.environ.get('MORAG_TRANSFER_KEY', 'default-key-change-in-production')
        self._fernet = None

    def _get_fernet(self) -> Fernet:
        """Get or create Fernet instance."""
        if not self._fernet:
            password = self.password.encode()
            salt = b'morag-salt-12345'  # In production, use random salt per file
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password))
            self._fernet = Fernet(key)
        return self._fernet

    async def encrypt_file(self, source_path: Path, dest_path: Path):
        """Encrypt a file."""
        fernet = self._get_fernet()

        with open(source_path, 'rb') as source:
            with open(dest_path, 'wb') as dest:
                while True:
                    chunk = source.read(8192)
                    if not chunk:
                        break
                    encrypted_chunk = fernet.encrypt(chunk)
                    dest.write(encrypted_chunk)

    async def decrypt_file(self, source_path: Path, dest_path: Path):
        """Decrypt a file."""
        fernet = self._get_fernet()

        with open(source_path, 'rb') as source:
            with open(dest_path, 'wb') as dest:
                while True:
                    chunk = source.read(8192)
                    if not chunk:
                        break
                    decrypted_chunk = fernet.decrypt(chunk)
                    dest.write(decrypted_chunk)
```

#### 4.2 Create Authentication Utility
**File**: `packages/morag/src/morag/utils/auth.py`

```python
"""Authentication utilities for worker communication."""

import jwt
import os
from datetime import datetime, timedelta
from typing import Optional
import structlog

logger = structlog.get_logger(__name__)

SECRET_KEY = os.environ.get('MORAG_JWT_SECRET', 'change-this-secret-in-production')

def generate_transfer_token(transfer_id: str, worker_id: str,
                          expires_hours: int = 1) -> str:
    """Generate JWT token for file transfer authentication."""
    payload = {
        'transfer_id': transfer_id,
        'worker_id': worker_id,
        'exp': datetime.utcnow() + timedelta(hours=expires_hours),
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')

def verify_transfer_token(token: str, expected_transfer_id: str) -> bool:
    """Verify JWT token for file transfer."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        return payload.get('transfer_id') == expected_transfer_id
    except jwt.ExpiredSignatureError:
        logger.warning("Transfer token expired", transfer_id=expected_transfer_id)
        return False
    except jwt.InvalidTokenError:
        logger.warning("Invalid transfer token", transfer_id=expected_transfer_id)
        return False
```

## Dependencies
- **New**: `cryptography` for file encryption
- **New**: `aiofiles` for async file operations
- **New**: `PyJWT` for authentication tokens
- **Existing**: Redis for transfer metadata storage

## Success Criteria
1. Files can be securely transferred to remote workers
2. Workers can upload processed results back to server
3. All transfers are encrypted and authenticated
4. Large files (up to 5GB) transfer successfully
5. Expired transfers are automatically cleaned up
6. Transfer status can be monitored in real-time

## Next Steps
After completing this task:
1. Proceed to Task 4: Worker Communication Protocol
2. Test file transfers with simulated remote workers
3. Validate encryption and authentication mechanisms

---

**Dependencies**: Task 2 (Worker Registration)
**Estimated Time**: 4-5 days
**Risk Level**: High (security and large file handling)
