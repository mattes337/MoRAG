"""HTTP-based file transfer service for remote workers."""

import os
import hashlib
import tempfile
import shutil
import json
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
        
    async def download_file(self, file_path: str, api_key: Optional[str] = None) -> str:
        """Download file from main server to local temp directory."""
        try:
            # Generate local temp file path
            file_hash = hashlib.md5(file_path.encode()).hexdigest()
            local_path = self.temp_dir / f"download_{file_hash}_{Path(file_path).name}"
            
            # Download file
            download_url = f"{self.server_url}/api/v1/files/download"
            
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            
            async with httpx.AsyncClient(timeout=300) as client:
                response = await client.post(
                    download_url,
                    json={"file_path": file_path},
                    headers=headers
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
    
    async def upload_result(self, local_path: str, result_data: Dict[str, Any], api_key: Optional[str] = None) -> bool:
        """Upload processing result back to main server."""
        try:
            upload_url = f"{self.server_url}/api/v1/files/upload-result"
            
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            
            # Prepare upload data
            files = {}
            if os.path.exists(local_path):
                files['file'] = open(local_path, 'rb')
            
            try:
                async with httpx.AsyncClient(timeout=300) as client:
                    response = await client.post(
                        upload_url,
                        files=files,
                        data={'result_data': json.dumps(result_data)},
                        headers=headers
                    )
                    response.raise_for_status()
                
                # Cleanup local file
                if local_path and os.path.exists(local_path):
                    os.unlink(local_path)
                
                logger.info("Result uploaded successfully", local_path=local_path)
                return True
                
            finally:
                # Close file handles
                for f in files.values():
                    if hasattr(f, 'close'):
                        f.close()
                        
        except Exception as e:
            logger.error("Failed to upload result",
                        local_path=local_path,
                        error=str(e))
            return False
    
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


# Global file transfer service instance
_file_transfer_service: Optional[FileTransferService] = None


def get_file_transfer_service() -> Optional[FileTransferService]:
    """Get file transfer service if configured for HTTP mode."""
    global _file_transfer_service
    
    main_server_url = os.getenv('MAIN_SERVER_URL')
    file_transfer_mode = os.getenv('FILE_TRANSFER_MODE', 'shared')
    
    if file_transfer_mode == 'http' and main_server_url:
        if _file_transfer_service is None:
            temp_dir = os.getenv('TEMP_DIR', '/tmp/morag-remote-worker')
            _file_transfer_service = FileTransferService(main_server_url, temp_dir)
        return _file_transfer_service
    
    return None
