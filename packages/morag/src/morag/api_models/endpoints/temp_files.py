"""Temporary file management endpoints for MoRAG API."""

import os
from typing import Optional
from pathlib import Path
import structlog

from fastapi import APIRouter, HTTPException, Request, Response, Header
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends

from morag.services.temporary_file_service import get_temp_file_service

logger = structlog.get_logger(__name__)

temp_files_router = APIRouter(prefix="/api/files", tags=["Temporary Files"])
security = HTTPBearer(auto_error=False)


def validate_session_access(
    session_id: str,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> str:
    """Validate session access (placeholder for real authentication).
    
    Args:
        session_id: Session identifier
        credentials: Optional bearer token
        
    Returns:
        Validated session ID
        
    Raises:
        HTTPException: If access is denied
    """
    # For now, allow all access - in production this should validate the token
    # and ensure the user has access to the specific session
    
    # Basic session ID validation
    if not session_id or len(session_id) < 3:
        raise HTTPException(status_code=400, detail="Invalid session ID")
    
    return session_id


def setup_temp_files_endpoints():
    """Setup temporary file management endpoints."""
    
    @temp_files_router.get("/temp/{session_id}")
    async def list_session_files(
        session_id: str = Depends(validate_session_access)
    ):
        """List all files in a session.
        
        Returns metadata for all files in the specified session including
        file names, sizes, content types, and expiration information.
        """
        try:
            temp_service = get_temp_file_service()
            session_info = await temp_service.list_session_files(session_id)
            
            logger.info("Listed session files",
                       session_id=session_id,
                       file_count=len(session_info["files"]),
                       total_size=session_info["total_size_bytes"])
            
            return session_info
            
        except Exception as e:
            logger.error("Failed to list session files",
                        session_id=session_id,
                        error=str(e))
            raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")
    
    @temp_files_router.get("/temp/{session_id}/{filename:path}")
    async def download_file(
        request: Request,
        session_id: str,
        filename: str,
        range_header: Optional[str] = Header(None, alias="range"),
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
    ):
        """Download a file from the session directory.
        
        Supports:
        - Direct download of files
        - Range requests for partial content
        - Proper content-type detection
        - Streaming for large files
        """
        # Validate session access
        validate_session_access(session_id, credentials)
        
        try:
            temp_service = get_temp_file_service()
            file_path = await temp_service.get_file_path(session_id, filename)
            
            # Get file stats
            file_stat = file_path.stat()
            file_size = file_stat.st_size
            
            # Detect content type
            import mimetypes
            content_type, _ = mimetypes.guess_type(filename)
            if not content_type:
                content_type = "application/octet-stream"
            
            # Handle range requests
            if range_header:
                try:
                    # Parse range header (e.g., "bytes=0-1023")
                    range_match = range_header.replace("bytes=", "").split("-")
                    start = int(range_match[0]) if range_match[0] else 0
                    end = int(range_match[1]) if range_match[1] else file_size - 1
                    
                    # Validate range
                    if start >= file_size or end >= file_size or start > end:
                        raise HTTPException(
                            status_code=416,
                            detail="Requested range not satisfiable",
                            headers={"Content-Range": f"bytes */{file_size}"}
                        )
                    
                    # Create streaming response for range
                    async def stream_range():
                        with open(file_path, 'rb') as f:
                            f.seek(start)
                            remaining = end - start + 1
                            chunk_size = 8192
                            
                            while remaining > 0:
                                chunk = f.read(min(chunk_size, remaining))
                                if not chunk:
                                    break
                                remaining -= len(chunk)
                                yield chunk
                    
                    content_length = end - start + 1
                    headers = {
                        "Content-Range": f"bytes {start}-{end}/{file_size}",
                        "Accept-Ranges": "bytes",
                        "Content-Length": str(content_length),
                        "Content-Type": content_type
                    }
                    
                    logger.info("Serving file range",
                               session_id=session_id,
                               filename=filename,
                               range=f"{start}-{end}",
                               size=content_length)
                    
                    return StreamingResponse(
                        stream_range(),
                        status_code=206,
                        headers=headers,
                        media_type=content_type
                    )
                    
                except ValueError:
                    # Invalid range format, fall back to full file
                    pass
            
            # For small files or no range request, use FileResponse
            if file_size < 10 * 1024 * 1024:  # 10MB threshold
                logger.info("Serving complete file",
                           session_id=session_id,
                           filename=filename,
                           size=file_size)
                
                return FileResponse(
                    path=file_path,
                    media_type=content_type,
                    filename=filename,
                    headers={"Accept-Ranges": "bytes"}
                )
            
            # For large files, use streaming
            async def stream_file():
                with open(file_path, 'rb') as f:
                    chunk_size = 8192
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        yield chunk
            
            headers = {
                "Content-Length": str(file_size),
                "Accept-Ranges": "bytes",
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
            
            logger.info("Streaming large file",
                       session_id=session_id,
                       filename=filename,
                       size=file_size)
            
            return StreamingResponse(
                stream_file(),
                media_type=content_type,
                headers=headers
            )
            
        except FileNotFoundError:
            logger.warning("File not found",
                          session_id=session_id,
                          filename=filename)
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")
        except Exception as e:
            logger.error("Failed to download file",
                        session_id=session_id,
                        filename=filename,
                        error=str(e))
            raise HTTPException(status_code=500, detail=f"Failed to download file: {str(e)}")
    
    @temp_files_router.delete("/temp/{session_id}")
    async def delete_session(
        session_id: str = Depends(validate_session_access)
    ):
        """Delete all files for a session.
        
        Immediately removes all files and directories associated with
        the specified session. This operation cannot be undone.
        """
        try:
            temp_service = get_temp_file_service()
            success = await temp_service.delete_session(session_id)
            
            if success:
                logger.info("Session deleted successfully", session_id=session_id)
                return {"success": True, "message": f"Session {session_id} deleted successfully"}
            else:
                raise HTTPException(status_code=500, detail="Failed to delete session")
                
        except Exception as e:
            logger.error("Failed to delete session",
                        session_id=session_id,
                        error=str(e))
            raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")
    
    @temp_files_router.post("/temp/cleanup")
    async def manual_cleanup(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
    ):
        """Manually trigger cleanup of expired sessions.
        
        This endpoint allows administrators to immediately clean up
        expired sessions without waiting for the automatic cleanup cycle.
        """
        # In production, this should require admin privileges
        try:
            temp_service = get_temp_file_service()
            cleaned_count = await temp_service.cleanup_expired_sessions()
            
            logger.info("Manual cleanup completed", cleaned_sessions=cleaned_count)
            return {
                "success": True,
                "cleaned_sessions": cleaned_count,
                "message": f"Cleaned up {cleaned_count} expired sessions"
            }
            
        except Exception as e:
            logger.error("Manual cleanup failed", error=str(e))
            raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")
    
    @temp_files_router.get("/temp/{session_id}/{filename:path}/info")
    async def get_file_info(
        session_id: str,
        filename: str,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
    ):
        """Get metadata information about a specific file.
        
        Returns detailed information about a file without downloading it,
        including size, content type, creation time, and other metadata.
        """
        # Validate session access
        validate_session_access(session_id, credentials)
        
        try:
            temp_service = get_temp_file_service()
            _, metadata = await temp_service.get_file(session_id, filename)
            
            logger.info("Retrieved file info",
                       session_id=session_id,
                       filename=filename)
            
            return metadata
            
        except FileNotFoundError:
            logger.warning("File not found for info request",
                          session_id=session_id,
                          filename=filename)
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")
        except Exception as e:
            logger.error("Failed to get file info",
                        session_id=session_id,
                        filename=filename,
                        error=str(e))
            raise HTTPException(status_code=500, detail=f"Failed to get file info: {str(e)}")

    return temp_files_router
