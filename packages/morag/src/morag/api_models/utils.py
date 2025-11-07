"""Utility functions for MoRAG API."""

import asyncio
import base64
import uuid
from typing import Dict, Any, List, Optional
from pathlib import Path

import aiohttp
import aiofiles
from urllib.parse import urlparse
from fastapi import HTTPException
import structlog

from morag_core.models import ProcessingResult

logger = structlog.get_logger(__name__)


async def download_remote_file(file_path: str, temp_dir: Path) -> Path:
    """Download a remote file (HTTP/HTTPS URL or UNC path) to local temp directory.

    Args:
        file_path: Remote file path (URL or UNC path)
        temp_dir: Local temporary directory

    Returns:
        Path to downloaded local file

    Raises:
        HTTPException: If download fails
    """
    # Determine if it's a URL or UNC path
    parsed = urlparse(file_path)
    is_url = parsed.scheme in ('http', 'https')
    is_unc = file_path.startswith('\\\\') or (len(file_path) > 1 and file_path[1] == ':' and '\\' in file_path)

    if not is_url and not is_unc:
        raise HTTPException(status_code=400, detail=f"Invalid remote file path: {file_path}. Must be HTTP/HTTPS URL or UNC path.")

    # Generate local filename
    if is_url:
        filename = Path(parsed.path).name or "downloaded_file"
    else:
        filename = Path(file_path).name

    if not filename or filename == '.':
        filename = f"remote_file_{uuid.uuid4().hex[:8]}"

    local_path = temp_dir / filename

    try:
        if is_url:
            # Download from HTTP/HTTPS URL
            async with aiohttp.ClientSession() as session:
                async with session.get(file_path) as response:
                    if response.status != 200:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Failed to download file from {file_path}: HTTP {response.status}"
                        )

                    # Check content length if available
                    content_length = response.headers.get('content-length')
                    if content_length:
                        try:
                            from morag_core.config import get_settings
                            settings = get_settings()
                            max_size = settings.get_max_upload_size_bytes()
                            if int(content_length) > max_size:
                                raise HTTPException(
                                    status_code=413,
                                    detail=f"Remote file too large: {content_length} bytes exceeds limit of {max_size} bytes"
                                )
                        except Exception as e:
                            logger.warning("Could not check file size limit", error=str(e))

                    # Download file
                    async with aiofiles.open(local_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            await f.write(chunk)

        else:
            # Copy from UNC path
            import shutil
            shutil.copy2(file_path, local_path)

        logger.info("Remote file downloaded successfully",
                   remote_path=file_path, local_path=str(local_path))
        return local_path

    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        logger.error("Failed to download remote file",
                    remote_path=file_path, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download remote file: {str(e)}"
        )


def normalize_content_type(content_type: Optional[str]) -> Optional[str]:
    """Normalize content type from MIME type to MoRAG content type.

    Args:
        content_type: MIME content type

    Returns:
        Normalized content type for MoRAG
    """
    if not content_type:
        return None

    # Mapping from MIME types to MoRAG content types
    mime_to_morag = {
        'application/pdf': 'document',
        'application/msword': 'document',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'document',
        'application/vnd.ms-excel': 'document',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'document',
        'application/vnd.ms-powerpoint': 'document',
        'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'document',
        'text/plain': 'document',
        'text/html': 'web',
        'text/markdown': 'document',
        'image/jpeg': 'image',
        'image/png': 'image',
        'image/gif': 'image',
        'image/bmp': 'image',
        'image/tiff': 'image',
        'audio/mpeg': 'audio',
        'audio/wav': 'audio',
        'audio/mp4': 'audio',
        'audio/flac': 'audio',
        'video/mp4': 'video',
        'video/avi': 'video',
        'video/quicktime': 'video',
        'video/x-msvideo': 'video',
    }

    # Clean content type (remove charset, etc.)
    clean_content_type = content_type.split(';')[0].strip().lower()

    return mime_to_morag.get(clean_content_type, 'document')


def normalize_processing_result(result: ProcessingResult) -> ProcessingResult:
    """Normalize ProcessingResult to ensure it has a content attribute.

    Args:
        result: Processing result to normalize

    Returns:
        Normalized processing result
    """
    # Ensure the result has a content attribute
    if not hasattr(result, 'content') or result.content is None:
        if hasattr(result, 'document') and result.document:
            if hasattr(result.document, 'raw_text') and result.document.raw_text:
                result.content = result.document.raw_text
            elif hasattr(result.document, 'chunks') and result.document.chunks:
                # Combine all chunks into content
                result.content = '\n\n'.join([chunk.content for chunk in result.document.chunks])
            else:
                result.content = "No content extracted"
        else:
            result.content = "No content available"

    return result


def encode_thumbnails_to_base64(thumbnail_paths: List[str]) -> List[str]:
    """Encode thumbnail images to base64 strings.

    Args:
        thumbnail_paths: List of paths to thumbnail images

    Returns:
        List of base64-encoded thumbnail strings
    """
    encoded_thumbnails = []

    for thumbnail_path in thumbnail_paths:
        try:
            with open(thumbnail_path, 'rb') as f:
                thumbnail_data = f.read()
                encoded_thumbnail = base64.b64encode(thumbnail_data).decode('utf-8')
                encoded_thumbnails.append(encoded_thumbnail)
        except Exception as e:
            logger.warning(f"Failed to encode thumbnail {thumbnail_path}: {e}")
            # Add placeholder for failed thumbnail
            encoded_thumbnails.append("")

    return encoded_thumbnails
