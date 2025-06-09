"""HTTP Remote Worker for MoRAG system - No Redis Required.

This module provides the HTTP-based remote worker functionality that eliminates
the need for Redis/Celery infrastructure. Workers connect directly to the main
server via HTTP API calls.
"""

import asyncio
import os
import tempfile
from typing import Dict, Any, List, Optional
from pathlib import Path

import structlog

from morag.api import MoRAGAPI

logger = structlog.get_logger(__name__)

# Global MoRAG API instance
morag_api: Optional[MoRAGAPI] = None


def get_morag_api() -> MoRAGAPI:
    """Get or create MoRAG API instance."""
    global morag_api
    if morag_api is None:
        morag_api = MoRAGAPI()
    return morag_api


async def cleanup():
    """Cleanup resources."""
    global morag_api
    if morag_api:
        await morag_api.cleanup()
        morag_api = None


# HTTP Worker Processing Functions
# These functions are used by the HTTP remote worker to process different content types

async def process_url(url: str, content_type: Optional[str] = None, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Process content from URL."""
    api = get_morag_api()
    try:
        result = await api.process_url(url, content_type, options)
        
        return {
            'success': result.success,
            'content': result.text_content or "",
            'metadata': result.metadata,
            'processing_time': result.processing_time,
            'error_message': result.error_message
        }
    except Exception as e:
        logger.error("URL processing failed", url=url, error=str(e))
        return {
            'success': False,
            'content': "",
            'metadata': {},
            'processing_time': 0,
            'error_message': str(e)
        }


async def process_file(file_path: str, content_type: Optional[str] = None, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Process file."""
    api = get_morag_api()
    try:
        result = await api.process_file(file_path, content_type, options)
        
        return {
            'success': result.success,
            'content': result.text_content or "",
            'metadata': result.metadata,
            'processing_time': result.processing_time,
            'error_message': result.error_message
        }
    except Exception as e:
        logger.error("File processing failed", file_path=file_path, error=str(e))
        return {
            'success': False,
            'content': "",
            'metadata': {},
            'processing_time': 0,
            'error_message': str(e)
        }


async def process_web_page(url: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Process web page."""
    api = get_morag_api()
    try:
        result = await api.process_web_page(url, options)
        
        return {
            'success': result.success,
            'content': result.text_content or "",
            'metadata': result.metadata,
            'processing_time': result.processing_time,
            'error_message': result.error_message
        }
    except Exception as e:
        logger.error("Web page processing failed", url=url, error=str(e))
        return {
            'success': False,
            'content': "",
            'metadata': {},
            'processing_time': 0,
            'error_message': str(e)
        }


async def process_youtube_video(url: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Process YouTube video."""
    api = get_morag_api()
    try:
        result = await api.process_youtube_video(url, options)
        
        return {
            'success': result.success,
            'content': result.text_content or "",
            'metadata': result.metadata,
            'processing_time': result.processing_time,
            'error_message': result.error_message
        }
    except Exception as e:
        logger.error("YouTube processing failed", url=url, error=str(e))
        return {
            'success': False,
            'content': "",
            'metadata': {},
            'processing_time': 0,
            'error_message': str(e)
        }


async def process_audio_file(file_path: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Process audio file."""
    api = get_morag_api()
    try:
        result = await api.process_audio_file(file_path, options)
        
        return {
            'success': result.success,
            'content': result.text_content or "",
            'metadata': result.metadata,
            'processing_time': result.processing_time,
            'error_message': result.error_message
        }
    except Exception as e:
        logger.error("Audio processing failed", file_path=file_path, error=str(e))
        return {
            'success': False,
            'content': "",
            'metadata': {},
            'processing_time': 0,
            'error_message': str(e)
        }


async def process_video_file(file_path: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Process video file."""
    api = get_morag_api()
    try:
        result = await api.process_video_file(file_path, options)
        
        return {
            'success': result.success,
            'content': result.text_content or "",
            'metadata': result.metadata,
            'processing_time': result.processing_time,
            'error_message': result.error_message
        }
    except Exception as e:
        logger.error("Video processing failed", file_path=file_path, error=str(e))
        return {
            'success': False,
            'content': "",
            'metadata': {},
            'processing_time': 0,
            'error_message': str(e)
        }


async def process_document(file_path: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Process document."""
    api = get_morag_api()
    try:
        result = await api.process_document(file_path, options)
        
        return {
            'success': result.success,
            'content': result.text_content or "",
            'metadata': result.metadata,
            'processing_time': result.processing_time,
            'error_message': result.error_message
        }
    except Exception as e:
        logger.error("Document processing failed", file_path=file_path, error=str(e))
        return {
            'success': False,
            'content': "",
            'metadata': {},
            'processing_time': 0,
            'error_message': str(e)
        }


async def process_image(file_path: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Process image."""
    api = get_morag_api()
    try:
        result = await api.process_image(file_path, options)
        
        return {
            'success': result.success,
            'content': result.text_content or "",
            'metadata': result.metadata,
            'processing_time': result.processing_time,
            'error_message': result.error_message
        }
    except Exception as e:
        logger.error("Image processing failed", file_path=file_path, error=str(e))
        return {
            'success': False,
            'content': "",
            'metadata': {},
            'processing_time': 0,
            'error_message': str(e)
        }


async def health_check() -> Dict[str, Any]:
    """Health check."""
    api = get_morag_api()
    try:
        status = await api.health_check()
        return status
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return {
            'status': 'unhealthy',
            'error': str(e)
        }


# Task type mapping for HTTP workers
TASK_HANDLERS = {
    'process_url': process_url,
    'process_file': process_file,
    'process_web': process_web_page,
    'process_youtube': process_youtube_video,
    'process_audio': process_audio_file,
    'process_video': process_video_file,
    'process_document': process_document,
    'process_image': process_image,
    'health_check': health_check,
}


async def execute_task(task_type: str, **kwargs) -> Dict[str, Any]:
    """Execute a task by type."""
    handler = TASK_HANDLERS.get(task_type)
    if not handler:
        return {
            'success': False,
            'content': "",
            'metadata': {},
            'processing_time': 0,
            'error_message': f"Unknown task type: {task_type}"
        }
    
    return await handler(**kwargs)
