"""HTTP Remote Worker for MoRAG system - No Redis Required.

This module has been replaced with HTTP-based workers that eliminate Redis dependency.
For HTTP worker functionality, use scripts/start_http_remote_worker.py instead.
"""

# Import HTTP worker functions for backward compatibility
from .http_worker import (
    get_morag_api,
    cleanup,
    process_url,
    process_file,
    process_web_page,
    process_youtube_video,
    process_audio_file,
    process_video_file,
    process_document,
    process_image,
    health_check,
    execute_task,
    TASK_HANDLERS
)

__all__ = [
    'get_morag_api',
    'cleanup',
    'process_url',
    'process_file',
    'process_web_page',
    'process_youtube_video',
    'process_audio_file',
    'process_video_file',
    'process_document',
    'process_image',
    'health_check',
    'execute_task',
    'TASK_HANDLERS'
]
