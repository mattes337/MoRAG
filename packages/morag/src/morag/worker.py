"""HTTP Remote Worker for MoRAG system - No Redis/Celery Required.

This module provides HTTP-based workers that eliminate Redis/Celery dependency.
For HTTP worker functionality, use scripts/start_http_remote_worker.py.
"""

from typing import Dict, Any
import structlog

# Import HTTP worker functions
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

logger = structlog.get_logger(__name__)


def submit_task_for_user(user_id: str, task_type: str, **kwargs) -> str:
    """Submit a task for a specific user (for GPU worker routing).

    This is a placeholder for HTTP-based worker routing.
    In the future, this would route to user-specific GPU workers via HTTP.
    """
    logger.info("Task submitted for user (HTTP routing not implemented yet)",
               user_id=user_id, task_type=task_type)

    # For now, return a mock task ID
    # TODO: Implement HTTP-based task submission to remote workers
    import uuid
    return str(uuid.uuid4())





def main():
    """Main entry point for HTTP worker."""
    logger.info("Starting HTTP worker...")
    # This would start the HTTP worker server
    # For now, this is a placeholder
    print("HTTP worker main - use scripts/start_http_remote_worker.py instead")


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
    'TASK_HANDLERS',
    'submit_task_for_user',
    'main'
]
