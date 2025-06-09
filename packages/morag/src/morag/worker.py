"""Background worker for MoRAG system using Celery."""

import asyncio
import os
import tempfile
import requests
from typing import Dict, Any, List, Optional
from pathlib import Path

from celery import Celery
import structlog
import redis

from morag.api import MoRAGAPI
from morag_services import ServiceConfig

logger = structlog.get_logger(__name__)

# Create Celery app
celery_app = Celery('morag_worker')

# Get Redis URL from environment or use default
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

# Configure Celery with basic settings (timeouts will be set in worker_init)
celery_app.conf.update(
    broker_url=redis_url,
    result_backend=redis_url,
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    broker_connection_retry_on_startup=True,  # Fix deprecation warning

    # Dynamic queue routing - queues created on demand
    task_default_queue='celery',
    task_default_exchange='celery',
    task_default_exchange_type='direct',
    task_default_routing_key='celery',

    # Enable dynamic queue creation
    task_create_missing_queues=True,
    worker_direct=True,
)

# Initialize Redis client and API key service for remote workers
redis_client = redis.from_url(redis_url)
api_key_service = None  # Will be initialized when needed

# Global MoRAG API instance
morag_api: Optional[MoRAGAPI] = None


def get_morag_api() -> MoRAGAPI:
    """Get or create MoRAG API instance."""
    global morag_api
    if morag_api is None:
        morag_api = MoRAGAPI()
    return morag_api


def get_api_key_service():
    """Get or create API key service instance."""
    global api_key_service
    if api_key_service is None:
        from morag.services.auth_service import APIKeyService
        api_key_service = APIKeyService(redis_client)
    return api_key_service


def get_task_for_user(base_task_name: str, user_id: Optional[str] = None,
                     use_remote: bool = False):
    """Get the appropriate task function and queue based on user and remote flag."""
    if use_remote and user_id:
        remote_task_name = f"{base_task_name}_remote"
        queue_name = get_api_key_service().get_user_queue_name(user_id, "gpu")
        return globals().get(remote_task_name, globals()[base_task_name]), queue_name

    # Default to local processing
    return globals()[base_task_name], 'celery'


def submit_task_for_user(task_func, args, kwargs, user_id: Optional[str] = None,
                        use_remote: bool = False):
    """Submit task to appropriate queue based on user."""
    task, queue = get_task_for_user(task_func.__name__, user_id, use_remote)

    return task.apply_async(
        args=args,
        kwargs=kwargs,
        queue=queue
    )


@celery_app.task(bind=True)
def process_url_task(self, url: str, content_type: Optional[str] = None, options: Optional[Dict[str, Any]] = None):
    """Process content from URL as background task."""
    async def _process():
        api = get_morag_api()
        try:
            self.update_state(state='PROCESSING', meta={'stage': 'starting'})
            
            result = await api.process_url(url, content_type, options)
            
            return {
                'success': result.success,
                'content': result.text_content or "",
                'metadata': result.metadata,
                'processing_time': result.processing_time,
                'error_message': result.error_message
            }
        except Exception as e:
            logger.error("URL processing task failed", url=url, error=str(e))
            self.update_state(state='FAILURE', meta={'error': str(e)})
            raise
    
    return asyncio.run(_process())


@celery_app.task(bind=True)
def process_file_task(self, file_path: str, content_type: Optional[str] = None, options: Optional[Dict[str, Any]] = None):
    """Process file as background task."""
    async def _process():
        api = get_morag_api()
        try:
            self.update_state(state='PROCESSING', meta={'stage': 'starting'})
            
            result = await api.process_file(file_path, content_type, options)
            
            return {
                'success': result.success,
                'content': result.text_content or "",
                'metadata': result.metadata,
                'processing_time': result.processing_time,
                'error_message': result.error_message
            }
        except Exception as e:
            logger.error("File processing task failed", file_path=file_path, error=str(e))
            self.update_state(state='FAILURE', meta={'error': str(e)})
            raise
    
    return asyncio.run(_process())


@celery_app.task(bind=True)
def process_web_page_task(self, url: str, options: Optional[Dict[str, Any]] = None):
    """Process web page as background task."""
    async def _process():
        api = get_morag_api()
        try:
            self.update_state(state='PROCESSING', meta={'stage': 'web_scraping'})
            
            result = await api.process_web_page(url, options)
            
            return {
                'success': result.success,
                'content': result.text_content or "",
                'metadata': result.metadata,
                'processing_time': result.processing_time,
                'error_message': result.error_message
            }
        except Exception as e:
            logger.error("Web page processing task failed", url=url, error=str(e))
            self.update_state(state='FAILURE', meta={'error': str(e)})
            raise
    
    return asyncio.run(_process())


@celery_app.task(bind=True)
def process_youtube_video_task(self, url: str, options: Optional[Dict[str, Any]] = None):
    """Process YouTube video as background task."""
    async def _process():
        api = get_morag_api()
        try:
            self.update_state(state='PROCESSING', meta={'stage': 'youtube_download'})
            
            result = await api.process_youtube_video(url, options)
            
            return {
                'success': result.success,
                'content': result.text_content or "",
                'metadata': result.metadata,
                'processing_time': result.processing_time,
                'error_message': result.error_message
            }
        except Exception as e:
            logger.error("YouTube processing task failed", url=url, error=str(e))
            self.update_state(state='FAILURE', meta={'error': str(e)})
            raise
    
    return asyncio.run(_process())


@celery_app.task(bind=True)
def process_batch_task(self, items: List[Dict[str, Any]], options: Optional[Dict[str, Any]] = None):
    """Process batch of items as background task."""
    async def _process():
        api = get_morag_api()
        try:
            self.update_state(state='PROCESSING', meta={'stage': 'batch_processing', 'total_items': len(items)})
            
            results = await api.process_batch(items, options)
            
            return [
                {
                    'success': result.success,
                    'content': result.text_content or "",
                    'metadata': result.metadata,
                    'processing_time': result.processing_time,
                    'error_message': result.error_message
                } for result in results
            ]
        except Exception as e:
            logger.error("Batch processing task failed", item_count=len(items), error=str(e))
            self.update_state(state='FAILURE', meta={'error': str(e)})
            raise
    
    return asyncio.run(_process())


@celery_app.task(bind=True)
def search_task(self, query: str, limit: int = 10, filters: Optional[Dict[str, Any]] = None):
    """Search for similar content as background task."""
    async def _search():
        api = get_morag_api()
        try:
            self.update_state(state='PROCESSING', meta={'stage': 'searching'})
            
            results = await api.search(query, limit, filters)
            
            return {'results': results}
        except Exception as e:
            logger.error("Search task failed", query=query, error=str(e))
            self.update_state(state='FAILURE', meta={'error': str(e)})
            raise
    
    return asyncio.run(_search())


@celery_app.task
def health_check_task():
    """Health check as background task."""
    async def _health():
        api = get_morag_api()
        try:
            status = await api.health_check()
            return status
        except Exception as e:
            logger.error("Health check task failed", error=str(e))
            raise
    
    return asyncio.run(_health())


# Remote worker task variants for HTTP file transfer
@celery_app.task(bind=True)
def process_file_task_remote(self, file_url: str, user_id: str, content_type: Optional[str] = None,
                           task_options: Optional[Dict[str, Any]] = None):
    """Remote worker variant - downloads file via HTTP and processes."""
    async def _process():
        api = get_morag_api()
        temp_path = None

        try:
            # Download file from server
            self.update_state(state='DOWNLOADING', meta={'stage': 'downloading_file'})

            response = requests.get(file_url, stream=True)
            response.raise_for_status()

            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file_url).suffix) as temp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                temp_path = temp_file.name

            self.update_state(state='PROCESSING', meta={'stage': 'processing'})

            # Process the file (only heavy lifting - no external services)
            result = await api.process_file(temp_path, content_type, task_options)

            # Return only markdown content and metadata - no vector storage
            return {
                'success': result.success,
                'content': result.text_content or result.content,
                'metadata': result.metadata,
                'processing_time': result.processing_time,
                'error_message': result.error_message,
                'user_id': user_id
            }

        except Exception as e:
            logger.error("Remote file processing failed", file_url=file_url, error=str(e))
            self.update_state(state='FAILURE', meta={'error': str(e)})
            raise
        finally:
            # Clean up temporary file
            if temp_path and Path(temp_path).exists():
                Path(temp_path).unlink()

    return asyncio.run(_process())


@celery_app.task(bind=True)
def process_url_task_remote(self, url: str, user_id: str, content_type: Optional[str] = None,
                          task_options: Optional[Dict[str, Any]] = None):
    """Remote worker variant - processes URL directly."""
    async def _process():
        api = get_morag_api()
        try:
            self.update_state(state='PROCESSING', meta={'stage': 'processing_url'})

            # Process URL directly (no external service calls)
            result = await api.process_url(url, content_type, task_options)

            # Return only markdown content and metadata
            return {
                'success': result.success,
                'content': result.text_content or result.content,
                'metadata': result.metadata,
                'processing_time': result.processing_time,
                'error_message': result.error_message,
                'user_id': user_id
            }

        except Exception as e:
            logger.error("Remote URL processing failed", url=url, error=str(e))
            self.update_state(state='FAILURE', meta={'error': str(e)})
            raise

    return asyncio.run(_process())


@celery_app.task(bind=True)
def process_web_page_task_remote(self, url: str, user_id: str, task_options: Optional[Dict[str, Any]] = None):
    """Remote worker variant - processes web page directly."""
    async def _process():
        api = get_morag_api()
        try:
            self.update_state(state='PROCESSING', meta={'stage': 'web_scraping'})

            result = await api.process_web_page(url, task_options)

            return {
                'success': result.success,
                'content': result.text_content or result.content,
                'metadata': result.metadata,
                'processing_time': result.processing_time,
                'error_message': result.error_message,
                'user_id': user_id
            }

        except Exception as e:
            logger.error("Remote web page processing failed", url=url, error=str(e))
            self.update_state(state='FAILURE', meta={'error': str(e)})
            raise

    return asyncio.run(_process())


@celery_app.task(bind=True)
def process_youtube_video_task_remote(self, url: str, user_id: str, task_options: Optional[Dict[str, Any]] = None):
    """Remote worker variant - processes YouTube video directly."""
    async def _process():
        api = get_morag_api()
        try:
            self.update_state(state='PROCESSING', meta={'stage': 'youtube_download'})

            result = await api.process_youtube_video(url, task_options)

            return {
                'success': result.success,
                'content': result.text_content or result.content,
                'metadata': result.metadata,
                'processing_time': result.processing_time,
                'error_message': result.error_message,
                'user_id': user_id
            }

        except Exception as e:
            logger.error("Remote YouTube processing failed", url=url, error=str(e))
            self.update_state(state='FAILURE', meta={'error': str(e)})
            raise

    return asyncio.run(_process())


# Worker event handlers
from celery.signals import worker_init, worker_ready, worker_shutdown

@worker_init.connect
def worker_init_handler(sender=None, conf=None, **kwargs):
    """Initialize worker."""
    # Import settings here to avoid module-level import issues
    from morag_core.config import settings

    # Update Celery configuration with timeout settings now that environment is loaded
    celery_app.conf.update(
        task_time_limit=settings.celery_task_time_limit,  # Configurable hard limit (default: 2.5 hours)
        task_soft_time_limit=settings.celery_task_soft_time_limit,  # Configurable soft limit (default: 2 hours)
        worker_prefetch_multiplier=settings.celery_worker_prefetch_multiplier,  # Configurable prefetch
        worker_max_tasks_per_child=settings.celery_worker_max_tasks_per_child,  # Configurable max tasks
    )

    logger.info(
        "MoRAG worker initializing",
        task_soft_time_limit=f"{settings.celery_task_soft_time_limit}s ({settings.celery_task_soft_time_limit/60:.1f}min)",
        task_time_limit=f"{settings.celery_task_time_limit}s ({settings.celery_task_time_limit/60:.1f}min)",
        worker_prefetch_multiplier=settings.celery_worker_prefetch_multiplier,
        worker_max_tasks_per_child=settings.celery_worker_max_tasks_per_child
    )


@worker_ready.connect
def worker_ready_handler(sender=None, **kwargs):
    """Worker ready handler."""
    logger.info("MoRAG worker ready")


@worker_shutdown.connect
def worker_shutdown_handler(sender=None, **kwargs):
    """Worker shutdown handler."""
    global morag_api
    if morag_api:
        asyncio.run(morag_api.cleanup())
    logger.info("MoRAG worker shutdown")


def main():
    """Main entry point for the worker."""
    import argparse

    parser = argparse.ArgumentParser(description="MoRAG Background Worker")
    parser.add_argument("--loglevel", default="info", help="Log level")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent workers")
    parser.add_argument("--queues", default="celery", help="Comma-separated list of queues")
    parser.add_argument("--broker", default=redis_url, help="Broker URL")
    parser.add_argument("--backend", default=redis_url, help="Result backend URL")

    args = parser.parse_args()

    # Update Celery configuration
    celery_app.conf.update(
        broker_url=args.broker,
        result_backend=args.backend
    )

    # Start worker
    celery_app.worker_main([
        'worker',
        f'--loglevel={args.loglevel}',
        f'--concurrency={args.concurrency}',
        f'--queues={args.queues}',
    ])


if __name__ == "__main__":
    main()
