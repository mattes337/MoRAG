"""Background worker for MoRAG system using Celery."""

import asyncio
import os
from typing import Dict, Any, List, Optional
from pathlib import Path

from celery import Celery
import structlog

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
)

# Global MoRAG API instance
morag_api: Optional[MoRAGAPI] = None


def get_morag_api() -> MoRAGAPI:
    """Get or create MoRAG API instance."""
    global morag_api
    if morag_api is None:
        morag_api = MoRAGAPI()
    return morag_api


@celery_app.task(bind=True)
def process_url_task(self, url: str, content_type: Optional[str] = None, options: Optional[Dict[str, Any]] = None):
    """Process content from URL as background task."""
    async def _process():
        api = get_morag_api()
        try:
            self.update_state(state='PROCESSING', meta={'stage': 'starting', 'progress': 0.0, 'message': 'Initializing URL processing'})

            # Create a progress callback for the API
            def progress_callback(progress: float, message: str = None):
                self.update_state(state='PROGRESS', meta={'progress': progress, 'message': message or f'Processing... {int(progress * 100)}%'})

            # Pass progress callback to API if supported
            if options is None:
                options = {}
            options['progress_callback'] = progress_callback

            result = await api.process_url(url, content_type, options)

            self.update_state(state='PROCESSING', meta={'stage': 'completing', 'progress': 0.95, 'message': 'Finalizing processing'})

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
            self.update_state(state='PROCESSING', meta={'stage': 'starting', 'progress': 0.0, 'message': 'Initializing file processing'})

            # Create a progress callback for the API
            def progress_callback(progress: float, message: str = None):
                self.update_state(state='PROGRESS', meta={'progress': progress, 'message': message or f'Processing... {int(progress * 100)}%'})

            # Pass progress callback to API if supported
            if options is None:
                options = {}
            options['progress_callback'] = progress_callback

            result = await api.process_file(file_path, content_type, options)

            self.update_state(state='PROCESSING', meta={'stage': 'completing', 'progress': 0.95, 'message': 'Finalizing processing'})

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
            self.update_state(state='PROCESSING', meta={'stage': 'web_scraping', 'progress': 0.0, 'message': 'Initializing web page processing'})

            # Create a progress callback for the API
            def progress_callback(progress: float, message: str = None):
                self.update_state(state='PROGRESS', meta={'progress': progress, 'message': message or f'Processing... {int(progress * 100)}%'})

            # Pass progress callback to API if supported
            if options is None:
                options = {}
            options['progress_callback'] = progress_callback

            result = await api.process_web_page(url, options)

            self.update_state(state='PROCESSING', meta={'stage': 'completing', 'progress': 0.95, 'message': 'Finalizing web page processing'})

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
            self.update_state(state='PROCESSING', meta={'stage': 'youtube_download', 'progress': 0.0, 'message': 'Initializing YouTube video processing'})

            # Create a progress callback for the API
            def progress_callback(progress: float, message: str = None):
                self.update_state(state='PROGRESS', meta={'progress': progress, 'message': message or f'Processing... {int(progress * 100)}%'})

            # Pass progress callback to API if supported
            if options is None:
                options = {}
            options['progress_callback'] = progress_callback

            result = await api.process_youtube_video(url, options)

            self.update_state(state='PROCESSING', meta={'stage': 'completing', 'progress': 0.95, 'message': 'Finalizing YouTube video processing'})

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
            total_items = len(items)
            self.update_state(state='PROCESSING', meta={
                'stage': 'batch_processing',
                'total_items': total_items,
                'progress': 0.0,
                'message': f'Starting batch processing of {total_items} items'
            })

            # Create a progress callback for batch processing
            def progress_callback(completed_items: int, message: str = None):
                progress = completed_items / total_items if total_items > 0 else 0.0
                self.update_state(state='PROGRESS', meta={
                    'progress': progress,
                    'completed_items': completed_items,
                    'total_items': total_items,
                    'message': message or f'Processed {completed_items}/{total_items} items'
                })

            # Pass progress callback to API if supported
            if options is None:
                options = {}
            options['batch_progress_callback'] = progress_callback

            results = await api.process_batch(items, options)

            self.update_state(state='PROCESSING', meta={
                'stage': 'completing',
                'progress': 0.95,
                'message': 'Finalizing batch processing'
            })

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
            self.update_state(state='PROCESSING', meta={'stage': 'searching', 'progress': 0.1, 'message': 'Initializing search'})

            # Create a progress callback for the API
            def progress_callback(progress: float, message: str = None):
                self.update_state(state='PROGRESS', meta={'progress': progress, 'message': message or f'Searching... {int(progress * 100)}%'})

            # Search doesn't typically need progress callbacks, but we can simulate progress
            self.update_state(state='PROGRESS', meta={'progress': 0.5, 'message': 'Executing search query'})

            results = await api.search(query, limit, filters)

            self.update_state(state='PROCESSING', meta={'stage': 'completing', 'progress': 0.9, 'message': 'Finalizing search results'})

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
