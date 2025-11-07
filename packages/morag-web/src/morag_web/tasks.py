"""Web content processing tasks for MoRAG."""

import asyncio
import threading
from typing import Any, Dict, List, Optional
import nest_asyncio

import structlog

from morag_core.interfaces.task import BaseTask
from morag_core.models.document import DocumentChunk

from .processor import WebProcessor, WebScrapingConfig

logger = structlog.get_logger(__name__)

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Create a shared event loop for all async operations
_event_loop = None
_loop_thread = None
_loop_lock = threading.Lock()

def get_shared_event_loop():
    """Get or create a shared event loop for async operations."""
    global _event_loop, _loop_thread

    with _loop_lock:
        if _event_loop is None or _event_loop.is_closed():
            # Create new event loop in a separate thread
            def run_event_loop():
                global _event_loop
                _event_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(_event_loop)
                _event_loop.run_forever()

            _loop_thread = threading.Thread(target=run_event_loop, daemon=True)
            _loop_thread.start()

            # Wait for loop to be ready
            import time
            while _event_loop is None:
                time.sleep(0.01)

    return _event_loop

def run_async(coroutine):
    """Run coroutine using the shared event loop."""
    loop = get_shared_event_loop()
    future = asyncio.run_coroutine_threadsafe(coroutine, loop)
    return future.result()


class WebProcessingTask(BaseTask):
    """Base class for web processing tasks."""

    def __init__(self):
        super().__init__()
        self.web_processor = WebProcessor()


class ProcessWebUrlTask(WebProcessingTask):
    """Task for processing a single web URL."""

    async def execute(
        self,
        url: str,
        config: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a single web URL and extract content."""

        logger.info("Starting web URL processing task",
                   task_id=task_id,
                   url=url)

        try:
            await self.update_status("PROCESSING", {"stage": "url_validation"})

            # Parse configuration
            web_config = WebScrapingConfig()
            if config:
                for key, value in config.items():
                    if hasattr(web_config, key):
                        setattr(web_config, key, value)

            await self.update_status("PROCESSING", {"stage": "content_extraction"})

            # Process the URL
            result = await self.web_processor.process_url(url, web_config)

            if not result.success:
                await self.update_status("FAILED", {"error": result.error_message})
                return {
                    "success": False,
                    "error": result.error_message,
                    "url": url
                }

            await self.update_status("PROCESSING", {"stage": "content_chunking"})

            # Prepare response data
            response_data = {
                "url": result.url,
                "title": result.content.title,
                "content_length": result.content.content_length,
                "content_type": result.content.content_type,
                "chunks_created": len(result.chunks),
                "links_found": len(result.content.links),
                "images_found": len(result.content.images),
                "processing_time": result.processing_time,
                "metadata": result.content.metadata,
                "success": True
            }

            await self.update_status("COMPLETED", response_data)

            logger.info(
                "Web URL processing completed",
                task_id=task_id,
                url=url,
                chunks_created=len(result.chunks),
                processing_time=result.processing_time
            )

            return response_data

        except Exception as e:
            error_msg = f"Web URL processing failed: {str(e)}"
            await self.update_status("FAILED", {"error": error_msg})

            logger.error(
                "Web URL processing failed",
                task_id=task_id,
                url=url,
                error=str(e)
            )

            return {
                "success": False,
                "error": error_msg,
                "url": url
            }


class ProcessWebUrlsBatchTask(WebProcessingTask):
    """Task for processing multiple web URLs in batch."""

    async def execute(
        self,
        urls: List[str],
        config: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process multiple web URLs in batch."""

        logger.info("Starting batch web URL processing task",
                   task_id=task_id,
                   url_count=len(urls))

        try:
            await self.update_status("PROCESSING", {"stage": "batch_initialization"})

            # Parse configuration
            web_config = WebScrapingConfig()
            if config:
                for key, value in config.items():
                    if hasattr(web_config, key):
                        setattr(web_config, key, value)

            await self.update_status("PROCESSING", {"stage": "batch_processing"})

            # Process all URLs
            results = await self.web_processor.process_urls(urls, web_config)

            # Aggregate results
            successful_results = [r for r in results if r.success]
            failed_results = [r for r in results if not r.success]

            total_chunks = sum(len(r.chunks) for r in successful_results)
            total_processing_time = sum(r.processing_time for r in results)

            response_data = {
                "total_urls": len(urls),
                "successful": len(successful_results),
                "failed": len(failed_results),
                "total_chunks": total_chunks,
                "total_processing_time": total_processing_time,
                "successful_urls": [r.url for r in successful_results],
                "failed_urls": [{"url": r.url, "error": r.error_message} for r in failed_results],
                "success": True
            }

            await self.update_status("COMPLETED", response_data)

            logger.info(
                "Batch web URL processing completed",
                task_id=task_id,
                total_urls=len(urls),
                successful=len(successful_results),
                failed=len(failed_results),
                total_chunks=total_chunks
            )

            return response_data

        except Exception as e:
            error_msg = f"Batch web URL processing failed: {str(e)}"
            await self.update_status("FAILED", {"error": error_msg})

            logger.error(
                "Batch web URL processing failed",
                task_id=task_id,
                urls=urls,
                error=str(e)
            )

            return {
                "success": False,
                "error": error_msg,
                "total_urls": len(urls)
            }


# Convenience functions for external use
async def process_web_url(
    url: str,
    config: Optional[Dict[str, Any]] = None,
    task_id: Optional[str] = None
) -> Dict[str, Any]:
    """Process a single web URL."""
    task = ProcessWebUrlTask()
    return await task.execute(url, config, task_id)


async def process_web_urls_batch(
    urls: List[str],
    config: Optional[Dict[str, Any]] = None,
    task_id: Optional[str] = None
) -> Dict[str, Any]:
    """Process multiple web URLs in batch."""
    task = ProcessWebUrlsBatchTask()
    return await task.execute(urls, config, task_id)


# For backward compatibility with Celery-based systems
def create_celery_tasks(celery_app):
    """Create Celery task wrappers for web processing."""

    @celery_app.task(bind=True)
    def process_web_url_celery(self, url: str, config: Optional[Dict[str, Any]] = None):
        """Celery wrapper for web URL processing."""
        return run_async(process_web_url(url, config, self.request.id))

    @celery_app.task(bind=True)
    def process_web_urls_batch_celery(self, urls: List[str], config: Optional[Dict[str, Any]] = None):
        """Celery wrapper for batch web URL processing."""
        return run_async(process_web_urls_batch(urls, config, self.request.id))

    return {
        'process_web_url': process_web_url_celery,
        'process_web_urls_batch': process_web_urls_batch_celery
    }
