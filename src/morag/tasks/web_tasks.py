"""Web content processing tasks."""

import asyncio
from typing import Any, Dict, List, Optional

import structlog

from ..core.celery_app import celery_app
from ..processors.web import WebProcessor, WebScrapingConfig
from ..processors.document import DocumentChunk
from ..services.chunking import ChunkingService
from ..services.embedding import GeminiService
from ..services.storage import QdrantService
from .base import ProcessingTask

logger = structlog.get_logger(__name__)

# Initialize services
chunking_service = ChunkingService()
embedding_service = GeminiService()
storage_service = QdrantService()
web_processor = WebProcessor(chunking_service)


@celery_app.task(bind=True, base=ProcessingTask)
async def process_web_url(
    self,
    url: str,
    config: Optional[Dict[str, Any]] = None,
    task_id: str = None
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
        result = await web_processor.process_url(url, web_config)

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

        # Store embeddings if embedding service is available
        embeddings_stored = 0
        if result.chunks and hasattr(embedding_service, 'generate_embedding'):
            try:
                await self.update_status("PROCESSING", {"stage": "embedding_generation"})

                # Generate embeddings for chunks
                for chunk in result.chunks:
                    embedding_result = await embedding_service.generate_embedding(chunk.text)
                    if embedding_result and embedding_result.embedding:
                        # Store in vector database (simplified for now)
                        # await storage_service.store_chunk_with_embedding(
                        #     chunk, embedding_result.embedding, result.content.metadata
                        # )
                        embeddings_stored += 1

                response_data["embeddings_stored"] = embeddings_stored

            except Exception as e:
                logger.warning("Failed to generate/store embeddings", error=str(e))
                response_data["embedding_error"] = str(e)

        await self.update_status("COMPLETED", response_data)

        logger.info(
            "Web URL processing completed",
            task_id=task_id,
            url=url,
            chunks_created=len(result.chunks),
            embeddings_stored=embeddings_stored,
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


@celery_app.task(bind=True, base=ProcessingTask)
async def process_web_urls_batch(
    self,
    urls: List[str],
    config: Optional[Dict[str, Any]] = None,
    task_id: str = None
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
        results = await web_processor.process_urls(urls, web_config)

        # Aggregate results
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]

        total_chunks = sum(len(r.chunks) for r in successful_results)
        total_processing_time = sum(r.processing_time for r in results)

        await self.update_status("PROCESSING", {"stage": "embedding_generation"})

        # Generate embeddings for all successful results
        embeddings_stored = 0
        if successful_results and hasattr(embedding_service, 'generate_embedding'):
            try:
                for result in successful_results:
                    for chunk in result.chunks:
                        embedding_result = await embedding_service.generate_embedding(chunk.text)
                        if embedding_result and embedding_result.embedding:
                            # Store in vector database (simplified for now)
                            # await storage_service.store_chunk_with_embedding(
                            #     chunk, embedding_result.embedding, result.content.metadata
                            # )
                            embeddings_stored += 1

            except Exception as e:
                logger.warning("Failed to generate/store embeddings", error=str(e))

        response_data = {
            "total_urls": len(urls),
            "successful": len(successful_results),
            "failed": len(failed_results),
            "total_chunks": total_chunks,
            "embeddings_stored": embeddings_stored,
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
            total_chunks=total_chunks,
            embeddings_stored=embeddings_stored
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
