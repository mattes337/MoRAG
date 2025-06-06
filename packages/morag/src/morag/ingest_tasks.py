"""Ingest tasks that process content and store in vector database."""

import asyncio
import json
import requests
from typing import Dict, Any, List, Optional
from pathlib import Path
import structlog

from morag.worker import celery_app, get_morag_api
from morag_services import QdrantVectorStorage, GeminiEmbeddingService
from morag_core.models import Document, DocumentChunk

logger = structlog.get_logger(__name__)


def send_webhook_notification(webhook_url: str, task_id: str, status: str, result: Optional[Dict[str, Any]] = None):
    """Send webhook notification when task completes."""
    if not webhook_url:
        return
    
    try:
        payload = {
            "task_id": task_id,
            "status": status,
            "result": result,
            "completed_at": "2024-01-01T12:05:30"  # Would use actual timestamp
        }
        
        response = requests.post(
            webhook_url,
            json=payload,
            timeout=10,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            logger.info("Webhook notification sent successfully", 
                       webhook_url=webhook_url, task_id=task_id)
        else:
            logger.warning("Webhook notification failed", 
                          webhook_url=webhook_url, 
                          status_code=response.status_code)
            
    except Exception as e:
        logger.error("Failed to send webhook notification", 
                    webhook_url=webhook_url, 
                    task_id=task_id, 
                    error=str(e))


async def store_content_in_vector_db(
    content: str, 
    metadata: Dict[str, Any],
    collection_name: str = "morag_vectors"
) -> List[str]:
    """Store processed content in vector database."""
    if not content.strip():
        logger.warning("Empty content provided for vector storage")
        return []
    
    try:
        # Initialize services
        vector_storage = QdrantVectorStorage()
        embedding_service = GeminiEmbeddingService()
        
        # Connect to vector storage
        await vector_storage.connect()
        
        # Create document chunks for better retrieval
        # Split content into chunks (simple implementation)
        chunk_size = 1000  # characters
        chunks = []
        
        if len(content) <= chunk_size:
            chunks = [content]
        else:
            # Split into overlapping chunks
            overlap = 200
            for i in range(0, len(content), chunk_size - overlap):
                chunk = content[i:i + chunk_size]
                if chunk.strip():
                    chunks.append(chunk)
        
        # Generate embeddings for each chunk
        embeddings = []
        chunk_metadata = []
        
        for i, chunk in enumerate(chunks):
            # Generate embedding
            embedding_result = await embedding_service.generate_embedding(
                chunk, 
                task_type="retrieval_document"
            )
            
            embeddings.append(embedding_result.embedding)
            
            # Prepare metadata for this chunk
            chunk_meta = {
                **metadata,
                "chunk_index": i,
                "chunk_count": len(chunks),
                "text": chunk,  # Store the actual text for retrieval
                "text_length": len(chunk)
            }
            chunk_metadata.append(chunk_meta)
        
        # Store vectors in Qdrant
        point_ids = await vector_storage.store_vectors(
            embeddings, 
            chunk_metadata, 
            collection_name
        )
        
        logger.info("Content stored in vector database successfully",
                   chunk_count=len(chunks),
                   point_ids_count=len(point_ids),
                   collection=collection_name)
        
        return point_ids
        
    except Exception as e:
        logger.error("Failed to store content in vector database", error=str(e))
        raise


@celery_app.task(bind=True)
def ingest_file_task(self, file_path: str, content_type: Optional[str] = None, task_options: Optional[Dict[str, Any]] = None):
    """Ingest file: process content and store in vector database."""
    async def _ingest():
        api = get_morag_api()
        options = task_options or {}
        
        try:
            self.update_state(state='PROGRESS', meta={'stage': 'processing', 'progress': 0.1})
            
            # Process the file
            result = await api.process_file(file_path, content_type, options)
            
            if not result.success:
                raise Exception(f"Processing failed: {result.error_message}")
            
            self.update_state(state='PROGRESS', meta={'stage': 'storing', 'progress': 0.7})
            
            # Store in vector database if requested
            if options.get('store_in_vector_db', True):
                # Prepare metadata for vector storage
                vector_metadata = {
                    "source_type": content_type or "unknown",
                    "source_path": file_path,
                    "processing_time": result.processing_time,
                    **result.metadata,
                    **(options.get('metadata', {}))
                }
                
                # Store content in vector database
                point_ids = await store_content_in_vector_db(
                    result.text_content or result.content,
                    vector_metadata
                )
                
                # Add vector storage info to result
                result.metadata['vector_point_ids'] = point_ids
                result.metadata['stored_in_vector_db'] = True
            
            self.update_state(state='PROGRESS', meta={'stage': 'completing', 'progress': 0.9})
            
            # Send webhook notification if requested
            webhook_url = options.get('webhook_url')
            if webhook_url:
                send_webhook_notification(
                    webhook_url, 
                    self.request.id, 
                    'SUCCESS',
                    {
                        'chunks_processed': len(result.metadata.get('vector_point_ids', [])),
                        'total_text_length': len(result.text_content or result.content),
                        'metadata': result.metadata
                    }
                )
            
            # Clean up temporary file
            try:
                Path(file_path).unlink(missing_ok=True)
                logger.debug("Cleaned up temporary file", file_path=file_path)
            except Exception as e:
                logger.warning("Failed to clean up temporary file", 
                             file_path=file_path, error=str(e))
            
            return {
                'success': result.success,
                'content': result.content,
                'metadata': result.metadata,
                'processing_time': result.processing_time,
                'error_message': result.error_message
            }
            
        except Exception as e:
            logger.error("File ingestion task failed", file_path=file_path, error=str(e))

            # Send failure webhook notification
            webhook_url = options.get('webhook_url')
            if webhook_url:
                send_webhook_notification(webhook_url, self.request.id, 'FAILURE')

            # Clean up temporary file on failure too
            try:
                Path(file_path).unlink(missing_ok=True)
            except:
                pass

            # Create proper exception info for Celery
            error_info = {
                'error': str(e),
                'error_type': e.__class__.__name__,
                'file_path': file_path
            }
            self.update_state(state='FAILURE', meta=error_info)

            # Re-raise with proper exception type information
            raise type(e)(str(e))
    
    return asyncio.run(_ingest())


@celery_app.task(bind=True)
def ingest_url_task(self, url: str, content_type: Optional[str] = None, task_options: Optional[Dict[str, Any]] = None):
    """Ingest URL: process content and store in vector database."""
    async def _ingest():
        api = get_morag_api()
        options = task_options or {}
        
        try:
            self.update_state(state='PROGRESS', meta={'stage': 'processing', 'progress': 0.1})
            
            # Process the URL
            result = await api.process_url(url, content_type, options)
            
            if not result.success:
                raise Exception(f"Processing failed: {result.error_message}")
            
            self.update_state(state='PROGRESS', meta={'stage': 'storing', 'progress': 0.7})
            
            # Store in vector database if requested
            if options.get('store_in_vector_db', True):
                # Prepare metadata for vector storage
                vector_metadata = {
                    "source_type": content_type or "url",
                    "source_url": url,
                    "processing_time": result.processing_time,
                    **result.metadata,
                    **(options.get('metadata', {}))
                }
                
                # Store content in vector database
                point_ids = await store_content_in_vector_db(
                    result.text_content or result.content,
                    vector_metadata
                )
                
                # Add vector storage info to result
                result.metadata['vector_point_ids'] = point_ids
                result.metadata['stored_in_vector_db'] = True
            
            self.update_state(state='PROGRESS', meta={'stage': 'completing', 'progress': 0.9})
            
            # Send webhook notification if requested
            webhook_url = options.get('webhook_url')
            if webhook_url:
                send_webhook_notification(
                    webhook_url, 
                    self.request.id, 
                    'SUCCESS',
                    {
                        'chunks_processed': len(result.metadata.get('vector_point_ids', [])),
                        'total_text_length': len(result.text_content or result.content),
                        'metadata': result.metadata
                    }
                )
            
            return {
                'success': result.success,
                'content': result.content,
                'metadata': result.metadata,
                'processing_time': result.processing_time,
                'error_message': result.error_message
            }
            
        except Exception as e:
            logger.error("URL ingestion task failed", url=url, error=str(e))

            # Send failure webhook notification
            webhook_url = options.get('webhook_url')
            if webhook_url:
                send_webhook_notification(webhook_url, self.request.id, 'FAILURE')

            # Create proper exception info for Celery
            error_info = {
                'error': str(e),
                'error_type': e.__class__.__name__,
                'url': url
            }
            self.update_state(state='FAILURE', meta=error_info)

            # Re-raise with proper exception type information
            raise type(e)(str(e))

    return asyncio.run(_ingest())


@celery_app.task(bind=True)
def ingest_batch_task(self, items: List[Dict[str, Any]], task_options: Optional[Dict[str, Any]] = None):
    """Ingest batch: process multiple items and store in vector database."""
    async def _ingest():
        api = get_morag_api()
        options = task_options or {}

        try:
            self.update_state(state='PROGRESS', meta={'stage': 'batch_processing', 'total_items': len(items), 'progress': 0.1})

            results = []
            for i, item in enumerate(items):
                try:
                    # Update progress
                    progress = 0.1 + (i / len(items)) * 0.8
                    self.update_state(state='PROGRESS', meta={
                        'stage': f'processing_item_{i+1}',
                        'total_items': len(items),
                        'current_item': i + 1,
                        'progress': progress
                    })

                    # Process item based on type
                    detected_source_type = None
                    if 'url' in item:
                        # Auto-detect source type if not provided
                        source_type = item.get('source_type')
                        if not source_type:
                            source_type = api._detect_content_type(item['url'])
                            detected_source_type = source_type
                            logger.info("Auto-detected content type for batch URL",
                                       batch_index=i,
                                       url=item['url'],
                                       detected_type=source_type)

                        result = await api.process_url(
                            item['url'],
                            source_type,
                            {**options, **item.get('options', {})}
                        )
                    elif 'file_path' in item:
                        # Auto-detect source type if not provided
                        source_type = item.get('source_type')
                        if not source_type:
                            source_type = api._detect_content_type_from_file(Path(item['file_path']))
                            detected_source_type = source_type
                            logger.info("Auto-detected content type for batch file",
                                       batch_index=i,
                                       file_path=item['file_path'],
                                       detected_type=source_type)

                        result = await api.process_file(
                            item['file_path'],
                            source_type,
                            {**options, **item.get('options', {})}
                        )
                    else:
                        raise ValueError(f"Invalid item format: {item}")

                    if result.success and options.get('store_in_vector_db', True):
                        # Store in vector database
                        vector_metadata = {
                            "source_type": detected_source_type or item.get('source_type', 'unknown'),
                            "batch_index": i,
                            "batch_size": len(items),
                            "processing_time": result.processing_time,
                            **result.metadata,
                            **(options.get('metadata', {}))
                        }

                        if 'url' in item:
                            vector_metadata['source_url'] = item['url']
                        elif 'file_path' in item:
                            vector_metadata['source_path'] = item['file_path']

                        point_ids = await store_content_in_vector_db(
                            result.text_content or result.content,
                            vector_metadata
                        )

                        result.metadata['vector_point_ids'] = point_ids
                        result.metadata['stored_in_vector_db'] = True

                    results.append({
                        'success': result.success,
                        'content': result.content,
                        'metadata': result.metadata,
                        'processing_time': result.processing_time,
                        'error_message': result.error_message
                    })

                except Exception as e:
                    logger.error("Failed to process batch item", item_index=i, error=str(e))
                    results.append({
                        'success': False,
                        'content': "",
                        'metadata': {"error": str(e)},
                        'processing_time': 0.0,
                        'error_message': str(e)
                    })

            self.update_state(state='PROGRESS', meta={'stage': 'completing', 'progress': 0.9})

            # Send webhook notification if requested
            webhook_url = options.get('webhook_url')
            if webhook_url:
                successful_items = len([r for r in results if r['success']])
                send_webhook_notification(
                    webhook_url,
                    self.request.id,
                    'SUCCESS',
                    {
                        'total_items': len(items),
                        'successful_items': successful_items,
                        'failed_items': len(items) - successful_items,
                        'results': results
                    }
                )

            return results

        except Exception as e:
            logger.error("Batch ingestion task failed", item_count=len(items), error=str(e))

            # Send failure webhook notification
            webhook_url = options.get('webhook_url')
            if webhook_url:
                send_webhook_notification(webhook_url, self.request.id, 'FAILURE')

            # Create proper exception info for Celery
            error_info = {
                'error': str(e),
                'error_type': e.__class__.__name__,
                'item_count': len(items)
            }
            self.update_state(state='FAILURE', meta=error_info)

            # Re-raise with proper exception type information
            raise type(e)(str(e))

    return asyncio.run(_ingest())
