"""Ingest tasks that process content and store in vector database."""

import asyncio
import json
import requests
import os
from typing import Dict, Any, List, Optional
from pathlib import Path
import structlog

from morag.worker import celery_app, get_morag_api
from morag_services import QdrantVectorStorage, GeminiEmbeddingService
from morag_core.models import Document, DocumentChunk
from morag_core.config import get_settings, validate_chunk_size
from morag.services.remote_job_service import RemoteJobService
from morag.models.remote_job_api import CreateRemoteJobRequest

logger = structlog.get_logger(__name__)


async def continue_ingestion_after_remote_processing(
    remote_job_id: str,
    content: str,
    metadata: Dict[str, Any],
    processing_time: float
) -> bool:
    """Continue the ingestion pipeline after remote processing completes."""
    try:
        # Get the remote job to get original task options
        remote_service = RemoteJobService()
        remote_job = remote_service.get_job_status(remote_job_id)

        if not remote_job:
            logger.error("Remote job not found for continuation", remote_job_id=remote_job_id)
            return False

        options = remote_job.task_options

        # Store in vector database if requested
        if options.get('store_in_vector_db', True):
            # Prepare metadata for vector storage
            options_metadata = options.get('metadata') or {}

            vector_metadata = {
                "source_type": remote_job.content_type,
                "source_path": remote_job.source_file_path,
                "processing_time": processing_time,
                "remote_processing": True,
                "remote_job_id": remote_job_id,
                **metadata,
                **options_metadata
            }

            # Generate document ID if not provided
            document_id = options.get('document_id')
            if not document_id:
                document_id = generate_document_id(remote_job.source_file_path, content)

            # Store content in vector database
            point_ids = await store_content_in_vector_db(
                content,
                vector_metadata,
                chunk_size=options.get('chunk_size'),
                chunk_overlap=options.get('chunk_overlap'),
                document_id=document_id,
                replace_existing=options.get('replace_existing', False),
                use_content_checksum=options.get('use_content_checksum', True)
            )

            logger.info("Remote processing result stored in vector database",
                       remote_job_id=remote_job_id,
                       point_ids_count=len(point_ids))

        # Send webhook notification if requested
        webhook_url = options.get('webhook_url')
        if webhook_url:
            send_webhook_notification(
                webhook_url,
                remote_job.ingestion_task_id,
                'SUCCESS',
                {
                    'remote_job_id': remote_job_id,
                    'chunks_processed': len(point_ids) if 'point_ids' in locals() else 0,
                    'total_text_length': len(content),
                    'metadata': {**metadata, 'remote_processing': True}
                }
            )

        return True

    except Exception as e:
        logger.error("Failed to continue ingestion after remote processing",
                    remote_job_id=remote_job_id,
                    error=str(e))
        return False


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


def generate_document_id(source: str, content: Optional[str] = None) -> str:
    """Generate consistent document ID from source and optionally content.

    Args:
        source: Source identifier (filename, URL, etc.)
        content: Optional content for hash generation

    Returns:
        Generated document ID
    """
    import hashlib
    import os
    from urllib.parse import urlparse, urlunparse

    # For URLs, use normalized URL as base
    if source.startswith(('http://', 'https://')):
        parsed = urlparse(source)
        normalized = urlunparse((
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path,
            parsed.params,
            parsed.query,
            ''  # Remove fragment
        ))
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    # For files, use filename and optionally content hash
    elif ('/' in source or '\\' in source or '.' in source) and not source.startswith(('http://', 'https://')):
        filename = os.path.basename(source)
        # Remove extension and replace special characters
        base_name = os.path.splitext(filename)[0].replace('.', '_').replace(' ', '_')
        extension = os.path.splitext(filename)[1].replace('.', '').replace(' ', '_')

        if content:
            content_hash = hashlib.sha256(content.encode()).hexdigest()[:8]
            return f"{base_name}_{extension}_{content_hash}"
        return f"{base_name}_{extension}" if extension else base_name

    # For other sources, use direct hash
    else:
        return hashlib.sha256(source.encode()).hexdigest()[:16]

async def store_content_in_vector_db(
    content: str,
    metadata: Dict[str, Any],
    collection_name: str = "morag_vectors",
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    document_id: Optional[str] = None,
    replace_existing: bool = False,
    use_content_checksum: bool = True
) -> List[str]:
    """Store processed content in vector database with document replacement support."""
    if not content.strip():
        logger.warning("Empty content provided for vector storage")
        return []

    try:
        # Get settings for chunk configuration
        settings = get_settings()

        # Use provided chunk size or default from settings
        chunk_size = chunk_size or settings.default_chunk_size
        chunk_overlap = chunk_overlap or settings.default_chunk_overlap

        # Validate chunk size
        is_valid, validation_message = validate_chunk_size(chunk_size, content)
        if not is_valid:
            logger.warning("Chunk size validation warning",
                         chunk_size=chunk_size,
                         message=validation_message)

        # Initialize services with environment configuration
        qdrant_host = os.getenv('QDRANT_HOST', 'localhost')
        qdrant_port = int(os.getenv('QDRANT_PORT', '6333'))
        qdrant_api_key = os.getenv('QDRANT_API_KEY')
        collection_name_env = os.getenv('QDRANT_COLLECTION_NAME')
        if not collection_name_env:
            raise ValueError("QDRANT_COLLECTION_NAME environment variable is required")

        vector_storage = QdrantVectorStorage(
            host=qdrant_host,
            port=qdrant_port,
            api_key=qdrant_api_key,
            collection_name=collection_name_env
        )

        # Get API key from environment (prefer GEMINI_API_KEY for consistency)
        api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")

        embedding_service = GeminiEmbeddingService(api_key=api_key)

        # Connect to vector storage
        await vector_storage.connect()

        # Generate content checksum for duplicate detection
        content_checksum = None
        if use_content_checksum:
            import hashlib
            content_checksum = hashlib.sha256(content.encode()).hexdigest()

            # Check if document with same checksum already exists
            if not replace_existing:
                existing_points = await vector_storage.search_by_metadata(
                    {"content_checksum": content_checksum},
                    limit=1
                )
                if existing_points:
                    logger.info("Document with same content checksum already exists, skipping",
                               content_checksum=content_checksum[:16],
                               existing_point_id=existing_points[0]["id"])
                    return [existing_points[0]["id"]]

        # Create document chunks for better retrieval
        chunks = []

        if len(content) <= chunk_size:
            chunks = [content]
        else:
            # Split into overlapping chunks with configured sizes
            for i in range(0, len(content), chunk_size - chunk_overlap):
                chunk = content[i:i + chunk_size]
                if chunk.strip():
                    chunks.append(chunk)
        
        # Generate embeddings for all chunks using batch processing
        logger.info("Generating embeddings for chunks", chunk_count=len(chunks))

        # Use batch embedding for better performance
        batch_result = await embedding_service.generate_embeddings_batch(
            chunks,
            task_type="retrieval_document"
        )

        # Extract embeddings from batch result
        embeddings = [result.embedding for result in batch_result]

        # Prepare metadata for each chunk
        chunk_metadata = []
        for i, chunk in enumerate(chunks):
            chunk_meta = {
                **metadata,
                "chunk_index": i,
                "chunk_count": len(chunks),
                "text": chunk,  # Store the actual text for retrieval
                "text_length": len(chunk)
            }

            # Add document_id if provided
            if document_id:
                chunk_meta["document_id"] = document_id

            # Add content checksum if generated
            if content_checksum:
                chunk_meta["content_checksum"] = content_checksum

            chunk_metadata.append(chunk_meta)

        # Store vectors in Qdrant with replacement support
        if document_id and replace_existing:
            # Use document replacement
            point_ids = await vector_storage.replace_document(
                document_id,
                embeddings,
                chunk_metadata,
                collection_name
            )
        else:
            # Regular storage
            point_ids = await vector_storage.store_vectors(
                embeddings,
                chunk_metadata,
                collection_name
            )
        
        logger.info("Content stored in vector database successfully",
                   chunk_count=len(chunks),
                   chunk_size=chunk_size,
                   chunk_overlap=chunk_overlap,
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

        # Log task start and file existence check with detailed timing
        file_path_obj = Path(file_path)
        file_exists = file_path_obj.exists()

        logger.info("Starting file ingestion task",
                   task_id=self.request.id,
                   file_path=file_path,
                   content_type=content_type,
                   file_exists=file_exists,
                   remote=options.get('remote', False))

        try:
            # Check if remote processing is requested
            if options.get('remote', False):
                # Check if content type supports remote processing (audio/video)
                if content_type in ['audio', 'video']:
                    logger.info("Creating remote job for processing",
                               task_id=self.request.id,
                               content_type=content_type)

                    # Create remote job
                    remote_service = RemoteJobService()
                    remote_request = CreateRemoteJobRequest(
                        source_file_path=file_path,
                        content_type=content_type,
                        task_options=options,
                        ingestion_task_id=self.request.id
                    )

                    remote_job = remote_service.create_job(remote_request)

                    logger.info("Remote job created, waiting for completion",
                               task_id=self.request.id,
                               remote_job_id=remote_job.id)

                    # Return early - the remote worker will continue the pipeline
                    return {
                        'success': True,
                        'content': "",
                        'metadata': {
                            'remote_job_id': remote_job.id,
                            'remote_processing': True,
                            'status': 'pending_remote_processing'
                        },
                        'processing_time': 0.0,
                        'error_message': None
                    }
                else:
                    logger.info("Content type does not support remote processing, falling back to local",
                               content_type=content_type)
                    # Fall through to local processing
            # Check if file still exists before processing
            if not file_exists:
                # Get additional debugging information
                parent_dir = file_path_obj.parent
                parent_exists = parent_dir.exists() if parent_dir else False

                if parent_exists:
                    # List files in parent directory for debugging
                    try:
                        files_in_dir = list(parent_dir.glob("*")) if parent_dir.exists() else []
                        logger.error("File not found but parent directory exists",
                                   file_path=file_path,
                                   parent_dir=str(parent_dir),
                                   files_in_parent=len(files_in_dir),
                                   sample_files=[str(f.name) for f in files_in_dir[:5]])
                    except Exception as list_error:
                        logger.warning("Could not list parent directory contents",
                                     parent_dir=str(parent_dir),
                                     error=str(list_error))
                else:
                    logger.error("File and parent directory do not exist",
                               file_path=file_path,
                               parent_dir=str(parent_dir) if parent_dir else "None")

                raise FileNotFoundError(f"Temporary file was cleaned up before processing: {file_path}")

            self.update_state(state='PROGRESS', meta={'stage': 'processing', 'progress': 0.1})

            # Process the file
            result = await api.process_file(file_path, content_type, options)
            
            if not result.success:
                raise Exception(f"Processing failed: {result.error_message}")

            # Ensure metadata is always a dictionary
            if result.metadata is None:
                result.metadata = {}

            self.update_state(state='PROGRESS', meta={'stage': 'storing', 'progress': 0.7})

            # Store in vector database if requested
            if options.get('store_in_vector_db', True):
                # Prepare metadata for vector storage
                # Ensure options metadata is also a dictionary
                options_metadata = options.get('metadata') or {}

                vector_metadata = {
                    "source_type": content_type or "unknown",
                    "source_path": file_path,
                    "processing_time": result.processing_time,
                    **result.metadata,
                    **options_metadata
                }

                # Generate document ID if not provided
                document_id = options.get('document_id')
                if not document_id:
                    document_id = generate_document_id(file_path, result.text_content or result.content)

                # Store content in vector database with chunk configuration and document replacement
                point_ids = await store_content_in_vector_db(
                    result.text_content or result.content,
                    vector_metadata,
                    chunk_size=options.get('chunk_size'),
                    chunk_overlap=options.get('chunk_overlap'),
                    document_id=document_id,
                    replace_existing=options.get('replace_existing', False),
                    use_content_checksum=options.get('use_content_checksum', True)
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
                'content': result.text_content or "",
                'metadata': result.metadata,
                'processing_time': result.processing_time,
                'error_message': result.error_message
            }
            
        except Exception as e:
            # Add specific context for file not found errors
            if isinstance(e, FileNotFoundError):
                logger.error("File ingestion task failed - temporary file was cleaned up prematurely",
                           file_path=file_path,
                           error=str(e),
                           error_type="FileNotFoundError",
                           suggestion="This indicates a race condition between file cleanup and task processing")
            else:
                logger.error("File ingestion task failed",
                           file_path=file_path,
                           error=str(e),
                           error_type=e.__class__.__name__)

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
            # Handle special cases for exceptions that require specific parameters
            if hasattr(e, 'service') and hasattr(type(e), '__init__'):
                # For ExternalServiceError and similar exceptions that need service parameter
                try:
                    raise type(e)(str(e).replace(f"{e.service} error: ", ""), e.service)
                except:
                    # Fallback to generic exception if reconstruction fails
                    raise Exception(str(e))
            else:
                # For other exceptions, try to recreate with just the message
                try:
                    raise type(e)(str(e))
                except:
                    # Fallback to generic exception if reconstruction fails
                    raise Exception(str(e))
    
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

            # Ensure metadata is always a dictionary
            if result.metadata is None:
                result.metadata = {}

            self.update_state(state='PROGRESS', meta={'stage': 'storing', 'progress': 0.7})

            # Store in vector database if requested
            if options.get('store_in_vector_db', True):
                # Prepare metadata for vector storage
                # Ensure options metadata is also a dictionary
                options_metadata = options.get('metadata') or {}

                vector_metadata = {
                    "source_type": content_type or "url",
                    "source_url": url,
                    "processing_time": result.processing_time,
                    **result.metadata,
                    **options_metadata
                }

                # Generate document ID if not provided
                document_id = options.get('document_id')
                if not document_id:
                    document_id = generate_document_id(url, result.text_content or result.content)

                # Store content in vector database with chunk configuration and document replacement
                point_ids = await store_content_in_vector_db(
                    result.text_content or result.content,
                    vector_metadata,
                    chunk_size=options.get('chunk_size'),
                    chunk_overlap=options.get('chunk_overlap'),
                    document_id=document_id,
                    replace_existing=options.get('replace_existing', False),
                    use_content_checksum=options.get('use_content_checksum', True)
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
                'content': result.text_content or "",
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
            # Handle special cases for exceptions that require specific parameters
            if hasattr(e, 'service') and hasattr(type(e), '__init__'):
                # For ExternalServiceError and similar exceptions that need service parameter
                try:
                    raise type(e)(str(e).replace(f"{e.service} error: ", ""), e.service)
                except:
                    # Fallback to generic exception if reconstruction fails
                    raise Exception(str(e))
            else:
                # For other exceptions, try to recreate with just the message
                try:
                    raise type(e)(str(e))
                except:
                    # Fallback to generic exception if reconstruction fails
                    raise Exception(str(e))

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

                    # Ensure metadata is always a dictionary
                    if result.metadata is None:
                        result.metadata = {}

                    if result.success and options.get('store_in_vector_db', True):
                        # Store in vector database
                        # Ensure options metadata is also a dictionary
                        options_metadata = options.get('metadata') or {}

                        vector_metadata = {
                            "source_type": detected_source_type or item.get('source_type', 'unknown'),
                            "batch_index": i,
                            "batch_size": len(items),
                            "processing_time": result.processing_time,
                            **result.metadata,
                            **options_metadata
                        }

                        if 'url' in item:
                            vector_metadata['source_url'] = item['url']
                        elif 'file_path' in item:
                            vector_metadata['source_path'] = item['file_path']

                        point_ids = await store_content_in_vector_db(
                            result.text_content or result.content,
                            vector_metadata,
                            chunk_size=options.get('chunk_size'),
                            chunk_overlap=options.get('chunk_overlap')
                        )

                        result.metadata['vector_point_ids'] = point_ids
                        result.metadata['stored_in_vector_db'] = True

                    results.append({
                        'success': result.success,
                        'content': result.text_content or "",
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
            # Handle special cases for exceptions that require specific parameters
            if hasattr(e, 'service') and hasattr(type(e), '__init__'):
                # For ExternalServiceError and similar exceptions that need service parameter
                try:
                    raise type(e)(str(e).replace(f"{e.service} error: ", ""), e.service)
                except:
                    # Fallback to generic exception if reconstruction fails
                    raise Exception(str(e))
            else:
                # For other exceptions, try to recreate with just the message
                try:
                    raise type(e)(str(e))
                except:
                    # Fallback to generic exception if reconstruction fails
                    raise Exception(str(e))

    return asyncio.run(_ingest())
