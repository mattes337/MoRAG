"""Enhanced processing task with detailed webhook notifications."""

import asyncio
import uuid
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import structlog

from morag.api import MoRAGAPI
from morag.services.enhanced_webhook_service import get_webhook_service
from morag.services.temporary_file_service import get_temp_file_service
from morag.worker import celery_app, run_async

logger = structlog.get_logger(__name__)


@celery_app.task(bind=True)
def enhanced_process_ingest_task(
    self,
    file_path: str,
    options: Dict[str, Any]
) -> Dict[str, Any]:
    """Enhanced processing task with detailed webhook notifications.
    
    Args:
        file_path: Path to file to process
        options: Processing options including webhook_url, document_id, etc.
        
    Returns:
        Processing result dictionary
    """
    async def _process():
        webhook_service = get_webhook_service()
        task_id = self.request.id
        
        # Extract options
        webhook_url = options.get('webhook_url')
        webhook_auth_token = options.get('webhook_auth_token')
        document_id = options.get('document_id')
        collection_name = options.get('collection_name')
        language = options.get('language')
        chunking_strategy = options.get('chunking_strategy')
        chunk_size = options.get('chunk_size')
        chunk_overlap = options.get('chunk_overlap')
        metadata = options.get('metadata', {})
        
        # Generate document ID if not provided
        if not document_id:
            document_id = str(uuid.uuid4())
        
        try:
            # Validate webhook URL
            if webhook_url and not webhook_service.validate_webhook_url(webhook_url, allow_localhost=True):
                raise ValueError(f"Invalid webhook URL: {webhook_url}")

            # Initialize services
            api = MoRAGAPI()
            temp_service = get_temp_file_service()
            file_path_obj = Path(file_path)

            if not file_path_obj.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            # Store original file in temp storage
            with open(file_path_obj, 'rb') as f:
                original_content = f.read()

            original_filename = f"original{file_path_obj.suffix}"
            await temp_service.store_file(task_id, original_filename, original_content)

            logger.info("Starting enhanced processing task",
                       task_id=task_id,
                       document_id=document_id,
                       file_path=file_path,
                       webhook_url=webhook_url)
            
            # Step 1: Markdown Conversion (0-20%)
            await webhook_service.send_step_started(
                webhook_url, task_id, document_id, "markdown_conversion", 0.0, webhook_auth_token
            )
            
            # Convert to markdown using the conversion service
            from morag_document.services.markitdown_service import MarkitdownService
            markitdown_service = MarkitdownService()
            markdown_content = await markitdown_service.convert_file(file_path_obj)

            # Store markdown in temp storage
            await temp_service.store_text_file(task_id, "markdown.md", markdown_content)
            
            conversion_metadata = {
                "original_format": file_path_obj.suffix.lower().lstrip('.'),
                "file_size_bytes": file_path_obj.stat().st_size,
                "content_length": len(markdown_content)
            }
            
            await webhook_service.send_step_completed(
                webhook_url, task_id, document_id, "markdown_conversion", 20.0,
                {
                    "markdown_file_url": f"/api/files/temp/{task_id}/markdown.md",
                    "original_file_url": f"/api/files/temp/{task_id}/{original_filename}",
                    "conversion_metadata": conversion_metadata
                },
                webhook_auth_token
            )
            
            # Step 2: Metadata Extraction (20-40%)
            await webhook_service.send_step_started(
                webhook_url, task_id, document_id, "metadata_extraction", 20.0, webhook_auth_token
            )
            
            # Extract metadata using existing services
            file_metadata = {
                "format": file_path_obj.suffix.lower().lstrip('.'),
                "file_size_bytes": file_path_obj.stat().st_size,
                "created_at": file_path_obj.stat().st_ctime,
                "modified_at": file_path_obj.stat().st_mtime
            }
            
            # Store metadata in temp storage
            import json
            metadata_json = json.dumps(file_metadata, indent=2)
            await temp_service.store_text_file(task_id, "metadata.json", metadata_json)
            
            await webhook_service.send_step_completed(
                webhook_url, task_id, document_id, "metadata_extraction", 40.0,
                {
                    "metadata_file_url": f"/api/files/temp/{task_id}/metadata.json",
                    "metadata": file_metadata
                },
                webhook_auth_token
            )
            
            # Step 3: Content Processing (40-70%)
            await webhook_service.send_step_started(
                webhook_url, task_id, document_id, "content_processing", 40.0, webhook_auth_token
            )
            
            # Process the file using the full MoRAG pipeline
            processing_options = {
                "document_id": document_id,
                "language": language,
                "chunking_strategy": chunking_strategy,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "metadata": {**metadata, **file_metadata}
            }
            
            # Determine content type
            content_type = None
            file_ext = file_path_obj.suffix.lower().lstrip('.')
            if file_ext in ['pdf', 'doc', 'docx', 'txt', 'md']:
                content_type = "document"
            elif file_ext in ['mp3', 'wav', 'm4a', 'flac', 'aac', 'ogg']:
                content_type = "audio"
            elif file_ext in ['mp4', 'avi', 'mov', 'mkv', 'webm']:
                content_type = "video"
            
            # Process the content
            result = await api.process_file(file_path, content_type, processing_options)
            
            if not result.success:
                raise Exception(f"Content processing failed: {result.error_message}")
            
            # Analyze content for summary
            summary_data = {
                "summary": result.content[:500] + "..." if len(result.content) > 500 else result.content,
                "content_length": len(result.content),
                "language": language or "auto-detected"
            }
            
            await webhook_service.send_step_completed(
                webhook_url, task_id, document_id, "content_processing", 70.0,
                summary_data,
                webhook_auth_token
            )
            
            # Step 4: Ingestion (70-100%)
            await webhook_service.send_step_started(
                webhook_url, task_id, document_id, "ingestion", 70.0, webhook_auth_token
            )

            # Ingest into vector database using the store_content_in_vector_db function
            from morag.ingest_tasks import store_content_in_vector_db

            ingestion_metadata = {
                **processing_options["metadata"],
                "document_id": document_id,
                "content_type": content_type,
                "processing_time": result.processing_time,
                "created_at": datetime.now(timezone.utc).isoformat()
            }

            point_ids = await store_content_in_vector_db(
                content=result.content,
                metadata=ingestion_metadata,
                collection_name=collection_name or "morag_documents",
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                document_id=document_id,
                replace_existing=True
            )

            # Calculate ingestion statistics
            ingestion_stats = {
                "chunks_processed": len(point_ids),
                "total_text_length": len(result.content),
                "database_collection": collection_name or "morag_documents",
                "processing_time_seconds": result.processing_time or 0.0
            }
            
            await webhook_service.send_step_completed(
                webhook_url, task_id, document_id, "ingestion", 100.0,
                ingestion_stats,
                webhook_auth_token
            )
            
            logger.info("Enhanced processing task completed successfully",
                       task_id=task_id,
                       document_id=document_id,
                       chunks_processed=ingestion_stats["chunks_processed"])
            
            return {
                "success": True,
                "document_id": document_id,
                "task_id": task_id,
                "content": result.content,
                "metadata": {
                    **result.metadata,
                    "vector_point_ids": point_ids,
                    "ingestion_stats": ingestion_stats
                },
                "ingestion_stats": ingestion_stats,
                "processing_time": result.processing_time
            }
            
        except Exception as e:
            logger.error("Enhanced processing task failed",
                        task_id=task_id,
                        document_id=document_id,
                        error=str(e))
            
            # Send failure notification
            if webhook_url:
                await webhook_service.send_step_failed(
                    webhook_url, task_id, document_id, "processing", 0.0,
                    str(e), webhook_auth_token
                )
            
            # Clean up temp files
            try:
                if file_path_obj.exists():
                    file_path_obj.unlink()
            except Exception:
                pass
            
            return {
                "success": False,
                "document_id": document_id,
                "task_id": task_id,
                "error_message": str(e),
                "content": "",
                "metadata": {},
                "processing_time": 0.0
            }
        
        finally:
            # Clean up original file
            try:
                if file_path_obj.exists():
                    file_path_obj.unlink()
                    logger.debug("Cleaned up temporary file", file_path=file_path)
            except Exception as e:
                logger.warning("Failed to clean up temporary file",
                             file_path=file_path, error=str(e))
    
    # Run the async function using shared event loop
    return run_async(_process())
