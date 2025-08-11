"""Conversion endpoints for MoRAG API."""

import time
from typing import Optional, Dict, Any
from pathlib import Path
import structlog

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse

from morag.api_models.models import (
    MarkdownConversionRequest, MarkdownConversionResponse,
    ProcessIngestRequest, ProcessIngestResponse
)
from morag.utils.file_upload import get_upload_handler, FileUploadError
from morag_document.services.markitdown_service import MarkitdownService
from morag.tasks.enhanced_processing_task import enhanced_process_ingest_task
from morag.services.document_deduplication_service import get_deduplication_service

logger = structlog.get_logger(__name__)

conversion_router = APIRouter(prefix="/api/convert", tags=["Conversion"])


def setup_conversion_endpoints():
    """Setup conversion endpoints."""
    
    @conversion_router.post("/markdown", response_model=MarkdownConversionResponse)
    async def convert_to_markdown(
        file: UploadFile = File(...),
        include_metadata: bool = Form(default=True),
        language: Optional[str] = Form(default=None),
        document_id: Optional[str] = Form(default=None)
    ):
        """Convert uploaded file to markdown format for UI preview.
        
        This endpoint provides fast conversion without full processing pipeline.
        Supports PDF, audio, video, text, and other document formats.
        """
        temp_path = None
        start_time = time.time()

        try:
            # Get deduplication service
            dedup_service = await get_deduplication_service()

            # Handle document ID
            if document_id:
                if not dedup_service.validate_document_id(document_id):
                    raise HTTPException(status_code=400, detail=f"Invalid document ID format: {document_id}")

                # Check for existing document
                existing_doc = await dedup_service.check_document_exists(document_id)
                if existing_doc:
                    # Return existing document info instead of processing again
                    return MarkdownConversionResponse(
                        success=True,
                        markdown="# Existing Document\n\nThis document has already been processed.",
                        metadata={
                            "document_id": document_id,
                            "status": "already_exists",
                            "original_format": existing_doc["metadata"].get("content_type", "unknown"),
                            "file_size_bytes": existing_doc["metadata"].get("file_size_bytes", 0),
                            "processing_time_ms": 0.0,
                            "existing_document": existing_doc
                        },
                        processing_time_ms=0.0
                    )
            else:
                # Generate document ID if not provided
                document_id = dedup_service.generate_document_id()

            # Get upload handler
            upload_handler = get_upload_handler()

            # Save uploaded file to temp directory
            temp_path = await upload_handler.save_upload(file)
            
            # Determine file format
            file_extension = temp_path.suffix.lower().lstrip('.')
            file_size = temp_path.stat().st_size
            
            logger.info("Converting file to markdown", 
                       filename=file.filename,
                       format=file_extension,
                       size=file_size)
            
            # Initialize conversion result
            markdown_content = ""
            conversion_metadata = {
                "document_id": document_id,
                "original_format": file_extension,
                "file_size_bytes": file_size,
                "original_filename": file.filename
            }
            
            # Route to appropriate converter based on file type
            if file_extension in ['pdf', 'doc', 'docx', 'txt', 'md', 'html', 'rtf']:
                # Use markitdown service for documents
                markitdown_service = MarkitdownService()
                markdown_content = await markitdown_service.convert_file(temp_path)
                
                # Add document-specific metadata
                if file_extension == 'pdf':
                    # Try to get page count (basic estimation)
                    try:
                        # Simple page count estimation based on file size
                        estimated_pages = max(1, file_size // 50000)  # Rough estimate
                        conversion_metadata["estimated_page_count"] = estimated_pages
                    except Exception:
                        pass
                        
            elif file_extension in ['mp3', 'wav', 'm4a', 'flac', 'aac', 'ogg', 'wma']:
                # Use markitdown service for audio files (faster than full audio processing)
                try:
                    markitdown_service = MarkitdownService()
                    markdown_content = await markitdown_service.convert_file(temp_path)

                    # Add basic audio metadata
                    conversion_metadata["content_type"] = "audio"
                except Exception as e:
                    logger.warning("Markitdown audio conversion failed, trying fallback", error=str(e))
                    # Fallback: create basic markdown structure
                    markdown_content = f"# Audio File: {file.filename}\n\n*Audio transcription not available in preview mode*"

            elif file_extension in ['mp4', 'avi', 'mov', 'mkv', 'webm', 'flv']:
                # Use markitdown service for video files (faster than full video processing)
                try:
                    markitdown_service = MarkitdownService()
                    markdown_content = await markitdown_service.convert_file(temp_path)

                    # Add basic video metadata
                    conversion_metadata["content_type"] = "video"
                except Exception as e:
                    logger.warning("Markitdown video conversion failed, trying fallback", error=str(e))
                    # Fallback: create basic markdown structure
                    markdown_content = f"# Video File: {file.filename}\n\n*Video transcription not available in preview mode*"
                    
            else:
                # Try markitdown as fallback for other formats
                try:
                    markitdown_service = MarkitdownService()
                    markdown_content = await markitdown_service.convert_file(temp_path)
                except Exception as e:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Unsupported file format: {file_extension}. Error: {str(e)}"
                    )
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            conversion_metadata["processing_time_ms"] = processing_time_ms
            
            # Add language if specified
            if language:
                conversion_metadata["language"] = language
            
            # Validate we got content
            if not markdown_content or not markdown_content.strip():
                raise Exception("Conversion produced empty content")
            
            logger.info("File converted successfully", 
                       filename=file.filename,
                       content_length=len(markdown_content),
                       processing_time_ms=processing_time_ms)
            
            return MarkdownConversionResponse(
                success=True,
                markdown=markdown_content,
                metadata=conversion_metadata,
                processing_time_ms=processing_time_ms
            )
            
        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            logger.error("File conversion failed", 
                        filename=file.filename, 
                        error=str(e),
                        processing_time_ms=processing_time_ms)
            
            return MarkdownConversionResponse(
                success=False,
                markdown="",
                metadata={
                    "original_format": temp_path.suffix.lower().lstrip('.') if temp_path else "unknown",
                    "file_size_bytes": temp_path.stat().st_size if temp_path and temp_path.exists() else 0,
                    "original_filename": file.filename,
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms,
                error_message=str(e)
            )
            
        finally:
            # Clean up temporary file
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                    logger.debug("Cleaned up temporary file", temp_path=str(temp_path))
                except Exception as e:
                    logger.warning("Failed to clean up temporary file",
                                 temp_path=str(temp_path),
                                 error=str(e))

    @conversion_router.post("/process-ingest", response_model=ProcessIngestResponse)
    async def process_with_ingestion(
        file: UploadFile = File(...),
        webhook_url: str = Form(...),
        document_id: Optional[str] = Form(default=None),
        webhook_auth_token: Optional[str] = Form(default=None),
        collection_name: Optional[str] = Form(default=None),
        language: Optional[str] = Form(default=None),
        chunking_strategy: Optional[str] = Form(default=None),
        chunk_size: Optional[int] = Form(default=None),
        chunk_overlap: Optional[int] = Form(default=None),
        metadata: Optional[str] = Form(default=None)  # JSON string
    ):
        """Process file with full MoRAG pipeline and webhook notifications.

        This endpoint runs the complete processing pipeline including:
        1. Markdown conversion
        2. Metadata extraction
        3. Content processing
        4. Vector database ingestion

        Progress is reported via webhooks at each step.
        """
        temp_path = None
        try:
            # Get deduplication service
            dedup_service = await get_deduplication_service()

            # Handle document ID
            if document_id:
                if not dedup_service.validate_document_id(document_id):
                    raise HTTPException(status_code=400, detail=f"Invalid document ID format: {document_id}")

                # Check for existing document
                existing_doc = await dedup_service.check_document_exists(document_id)
                if existing_doc:
                    # Return duplicate error response
                    duplicate_response = dedup_service.create_duplicate_error_response(existing_doc)
                    raise HTTPException(status_code=409, detail=duplicate_response)
            else:
                # Generate document ID if not provided
                document_id = dedup_service.generate_document_id()

            # Get upload handler
            upload_handler = get_upload_handler()

            # Save uploaded file to temp directory
            temp_path = await upload_handler.save_upload(file)

            # Parse metadata if provided
            parsed_metadata = {}
            if metadata:
                import json
                try:
                    parsed_metadata = json.loads(metadata)
                except json.JSONDecodeError as e:
                    logger.warning("Invalid JSON in metadata parameter", metadata=metadata, error=str(e))

            # Generate document ID if not provided
            if not document_id:
                import uuid
                document_id = str(uuid.uuid4())

            # Prepare task options
            task_options = {
                "webhook_url": webhook_url,
                "webhook_auth_token": webhook_auth_token,
                "document_id": document_id,
                "collection_name": collection_name,
                "language": language,
                "chunking_strategy": chunking_strategy,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "metadata": {
                    **parsed_metadata,
                    "original_filename": file.filename,
                    "content_type": file.content_type
                }
            }

            logger.info("Starting enhanced processing task",
                       filename=file.filename,
                       document_id=document_id,
                       webhook_url=webhook_url,
                       temp_path=str(temp_path))

            # Submit background task
            task = enhanced_process_ingest_task.delay(str(temp_path), task_options)

            # Estimate processing time based on file size
            file_size_mb = temp_path.stat().st_size / (1024 * 1024)
            estimated_time = max(60, int(file_size_mb * 30))  # 30 seconds per MB, minimum 1 minute

            return ProcessIngestResponse(
                success=True,
                task_id=task.id,
                document_id=document_id,
                estimated_time_seconds=estimated_time,
                status_url=f"/api/v1/status/{task.id}",
                message=f"Processing started for '{file.filename}'. Webhook notifications will be sent to {webhook_url}"
            )

        except Exception as e:
            logger.error("Failed to start enhanced processing task",
                        filename=file.filename,
                        error=str(e))

            # Clean up temp file on error
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception as cleanup_error:
                    logger.warning("Failed to clean up temp file after error",
                                 temp_path=str(temp_path),
                                 cleanup_error=str(cleanup_error))

            raise HTTPException(status_code=500, detail=f"Failed to start processing: {str(e)}")

    return conversion_router
