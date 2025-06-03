"""API routes for universal document conversion."""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any, List
from pathlib import Path
import tempfile
import structlog

from ...services.universal_converter import universal_converter_service
from ...converters import ConversionOptions, ChunkingStrategy
from ...core.exceptions import ValidationError, ProcessingError
from ..dependencies import get_current_user
from ..models.conversion import (
    ConversionRequest,
    ConversionResponse,
    ConversionStatus,
    SupportedFormatsResponse,
    ConverterInfoResponse
)

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/v1/conversion", tags=["conversion"])


@router.get("/formats", response_model=SupportedFormatsResponse)
async def get_supported_formats():
    """Get list of supported document formats."""
    try:
        formats = universal_converter_service.get_supported_formats()
        converter_info = universal_converter_service.get_converter_info()
        
        return SupportedFormatsResponse(
            supported_formats=formats,
            converter_info=converter_info
        )
    except Exception as e:
        logger.error("Failed to get supported formats", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve supported formats")


@router.get("/statistics")
async def get_conversion_statistics(current_user=Depends(get_current_user)):
    """Get conversion statistics."""
    try:
        stats = universal_converter_service.get_statistics()
        return JSONResponse(content=stats)
    except Exception as e:
        logger.error("Failed to get conversion statistics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")


@router.post("/convert", response_model=ConversionResponse)
async def convert_document(
    file: UploadFile = File(...),
    chunking_strategy: Optional[str] = "page",
    preserve_formatting: Optional[bool] = True,
    extract_images: Optional[bool] = True,
    include_metadata: Optional[bool] = True,
    min_quality_threshold: Optional[float] = 0.7,
    enable_fallback: Optional[bool] = True,
    generate_embeddings: Optional[bool] = True,
    format_options: Optional[str] = None,
    current_user=Depends(get_current_user)
):
    """Convert uploaded document to markdown."""
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = Path(tmp_file.name)
    
    try:
        # Parse format options
        parsed_format_options = {}
        if format_options:
            import json
            try:
                parsed_format_options = json.loads(format_options)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid format_options JSON")
        
        # Create conversion options
        try:
            chunking_enum = ChunkingStrategy(chunking_strategy)
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid chunking strategy. Must be one of: {[s.value for s in ChunkingStrategy]}"
            )
        
        options = ConversionOptions(
            preserve_formatting=preserve_formatting,
            extract_images=extract_images,
            include_metadata=include_metadata,
            chunking_strategy=chunking_enum,
            min_quality_threshold=min_quality_threshold,
            enable_fallback=enable_fallback,
            format_options=parsed_format_options
        )
        
        logger.info(
            "Starting document conversion via API",
            filename=file.filename,
            file_size=len(content),
            chunking_strategy=chunking_strategy,
            user_id=current_user.get('id') if current_user else None
        )
        
        # Convert document
        result = await universal_converter_service.convert_document(
            tmp_path,
            options=options,
            generate_embeddings=generate_embeddings
        )
        
        # Create response
        response = ConversionResponse(
            success=result['success'],
            content=result['content'],
            metadata=result['metadata'],
            quality_score=result['quality_score'],
            processing_time=result['total_processing_time'],
            chunks_count=len(result['chunks']),
            embeddings_count=len(result['embeddings']),
            warnings=result['warnings'],
            error_message=result['error_message'],
            converter_used=result['converter_used'],
            fallback_used=result['fallback_used'],
            original_format=result['original_format'],
            word_count=result['word_count']
        )
        
        logger.info(
            "Document conversion completed via API",
            success=result['success'],
            processing_time=result['total_processing_time'],
            quality_score=result['quality_score']['overall_score'] if result['quality_score'] else None,
            chunks_count=len(result['chunks'])
        )
        
        return response
        
    except ValidationError as e:
        logger.warning("Validation error in document conversion", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except ProcessingError as e:
        logger.error("Processing error in document conversion", error=str(e))
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("Unexpected error in document conversion", error=str(e), error_type=type(e).__name__)
        raise HTTPException(status_code=500, detail="Internal server error during conversion")
    finally:
        # Clean up temporary file
        if tmp_path.exists():
            tmp_path.unlink()


@router.post("/convert-batch")
async def convert_documents_batch(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    chunking_strategy: Optional[str] = "page",
    preserve_formatting: Optional[bool] = True,
    extract_images: Optional[bool] = True,
    include_metadata: Optional[bool] = True,
    min_quality_threshold: Optional[float] = 0.7,
    enable_fallback: Optional[bool] = True,
    generate_embeddings: Optional[bool] = True,
    current_user=Depends(get_current_user)
):
    """Convert multiple documents in batch."""
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch")
    
    # Validate all files first
    for file in files:
        if not file.filename:
            raise HTTPException(status_code=400, detail="All files must have filenames")
    
    # Create conversion options
    try:
        chunking_enum = ChunkingStrategy(chunking_strategy)
    except ValueError:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid chunking strategy. Must be one of: {[s.value for s in ChunkingStrategy]}"
        )
    
    options = ConversionOptions(
        preserve_formatting=preserve_formatting,
        extract_images=extract_images,
        include_metadata=include_metadata,
        chunking_strategy=chunking_enum,
        min_quality_threshold=min_quality_threshold,
        enable_fallback=enable_fallback
    )
    
    # Process files
    results = []
    temp_files = []
    
    try:
        # Save all files temporarily
        for file in files:
            content = await file.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
                tmp_file.write(content)
                temp_files.append((Path(tmp_file.name), file.filename))
        
        # Convert all files
        for tmp_path, original_filename in temp_files:
            try:
                result = await universal_converter_service.convert_document(
                    tmp_path,
                    options=options,
                    generate_embeddings=generate_embeddings
                )
                
                results.append({
                    'filename': original_filename,
                    'success': result['success'],
                    'content': result['content'],
                    'metadata': result['metadata'],
                    'quality_score': result['quality_score'],
                    'processing_time': result['total_processing_time'],
                    'chunks_count': len(result['chunks']),
                    'error_message': result['error_message']
                })
                
            except Exception as e:
                logger.error(f"Failed to convert {original_filename}", error=str(e))
                results.append({
                    'filename': original_filename,
                    'success': False,
                    'content': '',
                    'metadata': {},
                    'quality_score': None,
                    'processing_time': 0,
                    'chunks_count': 0,
                    'error_message': str(e)
                })
        
        # Calculate batch statistics
        successful = sum(1 for r in results if r['success'])
        total_time = sum(r['processing_time'] for r in results)
        avg_quality = sum(
            r['quality_score']['overall_score'] for r in results 
            if r['success'] and r['quality_score']
        ) / max(successful, 1)
        
        logger.info(
            "Batch conversion completed",
            total_files=len(files),
            successful=successful,
            failed=len(files) - successful,
            total_time=total_time,
            avg_quality=avg_quality
        )
        
        return {
            'batch_id': f"batch_{len(files)}_{int(total_time)}",
            'total_files': len(files),
            'successful_conversions': successful,
            'failed_conversions': len(files) - successful,
            'total_processing_time': total_time,
            'average_quality_score': avg_quality,
            'results': results
        }
        
    finally:
        # Clean up all temporary files
        for tmp_path, _ in temp_files:
            if tmp_path.exists():
                tmp_path.unlink()


@router.delete("/statistics")
async def reset_conversion_statistics(current_user=Depends(get_current_user)):
    """Reset conversion statistics (admin only)."""
    # Add admin check here if needed
    try:
        universal_converter_service.reset_statistics()
        return {"message": "Statistics reset successfully"}
    except Exception as e:
        logger.error("Failed to reset statistics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to reset statistics")


@router.get("/health")
async def conversion_health_check():
    """Health check for conversion service."""
    try:
        formats = universal_converter_service.get_supported_formats()
        stats = universal_converter_service.get_statistics()
        
        return {
            "status": "healthy",
            "supported_formats_count": len(formats),
            "total_conversions": stats.get('total_conversions', 0),
            "success_rate": (
                stats.get('successful_conversions', 0) / max(stats.get('total_conversions', 1), 1)
            ) * 100
        }
    except Exception as e:
        logger.error("Conversion service health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "error": str(e)
        }
