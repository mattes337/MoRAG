"""Celery tasks for image processing."""

from typing import Dict, Any, List, Optional
import structlog
from pathlib import Path

from morag_services.celery_app import celery_app
from morag_services.tasks import ProcessingTask
from morag_image import image_processor, ImageConfig
from morag_image.services import vision_service
from morag_image.services import ocr_service
from morag_services.embedding import gemini_service
from morag_services.storage import qdrant_service

logger = structlog.get_logger()

@celery_app.task(bind=True, base=ProcessingTask)
async def process_image_file(
    self,
    file_path: str,
    task_id: str,
    config: Optional[Dict[str, Any]] = None,
    store_embeddings: bool = True
) -> Dict[str, Any]:
    """Process image file with captioning, OCR, and metadata extraction."""
    
    logger.info("Starting image processing task",
               task_id=task_id,
               file_path=file_path)
    
    try:
        await self.update_status("PROCESSING", {"stage": "image_analysis"})
        
        # Parse configuration
        image_config = ImageConfig()
        if config:
            for key, value in config.items():
                if hasattr(image_config, key):
                    setattr(image_config, key, value)
        
        # Process image
        image_result = await image_processor.process_image(
            Path(file_path),
            image_config
        )
        
        result = {
            "image_metadata": {
                "width": image_result.metadata.width,
                "height": image_result.metadata.height,
                "format": image_result.metadata.format,
                "mode": image_result.metadata.mode,
                "file_size": image_result.metadata.file_size,
                "has_exif": image_result.metadata.has_exif,
                "creation_time": image_result.metadata.creation_time,
                "camera_make": image_result.metadata.camera_make,
                "camera_model": image_result.metadata.camera_model
            },
            "caption": image_result.caption,
            "extracted_text": image_result.extracted_text,
            "confidence_scores": image_result.confidence_scores,
            "processing_time": image_result.processing_time,
            "embeddings_stored": 0
        }
        
        # Store embeddings if requested
        if store_embeddings:
            await self.update_status("PROCESSING", {"stage": "embedding_generation"})
            
            # Combine caption and extracted text for embedding
            combined_text = ""
            if image_result.caption:
                combined_text += f"Image Caption: {image_result.caption}\n"
            if image_result.extracted_text:
                combined_text += f"Extracted Text: {image_result.extracted_text}\n"
            
            if combined_text.strip():
                # Generate embeddings
                embedding = await gemini_service.generate_embedding(combined_text)
                
                # Store in vector database
                metadata = {
                    "source_type": "image",
                    "file_path": file_path,
                    "image_width": image_result.metadata.width,
                    "image_height": image_result.metadata.height,
                    "image_format": image_result.metadata.format,
                    "has_caption": image_result.caption is not None,
                    "has_text": image_result.extracted_text is not None,
                    "caption_confidence": image_result.confidence_scores.get("caption", 0.0),
                    "ocr_confidence": image_result.confidence_scores.get("ocr", 0.0),
                    "processing_time": image_result.processing_time
                }
                
                await qdrant_service.store_embedding(
                    embedding=embedding,
                    text=combined_text,
                    metadata=metadata,
                    collection_name="images"
                )
                
                result["embeddings_stored"] = 1
        
        await self.update_status("SUCCESS", result)
        
        logger.info("Image processing task completed",
                   task_id=task_id,
                   has_caption=image_result.caption is not None,
                   has_text=image_result.extracted_text is not None,
                   embeddings_stored=result["embeddings_stored"])
        
        # Clean up temporary files
        image_processor.cleanup_temp_files(image_result.temp_files)
        
        return result
        
    except Exception as e:
        error_msg = f"Image processing failed: {str(e)}"
        logger.error("Image processing task failed",
                    task_id=task_id,
                    error=str(e))
        
        await self.update_status("FAILURE", {"error": error_msg})
        raise

@celery_app.task(bind=True, base=ProcessingTask)
async def generate_image_caption(
    self,
    file_path: str,
    task_id: str,
    custom_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """Generate caption for image file."""
    
    logger.info("Starting image caption generation",
               task_id=task_id,
               file_path=file_path)
    
    try:
        await self.update_status("PROCESSING", {"stage": "caption_generation"})
        
        # Generate caption using vision service
        caption = await vision_service.generate_caption(
            Path(file_path),
            custom_prompt=custom_prompt
        )
        
        result = {
            "caption": caption,
            "caption_length": len(caption),
            "custom_prompt_used": custom_prompt is not None
        }
        
        await self.update_status("SUCCESS", result)
        
        logger.info("Image caption generation completed",
                   task_id=task_id,
                   caption_length=len(caption))
        
        return result
        
    except Exception as e:
        error_msg = f"Image caption generation failed: {str(e)}"
        logger.error("Image caption generation task failed",
                    task_id=task_id,
                    error=str(e))
        
        await self.update_status("FAILURE", {"error": error_msg})
        raise

@celery_app.task(bind=True, base=ProcessingTask)
async def extract_image_text(
    self,
    file_path: str,
    task_id: str,
    ocr_engine: str = "auto",
    preprocess: bool = True
) -> Dict[str, Any]:
    """Extract text from image using OCR."""
    
    logger.info("Starting image text extraction",
               task_id=task_id,
               file_path=file_path,
               ocr_engine=ocr_engine)
    
    try:
        await self.update_status("PROCESSING", {"stage": "text_extraction"})
        
        if ocr_engine == "auto":
            extracted_text, engine_used = await ocr_service.extract_text_auto(
                Path(file_path),
                preprocess=preprocess
            )
        elif ocr_engine == "tesseract":
            extracted_text = await ocr_service.extract_text_tesseract(Path(file_path))
            engine_used = "tesseract"
        elif ocr_engine == "easyocr":
            extracted_text = await ocr_service.extract_text_easyocr(Path(file_path))
            engine_used = "easyocr"
        else:
            raise ValueError(f"Unknown OCR engine: {ocr_engine}")
        
        result = {
            "extracted_text": extracted_text,
            "text_length": len(extracted_text),
            "engine_used": engine_used,
            "preprocessing_applied": preprocess
        }
        
        await self.update_status("SUCCESS", result)
        
        logger.info("Image text extraction completed",
                   task_id=task_id,
                   text_length=len(extracted_text),
                   engine_used=engine_used)
        
        return result
        
    except Exception as e:
        error_msg = f"Image text extraction failed: {str(e)}"
        logger.error("Image text extraction task failed",
                    task_id=task_id,
                    error=str(e))
        
        await self.update_status("FAILURE", {"error": error_msg})
        raise

@celery_app.task(bind=True, base=ProcessingTask)
async def analyze_image_content(
    self,
    file_path: str,
    task_id: str
) -> Dict[str, Any]:
    """Analyze image content for detailed information."""
    
    logger.info("Starting image content analysis",
               task_id=task_id,
               file_path=file_path)
    
    try:
        await self.update_status("PROCESSING", {"stage": "content_analysis"})
        
        # Analyze image content
        content_analysis = await vision_service.analyze_image_content(Path(file_path))
        
        # Classify image type
        image_type = await vision_service.classify_image_type(Path(file_path))
        
        result = {
            "content_analysis": content_analysis,
            "image_type": image_type,
            "analysis_categories": list(content_analysis.keys())
        }
        
        await self.update_status("SUCCESS", result)
        
        logger.info("Image content analysis completed",
                   task_id=task_id,
                   image_type=image_type)
        
        return result
        
    except Exception as e:
        error_msg = f"Image content analysis failed: {str(e)}"
        logger.error("Image content analysis task failed",
                    task_id=task_id,
                    error=str(e))
        
        await self.update_status("FAILURE", {"error": error_msg})
        raise

@celery_app.task(bind=True, base=ProcessingTask)
async def detect_text_regions(
    self,
    file_path: str,
    task_id: str,
    engine: str = "tesseract"
) -> Dict[str, Any]:
    """Detect text regions in image with bounding boxes."""
    
    logger.info("Starting text region detection",
               task_id=task_id,
               file_path=file_path,
               engine=engine)
    
    try:
        await self.update_status("PROCESSING", {"stage": "region_detection"})
        
        # Detect text regions
        regions = await ocr_service.detect_text_regions(
            Path(file_path),
            engine=engine
        )
        
        result = {
            "text_regions": regions,
            "regions_count": len(regions),
            "engine_used": engine,
            "total_text_length": sum(len(region["text"]) for region in regions)
        }
        
        await self.update_status("SUCCESS", result)
        
        logger.info("Text region detection completed",
                   task_id=task_id,
                   regions_count=len(regions))
        
        return result
        
    except Exception as e:
        error_msg = f"Text region detection failed: {str(e)}"
        logger.error("Text region detection task failed",
                    task_id=task_id,
                    error=str(e))
        
        await self.update_status("FAILURE", {"error": error_msg})
        raise
