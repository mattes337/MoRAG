"""Image processing with vision models and OCR for text extraction and captioning."""

import os
import tempfile
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
import asyncio
import structlog
import time

from PIL import Image, ExifTags
import google.generativeai as genai

from morag_core.exceptions import ProcessingError, ExternalServiceError
from morag_core.interfaces.processor import BaseProcessor, ProcessingResult as BaseProcessingResult, ProcessingConfig

logger = structlog.get_logger()

@dataclass
class ImageConfig:
    """Configuration for image processing."""
    generate_caption: bool = True
    extract_text: bool = True
    extract_metadata: bool = True
    resize_max_dimension: Optional[int] = 1024
    ocr_engine: str = "tesseract"  # or "easyocr"
    vision_model: str = None  # Will be set from environment variable
    image_quality: int = 85  # JPEG quality for resizing

    def __post_init__(self):
        """Set default vision model from environment if not specified."""
        if self.vision_model is None:
            import os
            self.vision_model = os.environ.get("GEMINI_VISION_MODEL", "gemini-1.5-flash")

@dataclass
class ImageMetadata:
    """Image metadata information."""
    width: int
    height: int
    format: str
    mode: str
    file_size: int
    has_exif: bool
    exif_data: Dict[str, Any]
    creation_time: Optional[str]
    camera_make: Optional[str]
    camera_model: Optional[str]

@dataclass
class ImageProcessingResult:
    """Result of image processing operation."""
    caption: Optional[str]
    extracted_text: Optional[str]
    metadata: ImageMetadata
    processing_time: float
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    temp_files: List[Path] = field(default_factory=list)

class ImageProcessor(BaseProcessor):
    """Image processing service using vision models and OCR."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.temp_dir = Path(tempfile.gettempdir()) / "morag_image"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Configure Gemini for vision tasks
        if api_key:
            genai.configure(api_key=api_key)

    async def process(self, file_path, config: Optional[ProcessingConfig] = None) -> BaseProcessingResult:
        """Process image file according to BaseProcessor interface.

        Args:
            file_path: Path to image file
            config: Processing configuration

        Returns:
            BaseProcessingResult with processed content
        """
        try:
            # Convert to Path if string
            if isinstance(file_path, str):
                file_path = Path(file_path)

            # Create ImageConfig from ProcessingConfig
            image_config = ImageConfig()
            if config:
                if hasattr(config, 'extract_metadata'):
                    image_config.extract_metadata = config.extract_metadata

            # Process the image
            result = await self.process_image(file_path, image_config)

            # Convert to BaseProcessingResult
            return BaseProcessingResult(
                success=True,
                processing_time=result.processing_time,
                metadata={
                    'caption': result.caption,
                    'extracted_text': result.extracted_text,
                    'image_metadata': result.metadata.__dict__,
                    'confidence_scores': result.confidence_scores
                }
            )
        except Exception as e:
            return BaseProcessingResult(
                success=False,
                processing_time=0.0,
                error_message=str(e)
            )

    def supports_format(self, format_type: str) -> bool:
        """Check if processor supports the given format.

        Args:
            format_type: Format type to check (e.g., 'jpg', 'png', 'image/jpeg')

        Returns:
            True if format is supported
        """
        supported_formats = {
            'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp',
            'image/jpeg', 'image/png', 'image/gif', 'image/bmp',
            'image/tiff', 'image/webp'
        }
        return format_type.lower() in supported_formats
        
    async def process_image(
        self,
        file_path: Path,
        config: ImageConfig
    ) -> ImageProcessingResult:
        """Process image file with captioning, OCR, and metadata extraction."""
        start_time = time.time()
        
        try:
            logger.info("Starting image processing", file_path=str(file_path))
            
            # Validate input file
            if not file_path.exists():
                raise ProcessingError(f"Image file not found: {file_path}")
            
            # Extract metadata
            metadata = await self._extract_metadata(file_path)
            
            # Initialize result
            result = ImageProcessingResult(
                caption=None,
                extracted_text=None,
                metadata=metadata,
                processing_time=0.0,
                confidence_scores={},
                temp_files=[]
            )
            
            # Preprocess image if needed
            processed_image_path = await self._preprocess_image(file_path, config)
            if processed_image_path != file_path:
                result.temp_files.append(processed_image_path)
            
            # Generate caption if requested
            if config.generate_caption:
                caption, confidence = await self._generate_caption(processed_image_path, config)
                result.caption = caption
                result.confidence_scores["caption"] = confidence
            
            # Extract text if requested
            if config.extract_text:
                text, confidence = await self._extract_text(processed_image_path, config)
                result.extracted_text = text
                result.confidence_scores["ocr"] = confidence
            
            # Calculate processing time
            result.processing_time = time.time() - start_time
            
            logger.info("Image processing completed", 
                       file_path=str(file_path),
                       processing_time=result.processing_time,
                       has_caption=result.caption is not None,
                       has_text=result.extracted_text is not None)
            
            return result
            
        except Exception as e:
            logger.error("Image processing failed", 
                        file_path=str(file_path), 
                        error=str(e),
                        exc_info=True)
            raise ProcessingError(f"Image processing failed: {str(e)}") from e
        
    async def process_images(
        self,
        file_paths: List[Path],
        config: ImageConfig,
        max_concurrency: int = 3
    ) -> List[ImageProcessingResult]:
        """Process multiple images with concurrency control."""
        logger.info("Starting batch image processing", 
                   image_count=len(file_paths),
                   max_concurrency=max_concurrency)
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def process_with_semaphore(file_path: Path) -> Tuple[Path, Optional[ImageProcessingResult], Optional[Exception]]:
            async with semaphore:
                try:
                    result = await self.process_image(file_path, config)
                    return file_path, result, None
                except Exception as e:
                    logger.error("Failed to process image", 
                                file_path=str(file_path), 
                                error=str(e))
                    return file_path, None, e
        
        # Process all images concurrently with semaphore control
        tasks = [process_with_semaphore(file_path) for file_path in file_paths]
        results = await asyncio.gather(*tasks)
        
        # Collect successful results
        successful_results = []
        failed_count = 0
        
        for file_path, result, error in results:
            if result:
                successful_results.append(result)
            else:
                failed_count += 1
        
        logger.info("Batch image processing completed",
                   total=len(file_paths),
                   successful=len(successful_results),
                   failed=failed_count)
        
        return successful_results
    
    async def _extract_metadata(self, image_path: Path) -> ImageMetadata:
        """Extract metadata from image file."""
        try:
            with Image.open(image_path) as img:
                # Get basic metadata
                width, height = img.size
                img_format = img.format or "Unknown"
                img_mode = img.mode
                file_size = image_path.stat().st_size
                
                # Extract EXIF data if available
                exif_data = {}
                creation_time = None
                camera_make = None
                camera_model = None
                
                if hasattr(img, '_getexif') and img._getexif():
                    exif = img._getexif()
                    if exif:
                        # Convert EXIF tags to readable format
                        for tag_id, value in exif.items():
                            tag_name = ExifTags.TAGS.get(tag_id, tag_id)
                            exif_data[tag_name] = str(value)
                        
                        # Extract common EXIF fields
                        creation_time = exif_data.get('DateTime')
                        camera_make = exif_data.get('Make')
                        camera_model = exif_data.get('Model')
                
                return ImageMetadata(
                    width=width,
                    height=height,
                    format=img_format,
                    mode=img_mode,
                    file_size=file_size,
                    has_exif=bool(exif_data),
                    exif_data=exif_data,
                    creation_time=creation_time,
                    camera_make=camera_make,
                    camera_model=camera_model
                )
        except Exception as e:
            logger.error("Failed to extract image metadata", 
                        image_path=str(image_path), 
                        error=str(e))
            raise ProcessingError(f"Metadata extraction failed: {str(e)}") from e
    
    async def _preprocess_image(self, image_path: Path, config: ImageConfig) -> Path:
        """Preprocess image for further processing."""
        try:
            with Image.open(image_path) as img:
                # Check if resizing is needed
                if config.resize_max_dimension and (
                    img.width > config.resize_max_dimension or 
                    img.height > config.resize_max_dimension
                ):
                    # Calculate new dimensions while maintaining aspect ratio
                    if img.width > img.height:
                        new_width = config.resize_max_dimension
                        new_height = int(img.height * (config.resize_max_dimension / img.width))
                    else:
                        new_height = config.resize_max_dimension
                        new_width = int(img.width * (config.resize_max_dimension / img.height))
                    
                    # Resize image
                    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
                    
                    # Save to temporary file
                    output_path = self.temp_dir / f"resized_{image_path.name}"
                    resized_img.save(output_path, quality=config.image_quality)
                    
                    logger.debug("Image resized", 
                               original_size=f"{img.width}x{img.height}",
                               new_size=f"{new_width}x{new_height}",
                               output_path=str(output_path))
                    
                    return output_path
            
            # No preprocessing needed
            return image_path
            
        except Exception as e:
            logger.error("Image preprocessing failed", 
                        image_path=str(image_path), 
                        error=str(e))
            raise ProcessingError(f"Image preprocessing failed: {str(e)}") from e
    
    async def _generate_caption(self, image_path: Path, config: ImageConfig) -> Tuple[str, float]:
        """Generate caption for image using vision model."""
        try:
            # Check if Gemini API is configured
            import os
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ExternalServiceError("Gemini API key not configured. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.", "gemini")
            
            # Load image for Gemini
            with open(image_path, "rb") as f:
                image_data = f.read()
            
            # Create Gemini model
            model = genai.GenerativeModel(config.vision_model)
            
            # Generate caption
            prompt = "Describe this image in detail, including what you see and any relevant context."
            
            response = await asyncio.to_thread(
                model.generate_content,
                [prompt, {"mime_type": "image/jpeg", "data": image_data}]
            )
            
            # Extract caption from response
            caption = response.text.strip()
            
            # Use a default confidence score since Gemini doesn't provide one
            confidence = 0.9
            
            logger.debug("Caption generated", 
                       image_path=str(image_path),
                       caption_length=len(caption))
            
            return caption, confidence
            
        except Exception as e:
            logger.error("Caption generation failed", 
                        image_path=str(image_path), 
                        error=str(e))
            raise ExternalServiceError(f"Caption generation failed: {str(e)}", "gemini") from e
    
    async def _extract_text(self, image_path: Path, config: ImageConfig) -> Tuple[str, float]:
        """Extract text from image using OCR."""
        if config.ocr_engine == "tesseract":
            return await self._extract_text_tesseract(image_path)
        elif config.ocr_engine == "easyocr":
            return await self._extract_text_easyocr(image_path)
        else:
            raise ProcessingError(f"Unknown OCR engine: {config.ocr_engine}")
    
    async def _extract_text_tesseract(self, image_path: Path) -> Tuple[str, float]:
        """Extract text using Tesseract OCR."""
        try:
            import pytesseract
            
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != "RGB":
                    img = img.convert("RGB")
                
                # Extract text with confidence
                data = await asyncio.to_thread(
                    pytesseract.image_to_data,
                    img,
                    output_type=pytesseract.Output.DICT
                )
                
                # Filter out low-confidence text
                texts = []
                confidences = []
                
                for i, conf in enumerate(data['conf']):
                    if int(conf) > 30:  # Minimum confidence threshold
                        text = data['text'][i].strip()
                        if text:  # Only include non-empty text
                            texts.append(text)
                            confidences.append(float(conf) / 100.0)
                
                # Combine text and calculate average confidence
                full_text = " ".join(texts)
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                
                logger.debug("Text extracted with Tesseract", 
                           image_path=str(image_path),
                           text_length=len(full_text),
                           confidence=avg_confidence)
                
                return full_text, avg_confidence
                
        except ImportError:
            logger.error("Tesseract OCR not available")
            raise ExternalServiceError("Tesseract OCR not available", "tesseract")
        except Exception as e:
            logger.error("Tesseract OCR failed", 
                        image_path=str(image_path), 
                        error=str(e))
            raise ExternalServiceError(f"Tesseract OCR failed: {str(e)}", "tesseract") from e
    
    async def _extract_text_easyocr(self, image_path: Path) -> Tuple[str, float]:
        """Extract text using EasyOCR."""
        try:
            import easyocr
            import numpy as np
            import cv2
            
            # Initialize EasyOCR reader (lazy loading)
            reader = easyocr.Reader(['en'])
            
            # Read image with OpenCV
            image = cv2.imread(str(image_path))
            
            # Extract text
            results = await asyncio.to_thread(reader.readtext, image)
            
            # Process results
            texts = []
            confidences = []
            
            for bbox, text, conf in results:
                if text.strip() and conf > 0.3:  # Minimum confidence threshold
                    texts.append(text.strip())
                    confidences.append(conf)
            
            # Combine text and calculate average confidence
            full_text = " ".join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            logger.debug("Text extracted with EasyOCR", 
                       image_path=str(image_path),
                       text_length=len(full_text),
                       confidence=avg_confidence)
            
            return full_text, avg_confidence
            
        except ImportError:
            logger.error("EasyOCR not available")
            raise ExternalServiceError("EasyOCR not available", "easyocr")
        except Exception as e:
            logger.error("EasyOCR failed", 
                        image_path=str(image_path), 
                        error=str(e))
            raise ExternalServiceError(f"EasyOCR failed: {str(e)}", "easyocr") from e
    
    def cleanup_temp_files(self, result: ImageProcessingResult) -> None:
        """Clean up temporary files created during processing."""
        for temp_file in result.temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
                    logger.debug("Temporary file removed", file_path=str(temp_file))
            except Exception as e:
                logger.warning("Failed to remove temporary file", 
                             file_path=str(temp_file),
                             error=str(e))