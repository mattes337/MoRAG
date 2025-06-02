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

from morag.core.config import settings
from morag.core.exceptions import ProcessingError, ExternalServiceError

logger = structlog.get_logger()

@dataclass
class ImageConfig:
    """Configuration for image processing."""
    generate_caption: bool = True
    extract_text: bool = True
    extract_metadata: bool = True
    resize_max_dimension: Optional[int] = 1024
    ocr_engine: str = "tesseract"  # or "easyocr"
    vision_model: str = "gemini-pro-vision"
    image_quality: int = 85  # JPEG quality for resizing

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

class ImageProcessor:
    """Image processing service using vision models and OCR."""
    
    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "morag_image"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Configure Gemini for vision tasks
        if settings.gemini_api_key:
            genai.configure(api_key=settings.gemini_api_key)
        
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
                caption, caption_confidence = await self._generate_caption(
                    processed_image_path, config.vision_model
                )
                result.caption = caption
                result.confidence_scores["caption"] = caption_confidence
            
            # Extract text if requested
            if config.extract_text:
                extracted_text, ocr_confidence = await self._extract_text(
                    processed_image_path, config.ocr_engine
                )
                result.extracted_text = extracted_text
                result.confidence_scores["ocr"] = ocr_confidence
            
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            
            logger.info("Image processing completed",
                       file_path=str(file_path),
                       processing_time=processing_time,
                       has_caption=result.caption is not None,
                       has_text=result.extracted_text is not None,
                       text_length=len(result.extracted_text) if result.extracted_text else 0)
            
            return result
            
        except Exception as e:
            logger.error("Image processing failed", 
                        file_path=str(file_path),
                        error=str(e))
            raise ProcessingError(f"Image processing failed: {str(e)}")
    
    async def _extract_metadata(self, file_path: Path) -> ImageMetadata:
        """Extract comprehensive image metadata."""
        try:
            with Image.open(file_path) as img:
                # Basic image info
                width, height = img.size
                format_name = img.format or "unknown"
                mode = img.mode
                file_size = file_path.stat().st_size
                
                # EXIF data extraction
                exif_data = {}
                has_exif = False
                creation_time = None
                camera_make = None
                camera_model = None
                
                if hasattr(img, '_getexif') and img._getexif() is not None:
                    has_exif = True
                    exif = img._getexif()
                    
                    for tag_id, value in exif.items():
                        tag = ExifTags.TAGS.get(tag_id, tag_id)
                        exif_data[tag] = value
                        
                        # Extract specific metadata
                        if tag == "DateTime":
                            creation_time = str(value)
                        elif tag == "Make":
                            camera_make = str(value)
                        elif tag == "Model":
                            camera_model = str(value)
                
                metadata = ImageMetadata(
                    width=width,
                    height=height,
                    format=format_name,
                    mode=mode,
                    file_size=file_size,
                    has_exif=has_exif,
                    exif_data=exif_data,
                    creation_time=creation_time,
                    camera_make=camera_make,
                    camera_model=camera_model
                )
                
                logger.debug("Image metadata extracted",
                            width=width,
                            height=height,
                            format=format_name,
                            file_size=file_size,
                            has_exif=has_exif)
                
                return metadata
                
        except Exception as e:
            logger.error("Metadata extraction failed", 
                        file_path=str(file_path),
                        error=str(e))
            raise ProcessingError(f"Metadata extraction failed: {str(e)}")
    
    async def _preprocess_image(self, file_path: Path, config: ImageConfig) -> Path:
        """Preprocess image for optimal processing."""
        try:
            with Image.open(file_path) as img:
                # Check if resizing is needed
                max_dim = config.resize_max_dimension
                if max_dim and (img.width > max_dim or img.height > max_dim):
                    # Calculate new dimensions maintaining aspect ratio
                    ratio = min(max_dim / img.width, max_dim / img.height)
                    new_width = int(img.width * ratio)
                    new_height = int(img.height * ratio)
                    
                    # Resize image
                    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    
                    # Save resized image
                    output_path = self.temp_dir / f"resized_{int(time.time())}_{file_path.name}"
                    
                    # Convert to RGB if necessary for JPEG
                    if resized_img.mode in ("RGBA", "P"):
                        resized_img = resized_img.convert("RGB")
                    
                    resized_img.save(output_path, "JPEG", quality=config.image_quality, optimize=True)
                    
                    logger.debug("Image resized",
                                original_size=(img.width, img.height),
                                new_size=(new_width, new_height),
                                output_path=str(output_path))
                    
                    return output_path
                else:
                    # No resizing needed
                    return file_path
                    
        except Exception as e:
            logger.error("Image preprocessing failed",
                        file_path=str(file_path),
                        error=str(e))
            # Return original file if preprocessing fails
            return file_path
    
    async def _generate_caption(self, image_path: Path, model_name: str) -> Tuple[str, float]:
        """Generate descriptive caption using vision model."""
        try:
            logger.debug("Generating image caption",
                        image_path=str(image_path),
                        model=model_name)
            
            # Load image for Gemini
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != "RGB":
                    img = img.convert("RGB")
                
                # Use Gemini Vision model
                model = genai.GenerativeModel('gemini-pro-vision')
                
                prompt = """Analyze this image and provide a detailed, descriptive caption. 
                Focus on:
                - Main subjects and objects
                - Setting and environment
                - Actions or activities
                - Visual style and composition
                - Any text visible in the image
                
                Provide a comprehensive but concise description."""
                
                response = await asyncio.to_thread(
                    model.generate_content,
                    [prompt, img]
                )
                
                caption = response.text.strip() if response.text else ""
                
                # Estimate confidence based on response length and content
                confidence = min(0.95, len(caption) / 200.0) if caption else 0.0
                
                logger.debug("Caption generated",
                            caption_length=len(caption),
                            confidence=confidence)
                
                return caption, confidence
                
        except Exception as e:
            logger.error("Caption generation failed",
                        image_path=str(image_path),
                        error=str(e))
            return "", 0.0
    
    async def _extract_text(self, image_path: Path, ocr_engine: str) -> Tuple[str, float]:
        """Extract text from image using OCR."""
        try:
            logger.debug("Extracting text from image",
                        image_path=str(image_path),
                        ocr_engine=ocr_engine)
            
            if ocr_engine == "tesseract":
                return await self._extract_text_tesseract(image_path)
            elif ocr_engine == "easyocr":
                return await self._extract_text_easyocr(image_path)
            else:
                logger.warning("Unknown OCR engine, falling back to tesseract",
                             ocr_engine=ocr_engine)
                return await self._extract_text_tesseract(image_path)
                
        except Exception as e:
            logger.error("Text extraction failed",
                        image_path=str(image_path),
                        error=str(e))
            return "", 0.0
    
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
                        if text:
                            texts.append(text)
                            confidences.append(int(conf))
                
                extracted_text = " ".join(texts)
                avg_confidence = sum(confidences) / len(confidences) / 100.0 if confidences else 0.0
                
                logger.debug("Tesseract OCR completed",
                            text_length=len(extracted_text),
                            avg_confidence=avg_confidence)
                
                return extracted_text, avg_confidence
                
        except ImportError:
            logger.error("pytesseract not available")
            return "", 0.0
        except Exception as e:
            logger.error("Tesseract OCR failed", error=str(e))
            return "", 0.0
    
    async def _extract_text_easyocr(self, image_path: Path) -> Tuple[str, float]:
        """Extract text using EasyOCR."""
        try:
            import easyocr
            
            # Initialize EasyOCR reader (cached)
            if not hasattr(self, '_easyocr_reader'):
                self._easyocr_reader = easyocr.Reader(['en'])
            
            # Extract text
            results = await asyncio.to_thread(
                self._easyocr_reader.readtext,
                str(image_path)
            )
            
            # Process results
            texts = []
            confidences = []
            
            for (bbox, text, confidence) in results:
                if confidence > 0.3:  # Minimum confidence threshold
                    texts.append(text.strip())
                    confidences.append(confidence)
            
            extracted_text = " ".join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            logger.debug("EasyOCR completed",
                        text_length=len(extracted_text),
                        avg_confidence=avg_confidence)
            
            return extracted_text, avg_confidence
            
        except ImportError:
            logger.error("easyocr not available")
            return "", 0.0
        except Exception as e:
            logger.error("EasyOCR failed", error=str(e))
            return "", 0.0
    
    def cleanup_temp_files(self, temp_files: List[Path]):
        """Clean up temporary files."""
        for file_path in temp_files:
            try:
                if file_path.exists():
                    file_path.unlink()
                    logger.debug("Temporary file cleaned up", file_path=str(file_path))
            except Exception as e:
                logger.warning("Failed to clean up temporary file",
                             file_path=str(file_path),
                             error=str(e))

# Global instance
image_processor = ImageProcessor()
