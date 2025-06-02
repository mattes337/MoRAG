"""OCR service for text extraction from images using Tesseract and EasyOCR."""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import structlog

from PIL import Image
import cv2
import numpy as np

from morag.core.exceptions import ExternalServiceError, ProcessingError

logger = structlog.get_logger()

class OCRService:
    """Service for optical character recognition using multiple OCR engines."""
    
    def __init__(self):
        self._tesseract_available = self._check_tesseract()
        self._easyocr_available = self._check_easyocr()
        self._easyocr_reader = None
        
        logger.info("OCR service initialized",
                   tesseract_available=self._tesseract_available,
                   easyocr_available=self._easyocr_available)
    
    def _check_tesseract(self) -> bool:
        """Check if Tesseract is available."""
        try:
            import pytesseract
            # Try to get version to verify installation
            pytesseract.get_tesseract_version()
            return True
        except (ImportError, Exception):
            return False
    
    def _check_easyocr(self) -> bool:
        """Check if EasyOCR is available."""
        try:
            import easyocr
            return True
        except ImportError:
            return False
    
    async def extract_text_tesseract(
        self, 
        image_path: Path,
        language: str = "eng",
        psm: int = 3,
        confidence_threshold: int = 30
    ) -> str:
        """Extract text using Tesseract OCR."""
        if not self._tesseract_available:
            raise ExternalServiceError("Tesseract OCR not available", "tesseract")
        
        try:
            import pytesseract
            
            logger.debug("Extracting text with Tesseract",
                        image_path=str(image_path),
                        language=language,
                        psm=psm)
            
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != "RGB":
                    img = img.convert("RGB")
                
                # Configure Tesseract
                config = f'--psm {psm} -l {language}'
                
                # Extract text with confidence data
                data = await asyncio.to_thread(
                    pytesseract.image_to_data,
                    img,
                    config=config,
                    output_type=pytesseract.Output.DICT
                )
                
                # Filter text by confidence
                texts = []
                for i, conf in enumerate(data['conf']):
                    if int(conf) > confidence_threshold:
                        text = data['text'][i].strip()
                        if text:
                            texts.append(text)
                
                extracted_text = " ".join(texts)
                
                logger.debug("Tesseract extraction completed",
                           text_length=len(extracted_text),
                           words_found=len(texts))
                
                return extracted_text
                
        except Exception as e:
            logger.error("Tesseract OCR failed",
                        image_path=str(image_path),
                        error=str(e))
            raise ExternalServiceError(f"Tesseract OCR failed: {str(e)}", "tesseract")
    
    async def extract_text_easyocr(
        self,
        image_path: Path,
        languages: List[str] = None,
        confidence_threshold: float = 0.3
    ) -> str:
        """Extract text using EasyOCR."""
        if not self._easyocr_available:
            raise ExternalServiceError("EasyOCR not available", "easyocr")
        
        try:
            import easyocr
            
            if languages is None:
                languages = ['en']
            
            logger.debug("Extracting text with EasyOCR",
                        image_path=str(image_path),
                        languages=languages)
            
            # Initialize reader if not already done
            if self._easyocr_reader is None:
                self._easyocr_reader = easyocr.Reader(languages)
            
            # Extract text
            results = await asyncio.to_thread(
                self._easyocr_reader.readtext,
                str(image_path)
            )
            
            # Filter and collect text
            texts = []
            for (bbox, text, confidence) in results:
                if confidence > confidence_threshold:
                    texts.append(text.strip())
            
            extracted_text = " ".join(texts)
            
            logger.debug("EasyOCR extraction completed",
                        text_length=len(extracted_text),
                        detections=len(results),
                        filtered_detections=len(texts))
            
            return extracted_text
            
        except Exception as e:
            logger.error("EasyOCR failed",
                        image_path=str(image_path),
                        error=str(e))
            raise ExternalServiceError(f"EasyOCR failed: {str(e)}", "easyocr")
    
    async def detect_text_regions(
        self,
        image_path: Path,
        engine: str = "tesseract"
    ) -> List[Dict[str, Any]]:
        """Detect text regions with bounding boxes and confidence scores."""
        try:
            logger.debug("Detecting text regions",
                        image_path=str(image_path),
                        engine=engine)
            
            if engine == "tesseract" and self._tesseract_available:
                return await self._detect_regions_tesseract(image_path)
            elif engine == "easyocr" and self._easyocr_available:
                return await self._detect_regions_easyocr(image_path)
            else:
                raise ExternalServiceError(f"OCR engine '{engine}' not available", engine)
                
        except Exception as e:
            logger.error("Text region detection failed",
                        image_path=str(image_path),
                        error=str(e))
            return []
    
    async def _detect_regions_tesseract(self, image_path: Path) -> List[Dict[str, Any]]:
        """Detect text regions using Tesseract."""
        import pytesseract
        
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            data = await asyncio.to_thread(
                pytesseract.image_to_data,
                img,
                output_type=pytesseract.Output.DICT
            )
            
            regions = []
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                conf = int(data['conf'][i])
                
                if text and conf > 30:
                    region = {
                        'text': text,
                        'confidence': conf / 100.0,
                        'bbox': {
                            'x': int(data['left'][i]),
                            'y': int(data['top'][i]),
                            'width': int(data['width'][i]),
                            'height': int(data['height'][i])
                        },
                        'engine': 'tesseract'
                    }
                    regions.append(region)
            
            return regions
    
    async def _detect_regions_easyocr(self, image_path: Path) -> List[Dict[str, Any]]:
        """Detect text regions using EasyOCR."""
        import easyocr
        
        if self._easyocr_reader is None:
            self._easyocr_reader = easyocr.Reader(['en'])
        
        results = await asyncio.to_thread(
            self._easyocr_reader.readtext,
            str(image_path)
        )
        
        regions = []
        for (bbox, text, confidence) in results:
            if confidence > 0.3:
                # Convert bbox format
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                
                region = {
                    'text': text.strip(),
                    'confidence': confidence,
                    'bbox': {
                        'x': int(min(x_coords)),
                        'y': int(min(y_coords)),
                        'width': int(max(x_coords) - min(x_coords)),
                        'height': int(max(y_coords) - min(y_coords))
                    },
                    'engine': 'easyocr'
                }
                regions.append(region)
        
        return regions
    
    async def preprocess_image_for_ocr(self, image_path: Path) -> Path:
        """Preprocess image to improve OCR accuracy."""
        try:
            logger.debug("Preprocessing image for OCR", image_path=str(image_path))
            
            # Read image with OpenCV
            img = cv2.imread(str(image_path))
            if img is None:
                raise ProcessingError("Could not read image file")
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply denoising
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Morphological operations to clean up
            kernel = np.ones((1, 1), np.uint8)
            processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Save processed image
            output_path = image_path.parent / f"ocr_preprocessed_{image_path.name}"
            cv2.imwrite(str(output_path), processed)
            
            logger.debug("Image preprocessing completed", output_path=str(output_path))
            
            return output_path
            
        except Exception as e:
            logger.error("Image preprocessing failed",
                        image_path=str(image_path),
                        error=str(e))
            # Return original image if preprocessing fails
            return image_path
    
    async def extract_text_auto(
        self,
        image_path: Path,
        preprocess: bool = True
    ) -> Tuple[str, str]:
        """Extract text using the best available OCR engine."""
        try:
            # Preprocess image if requested
            if preprocess:
                processed_path = await self.preprocess_image_for_ocr(image_path)
            else:
                processed_path = image_path
            
            # Try EasyOCR first (generally more accurate)
            if self._easyocr_available:
                try:
                    text = await self.extract_text_easyocr(processed_path)
                    if text.strip():
                        return text, "easyocr"
                except Exception as e:
                    logger.warning("EasyOCR failed, trying Tesseract", error=str(e))
            
            # Fall back to Tesseract
            if self._tesseract_available:
                text = await self.extract_text_tesseract(processed_path)
                return text, "tesseract"
            
            # No OCR engines available
            raise ExternalServiceError("No OCR engines available", "ocr")
            
        except Exception as e:
            logger.error("Auto OCR extraction failed",
                        image_path=str(image_path),
                        error=str(e))
            return "", "none"
        finally:
            # Clean up preprocessed file if it was created
            if preprocess and processed_path != image_path and processed_path.exists():
                try:
                    processed_path.unlink()
                except Exception:
                    pass

# Global instance
ocr_service = OCRService()
