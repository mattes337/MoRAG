"""Image processing service for MoRAG."""

from pathlib import Path
from typing import List, Optional, Dict, Any
import asyncio
import structlog

from morag_core.interfaces.service import BaseService
from morag_core.exceptions import ProcessingError

from .processor import ImageProcessor, ImageConfig, ImageProcessingResult

logger = structlog.get_logger()

class ImageService(BaseService):
    """Service for processing images with OCR and captioning."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the image service.

        Args:
            api_key: Optional API key for Gemini vision model
        """
        self.processor = ImageProcessor(api_key=api_key)

    async def initialize(self) -> bool:
        """Initialize the service.

        Returns:
            True if initialization was successful
        """
        return True

    async def shutdown(self) -> None:
        """Shutdown the service and release resources."""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check service health.

        Returns:
            Dictionary with health status information
        """
        return {"status": "healthy", "processor": "ready"}
    
    async def process_image(self, 
                          file_path: Path, 
                          config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a single image file.
        
        Args:
            file_path: Path to the image file
            config: Optional configuration parameters
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Convert config dict to ImageConfig if provided
            image_config = ImageConfig()
            if config:
                for key, value in config.items():
                    if hasattr(image_config, key):
                        setattr(image_config, key, value)
            
            # Process the image
            result = await self.processor.process_image(file_path, image_config)

            # Convert result to dictionary with file path information
            return self._result_to_dict(result, file_path)
            
        except Exception as e:
            logger.error("Image processing failed", 
                        file_path=str(file_path),
                        error=str(e))
            raise ProcessingError(f"Image processing failed: {str(e)}") from e
        
    async def process_batch(self, 
                           file_paths: List[Path], 
                           config: Optional[Dict[str, Any]] = None,
                           max_concurrency: int = 3) -> List[Dict[str, Any]]:
        """Process multiple image files concurrently.
        
        Args:
            file_paths: List of paths to image files
            config: Optional configuration parameters
            max_concurrency: Maximum number of concurrent processing tasks
            
        Returns:
            List of dictionaries with processing results
        """
        try:
            # Convert config dict to ImageConfig if provided
            image_config = ImageConfig()
            if config:
                for key, value in config.items():
                    if hasattr(image_config, key):
                        setattr(image_config, key, value)
            
            # Process images in batch
            results = await self.processor.process_images(
                file_paths, 
                image_config,
                max_concurrency=max_concurrency
            )
            
            # Convert results to dictionaries
            return [self._result_to_dict(result) for result in results]
            
        except Exception as e:
            logger.error("Batch image processing failed", 
                        file_count=len(file_paths),
                        error=str(e))
            raise ProcessingError(f"Batch image processing failed: {str(e)}") from e
    
    def _result_to_dict(self, result: ImageProcessingResult, file_path: Optional[Path] = None) -> Dict[str, Any]:
        """Convert ImageProcessingResult to a dictionary."""
        # Convert metadata to dictionary with required document fields
        metadata_dict = {
            # Image-specific metadata
            "width": result.metadata.width,
            "height": result.metadata.height,
            "format": result.metadata.format,
            "mode": result.metadata.mode,
            "file_size": result.metadata.file_size,
            "has_exif": result.metadata.has_exif,
            "exif_data": result.metadata.exif_data,
            "creation_time": result.metadata.creation_time,
            "camera_make": result.metadata.camera_make,
            "camera_model": result.metadata.camera_model
        }

        # Add core document metadata fields if file_path is provided
        if file_path:
            metadata_dict.update({
                "source_path": str(file_path.absolute()),
                "source_name": file_path.name,
                "file_name": file_path.name,
                "mime_type": self._get_image_mime_type(file_path),
                "checksum": self._calculate_file_checksum(file_path)
            })

        # Add legacy filename field for compatibility
        if file_path:
            metadata_dict["filename"] = file_path.name
        
        # Create result dictionary
        return {
            "caption": result.caption,
            "extracted_text": result.extracted_text,
            "metadata": metadata_dict,
            "processing_time": result.processing_time,
            "confidence_scores": result.confidence_scores
        }

    def _get_image_mime_type(self, file_path: Path) -> str:
        """Get MIME type for image file based on extension."""
        ext = file_path.suffix.lower()
        mime_type_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.tiff': 'image/tiff',
            '.tif': 'image/tiff',
            '.webp': 'image/webp',
            '.svg': 'image/svg+xml'
        }
        return mime_type_map.get(ext, 'image/jpeg')

    def _calculate_file_checksum(self, file_path: Path) -> Optional[str]:
        """Calculate SHA256 checksum of file."""
        try:
            import hashlib
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                # Read file in chunks to handle large files efficiently
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.warning(f"Failed to calculate checksum for {file_path}: {e}")
            return None