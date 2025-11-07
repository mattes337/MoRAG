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

    def __init__(self, api_key: Optional[str] = None, output_dir: Optional[Path] = None):
        """Initialize the image service.

        Args:
            api_key: Optional API key for Gemini vision model
            output_dir: Directory to store processed files
        """
        self.processor = ImageProcessor(api_key=api_key)

        # Set up output directory
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(exist_ok=True)
        else:
            self.output_dir = None

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

    async def process_file(
        self,
        file_path: Path,
        save_output: bool = True,
        output_format: str = "markdown"
    ) -> Dict[str, Any]:
        """Process an image file and optionally save output files.

        Args:
            file_path: Path to the image file
            save_output: Whether to save output files
            output_format: Output format ('markdown', 'json', or 'both')

        Returns:
            Dictionary containing processing results and output file paths
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise ProcessingError(f"File not found: {file_path}")

        logger.info("Processing image file", file_path=str(file_path))

        try:
            # Create default config
            config = ImageConfig(
                extract_metadata=True,
                extract_text=True,
                generate_caption=True
            )

            # Process the image
            result = await self.processor.process_image(file_path, config)

            # Create markdown content
            markdown_content = self._create_markdown_content(file_path, result)

            # Prepare response
            response = {
                "success": True,
                "processing_time": result.processing_time,
                "result": {
                    "caption": result.caption,
                    "extracted_text": result.extracted_text,
                    "metadata": result.metadata.__dict__ if result.metadata else {},
                    "confidence_scores": result.confidence_scores
                },
                "content": markdown_content
            }

            # Save output files if requested
            if save_output and self.output_dir:
                output_files = await self._save_output_files(file_path, result, markdown_content, output_format)
                response["output_files"] = output_files

            return response

        except Exception as e:
            logger.error("Image processing failed", error=str(e), file_path=str(file_path))
            return {
                "success": False,
                "error": str(e),
                "processing_time": 0
            }

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

    def _create_markdown_content(self, file_path: Path, result: ImageProcessingResult) -> str:
        """Create unified markdown content for the image."""
        lines = []

        # Title
        lines.append(f"# {file_path.stem}")
        lines.append("")

        # Image Information
        lines.append("## Image Information")
        lines.append("| Property | Value |")
        lines.append("|----------|-------|")

        if result.metadata:
            lines.append(f"| Dimensions | {result.metadata.width}x{result.metadata.height} |")
            lines.append(f"| Format | {result.metadata.format} |")
            lines.append(f"| Mode | {result.metadata.mode} |")
            lines.append(f"| File Size | {result.metadata.file_size} bytes |")

            if result.metadata.camera_make:
                lines.append(f"| Camera Make | {result.metadata.camera_make} |")
            if result.metadata.camera_model:
                lines.append(f"| Camera Model | {result.metadata.camera_model} |")
            if result.metadata.creation_time:
                lines.append(f"| Creation Time | {result.metadata.creation_time} |")

        lines.append("")

        # Caption
        if result.caption:
            lines.append("## Image Caption")
            lines.append(result.caption)
            lines.append("")

        # Extracted Text (OCR)
        if result.extracted_text and result.extracted_text.strip():
            lines.append("## Extracted Text")
            lines.append(result.extracted_text)
            lines.append("")

        # Confidence Scores
        if result.confidence_scores:
            lines.append("## Confidence Scores")
            for key, score in result.confidence_scores.items():
                lines.append(f"- **{key.title()}**: {score:.3f}")
            lines.append("")

        return "\n".join(lines)

    async def _save_output_files(
        self,
        file_path: Path,
        result: ImageProcessingResult,
        markdown_content: str,
        output_format: str
    ) -> Dict[str, str]:
        """Save processing results to output files."""
        output_files = {}
        base_name = file_path.stem

        # Sanitize the base name for directory creation
        # Remove problematic characters and limit length
        sanitized_name = "".join(c for c in base_name if c.isalnum() or c in (' ', '-', '_')).strip()
        if len(sanitized_name) > 100:  # Limit directory name length
            sanitized_name = sanitized_name[:100].strip()
        if not sanitized_name:  # Fallback if name becomes empty
            sanitized_name = f"image_{hash(base_name) % 10000}"

        # Create output directory for this file
        file_output_dir = self.output_dir / sanitized_name
        try:
            file_output_dir.mkdir(exist_ok=True, parents=True)
            logger.debug("Created output directory", output_dir=str(file_output_dir))
        except Exception as e:
            logger.error("Failed to create output directory",
                        output_dir=str(file_output_dir),
                        error=str(e))
            raise ProcessingError(f"Failed to create output directory: {str(e)}")

        # Save markdown content
        if output_format in ["markdown", "both"]:
            markdown_path = file_output_dir / f"{sanitized_name}.md"
            try:
                with open(markdown_path, "w", encoding="utf-8") as f:
                    f.write(markdown_content)
                output_files["markdown"] = str(markdown_path)
                logger.debug("Saved markdown", path=str(markdown_path))
            except Exception as e:
                logger.error("Failed to save markdown",
                           path=str(markdown_path),
                           error=str(e))

        # Save metadata as JSON
        import json
        metadata_path = file_output_dir / f"{sanitized_name}_metadata.json"
        metadata_dict = {
            "file_path": str(file_path),
            "file_size": file_path.stat().st_size,
            "processing_time": result.processing_time,
            "caption": result.caption,
            "extracted_text": result.extracted_text,
            "confidence_scores": result.confidence_scores,
            "image_metadata": result.metadata.__dict__ if result.metadata else {}
        }
        try:
            metadata_path.write_text(json.dumps(metadata_dict, indent=2, ensure_ascii=False), encoding='utf-8')
            output_files["metadata"] = str(metadata_path)
            logger.debug("Saved metadata", path=str(metadata_path))
        except Exception as e:
            logger.error("Failed to save metadata",
                       path=str(metadata_path),
                       error=str(e))

        logger.info("Saved image output files",
                   output_dir=str(file_output_dir),
                   files_created=list(output_files.keys()))

        return output_files
