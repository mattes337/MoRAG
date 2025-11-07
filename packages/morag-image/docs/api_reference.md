# Morag Image Package API Reference

## Table of Contents

- [ImageProcessor](#imageprocessor)
- [ImageService](#imageservice)
- [Data Classes](#data-classes)
  - [ImageConfig](#imageconfig)
  - [ImageMetadata](#imagemetadata)
  - [ImageProcessingResult](#imageprocessingresult)
- [Exceptions](#exceptions)
  - [ProcessingError](#processingerror)

## ImageProcessor

The core class for processing images.

```python
class ImageProcessor:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the image processor.

        Args:
            api_key: Google API key for Gemini Vision API (for captioning)
        """

    async def process_image(self, image_path: str, config: ImageConfig) -> ImageProcessingResult:
        """Process a single image according to the provided configuration.

        Args:
            image_path: Path to the image file
            config: Configuration for image processing

        Returns:
            ImageProcessingResult object containing the processing results

        Raises:
            ProcessingError: If an error occurs during processing
            FileNotFoundError: If the image file does not exist
        """

    async def process_batch(self, image_paths: List[str], configs: Optional[Dict[str, ImageConfig]] = None,
                           default_config: Optional[ImageConfig] = None, max_concurrency: int = 4) -> List[ImageProcessingResult]:
        """Process multiple images concurrently.

        Args:
            image_paths: List of paths to image files
            configs: Optional dictionary mapping image paths to specific configurations
            default_config: Default configuration to use for images without specific configs
            max_concurrency: Maximum number of concurrent processing tasks

        Returns:
            List of ImageProcessingResult objects
        """

    async def extract_metadata(self, image_path: str) -> Dict[str, Any]:
        """Extract metadata from an image.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary containing image metadata

        Raises:
            ProcessingError: If metadata extraction fails
        """

    async def preprocess_image(self, image_path: str, max_dimension: Optional[int] = None) -> Tuple[Image.Image, str]:
        """Preprocess an image (resize if needed).

        Args:
            image_path: Path to the image file
            max_dimension: Maximum dimension (width or height) for resizing

        Returns:
            Tuple of (PIL Image object, temporary file path if created)

        Raises:
            ProcessingError: If preprocessing fails
        """

    async def generate_caption(self, image_path: str) -> str:
        """Generate a caption for an image using Gemini Vision API.

        Args:
            image_path: Path to the image file

        Returns:
            Generated caption string

        Raises:
            ProcessingError: If caption generation fails
        """

    async def extract_text(self, image_path: str, engine: str = "tesseract") -> str:
        """Extract text from an image using OCR.

        Args:
            image_path: Path to the image file
            engine: OCR engine to use ("tesseract" or "easyocr")

        Returns:
            Extracted text string

        Raises:
            ProcessingError: If text extraction fails
        """
```

## ImageService

Service class for integrating with the Morag framework.

```python
class ImageService(BaseService):
    def __init__(self, config: ServiceConfig):
        """Initialize the image service.

        Args:
            config: Service configuration object
        """

    async def process_image(self, image_path: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a single image.

        Args:
            image_path: Path to the image file
            config: Optional configuration dictionary

        Returns:
            Dictionary containing processing results
        """

    async def process_batch(self, image_paths: List[str], configs: Optional[Dict[str, Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """Process multiple images.

        Args:
            image_paths: List of paths to image files
            configs: Optional dictionary mapping image paths to configurations

        Returns:
            List of dictionaries containing processing results
        """

    def _result_to_dict(self, result: ImageProcessingResult) -> Dict[str, Any]:
        """Convert an ImageProcessingResult to a dictionary.

        Args:
            result: ImageProcessingResult object

        Returns:
            Dictionary representation of the result
        """
```

## Data Classes

### ImageConfig

Configuration class for customizing image processing options.

```python
@dataclass
class ImageConfig:
    extract_metadata: bool = True  # Whether to extract image metadata
    extract_text: bool = False     # Whether to extract text using OCR
    generate_caption: bool = False # Whether to generate image caption
    ocr_engine: str = "tesseract"  # OCR engine to use ("tesseract" or "easyocr")
    max_dimension: Optional[int] = 1024  # Maximum dimension for resizing
```

### ImageMetadata

Dataclass for storing image metadata.

```python
@dataclass
class ImageMetadata:
    width: int           # Image width in pixels
    height: int          # Image height in pixels
    format: str          # Image format (e.g., "JPEG", "PNG")
    mode: str            # Image mode (e.g., "RGB", "RGBA")
    exif: Dict[str, Any] # EXIF metadata (if available)
```

### ImageProcessingResult

Dataclass for storing image processing results.

```python
@dataclass
class ImageProcessingResult:
    image_path: str                      # Path to the processed image
    metadata: Optional[Dict[str, Any]]   # Image metadata (if extracted)
    text: Optional[str]                  # Extracted text (if OCR was performed)
    caption: Optional[str]               # Generated caption (if captioning was enabled)
    error: Optional[str] = None          # Error message (if processing failed)
```

## Exceptions

### ProcessingError

Exception raised when an error occurs during image processing.

```python
class ProcessingError(Exception):
    """Exception raised when an error occurs during image processing."""
    pass
```

## Command Line Interface

The package provides a command-line interface through the `cli.py` module.

```
usage: python -m morag_image.cli [-h] [--output OUTPUT] [--metadata] [--ocr] [--caption]
                                 [--ocr-engine {tesseract,easyocr}] [--max-dimension MAX_DIMENSION]
                                 [--max-concurrency MAX_CONCURRENCY] [--api-key API_KEY]
                                 input

Process images to extract metadata, text, and generate captions.

positional arguments:
  input                 Path to image file or directory containing images

options:
  -h, --help            show this help message and exit
  --output OUTPUT, -o OUTPUT
                        Output file path for results (JSON format)
  --metadata, -m        Extract image metadata
  --ocr, -t             Extract text using OCR
  --caption, -c         Generate image caption using Gemini Vision
  --ocr-engine {tesseract,easyocr}
                        OCR engine to use (default: tesseract)
  --max-dimension MAX_DIMENSION
                        Maximum dimension for image resizing
  --max-concurrency MAX_CONCURRENCY
                        Maximum number of concurrent processing tasks
  --api-key API_KEY     Google API key for Gemini Vision (overrides environment variable)
```

For more details on using the CLI, see the [CLI Usage Guide](../examples/cli_usage.md).
