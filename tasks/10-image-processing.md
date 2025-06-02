# Task 10: Image Processing

## Overview
Implement image processing capabilities including image captioning, OCR text extraction, and visual content analysis. This task enables the MoRAG system to process images, extract text content, and generate descriptive captions for visual understanding.

## Objectives
- [ ] Implement image captioning using vision language models
- [ ] Add OCR capabilities for text extraction from images
- [ ] Extract image metadata (EXIF, dimensions, format, etc.)
- [ ] Support multiple image formats (JPEG, PNG, GIF, TIFF, WebP, etc.)
- [ ] Implement image preprocessing and optimization
- [ ] Add visual content analysis and classification
- [ ] Create comprehensive testing for image processing pipeline

## Current State Analysis
The project currently has:
- ✅ Text processing and embedding pipeline
- ✅ Async task processing with Celery (Task 04)
- ✅ Vector storage with Qdrant (Task 03)
- ✅ Gemini API integration for text embeddings (can extend for vision)
- ✅ Enhanced summarization capabilities

## Implementation Plan

### Phase 1: Core Image Processing Infrastructure
1. **Image Processor Setup** - Create image processing service with vision models
2. **Format Support** - Add support for common image formats
3. **Image Preprocessing** - Implement image optimization and normalization
4. **Metadata Extraction** - Extract comprehensive image metadata

### Phase 2: Vision and Text Analysis
1. **Image Captioning** - Generate descriptive captions using vision models
2. **OCR Integration** - Extract text content from images
3. **Visual Classification** - Classify image content and scenes
4. **Quality Assessment** - Detect and handle poor quality images

### Phase 3: Integration with Processing Pipeline
1. **Celery Task Integration** - Async image processing tasks
2. **Text Pipeline Integration** - Route extracted text to existing pipeline
3. **Storage Integration** - Store processed image content and metadata
4. **Embedding Generation** - Create embeddings for visual and text content

## Technical Requirements

### Dependencies
- **Pillow** - Core image processing and format support
- **pytesseract** - OCR text extraction
- **google-generativeai** - Vision capabilities (already available)
- **opencv-python** - Computer vision operations (from video task)
- **easyocr** - Alternative OCR engine for better accuracy

### Image Processing Capabilities
- **Image Captioning**: Generate descriptive text for images
- **OCR Text Extraction**: Extract text content from images
- **Metadata Extraction**: EXIF data, dimensions, format, color profile
- **Image Classification**: Scene detection, object recognition
- **Quality Assessment**: Blur detection, brightness analysis
- **Format Conversion**: Convert between image formats as needed

### Integration Points
- **Text Pipeline**: Route extracted text to existing chunking and embedding
- **Vision Models**: Use Gemini Vision for image understanding
- **Storage**: Store image metadata and processed content in Qdrant
- **Task Queue**: Async processing with progress tracking

## File Structure
```
src/morag/processors/
├── image.py                # Main image processor
└── image_utils.py         # Image utility functions

src/morag/services/
├── vision_service.py      # Vision model integration
├── ocr_service.py         # OCR service integration
└── image_metadata.py     # Image metadata extraction

src/morag/tasks/
└── image_tasks.py        # Celery tasks for image processing

tests/unit/
├── test_image_processor.py
├── test_vision_service.py
├── test_ocr_service.py
└── test_image_tasks.py

tests/integration/
└── test_image_pipeline.py

tests/fixtures/
└── images/               # Test image files
    ├── sample.jpg
    ├── text_image.png
    ├── chart.png
    └── photo.jpeg
```

## Implementation Steps

### Step 1: Image Processor Core
```python
# src/morag/processors/image.py
@dataclass
class ImageConfig:
    generate_caption: bool = True
    extract_text: bool = True
    extract_metadata: bool = True
    resize_max_dimension: Optional[int] = 1024
    ocr_engine: str = "tesseract"  # or "easyocr"
    vision_model: str = "gemini-pro-vision"

@dataclass
class ImageProcessingResult:
    caption: Optional[str]
    extracted_text: Optional[str]
    metadata: Dict[str, Any]
    dimensions: Tuple[int, int]
    file_size: int
    processing_time: float
    confidence_scores: Dict[str, float]
```

### Step 2: Vision Service Integration
```python
# src/morag/services/vision_service.py
class VisionService:
    async def generate_caption(self, image_path: Path) -> str
    async def analyze_image_content(self, image_path: Path) -> Dict[str, Any]
    async def classify_image_type(self, image_path: Path) -> str
```

### Step 3: OCR Service Integration
```python
# src/morag/services/ocr_service.py
class OCRService:
    async def extract_text_tesseract(self, image_path: Path) -> str
    async def extract_text_easyocr(self, image_path: Path) -> str
    async def detect_text_regions(self, image_path: Path) -> List[Dict]
```

### Step 4: Image Tasks Implementation
```python
# src/morag/tasks/image_tasks.py
@celery_app.task(bind=True, base=ProcessingTask)
async def process_image_file(self, file_path: str, task_id: str, config: Optional[Dict] = None)

@celery_app.task(bind=True, base=ProcessingTask)
async def generate_image_caption(self, file_path: str, task_id: str)

@celery_app.task(bind=True, base=ProcessingTask)
async def extract_image_text(self, file_path: str, task_id: str, ocr_engine: str = "tesseract")
```

## Testing Requirements
- **Unit tests** for image processing components (>95% coverage)
- **Integration tests** with full image processing pipeline
- **Performance tests** with various image sizes and formats
- **Quality tests** with different image types (photos, documents, charts)
- **OCR accuracy tests** with text-heavy images
- **Vision model tests** for caption generation accuracy
- **Error handling tests** for corrupted or unsupported files

## Success Criteria
- [ ] Successfully generate descriptive captions for images
- [ ] Extract text content from images with high accuracy
- [ ] Extract comprehensive image metadata
- [ ] Support multiple image formats reliably
- [ ] Integrate with existing text processing pipeline
- [ ] Pass all unit and integration tests (>95% coverage)
- [ ] Process images asynchronously with progress tracking
- [ ] Robust error handling for edge cases

## Dependencies to Add
```toml
[project.optional-dependencies]
image = [
    "Pillow>=10.0.0",
    "pytesseract>=0.3.10",
    "easyocr>=1.7.0",
    "opencv-python>=4.8.0",
]
```

## Integration with Gemini Vision

### Vision API Usage
```python
# Use existing Gemini service with vision capabilities
import google.generativeai as genai

# Configure for vision tasks
model = genai.GenerativeModel('gemini-pro-vision')

# Generate image caption
response = model.generate_content([
    "Describe this image in detail, focusing on key visual elements and context.",
    image_data
])
```

## Notes
- Leverage existing Gemini API integration for vision capabilities
- Reuse text processing pipeline for extracted OCR text
- Consider image file size limits and processing timeouts
- Implement efficient image preprocessing and optimization
- Support both local and cloud-based vision models
- Prepare for integration with video frame processing (Task 09)
