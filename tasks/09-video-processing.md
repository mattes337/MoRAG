# Task 09: Video Processing

## Overview
Implement video processing capabilities to extract audio tracks, generate thumbnails, and extract metadata from video files. This task enables the MoRAG system to process video content by leveraging existing audio processing pipeline and adding visual content analysis.

## Objectives
- [ ] Extract audio tracks from video files for speech-to-text processing
- [ ] Generate video thumbnails and keyframes for visual analysis
- [ ] Extract video metadata (duration, resolution, codec, etc.)
- [ ] Support multiple video formats (MP4, AVI, MOV, MKV, WebM, etc.)
- [ ] Implement frame extraction for image processing pipeline
- [ ] Add video quality assessment and optimization
- [ ] Create comprehensive testing for video processing pipeline

## Current State Analysis
The project currently has:
- ✅ Audio processing pipeline (Task 08) - Can reuse for extracted audio
- ✅ Async task processing with Celery (Task 04)
- ✅ Vector storage with Qdrant (Task 03)
- ✅ Enhanced summarization (Task 07)
- ✅ FFmpeg dependencies available through audio processing

## Implementation Plan

### Phase 1: Core Video Processing Infrastructure
1. **Video Processor Setup** - Create video processing service with FFmpeg integration
2. **Format Support** - Add support for common video formats
3. **Audio Extraction** - Extract audio tracks for speech-to-text processing
4. **Metadata Extraction** - Extract comprehensive video metadata

### Phase 2: Visual Content Processing
1. **Thumbnail Generation** - Generate representative thumbnails
2. **Keyframe Extraction** - Extract important frames for analysis
3. **Frame Sampling** - Intelligent frame sampling strategies
4. **Video Quality Assessment** - Detect and handle poor quality video

### Phase 3: Integration with Processing Pipeline
1. **Celery Task Integration** - Async video processing tasks
2. **Audio Pipeline Integration** - Route extracted audio to existing pipeline
3. **Storage Integration** - Store processed video content and metadata
4. **Error Handling** - Robust error handling for video processing failures

## Technical Requirements

### Dependencies
- **ffmpeg-python** - Video processing and format conversion (already available)
- **opencv-python** - Computer vision operations for frame extraction
- **Pillow** - Image processing for thumbnails
- **moviepy** - High-level video editing (optional, for complex operations)

### Video Processing Capabilities
- **Audio Extraction**: Extract audio tracks in various formats
- **Thumbnail Generation**: Create representative thumbnails at specified timestamps
- **Keyframe Detection**: Identify scene changes and important frames
- **Metadata Extraction**: Duration, resolution, codec, bitrate, frame rate
- **Format Conversion**: Convert between video formats as needed
- **Quality Assessment**: Analyze video quality metrics

### Integration Points
- **Audio Pipeline**: Route extracted audio to existing Whisper processing
- **Image Pipeline**: Route extracted frames to image processing (Task 10)
- **Storage**: Store video metadata and processed content in Qdrant
- **Task Queue**: Async processing with progress tracking

## File Structure
```
src/morag/processors/
├── video.py                 # Main video processor
└── video_utils.py          # Video utility functions

src/morag/services/
├── ffmpeg_service.py       # FFmpeg integration service
└── video_metadata.py      # Video metadata extraction

src/morag/tasks/
└── video_tasks.py         # Celery tasks for video processing (update existing)

tests/unit/
├── test_video_processor.py
├── test_ffmpeg_service.py
└── test_video_tasks.py

tests/integration/
└── test_video_pipeline.py

tests/fixtures/
└── video/                 # Test video files
    ├── sample.mp4
    ├── sample.avi
    └── sample_with_audio.mov
```

## Implementation Steps

### Step 1: Video Processor Core
```python
# src/morag/processors/video.py
@dataclass
class VideoConfig:
    extract_audio: bool = True
    generate_thumbnails: bool = True
    thumbnail_count: int = 5
    extract_keyframes: bool = False
    max_keyframes: int = 10
    audio_format: str = "wav"
    thumbnail_size: tuple = (320, 240)

@dataclass 
class VideoProcessingResult:
    audio_path: Optional[Path]
    thumbnails: List[Path]
    keyframes: List[Path]
    metadata: Dict[str, Any]
    duration: float
    processing_time: float
```

### Step 2: FFmpeg Service Integration
```python
# src/morag/services/ffmpeg_service.py
class FFmpegService:
    async def extract_audio(self, video_path: Path, output_format: str = "wav") -> Path
    async def generate_thumbnails(self, video_path: Path, count: int = 5) -> List[Path]
    async def extract_metadata(self, video_path: Path) -> Dict[str, Any]
    async def extract_keyframes(self, video_path: Path, max_frames: int = 10) -> List[Path]
```

### Step 3: Video Tasks Implementation
```python
# src/morag/tasks/video_tasks.py (replace placeholder)
@celery_app.task(bind=True, base=ProcessingTask)
async def process_video_file(self, file_path: str, task_id: str, config: Optional[Dict] = None)

@celery_app.task(bind=True, base=ProcessingTask) 
async def extract_video_audio(self, file_path: str, task_id: str)

@celery_app.task(bind=True, base=ProcessingTask)
async def generate_video_thumbnails(self, file_path: str, task_id: str, count: int = 5)
```

## Testing Requirements
- **Unit tests** for video processing components (>95% coverage)
- **Integration tests** with full video processing pipeline
- **Performance tests** with various video file sizes and formats
- **Quality tests** with different video quality samples
- **Error handling tests** for corrupted or unsupported files
- **Audio integration tests** with existing audio pipeline

## Success Criteria
- [ ] Successfully extract audio from video files
- [ ] Generate high-quality thumbnails and keyframes
- [ ] Extract comprehensive video metadata
- [ ] Integrate with existing audio processing pipeline
- [ ] Handle multiple video formats reliably
- [ ] Pass all unit and integration tests (>95% coverage)
- [ ] Process video files asynchronously with progress tracking
- [ ] Robust error handling for edge cases

## Dependencies to Add
```toml
[project.optional-dependencies]
video = [
    "opencv-python>=4.8.0",
    "Pillow>=10.0.0",
    "moviepy>=1.0.3",  # Optional for advanced operations
]
```

## Notes
- Reuse existing FFmpeg dependencies from audio processing
- Leverage existing audio pipeline for extracted audio tracks
- Prepare for integration with image processing (Task 10)
- Consider video file size limits and processing timeouts
- Implement efficient temporary file management
