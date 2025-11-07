# MoRAG Video Processing Package

This package provides video processing capabilities for the MoRAG (Modular Retrieval Augmented Generation) system.

## Features

- **Video Transcription**: High-quality audio transcription using markitdown framework
- **Format Support**: MP4, AVI, MOV, MKV, WEBM, FLV, WMV video formats
- **Document Conversion**: Convert videos to structured markdown with transcripts
- **Scene Detection**: Keyframe extraction and thumbnail generation
- **Text Extraction**: OCR from video frames
- **Video Metadata**: Comprehensive metadata extraction
- **Integration**: Seamless integration with MoRAG document processing pipeline

## Installation

### Basic Installation

```bash
pip install morag-video
```

### With Scene Detection

```bash
pip install "morag-video[scene]"
```

### With OCR Support

```bash
pip install "morag-video[ocr]"
```

### Full Installation

```bash
pip install "morag-video[scene,ocr]"
```

## Usage

### Document Conversion (Recommended)

```python
from morag_video.converters.video import VideoConverter
from morag_core.interfaces.converter import ConversionOptions

# Create video converter
converter = VideoConverter()

# Convert video to markdown document
options = ConversionOptions(title="My Video")
result = await converter.convert("path/to/video.mp4", options)

# Access the converted document
document = result.document
print(f"Title: {document.title}")
print(f"Transcript: {document.raw_text}")
```

### Advanced Video Processing

```python
from morag_video import VideoProcessor, VideoConfig

# Create processor with custom configuration
config = VideoConfig(
    extract_audio=True,
    generate_thumbnails=True,
    extract_keyframes=True,
    enable_enhanced_audio=True
)
processor = VideoProcessor(config)

# Process a video file
result = await processor.process_video("path/to/video.mp4")

# Access the transcription
print(result.transcript)

# Access extracted keyframes
for i, frame in enumerate(result.keyframes):
    frame.save(f"keyframe_{i}.jpg")

# Access scene information
for i, scene in enumerate(result.scenes):
    print(f"Scene {i}: {scene.start_time} - {scene.end_time}")
```

## Dependencies

- morag-core: Core interfaces and utilities
- morag-embedding: Embedding services for scene analysis
- morag-audio: Audio processing for transcription
- opencv-python: Video frame processing
- ffmpeg-python: Video manipulation
- scenedetect (optional): Advanced scene detection
- pytesseract, easyocr (optional): OCR capabilities

## License

MIT
