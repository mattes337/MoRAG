# MoRAG Video Processing Package

This package provides video processing capabilities for the MoRAG (Modular Retrieval Augmented Generation) system.

## Features

- Video transcription and audio extraction
- Scene detection and keyframe extraction
- Thumbnail generation
- Text extraction from video frames (OCR)
- Video metadata extraction
- Integration with MoRAG core services

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

```python
from morag_video import VideoProcessor, VideoConfig

# Create processor with default configuration
processor = VideoProcessor()

# Process a video file
result = await processor.process("path/to/video.mp4")

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