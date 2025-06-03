# Task 27: Video to Markdown Conversion with Keyframe Extraction

## Objective
Implement comprehensive video-to-markdown conversion that extracts audio for speech-to-text, analyzes keyframes for visual content description, and combines everything into structured markdown with timestamps.

## Research Phase

### Video Processing Libraries
1. **FFmpeg** (Current implementation)
   - Pros: Comprehensive format support, reliable, fast
   - Cons: Command-line interface, complex for advanced features
   
2. **OpenCV**
   - Pros: Advanced computer vision, frame analysis
   - Cons: Limited codec support, more complex setup
   
3. **MoviePy**
   - Pros: Python-native, easy to use, good for editing
   - Cons: Slower than FFmpeg, limited format support
   
4. **PyAV**
   - Pros: Python bindings for FFmpeg, more control
   - Cons: Steeper learning curve, less documentation

### Keyframe Detection Strategies
1. **Scene Change Detection**: Detect significant visual changes
2. **Temporal Sampling**: Extract frames at regular intervals
3. **Content-Based**: Extract frames with important visual content
4. **Motion Analysis**: Detect frames with significant motion changes

### Visual Content Analysis
1. **CLIP Models**: Image-text understanding for scene description
2. **BLIP/BLIP-2**: Image captioning and visual question answering
3. **GPT-4 Vision**: Advanced image understanding and description
4. **Custom Vision Models**: Specialized for specific content types

## Implementation Strategy

### Phase 1: Enhanced Video Processing Pipeline
```python
class VideoToMarkdownConverter(BaseConverter):
    def __init__(self):
        self.video_processor = VideoProcessor()
        self.audio_extractor = AudioExtractor()
        self.keyframe_detector = KeyframeDetector()
        self.vision_analyzer = VisionAnalyzer()
        self.audio_converter = AudioToMarkdownConverter()  # From Task 26
        self.markdown_formatter = VideoMarkdownFormatter()
    
    async def convert(self, video_path: str, options: ConversionOptions) -> ConversionResult:
        # Step 1: Extract video metadata
        metadata = await self.video_processor.extract_metadata(video_path)
        
        # Step 2: Extract audio track
        audio_path = await self.audio_extractor.extract_audio(video_path)
        
        # Step 3: Process audio to text (reuse audio converter)
        audio_result = await self.audio_converter.convert(audio_path, options.audio_options)
        
        # Step 4: Extract and analyze keyframes
        keyframes = await self.keyframe_detector.extract_keyframes(video_path, options.keyframe_options)
        keyframe_analysis = await self.vision_analyzer.analyze_keyframes(keyframes, options.vision_options)
        
        # Step 5: Synchronize audio and visual content
        synchronized_content = await self.synchronize_content(audio_result, keyframe_analysis, metadata)
        
        # Step 6: Format as structured markdown
        markdown_content = await self.markdown_formatter.format(synchronized_content, options)
        
        return ConversionResult(
            content=markdown_content,
            metadata=self.combine_metadata(metadata, audio_result.metadata),
            quality_score=self.calculate_video_quality(audio_result, keyframe_analysis)
        )
```

### Phase 2: Keyframe Detection and Extraction
```python
import cv2
import numpy as np
from typing import List, Tuple

class KeyframeDetector:
    def __init__(self):
        self.scene_threshold = 0.3  # Threshold for scene change detection
        self.min_interval = 5.0     # Minimum seconds between keyframes
    
    async def extract_keyframes(self, video_path: str, options: KeyframeOptions) -> List[Keyframe]:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        keyframes = []
        prev_frame = None
        frame_count = 0
        last_keyframe_time = -self.min_interval
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = frame_count / fps
            
            # Scene change detection
            if prev_frame is not None:
                scene_change_score = self.calculate_scene_change(prev_frame, frame)
                
                # Extract keyframe if scene change detected and minimum interval passed
                if (scene_change_score > self.scene_threshold and 
                    current_time - last_keyframe_time >= self.min_interval):
                    
                    keyframe = Keyframe(
                        timestamp=current_time,
                        frame_number=frame_count,
                        image_data=frame,
                        scene_change_score=scene_change_score
                    )
                    keyframes.append(keyframe)
                    last_keyframe_time = current_time
            
            # Also extract frames at regular intervals as backup
            if options.regular_interval and current_time % options.interval_seconds < (1/fps):
                if current_time - last_keyframe_time >= self.min_interval:
                    keyframe = Keyframe(
                        timestamp=current_time,
                        frame_number=frame_count,
                        image_data=frame,
                        scene_change_score=0.0,
                        extraction_method="interval"
                    )
                    keyframes.append(keyframe)
                    last_keyframe_time = current_time
            
            prev_frame = frame
            frame_count += 1
        
        cap.release()
        return keyframes
    
    def calculate_scene_change(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        # Convert to grayscale and calculate histogram difference
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
        
        # Calculate correlation coefficient
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # Return scene change score (1 - correlation)
        return 1.0 - correlation
```

### Phase 3: Visual Content Analysis
```python
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

class VisionAnalyzer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model.to(self.device)
    
    async def analyze_keyframes(self, keyframes: List[Keyframe], options: VisionOptions) -> List[KeyframeAnalysis]:
        analyses = []
        
        for keyframe in keyframes:
            analysis = await self.analyze_single_keyframe(keyframe, options)
            analyses.append(analysis)
        
        return analyses
    
    async def analyze_single_keyframe(self, keyframe: Keyframe, options: VisionOptions) -> KeyframeAnalysis:
        # Convert OpenCV frame to PIL Image
        image = self.opencv_to_pil(keyframe.image_data)
        
        # Generate image caption
        caption = await self.generate_caption(image)
        
        # Detect objects (optional)
        objects = []
        if options.detect_objects:
            objects = await self.detect_objects(image)
        
        # Extract text from image (OCR)
        text_content = ""
        if options.extract_text:
            text_content = await self.extract_text_from_image(image)
        
        # Analyze scene type
        scene_type = await self.classify_scene_type(image)
        
        return KeyframeAnalysis(
            keyframe=keyframe,
            caption=caption,
            objects=objects,
            text_content=text_content,
            scene_type=scene_type,
            confidence_score=self.calculate_analysis_confidence(caption, objects, text_content)
        )
    
    async def generate_caption(self, image) -> str:
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            out = self.model.generate(**inputs, max_length=50, num_beams=5)
        
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption
```

## Structured Video Markdown Output

### Video Markdown Template
```markdown
# Video Analysis: {filename}

**Source**: {original_filename}
**Duration**: {duration} minutes
**Resolution**: {width}x{height}
**Frame Rate**: {fps} fps
**Audio Tracks**: {num_audio_tracks}
**Processed**: {timestamp}
**Quality Score**: {overall_quality}

## Summary
{ai_generated_summary_of_video_content}

## Visual Overview
- **Scene Changes**: {num_scene_changes} detected
- **Keyframes Extracted**: {num_keyframes}
- **Primary Content Type**: {content_type} (presentation, conversation, demonstration, etc.)
- **Visual Quality**: {visual_quality_assessment}

## Audio Transcript
{audio_transcript_from_task_26}

## Visual Timeline

### 00:00 - Opening Scene
![Keyframe 1](keyframe_001.jpg)
**Scene**: {scene_description}
**Visual Elements**: {detected_objects}
**Text Visible**: {ocr_text}

### 02:30 - Topic Introduction
![Keyframe 2](keyframe_002.jpg)
**Scene**: {scene_description}
**Visual Elements**: {detected_objects}
**Text Visible**: {ocr_text}

**Audio Context**: {corresponding_audio_transcript}

### 05:15 - Main Content
![Keyframe 3](keyframe_003.jpg)
**Scene**: {scene_description}
**Visual Elements**: {detected_objects}
**Text Visible**: {ocr_text}

**Audio Context**: {corresponding_audio_transcript}

## Synchronized Content

### Segment 1: Introduction (00:00 - 02:30)
**Visual**: Opening slide with title and presenter
**Audio**: "Welcome to today's presentation on..."
**Key Points**: 
- Presenter introduction
- Topic overview
- Agenda presentation

### Segment 2: Main Topic (02:30 - 15:45)
**Visual**: Slides with diagrams and charts
**Audio**: Detailed explanation of concepts
**Key Points**:
- Technical concepts explained
- Visual aids support audio content
- Interactive demonstrations

## Technical Metadata
- **Video Codec**: {video_codec}
- **Audio Codec**: {audio_codec}
- **Bitrate**: {bitrate}
- **File Size**: {file_size}
- **Color Space**: {color_space}
```

### Configuration Options
```yaml
video_conversion:
  keyframe_extraction:
    method: "scene_change"  # scene_change, interval, hybrid
    scene_threshold: 0.3
    min_interval_seconds: 5
    max_keyframes: 100
    regular_interval_seconds: 30
    
  visual_analysis:
    generate_captions: true
    detect_objects: true
    extract_text: true
    classify_scenes: true
    confidence_threshold: 0.7
    
  audio_processing:
    extract_audio: true
    audio_format: "wav"
    sample_rate: 16000
    # Inherit audio options from Task 26
    
  synchronization:
    align_audio_visual: true
    segment_by_scenes: true
    min_segment_length: 10  # seconds
    
  output_formatting:
    include_keyframe_images: true
    include_timestamps: true
    include_visual_descriptions: true
    include_synchronized_segments: true
```

## Integration with MoRAG System

### Enhanced Video Processing Service
```python
# Update services/video_processor.py
class VideoProcessor:
    def __init__(self):
        self.converter = VideoToMarkdownConverter()
        self.chunker = VideoChunker()
        self.embedder = EmbeddingService()
    
    async def process_video_file(self, file_path: str, options: ProcessingOptions) -> ProcessedVideo:
        # Convert to markdown
        conversion_result = await self.converter.convert(file_path, options.conversion)
        
        # Create chunks based on scenes and audio segments
        chunks = await self.chunker.create_scene_audio_chunks(
            conversion_result.content,
            options.chunking
        )
        
        # Generate embeddings for each chunk
        for chunk in chunks:
            chunk.embedding = await self.embedder.embed_text(chunk.content)
        
        return ProcessedVideo(
            markdown_content=conversion_result.content,
            chunks=chunks,
            keyframes=conversion_result.keyframes,
            metadata=conversion_result.metadata,
            quality_score=conversion_result.quality_score
        )
```

### Database Schema Updates
```sql
-- Video-specific metadata
ALTER TABLE documents ADD COLUMN video_duration FLOAT;
ALTER TABLE documents ADD COLUMN video_resolution VARCHAR(20);
ALTER TABLE documents ADD COLUMN video_fps FLOAT;
ALTER TABLE documents ADD COLUMN num_keyframes INTEGER;
ALTER TABLE documents ADD COLUMN num_scene_changes INTEGER;

-- Keyframes table
CREATE TABLE video_keyframes (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id),
    timestamp FLOAT,
    frame_number INTEGER,
    image_path VARCHAR(500),
    caption TEXT,
    scene_type VARCHAR(100),
    confidence_score FLOAT,
    extraction_method VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Visual elements detected in keyframes
CREATE TABLE keyframe_objects (
    id SERIAL PRIMARY KEY,
    keyframe_id INTEGER REFERENCES video_keyframes(id),
    object_type VARCHAR(100),
    confidence FLOAT,
    bounding_box JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Scene segments
CREATE TABLE video_scenes (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id),
    start_time FLOAT,
    end_time FLOAT,
    scene_type VARCHAR(100),
    description TEXT,
    keyframe_id INTEGER REFERENCES video_keyframes(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Testing Requirements

### Unit Tests
- [ ] Test keyframe extraction with various video types
- [ ] Test scene change detection accuracy
- [ ] Test visual content analysis
- [ ] Test audio-visual synchronization
- [ ] Test markdown formatting

### Integration Tests
- [ ] Test with different video formats (MP4, AVI, MOV)
- [ ] Test with various resolutions and frame rates
- [ ] Test with videos containing presentations, conversations, demonstrations
- [ ] Test with poor quality or corrupted videos
- [ ] Test with very long videos (>2 hours)

### Quality Validation Tests
- [ ] Validate keyframe relevance and coverage
- [ ] Test caption accuracy for extracted frames
- [ ] Validate audio-visual synchronization accuracy
- [ ] Test overall content comprehension

## Performance Optimization

### Processing Strategies
1. **Parallel Processing**: Process keyframes concurrently
2. **Smart Sampling**: Adaptive keyframe extraction based on content
3. **Caching**: Cache visual analysis results
4. **GPU Acceleration**: Use GPU for vision models when available

### Performance Targets
- **Speed**: Process 1 hour video in <20 minutes
- **Memory**: <4GB peak usage for HD videos
- **Keyframe Quality**: >85% relevant keyframes extracted
- **Caption Accuracy**: >80% meaningful captions

## Implementation Steps

### Step 1: Enhanced Keyframe Extraction (2-3 days)
- [ ] Implement scene change detection
- [ ] Add interval-based extraction
- [ ] Optimize keyframe selection algorithms
- [ ] Add quality assessment for keyframes

### Step 2: Visual Content Analysis (3-4 days)
- [ ] Integrate BLIP for image captioning
- [ ] Add object detection capabilities
- [ ] Implement OCR for text extraction
- [ ] Add scene classification

### Step 3: Audio-Visual Synchronization (2-3 days)
- [ ] Align keyframes with audio transcript
- [ ] Create synchronized segments
- [ ] Implement content correlation
- [ ] Add temporal consistency checks

### Step 4: Markdown Formatting (1-2 days)
- [ ] Create structured video markdown templates
- [ ] Add visual timeline formatting
- [ ] Implement synchronized content sections
- [ ] Add metadata and technical details

### Step 5: Integration and Testing (2-3 days)
- [ ] Integrate with MoRAG system
- [ ] Update database schema
- [ ] Comprehensive testing
- [ ] Performance optimization

## Success Criteria
- [ ] Keyframe extraction captures important visual moments
- [ ] Visual content analysis provides meaningful descriptions
- [ ] Audio-visual synchronization works accurately
- [ ] Structured markdown output includes all required elements
- [ ] Processing time meets performance targets
- [ ] Integration with existing MoRAG pipeline successful

## Dependencies
- Enhanced FFmpeg integration
- BLIP or similar vision models
- OCR capabilities (Tesseract or cloud service)
- Audio processing from Task 26
- Updated database schema

## Risks and Mitigation
- **Risk**: Poor keyframe selection missing important content
  - **Mitigation**: Multiple extraction methods, quality validation
- **Risk**: Inaccurate visual content analysis
  - **Mitigation**: Multiple vision models, confidence thresholds
- **Risk**: Performance issues with large video files
  - **Mitigation**: Chunked processing, GPU acceleration, optimization
