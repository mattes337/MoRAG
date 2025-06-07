# Enhanced Audio Processing with Speaker Diarization and Topic Segmentation

This document describes the enhanced audio processing capabilities in MoRAG, including advanced speaker diarization and topic segmentation features.

## Overview

The enhanced audio processing system provides:

- **Advanced Speaker Diarization**: Identify and track different speakers in audio/video content
- **Semantic Topic Segmentation**: Automatically segment content into coherent topics
- **Integrated Processing**: Seamless integration with existing MoRAG audio pipeline
- **Robust Fallbacks**: Graceful degradation when advanced models are not available
- **Comprehensive Configuration**: Fine-tuned control over processing parameters

## Features

### Speaker Diarization

#### Capabilities
- **Multi-speaker Detection**: Automatically identify multiple speakers in audio
- **Speaker Statistics**: Detailed analysis of speaking time, segments, and patterns
- **Timing Information**: Precise start/end times for each speaker segment
- **Overlap Detection**: Identify periods where multiple speakers talk simultaneously
- **Fallback Support**: Basic speaker detection when advanced models unavailable

#### Models Supported
- **Primary**: pyannote.audio speaker-diarization-3.1 (requires Hugging Face token)
- **Fallback**: Simple duration-based speaker assignment

#### Configuration Options
```python
# In settings or environment variables
enable_speaker_diarization = True
speaker_diarization_model = "pyannote/speaker-diarization-3.1"
min_speakers = 1
max_speakers = 10
huggingface_token = "your_hf_token_here"
```

### Topic Segmentation

#### Capabilities
- **Semantic Analysis**: Use sentence embeddings to detect topic boundaries
- **Adaptive Clustering**: Automatically determine optimal number of topics
- **Topic Summarization**: Generate summaries for each topic using LLM
- **Keyword Extraction**: Identify key terms for each topic
- **Speaker-Aware Boundaries**: Consider speaker changes when detecting topics
- **Timing Integration**: Map topics to audio timestamps

#### Models Supported
- **Primary**: SentenceTransformers with all-MiniLM-L6-v2
- **NLP Enhancement**: spaCy for advanced text processing
- **Fallback**: Simple sentence-based segmentation

#### Configuration Options
```python
# In settings or environment variables
enable_topic_segmentation = True
topic_similarity_threshold = 0.7
min_topic_sentences = 3
max_topics = 10
topic_embedding_model = "all-MiniLM-L6-v2"
use_llm_topic_summarization = True
```

## Installation

### Required Dependencies

#### Core Dependencies (always required)
```bash
pip install faster-whisper
pip install pydub
pip install librosa
pip install mutagen
```

#### Enhanced Features (optional)
```bash
# For speaker diarization
pip install pyannote.audio

# For topic segmentation
pip install sentence-transformers
pip install scikit-learn
pip install nltk

# For advanced NLP
pip install spacy
python -m spacy download en_core_web_sm
```

### Hugging Face Setup

For speaker diarization, you need a Hugging Face token:

1. Visit [Hugging Face](https://huggingface.co/) and create an account
2. Accept the user conditions for:
   - [pyannote/speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://hf.co/pyannote/segmentation-3.0)
3. Create a token at [Settings > Tokens](https://hf.co/settings/tokens)
4. Set the token in your environment:
   ```bash
   export HUGGINGFACE_TOKEN="your_token_here"
   ```

## Usage

### Basic Usage

```python
from morag.processors.audio import AudioProcessor

# Create processor
processor = AudioProcessor()

# Process with enhanced features
result = await processor.process_audio_file(
    "audio.wav",
    enable_diarization=True,
    enable_topic_segmentation=True
)

# Access results
print(f"Speakers detected: {result.speaker_diarization.total_speakers}")
print(f"Topics detected: {result.topic_segmentation.total_topics}")
```

### Advanced Configuration

```python
from morag.processors.audio import AudioProcessor, AudioConfig

# Custom configuration
config = AudioConfig(
    model_size="base",
    enable_diarization=True,
    device="auto"  # Auto-detect best available device (GPU/CPU)
)

processor = AudioProcessor(config)

# Process with custom settings
result = await processor.process_audio_file(
    "audio.wav",
    config=config,
    enable_diarization=True,
    enable_topic_segmentation=True
)
```

### Using the Converter

```python
from morag.converters.audio import AudioConverter
from morag.converters.base import ConversionOptions

converter = AudioConverter()

options = ConversionOptions(
    include_metadata=True,
    format_options={
        'enable_diarization': True,
        'enable_topic_segmentation': True,
        'include_timestamps': True,
        'include_speaker_info': True,
        'include_topic_info': True
    }
)

result = await converter.convert("audio.wav", options)
print(result.content)  # Enhanced markdown with speakers and topics
```

## Output Format

### Speaker Diarization Results

```python
# Speaker information
for speaker in result.speaker_diarization.speakers:
    print(f"Speaker {speaker.speaker_id}:")
    print(f"  Speaking time: {speaker.total_speaking_time:.2f}s")
    print(f"  Segments: {speaker.segment_count}")
    print(f"  Average segment: {speaker.average_segment_duration:.2f}s")

# Speaker segments with timing
for segment in result.speaker_diarization.segments:
    print(f"{segment.start_time:.1f}s - {segment.end_time:.1f}s: {segment.speaker_id}")
```

### Topic Segmentation Results

```python
# Topic information
for topic in result.topic_segmentation.topics:
    print(f"Topic: {topic.title}")
    print(f"  Summary: {topic.summary}")
    print(f"  Sentences: {len(topic.sentences)}")
    print(f"  Keywords: {', '.join(topic.keywords)}")
    if topic.start_time:
        print(f"  Timing: {topic.start_time:.1f}s - {topic.end_time:.1f}s")
```

### Enhanced Markdown Output

The enhanced markdown includes:

```markdown
# Audio Transcription: filename

## Audio Information
- **Duration**: 5.2 minutes
- **Speakers Detected**: 2
- **Topics Identified**: 3
- **Speaker Diarization**: Yes
- **Topic Segmentation**: Yes

## Speakers
- **Speaker 1** (SPEAKER_00): 3.2 minutes speaking time, 15 segments
- **Speaker 2** (SPEAKER_01): 2.0 minutes speaking time, 12 segments

## Topics
### Technology and AI
- Artificial intelligence is transforming industries
- Machine learning enables pattern recognition
- Deep learning improves accuracy

### Climate Change
- Global warming affects weather patterns
- Renewable energy reduces emissions
- Policy changes are necessary

## Transcript
**[0:00 - 0:15]**
Welcome to today's discussion about technology and climate change.

**[0:15 - 0:30]**
Let's start with how AI is being used in environmental research.
```

## Performance Considerations

### Processing Times
- **Basic transcription**: ~1x real-time
- **With speaker diarization**: ~2-3x real-time
- **With topic segmentation**: ~1.5x real-time
- **Full enhanced processing**: ~3-4x real-time

### Memory Usage
- **Basic processing**: ~500MB
- **Enhanced processing**: ~1-2GB (depending on models)

### Optimization Tips
1. Use smaller Whisper models for faster processing
2. Disable features not needed for your use case
3. Process shorter audio segments for better memory usage
4. Use CPU for small files, GPU for large batches

## Troubleshooting

### Common Issues

#### Speaker Diarization Not Working
```
Error: pyannote.audio not available
```
**Solution**: Install pyannote.audio and set up Hugging Face token

#### Topic Segmentation Failing
```
Error: Topic segmentation dependencies not available
```
**Solution**: Install sentence-transformers and scikit-learn

#### Poor Speaker Detection
**Symptoms**: All audio assigned to single speaker
**Solutions**:
- Check audio quality (clear speech, minimal background noise)
- Adjust min_speakers and max_speakers settings
- Ensure speakers have distinct voices

#### Inaccurate Topic Boundaries
**Symptoms**: Topics don't align with content changes
**Solutions**:
- Adjust topic_similarity_threshold (lower = more topics)
- Increase min_topic_sentences for longer topics
- Enable LLM summarization for better topic titles

### Fallback Behavior

When advanced models are not available:
- **Speaker diarization**: Falls back to simple duration-based splitting
- **Topic segmentation**: Falls back to sentence-count-based splitting
- **Processing continues**: No failures, just reduced functionality

## Testing

### Running Tests
```bash
# Unit tests
python -m pytest tests/unit/test_enhanced_audio_processing.py -v

# Integration tests
python -m pytest tests/integration/test_enhanced_audio_pipeline.py -v

# Demo script
python scripts/demo_enhanced_audio_processing.py sample_audio.wav
```

### Test Coverage
- Speaker diarization: Fallback and advanced modes
- Topic segmentation: Various text lengths and complexities
- Integration: Full pipeline with real audio files
- Error handling: Missing dependencies and invalid inputs

## API Reference

See the following modules for detailed API documentation:
- `morag.services.speaker_diarization`
- `morag.services.topic_segmentation`
- `morag.processors.audio`
- `morag.converters.audio`
