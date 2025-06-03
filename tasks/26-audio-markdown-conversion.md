# Task 26: Audio/Voice to Markdown Conversion with Speaker Diarization

## Objective
Implement comprehensive audio-to-markdown conversion with speech-to-text, speaker diarization, topic segmentation, and structured markdown output with timestamps and confidence scores.

## Research Phase

### Speech-to-Text Services
1. **OpenAI Whisper** (Current implementation)
   - Pros: High accuracy, local processing, multilingual
   - Cons: No built-in diarization, slower processing
   
2. **AssemblyAI**
   - Pros: Built-in diarization, topic detection, real-time processing
   - Cons: Cloud-only, cost per minute
   
3. **Google Speech-to-Text**
   - Pros: Excellent accuracy, diarization support, streaming
   - Cons: Cloud dependency, privacy concerns
   
4. **Azure Speech Services**
   - Pros: Good diarization, conversation analytics
   - Cons: Microsoft ecosystem dependency

### Speaker Diarization Libraries
1. **pyannote.audio** - State-of-the-art speaker diarization
2. **speechbrain** - Comprehensive speech processing toolkit
3. **resemblyzer** - Speaker verification and diarization
4. **NVIDIA NeMo** - Advanced speech AI toolkit

## Implementation Strategy

### Phase 1: Enhanced Audio Processing Pipeline
```python
class AudioToMarkdownConverter(BaseConverter):
    def __init__(self):
        self.transcriber = WhisperTranscriber()
        self.diarizer = SpeakerDiarizer()
        self.topic_segmenter = TopicSegmenter()
        self.markdown_formatter = AudioMarkdownFormatter()
    
    async def convert(self, audio_path: str, options: ConversionOptions) -> ConversionResult:
        # Step 1: Audio preprocessing
        processed_audio = await self.preprocess_audio(audio_path)
        
        # Step 2: Speech-to-text transcription
        transcription = await self.transcriber.transcribe(processed_audio, options)
        
        # Step 3: Speaker diarization
        if options.enable_diarization:
            diarization = await self.diarizer.identify_speakers(processed_audio)
            transcription = self.merge_transcription_with_diarization(transcription, diarization)
        
        # Step 4: Topic segmentation
        if options.enable_topic_segmentation:
            topics = await self.topic_segmenter.segment_topics(transcription)
            transcription = self.add_topic_headers(transcription, topics)
        
        # Step 5: Format as structured markdown
        markdown_content = await self.markdown_formatter.format(transcription, options)
        
        return ConversionResult(
            content=markdown_content,
            metadata=self.extract_audio_metadata(audio_path, transcription),
            quality_score=self.calculate_transcription_quality(transcription)
        )
```

### Phase 2: Speaker Diarization Implementation
```python
from pyannote.audio import Pipeline

class SpeakerDiarizer:
    def __init__(self):
        # Load pre-trained diarization pipeline
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token="YOUR_HF_TOKEN"
        )
    
    async def identify_speakers(self, audio_path: str) -> DiarizationResult:
        # Run speaker diarization
        diarization = self.pipeline(audio_path)
        
        # Convert to structured format
        speakers = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speakers.append({
                'speaker': speaker,
                'start_time': turn.start,
                'end_time': turn.end,
                'duration': turn.end - turn.start
            })
        
        return DiarizationResult(
            speakers=speakers,
            num_speakers=len(set(s['speaker'] for s in speakers)),
            total_duration=max(s['end_time'] for s in speakers)
        )
```

### Phase 3: Advanced Transcription with Timestamps
```python
class EnhancedWhisperTranscriber:
    def __init__(self):
        self.model = whisper.load_model("large-v3")
    
    async def transcribe_with_timestamps(self, audio_path: str, options: ConversionOptions) -> TranscriptionResult:
        # Transcribe with word-level timestamps
        result = self.model.transcribe(
            audio_path,
            word_timestamps=True,
            language=options.language,
            task="transcribe"
        )
        
        # Process segments with detailed timing
        segments = []
        for segment in result["segments"]:
            processed_segment = {
                'text': segment['text'].strip(),
                'start': segment['start'],
                'end': segment['end'],
                'confidence': segment.get('avg_logprob', 0.0),
                'words': segment.get('words', [])
            }
            segments.append(processed_segment)
        
        return TranscriptionResult(
            segments=segments,
            language=result.get('language', 'unknown'),
            duration=result.get('duration', 0)
        )
```

## Structured Markdown Output Format

### Standard Audio Markdown Template
```markdown
# Audio Transcription: {filename}

**Source**: {original_filename}
**Duration**: {duration} minutes
**Language**: {detected_language}
**Speakers**: {num_speakers} identified
**Transcribed**: {timestamp}
**Quality Score**: {confidence_score}

## Summary
{ai_generated_summary}

## Speakers
- **Speaker 1**: {speaker_characteristics}
- **Speaker 2**: {speaker_characteristics}

## Transcript

### Topic: {topic_name} ({start_time} - {end_time})

**Speaker 1** ({timestamp}): {transcribed_text}
*Confidence: {confidence_score}*

**Speaker 2** ({timestamp}): {transcribed_text}
*Confidence: {confidence_score}*

### Topic: {next_topic_name} ({start_time} - {end_time})

[Continue with more speakers and topics...]

## Timestamps Reference
- 00:00 - Introduction
- 05:30 - Main discussion begins
- 15:45 - Topic change: Technical details
- 25:10 - Q&A session
- 30:00 - Conclusion

## Metadata
- File format: {audio_format}
- Sample rate: {sample_rate}
- Channels: {channels}
- Bitrate: {bitrate}
```

### Configuration Options
```yaml
audio_conversion:
  transcription:
    model: "whisper-large-v3"
    language: "auto"  # or specific language code
    task: "transcribe"  # or "translate"
    word_timestamps: true
    
  diarization:
    enabled: true
    min_speakers: 1
    max_speakers: 10
    clustering_threshold: 0.7
    
  topic_segmentation:
    enabled: true
    min_segment_length: 30  # seconds
    similarity_threshold: 0.8
    
  output_formatting:
    include_timestamps: true
    include_confidence_scores: true
    include_speaker_labels: true
    include_topic_headers: true
    timestamp_format: "MM:SS"
    
  quality_filtering:
    min_confidence: 0.6
    filter_low_confidence: false
    mark_uncertain_text: true
```

## Integration with MoRAG System

### Enhanced Audio Processing Service
```python
# Update services/audio_processor.py
class AudioProcessor:
    def __init__(self):
        self.converter = AudioToMarkdownConverter()
        self.chunker = AudioChunker()
        self.embedder = EmbeddingService()
    
    async def process_audio_file(self, file_path: str, options: ProcessingOptions) -> ProcessedAudio:
        # Convert to markdown
        conversion_result = await self.converter.convert(file_path, options.conversion)
        
        # Create chunks based on speakers/topics
        chunks = await self.chunker.create_speaker_topic_chunks(
            conversion_result.content,
            options.chunking
        )
        
        # Generate embeddings for each chunk
        for chunk in chunks:
            chunk.embedding = await self.embedder.embed_text(chunk.content)
        
        return ProcessedAudio(
            markdown_content=conversion_result.content,
            chunks=chunks,
            metadata=conversion_result.metadata,
            quality_score=conversion_result.quality_score
        )
```

### Database Schema Updates
```sql
-- Audio-specific metadata
ALTER TABLE documents ADD COLUMN audio_duration FLOAT;
ALTER TABLE documents ADD COLUMN audio_language VARCHAR(10);
ALTER TABLE documents ADD COLUMN num_speakers INTEGER;
ALTER TABLE documents ADD COLUMN transcription_confidence FLOAT;

-- Speaker information
CREATE TABLE audio_speakers (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id),
    speaker_id VARCHAR(50),
    total_speaking_time FLOAT,
    characteristics JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Topic segments
CREATE TABLE audio_topics (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id),
    topic_name VARCHAR(200),
    start_time FLOAT,
    end_time FLOAT,
    summary TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Transcript segments with speaker attribution
CREATE TABLE transcript_segments (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id),
    speaker_id VARCHAR(50),
    start_time FLOAT,
    end_time FLOAT,
    text_content TEXT,
    confidence_score FLOAT,
    embedding VECTOR(768),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Advanced Features

### Topic Segmentation
```python
class TopicSegmenter:
    def __init__(self):
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.clustering_model = AgglomerativeClustering()
    
    async def segment_topics(self, transcription: TranscriptionResult) -> List[TopicSegment]:
        # Extract sentences with timestamps
        sentences = self.extract_sentences_with_timestamps(transcription)
        
        # Generate embeddings for sentences
        embeddings = self.sentence_transformer.encode([s.text for s in sentences])
        
        # Cluster similar sentences into topics
        clusters = self.clustering_model.fit_predict(embeddings)
        
        # Create topic segments
        topics = self.create_topic_segments(sentences, clusters)
        
        # Generate topic names using LLM
        for topic in topics:
            topic.name = await self.generate_topic_name(topic.sentences)
        
        return topics
```

### Quality Assessment
```python
class TranscriptionQualityAssessment:
    def assess_quality(self, transcription: TranscriptionResult) -> QualityMetrics:
        return QualityMetrics(
            overall_confidence=self.calculate_average_confidence(transcription),
            speaker_separation_quality=self.assess_speaker_separation(transcription),
            timestamp_accuracy=self.assess_timestamp_consistency(transcription),
            text_coherence=self.assess_text_coherence(transcription),
            completeness_score=self.assess_completeness(transcription)
        )
```

## Testing Requirements

### Unit Tests
- [ ] Test Whisper transcription accuracy
- [ ] Test speaker diarization with known audio
- [ ] Test topic segmentation logic
- [ ] Test markdown formatting
- [ ] Test timestamp accuracy

### Integration Tests
- [ ] Test with various audio formats (MP3, WAV, M4A)
- [ ] Test with different numbers of speakers
- [ ] Test with different languages
- [ ] Test with background noise and poor quality audio
- [ ] Test with long-form content (>1 hour)

### Quality Validation Tests
- [ ] Compare transcription accuracy with ground truth
- [ ] Validate speaker identification accuracy
- [ ] Test topic segmentation relevance
- [ ] Validate timestamp precision

## Performance Optimization

### Processing Strategies
1. **Parallel Processing**: Process audio chunks in parallel
2. **Model Optimization**: Use quantized models for faster inference
3. **Streaming**: Real-time processing for live audio
4. **Caching**: Cache speaker models and embeddings

### Performance Targets
- **Speed**: Real-time factor <0.5 (process 1 hour in <30 minutes)
- **Memory**: <2GB peak usage for 1-hour audio
- **Accuracy**: >95% word accuracy for clear speech
- **Diarization**: >90% speaker identification accuracy

## Implementation Steps

### Step 1: Enhanced Transcription (2-3 days)
- [ ] Upgrade Whisper integration with timestamps
- [ ] Add confidence scoring
- [ ] Implement language detection
- [ ] Add quality assessment

### Step 2: Speaker Diarization (3-4 days)
- [ ] Integrate pyannote.audio
- [ ] Implement speaker clustering
- [ ] Add speaker characteristic extraction
- [ ] Test with multi-speaker audio

### Step 3: Topic Segmentation (2-3 days)
- [ ] Implement sentence-level topic clustering
- [ ] Add topic name generation
- [ ] Create topic-based chunking
- [ ] Test segmentation quality

### Step 4: Markdown Formatting (1-2 days)
- [ ] Create structured markdown templates
- [ ] Add timestamp formatting
- [ ] Implement speaker attribution
- [ ] Add metadata sections

### Step 5: Integration and Testing (2-3 days)
- [ ] Integrate with MoRAG system
- [ ] Update database schema
- [ ] Comprehensive testing
- [ ] Performance optimization

## Success Criteria
- [ ] Speaker diarization working with >90% accuracy
- [ ] Topic segmentation creates meaningful segments
- [ ] Structured markdown output with all required elements
- [ ] Processing time <0.5x real-time for typical audio
- [ ] Integration with existing MoRAG pipeline
- [ ] Quality scores consistently >0.8

## Dependencies
- pyannote.audio library and models
- Enhanced Whisper integration
- Sentence transformer models
- Topic modeling capabilities
- Updated database schema

## Risks and Mitigation
- **Risk**: Diarization accuracy issues with similar voices
  - **Mitigation**: Multiple diarization models, manual review options
- **Risk**: Performance degradation with long audio files
  - **Mitigation**: Chunked processing, streaming capabilities
- **Risk**: Topic segmentation creating irrelevant segments
  - **Mitigation**: Configurable thresholds, manual override options
