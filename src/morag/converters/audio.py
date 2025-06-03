"""Enhanced Audio to Markdown converter with speaker diarization and topic segmentation."""

import time
from pathlib import Path
from typing import Union, List, Dict, Any, Optional
import structlog
import asyncio

from .base import BaseConverter, ConversionOptions, ConversionResult, QualityScore
from .quality import ConversionQualityValidator
from ..processors.audio import audio_processor

logger = structlog.get_logger(__name__)

try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    logger.warning("pyannote.audio not available, speaker diarization disabled")

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import AgglomerativeClustering
    TOPIC_SEGMENTATION_AVAILABLE = True
except ImportError:
    TOPIC_SEGMENTATION_AVAILABLE = False
    logger.warning("Topic segmentation dependencies not available")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class AudioConverter(BaseConverter):
    """Enhanced Audio to Markdown converter with speaker diarization and topic segmentation."""

    def __init__(self):
        super().__init__("Enhanced MoRAG Audio Converter")
        self.supported_formats = ['audio', 'mp3', 'wav', 'm4a', 'flac']
        self.quality_validator = ConversionQualityValidator()

        # Initialize speaker diarization pipeline
        self.diarization_pipeline = None
        if PYANNOTE_AVAILABLE:
            self._initialize_diarization_pipeline()

        # Initialize topic segmentation
        self.topic_segmenter = None
        if TOPIC_SEGMENTATION_AVAILABLE:
            self._initialize_topic_segmenter()

    def _initialize_diarization_pipeline(self):
        """Initialize the speaker diarization pipeline."""
        try:
            # Note: This requires a Hugging Face token for the pretrained model
            # Users should set HF_TOKEN environment variable
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1"
                # use_auth_token="YOUR_HF_TOKEN"  # Users need to configure this
            )
            logger.info("Speaker diarization pipeline initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize speaker diarization: {e}")
            self.diarization_pipeline = None

    def _initialize_topic_segmenter(self):
        """Initialize the topic segmentation model."""
        try:
            self.topic_segmenter = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Topic segmentation model initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize topic segmentation: {e}")
            self.topic_segmenter = None
    
    def supports_format(self, format_type: str) -> bool:
        """Check if this converter supports the given format."""
        return format_type.lower() in self.supported_formats
    
    async def convert(self, file_path: Union[str, Path], options: ConversionOptions) -> ConversionResult:
        """Convert audio to structured markdown with transcription.
        
        Args:
            file_path: Path to audio file
            options: Conversion options
            
        Returns:
            ConversionResult with markdown content
        """
        start_time = time.time()
        file_path = Path(file_path)
        
        await self.validate_input(file_path)
        
        logger.info(
            "Starting audio conversion",
            file_path=str(file_path),
            enable_diarization=options.format_options.get('enable_diarization', False),
            include_timestamps=options.format_options.get('include_timestamps', True)
        )
        
        try:
            # Use existing MoRAG audio processor for basic transcription
            audio_result = await audio_processor.process_audio_file(str(file_path))

            # Enhanced processing with speaker diarization and topic segmentation
            enhanced_result = await self._enhance_audio_processing(file_path, audio_result, options)

            # Convert to structured markdown
            markdown_content = await self._create_enhanced_structured_markdown(enhanced_result, options)

            # Calculate quality score
            quality_score = self.quality_validator.validate_conversion(str(file_path), ConversionResult(
                content=markdown_content,
                metadata=enhanced_result.metadata
            ))

            processing_time = time.time() - start_time

            result = ConversionResult(
                content=markdown_content,
                metadata=self._enhance_metadata(enhanced_result.metadata, file_path),
                quality_score=quality_score,
                processing_time=processing_time,
                success=True,
                original_format='audio',
                converter_used=self.name
            )

            logger.info(
                "Enhanced audio conversion completed",
                processing_time=processing_time,
                quality_score=quality_score.overall_score,
                word_count=result.word_count,
                duration=enhanced_result.metadata.get('duration', 0),
                speakers_detected=enhanced_result.metadata.get('num_speakers', 0),
                topics_detected=enhanced_result.metadata.get('num_topics', 0)
            )

            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Audio conversion failed: {str(e)}"
            
            logger.error(
                "Audio conversion failed",
                error=str(e),
                error_type=type(e).__name__,
                processing_time=processing_time
            )
            
            return ConversionResult(
                content="",
                metadata={},
                processing_time=processing_time,
                success=False,
                error_message=error_msg,
                original_format='audio',
                converter_used=self.name
            )

    async def _enhance_audio_processing(self, file_path: Path, audio_result, options: ConversionOptions):
        """Enhance audio processing with speaker diarization and topic segmentation.

        Args:
            file_path: Path to audio file
            audio_result: Basic audio processing result
            options: Conversion options

        Returns:
            Enhanced audio result with speaker and topic information
        """
        enhanced_result = type('EnhancedAudioResult', (), {
            'transcript': audio_result.transcript if hasattr(audio_result, 'transcript') else audio_result.text,
            'metadata': audio_result.metadata.copy(),
            'summary': getattr(audio_result, 'summary', ''),
            'segments': getattr(audio_result, 'segments', []),
            'speakers': [],
            'topics': [],
            'speaker_segments': []
        })()

        # Speaker diarization
        if (self.diarization_pipeline and
            options.format_options.get('enable_diarization', True)):
            try:
                speakers_info = await self._perform_speaker_diarization(file_path)
                enhanced_result.speakers = speakers_info['speakers']
                enhanced_result.speaker_segments = speakers_info['segments']
                enhanced_result.metadata['num_speakers'] = len(speakers_info['speakers'])
                enhanced_result.metadata['diarization_used'] = True
                logger.info(f"Speaker diarization completed: {len(speakers_info['speakers'])} speakers detected")
            except Exception as e:
                logger.warning(f"Speaker diarization failed: {e}")
                enhanced_result.metadata['diarization_used'] = False

        # Topic segmentation
        if (self.topic_segmenter and
            options.format_options.get('enable_topic_segmentation', True)):
            try:
                topics_info = await self._perform_topic_segmentation(enhanced_result.transcript)
                enhanced_result.topics = topics_info
                enhanced_result.metadata['num_topics'] = len(topics_info)
                enhanced_result.metadata['topic_segmentation_used'] = True
                logger.info(f"Topic segmentation completed: {len(topics_info)} topics detected")
            except Exception as e:
                logger.warning(f"Topic segmentation failed: {e}")
                enhanced_result.metadata['topic_segmentation_used'] = False

        return enhanced_result

    async def _perform_speaker_diarization(self, file_path: Path) -> Dict[str, Any]:
        """Perform speaker diarization on audio file.

        Args:
            file_path: Path to audio file

        Returns:
            Dictionary with speaker information and segments
        """
        # Run diarization
        diarization = self.diarization_pipeline(str(file_path))

        # Extract speaker information
        speakers = {}
        segments = []

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in speakers:
                speakers[speaker] = {
                    'id': speaker,
                    'total_speaking_time': 0,
                    'segments_count': 0
                }

            speakers[speaker]['total_speaking_time'] += turn.end - turn.start
            speakers[speaker]['segments_count'] += 1

            segments.append({
                'speaker': speaker,
                'start_time': turn.start,
                'end_time': turn.end,
                'duration': turn.end - turn.start
            })

        return {
            'speakers': list(speakers.values()),
            'segments': segments
        }

    async def _perform_topic_segmentation(self, transcript: str) -> List[Dict[str, Any]]:
        """Perform topic segmentation on transcript.

        Args:
            transcript: Full transcript text

        Returns:
            List of topic segments
        """
        # Split transcript into sentences
        sentences = [s.strip() for s in transcript.split('.') if s.strip()]

        if len(sentences) < 3:
            return [{'topic': 'Main Content', 'sentences': sentences, 'start_sentence': 0}]

        # Generate embeddings
        embeddings = self.topic_segmenter.encode(sentences)

        # Cluster sentences into topics
        n_clusters = min(max(2, len(sentences) // 10), 8)  # Adaptive number of clusters
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        clusters = clustering.fit_predict(embeddings)

        # Group sentences by cluster
        topics = {}
        for i, cluster in enumerate(clusters):
            if cluster not in topics:
                topics[cluster] = []
            topics[cluster].append((i, sentences[i]))

        # Create topic segments
        topic_segments = []
        for cluster_id, sentence_list in topics.items():
            topic_segments.append({
                'topic': f'Topic {cluster_id + 1}',
                'sentences': [s[1] for s in sentence_list],
                'start_sentence': min(s[0] for s in sentence_list),
                'sentence_count': len(sentence_list)
            })

        # Sort by start sentence
        topic_segments.sort(key=lambda x: x['start_sentence'])

        return topic_segments

    async def _create_enhanced_structured_markdown(self, enhanced_result, options: ConversionOptions) -> str:
        """Create enhanced structured markdown with speaker and topic information.

        Args:
            enhanced_result: Enhanced audio processing result
            options: Conversion options

        Returns:
            Enhanced structured markdown content
        """
        sections = []

        # Document header
        filename = enhanced_result.metadata.get('filename', 'Audio File')
        title = filename.replace('.mp3', '').replace('.wav', '').replace('.m4a', '').replace('.flac', '')
        sections.append(f"# Audio Transcription: {title}")
        sections.append("")

        # Enhanced metadata section
        if options.include_metadata:
            sections.append("## Audio Information")
            sections.append("")

            duration = enhanced_result.metadata.get('duration', 0)
            duration_str = f"{duration:.1f} seconds" if duration < 60 else f"{duration/60:.1f} minutes"

            metadata_items = [
                ("**Source**", enhanced_result.metadata.get('filename', 'Unknown')),
                ("**Duration**", duration_str),
                ("**Language**", enhanced_result.metadata.get('language', 'Auto-detected')),
                ("**Speakers Detected**", str(enhanced_result.metadata.get('num_speakers', 'N/A'))),
                ("**Topics Identified**", str(enhanced_result.metadata.get('num_topics', 'N/A'))),
                ("**Transcription Model**", enhanced_result.metadata.get('model_used', 'Whisper')),
                ("**Speaker Diarization**", "Yes" if enhanced_result.metadata.get('diarization_used') else "No"),
                ("**Topic Segmentation**", "Yes" if enhanced_result.metadata.get('topic_segmentation_used') else "No")
            ]

            for label, value in metadata_items:
                if value and value != 'Unknown' and value != 'N/A':
                    sections.append(f"{label}: {value}")

            sections.append("")

        # Summary section
        if enhanced_result.summary:
            sections.append("## Summary")
            sections.append("")
            sections.append(enhanced_result.summary)
            sections.append("")

        # Speakers section
        if enhanced_result.speakers and options.format_options.get('include_speaker_info', True):
            sections.append("## Speakers")
            sections.append("")

            for i, speaker in enumerate(enhanced_result.speakers, 1):
                speaking_time = speaker['total_speaking_time']
                time_str = f"{speaking_time:.1f} seconds" if speaking_time < 60 else f"{speaking_time/60:.1f} minutes"
                sections.append(f"- **Speaker {i}** ({speaker['id']}): {time_str} speaking time, {speaker['segments_count']} segments")

            sections.append("")

        # Transcript section
        sections.append("## Transcript")
        sections.append("")

        if hasattr(enhanced_result, 'segments') and enhanced_result.segments:
            # Detailed transcript with timestamps
            for segment in enhanced_result.segments:
                if options.format_options.get('include_timestamps', True):
                    start_time = self._format_timestamp(segment.start_time)
                    end_time = self._format_timestamp(segment.end_time)
                    sections.append(f"**[{start_time} - {end_time}]**")

                text = segment.text.strip()
                if text:
                    sections.append(text)
                    sections.append("")
        else:
            # Simple transcript
            if hasattr(enhanced_result, 'transcript') and enhanced_result.transcript:
                sections.append(enhanced_result.transcript)
            elif hasattr(enhanced_result, 'text') and enhanced_result.text:
                sections.append(enhanced_result.text)
            else:
                sections.append("*No transcript available*")
            sections.append("")

        # Topics section
        if enhanced_result.topics and options.format_options.get('include_topic_info', True):
            sections.append("## Topics")
            sections.append("")

            for i, topic in enumerate(enhanced_result.topics, 1):
                sections.append(f"### {topic.get('topic', f'Topic {i}')}")
                sections.append("")

                for sentence in topic.get('sentences', []):
                    sections.append(f"- {sentence}")

                sections.append("")

        # Processing details
        sections.append("## Processing Details")
        sections.append("")
        sections.append(f"**Transcription Engine**: {enhanced_result.metadata.get('model_used', 'Whisper')}")

        if 'confidence' in enhanced_result.metadata:
            confidence = enhanced_result.metadata['confidence']
            sections.append(f"**Average Confidence**: {confidence:.2f}")

        if 'word_count' in enhanced_result.metadata:
            sections.append(f"**Word Count**: {enhanced_result.metadata['word_count']}")

        return "\n".join(sections)

    async def _create_structured_markdown(self, audio_result, options: ConversionOptions) -> str:
        """Create structured markdown from audio processing result.
        
        Args:
            audio_result: Audio processing result from MoRAG processor
            options: Conversion options
            
        Returns:
            Structured markdown content
        """
        sections = []
        
        # Document header
        filename = audio_result.metadata.get('filename', 'Audio File')
        title = filename.replace('.mp3', '').replace('.wav', '').replace('.m4a', '').replace('.flac', '')
        sections.append(f"# Audio Transcription: {title}")
        sections.append("")
        
        # Metadata section
        if options.include_metadata:
            sections.append("## Audio Information")
            sections.append("")
            
            duration = audio_result.metadata.get('duration', 0)
            duration_str = f"{duration:.1f} seconds" if duration < 60 else f"{duration/60:.1f} minutes"
            
            metadata_items = [
                ("**Source**", audio_result.metadata.get('filename', 'Unknown')),
                ("**Duration**", duration_str),
                ("**Language**", audio_result.metadata.get('language', 'Auto-detected')),
                ("**Transcription Model**", audio_result.metadata.get('model_used', 'Whisper')),
                ("**Processing Method**", audio_result.metadata.get('processor_used', 'MoRAG Audio Processor'))
            ]
            
            for label, value in metadata_items:
                if value and value != 'Unknown':
                    sections.append(f"{label}: {value}")
            
            sections.append("")
        
        # Summary section (if available)
        if hasattr(audio_result, 'summary') and audio_result.summary:
            sections.append("## Summary")
            sections.append("")
            sections.append(audio_result.summary)
            sections.append("")
        
        # Transcript section
        sections.append("## Transcript")
        sections.append("")
        
        if hasattr(audio_result, 'segments') and audio_result.segments:
            # Detailed transcript with timestamps
            for segment in audio_result.segments:
                if options.format_options.get('include_timestamps', True):
                    start_time = self._format_timestamp(segment.start_time)
                    end_time = self._format_timestamp(segment.end_time)
                    sections.append(f"**[{start_time} - {end_time}]**")

                text = segment.text.strip()
                if text:
                    sections.append(text)
                    sections.append("")
        else:
            # Simple transcript
            if hasattr(audio_result, 'transcript') and audio_result.transcript:
                sections.append(audio_result.transcript)
            elif hasattr(audio_result, 'text') and audio_result.text:
                sections.append(audio_result.text)
            else:
                sections.append("*No transcript available*")
            sections.append("")
        
        # Processing details
        sections.append("## Processing Details")
        sections.append("")
        sections.append(f"**Transcription Engine**: {audio_result.metadata.get('model_used', 'Whisper')}")
        
        if 'confidence' in audio_result.metadata:
            confidence = audio_result.metadata['confidence']
            sections.append(f"**Average Confidence**: {confidence:.2f}")
        
        if 'word_count' in audio_result.metadata:
            sections.append(f"**Word Count**: {audio_result.metadata['word_count']}")
        
        return "\n".join(sections)
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format timestamp in MM:SS format.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted timestamp string
        """
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def _enhance_metadata(self, original_metadata: dict, file_path: Path) -> dict:
        """Enhance metadata with additional information.
        
        Args:
            original_metadata: Original metadata from processor
            file_path: Path to original file
            
        Returns:
            Enhanced metadata dictionary
        """
        enhanced = original_metadata.copy()
        
        # Add file information
        enhanced.update({
            'original_filename': file_path.name,
            'file_size': file_path.stat().st_size,
            'conversion_format': 'audio_to_markdown',
            'converter_version': '1.0.0',
            'file_extension': file_path.suffix.lower()
        })
        
        return enhanced
