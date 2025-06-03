"""Enhanced Audio to Markdown converter with speaker diarization and topic segmentation."""

import time
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Tuple
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
            # Use enhanced MoRAG audio processor with integrated features
            enable_diarization = options.format_options.get('enable_diarization', True)
            enable_topic_segmentation = options.format_options.get('enable_topic_segmentation', True)

            audio_result = await audio_processor.process_audio_file(
                str(file_path),
                enable_diarization=enable_diarization,
                enable_topic_segmentation=enable_topic_segmentation
            )

            # Create enhanced result structure for compatibility
            enhanced_result = await self._create_enhanced_result_structure(audio_result, options)

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

    async def _create_enhanced_result_structure(self, audio_result, options: ConversionOptions):
        """Create enhanced result structure from audio processing result.

        Args:
            audio_result: Audio processing result from enhanced processor
            options: Conversion options

        Returns:
            Enhanced audio result with speaker and topic information
        """
        enhanced_result = type('EnhancedAudioResult', (), {
            'transcript': audio_result.text,
            'metadata': audio_result.metadata.copy(),
            'summary': getattr(audio_result, 'summary', ''),
            'segments': audio_result.segments,
            'speakers': [],
            'topics': [],
            'speaker_segments': []
        })()

        # Extract speaker information if available
        if audio_result.speaker_diarization:
            diarization = audio_result.speaker_diarization
            enhanced_result.speakers = [
                {
                    'id': speaker.speaker_id,
                    'total_speaking_time': speaker.total_speaking_time,
                    'segments_count': speaker.segment_count
                }
                for speaker in diarization.speakers
            ]
            enhanced_result.speaker_segments = [
                {
                    'speaker': segment.speaker_id,
                    'start_time': segment.start_time,
                    'end_time': segment.end_time,
                    'duration': segment.duration
                }
                for segment in diarization.segments
            ]
            enhanced_result.metadata['num_speakers'] = diarization.total_speakers
            enhanced_result.metadata['diarization_used'] = True
        else:
            enhanced_result.metadata['diarization_used'] = False

        # Extract topic information if available
        if audio_result.topic_segmentation:
            segmentation = audio_result.topic_segmentation
            enhanced_result.topics = [
                {
                    'topic': topic.title,
                    'sentences': topic.sentences,
                    'start_sentence': 0,  # Simplified
                    'sentence_count': len(topic.sentences),
                    'summary': topic.summary,
                    'keywords': topic.keywords or []
                }
                for topic in segmentation.topics
            ]
            enhanced_result.metadata['num_topics'] = segmentation.total_topics
            enhanced_result.metadata['topic_segmentation_used'] = True
        else:
            enhanced_result.metadata['topic_segmentation_used'] = False

        return enhanced_result

    def _create_topic_dialogue(
        self,
        topic: Dict[str, Any],
        speaker_segments: List[Dict[str, Any]],
        transcript_segments: List
    ) -> List[Dict[str, str]]:
        """Create conversational dialogue format for a topic.

        Args:
            topic: Topic information with sentences
            speaker_segments: Speaker diarization segments
            transcript_segments: Transcript segments with timing

        Returns:
            List of dialogue entries with speaker and text
        """
        try:
            dialogue = []
            topic_sentences = topic.get('sentences', [])

            if not topic_sentences:
                return dialogue

            # If we have speaker segments and transcript segments, try to map them
            if speaker_segments and transcript_segments:
                # Create a mapping of text to speaker based on timing
                text_to_speaker = self._map_text_to_speakers(
                    topic_sentences, speaker_segments, transcript_segments
                )

                for sentence in topic_sentences:
                    speaker_id = text_to_speaker.get(sentence, 'Speaker_00')
                    dialogue.append({
                        'speaker': speaker_id,
                        'text': sentence.strip()
                    })
            else:
                # Fallback: assign all to Speaker_00
                for sentence in topic_sentences:
                    dialogue.append({
                        'speaker': 'Speaker_00',
                        'text': sentence.strip()
                    })

            return dialogue

        except Exception as e:
            logger.warning("Failed to create topic dialogue", error=str(e))
            # Fallback to simple format
            return [{'speaker': 'Speaker_00', 'text': sentence}
                   for sentence in topic.get('sentences', [])]

    def _map_text_to_speakers(
        self,
        sentences: List[str],
        speaker_segments: List[Dict[str, Any]],
        transcript_segments: List
    ) -> Dict[str, str]:
        """Map sentences to speakers based on timing information.

        Args:
            sentences: List of sentences to map
            speaker_segments: Speaker diarization segments
            transcript_segments: Transcript segments with timing

        Returns:
            Dictionary mapping sentence text to speaker ID
        """
        text_to_speaker = {}

        try:
            # Create a mapping of transcript text to timing
            transcript_timing = {}
            for segment in transcript_segments:
                if hasattr(segment, 'text') and hasattr(segment, 'start_time'):
                    transcript_timing[segment.text.strip()] = segment.start_time

            # For each sentence, find the best matching transcript segment
            for sentence in sentences:
                sentence_clean = sentence.strip()
                best_match_time = None

                # Try to find exact or partial match in transcript
                for transcript_text, start_time in transcript_timing.items():
                    if (sentence_clean in transcript_text or
                        transcript_text in sentence_clean or
                        self._text_similarity(sentence_clean, transcript_text) > 0.8):
                        best_match_time = start_time
                        break

                # If we found a timing, map to speaker
                if best_match_time is not None:
                    speaker_id = self._find_speaker_at_time(best_match_time, speaker_segments)
                    text_to_speaker[sentence] = speaker_id
                else:
                    # Default to first speaker
                    text_to_speaker[sentence] = 'Speaker_00'

        except Exception as e:
            logger.warning("Failed to map text to speakers", error=str(e))
            # Fallback: assign all to Speaker_00
            for sentence in sentences:
                text_to_speaker[sentence] = 'Speaker_00'

        return text_to_speaker

    def _find_speaker_at_time(self, time: float, speaker_segments: List[Dict[str, Any]]) -> str:
        """Find which speaker is active at a given time.

        Args:
            time: Time in seconds
            speaker_segments: Speaker diarization segments

        Returns:
            Speaker ID
        """
        for segment in speaker_segments:
            start_time = segment.get('start_time', 0)
            end_time = segment.get('end_time', 0)

            if start_time <= time <= end_time:
                return segment.get('speaker', 'Speaker_00')

        # Default to first speaker if no match found
        return 'Speaker_00'

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity between two strings.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Simple word-based similarity
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())

            if not words1 or not words2:
                return 0.0

            intersection = words1.intersection(words2)
            union = words1.union(words2)

            return len(intersection) / len(union) if union else 0.0

        except Exception:
            return 0.0

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

    def _calculate_topic_timestamps(
        self,
        topic: Dict[str, Any],
        speaker_segments: List[Dict[str, Any]],
        transcript_segments: List
    ) -> Tuple[float, float]:
        """Calculate start and end timestamps for a topic.

        Args:
            topic: Topic information with sentences
            speaker_segments: Speaker diarization segments
            transcript_segments: Transcript segments with timing

        Returns:
            Tuple of (start_time, end_time) in seconds, or (None, None) if not found
        """
        try:
            topic_sentences = topic.get('sentences', [])
            if not topic_sentences or not transcript_segments:
                return None, None

            # Find the earliest and latest timestamps for sentences in this topic
            topic_times = []

            for sentence in topic_sentences:
                sentence_clean = sentence.strip()

                # Try to find matching transcript segment
                for segment in transcript_segments:
                    if hasattr(segment, 'text') and hasattr(segment, 'start_time') and hasattr(segment, 'end_time'):
                        segment_text = segment.text.strip()

                        # Check for text similarity
                        if (sentence_clean in segment_text or
                            segment_text in sentence_clean or
                            self._text_similarity(sentence_clean, segment_text) > 0.6):
                            topic_times.append((segment.start_time, segment.end_time))
                            break

            if topic_times:
                start_time = min(t[0] for t in topic_times)
                end_time = max(t[1] for t in topic_times)
                return start_time, end_time

            return None, None

        except Exception as e:
            logger.warning("Failed to calculate topic timestamps", error=str(e))
            return None, None

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

        # Topics section with conversational format and timestamps
        if enhanced_result.topics and options.format_options.get('include_topic_info', True):
            for i, topic in enumerate(enhanced_result.topics, 1):
                topic_title = topic.get('topic', f'Topic {i}')

                # Calculate topic timestamp range
                topic_start_time, topic_end_time = self._calculate_topic_timestamps(
                    topic, enhanced_result.speaker_segments, enhanced_result.segments
                )

                # Add timestamp to topic header
                if topic_start_time is not None and topic_end_time is not None:
                    start_str = self._format_timestamp(topic_start_time)
                    end_str = self._format_timestamp(topic_end_time)
                    sections.append(f"# {topic_title} [{start_str} - {end_str}]")
                else:
                    sections.append(f"# {topic_title}")
                sections.append("")

                # Create conversational format by mapping sentences to speakers and timing
                topic_dialogue = self._create_topic_dialogue(
                    topic, enhanced_result.speaker_segments, enhanced_result.segments
                )

                if topic_dialogue:
                    for dialogue_entry in topic_dialogue:
                        speaker_id = dialogue_entry.get('speaker', 'Speaker_00')
                        text = dialogue_entry.get('text', '')
                        if text.strip():
                            sections.append(f"{speaker_id}: {text}")
                else:
                    # Fallback to sentence list if dialogue creation fails
                    for sentence in topic.get('sentences', []):
                        sections.append(f"Speaker_00: {sentence}")

                sections.append("")

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

        # Simple topic section for basic transcripts (without speaker diarization)
        if hasattr(audio_result, 'segments') and audio_result.segments:
            # Create a single topic with all content
            sections.append("# Main Content [00:00 - ")

            # Calculate total duration
            total_duration = 0
            if audio_result.segments:
                total_duration = max(segment.end_time for segment in audio_result.segments if hasattr(segment, 'end_time'))

            duration_str = self._format_timestamp(total_duration)
            sections[-1] += f"{duration_str}]"
            sections.append("")

            # Add all transcript content with speaker labels
            for segment in audio_result.segments:
                text = segment.text.strip()
                if text:
                    sections.append(f"Speaker_00: {text}")
        else:
            # Simple transcript fallback
            sections.append("# Main Content")
            sections.append("")

            if hasattr(audio_result, 'transcript') and audio_result.transcript:
                sections.append(f"Speaker_00: {audio_result.transcript}")
            elif hasattr(audio_result, 'text') and audio_result.text:
                sections.append(f"Speaker_00: {audio_result.text}")
            else:
                sections.append("Speaker_00: *No transcript available*")

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
