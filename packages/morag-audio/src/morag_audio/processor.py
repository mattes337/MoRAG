"""Audio processor module for MoRAG."""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
import asyncio
import structlog

from morag_core.exceptions import ProcessingError
from morag_core.utils import get_safe_device

logger = structlog.get_logger(__name__)


@dataclass
class AudioSegment:
    """Represents a segment of audio with speaker information."""
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    confidence: float = 1.0
    topic_id: Optional[int] = None
    topic_label: Optional[str] = None


@dataclass
class AudioConfig:
    """Configuration for audio processing."""
    model_size: str = "medium"  # tiny, base, small, medium, large-v2, large-v3
    language: Optional[str] = None  # Auto-detect if None
    enable_diarization: bool = True  # Enable by default
    enable_topic_segmentation: bool = True  # Enable by default
    min_speakers: int = 1
    max_speakers: int = 5
    device: str = "auto"  # auto, cpu, cuda
    compute_type: str = "default"  # default, int8, float16, float32
    beam_size: int = 5
    vad_filter: bool = True
    vad_parameters: Dict[str, Any] = field(default_factory=lambda: {
        "threshold": 0.5,
        "min_speech_duration_ms": 250,
        "min_silence_duration_ms": 500
    })
    word_timestamps: bool = True
    include_metadata: bool = True

    def __post_init__(self):
        """Load configuration from environment variables if not explicitly set."""
        import os

        # Override with environment variables if they exist
        # Support both WHISPER_MODEL_SIZE and MORAG_WHISPER_MODEL_SIZE
        env_model_size = (
            os.environ.get("WHISPER_MODEL_SIZE") or
            os.environ.get("MORAG_WHISPER_MODEL_SIZE")
        )
        if env_model_size and self.model_size == "medium":  # Only override if using default
            self.model_size = env_model_size

        # Override language if set in environment
        env_language = os.environ.get("MORAG_AUDIO_LANGUAGE")
        if env_language and self.language is None:
            self.language = env_language

        # Override device if set in environment
        env_device = os.environ.get("MORAG_AUDIO_DEVICE")
        if env_device and self.device == "auto":  # Only override if using default
            self.device = env_device

        # Override diarization setting
        env_diarization = os.environ.get("MORAG_ENABLE_SPEAKER_DIARIZATION")
        if env_diarization is not None:
            self.enable_diarization = env_diarization.lower() in ("true", "1", "yes", "on")

        # Override topic segmentation setting
        env_topic_seg = os.environ.get("MORAG_ENABLE_TOPIC_SEGMENTATION")
        if env_topic_seg is not None:
            self.enable_topic_segmentation = env_topic_seg.lower() in ("true", "1", "yes", "on")


@dataclass
class AudioProcessingResult:
    """Result of audio processing."""
    transcript: str
    segments: List[AudioSegment]
    metadata: Dict[str, Any]
    file_path: str
    processing_time: float
    success: bool = True
    error_message: Optional[str] = None


class AudioProcessingError(ProcessingError):
    """Error raised during audio processing."""
    pass


class AudioProcessor:
    """Processes audio files to extract transcription, speaker information, and topics."""

    def __init__(self, config: Optional[AudioConfig] = None):
        """Initialize the audio processor.
        
        Args:
            config: Configuration for audio processing. If None, default config is used.
        """
        self.config = config or AudioConfig()
        self.metadata = {}
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize the required components based on configuration."""
        self.transcriber = None
        self.transcriber_type = None
        self.diarization_pipeline = None
        self.topic_segmenter = None

        # Initialize transcriber with fallback mechanism
        try:
            from faster_whisper import WhisperModel

            device = get_safe_device(self.config.device)
            logger.info("Initializing faster-whisper model",
                       model_size=self.config.model_size,
                       device=device,
                       compute_type=self.config.compute_type)

            self.transcriber = WhisperModel(
                model_size_or_path=self.config.model_size,
                device=device,
                compute_type=self.config.compute_type
            )
            self.transcriber_type = "faster_whisper"
            logger.info("faster-whisper model initialized successfully")
        except ImportError:
            logger.warning("faster-whisper not installed, trying OpenAI whisper fallback")
            self._initialize_openai_whisper()
        except Exception as e:
            logger.warning("Failed to initialize faster-whisper model, trying OpenAI whisper fallback", error=str(e))
            self._initialize_openai_whisper()

    def _initialize_openai_whisper(self):
        """Initialize OpenAI whisper as fallback."""
        try:
            import whisper

            logger.info("Initializing OpenAI whisper model", model_size=self.config.model_size)
            self.transcriber = whisper.load_model(self.config.model_size)
            self.transcriber_type = "openai_whisper"
            logger.info("OpenAI whisper model initialized successfully")
        except ImportError:
            logger.error("OpenAI whisper not installed. Please install with 'pip install openai-whisper'")
            raise AudioProcessingError("No whisper implementation available. Please install either faster-whisper or openai-whisper")
        except Exception as e:
            logger.error("Failed to initialize OpenAI whisper model", error=str(e))
            raise AudioProcessingError(f"Failed to initialize any whisper model: {str(e)}")

        # Initialize diarization if enabled
        if self.config.enable_diarization:
            try:
                from morag_audio.services import SpeakerDiarizationService
                
                self.diarization_service = SpeakerDiarizationService(
                    model_name="pyannote/speaker-diarization-3.1",
                    device=self.config.device,
                    min_speakers=self.config.min_speakers,
                    max_speakers=self.config.max_speakers
                    # Users need to configure HF_TOKEN in their environment
                )
                logger.info("Speaker diarization service initialized successfully")
            except ImportError:
                logger.warning("Speaker diarization dependencies not available, diarization disabled")
                self.config.enable_diarization = False
            except Exception as e:
                logger.warning(f"Failed to initialize speaker diarization: {e}")
                self.config.enable_diarization = False
        
        # Initialize topic segmentation if enabled
        if self.config.enable_topic_segmentation:
            try:
                from morag_audio.services import TopicSegmentationService
                
                self.topic_segmentation_service = TopicSegmentationService(
                    embedding_model="all-MiniLM-L6-v2",
                    device=self.config.device,
                    max_segments=5  # Reasonable default
                )
                logger.info("Topic segmentation service initialized successfully")
            except ImportError:
                logger.warning("Topic segmentation dependencies not available")
                self.config.enable_topic_segmentation = False
            except Exception as e:
                logger.warning(f"Failed to initialize topic segmentation: {e}")
                self.config.enable_topic_segmentation = False
    
    async def process(self, file_path: Union[str, Path], progress_callback: callable = None) -> AudioProcessingResult:
        """Process an audio file to extract transcription and metadata.

        Args:
            file_path: Path to the audio file
            progress_callback: Optional callback for progress updates

        Returns:
            AudioProcessingResult containing transcript, segments, and metadata

        Raises:
            AudioProcessingError: If processing fails
        """
        start_time = time.time()
        file_path = Path(file_path)

        if not file_path.exists():
            raise AudioProcessingError(f"File not found: {file_path}")

        logger.info("Starting audio processing",
                   file_path=str(file_path),
                   enable_diarization=self.config.enable_diarization,
                   enable_topic_segmentation=self.config.enable_topic_segmentation)

        if progress_callback:
            progress_callback(0.1, "Initializing audio processing")

        try:
            # Reset metadata for this processing run
            self.metadata = {}

            # Extract audio metadata
            if progress_callback:
                progress_callback(0.2, "Extracting audio metadata")
            self.metadata = await self._extract_metadata(file_path)

            # Transcribe audio
            if progress_callback:
                progress_callback(0.3, "Transcribing audio content")
            segments, transcript = await self._transcribe_audio(file_path)

            # Apply speaker diarization if enabled
            if self.config.enable_diarization and hasattr(self, 'diarization_service'):
                if progress_callback:
                    progress_callback(0.7, "Applying speaker diarization")
                segments = await self._apply_diarization(file_path, segments)

            # Apply topic segmentation if enabled
            if self.config.enable_topic_segmentation and hasattr(self, 'topic_segmentation_service'):
                if progress_callback:
                    progress_callback(0.85, "Applying topic segmentation")
                segments = await self._apply_topic_segmentation(segments)
            
            processing_time = time.time() - start_time

            if progress_callback:
                progress_callback(0.95, "Finalizing audio processing")

            # Update metadata with processing info
            self.metadata.update({
                "processing_time": processing_time,
                "word_count": len(transcript.split()),
                "segment_count": len(segments),
                "has_speaker_info": self.config.enable_diarization,
                "has_topic_info": self.config.enable_topic_segmentation
            })

            if self.config.enable_diarization:
                speakers = set(segment.speaker for segment in segments if segment.speaker)
                self.metadata["num_speakers"] = len(speakers)
                self.metadata["speakers"] = list(speakers)

            if self.config.enable_topic_segmentation:
                topics = set(segment.topic_id for segment in segments if segment.topic_id is not None)
                self.metadata["num_topics"] = len(topics)

            logger.info("Audio processing completed",
                       file_path=str(file_path),
                       processing_time=processing_time,
                       word_count=self.metadata["word_count"],
                       segment_count=self.metadata["segment_count"],
                       num_speakers=self.metadata.get("num_speakers", 0))
            
            return AudioProcessingResult(
                transcript=transcript,
                segments=segments,
                metadata=self.metadata,
                file_path=str(file_path),
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error("Audio processing failed", 
                        file_path=str(file_path),
                        error=str(e))
            
            processing_time = time.time() - start_time
            self.metadata["error"] = str(e)
            return AudioProcessingResult(
                transcript="",
                segments=[],
                metadata=self.metadata,
                file_path=str(file_path),
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    async def _extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from audio file."""
        metadata = {}
        
        try:
            import mutagen
            
            # Basic file info
            metadata["filename"] = file_path.name
            metadata["file_size"] = file_path.stat().st_size
            metadata["file_extension"] = file_path.suffix.lower()[1:]
            
            # Try to extract audio properties with pydub (may fail on Python 3.13+ due to missing audioop)
            try:
                from pydub import AudioSegment as PydubSegment
                audio = PydubSegment.from_file(file_path)
                metadata["duration"] = len(audio) / 1000.0  # Convert ms to seconds
                metadata["channels"] = audio.channels
                metadata["sample_rate"] = audio.frame_rate
                metadata["bit_depth"] = audio.sample_width * 8
            except (ImportError, ModuleNotFoundError) as pydub_error:
                logger.debug("Could not extract audio properties with pydub", error=str(pydub_error))
                # Fallback: try to get basic info from file extension and size
                metadata["pydub_unavailable"] = str(pydub_error)
            
            # Try to get additional metadata from mutagen
            try:
                audio_file = mutagen.File(file_path)
                if audio_file:
                    for key, value in audio_file.items():
                        if key.lower() not in ["cover", "picture", "artwork"]:
                            metadata[f"tag_{key.lower()}"] = str(value)
            except Exception as mutagen_error:
                logger.debug("Could not extract additional metadata", error=str(mutagen_error))
                
        except Exception as e:
            logger.warning("Error extracting audio metadata", error=str(e))
            metadata["metadata_extraction_error"] = str(e)
            
        return metadata
    
    async def _transcribe_audio(self, file_path: Path) -> tuple[List[AudioSegment], str]:
        """Transcribe audio file using Whisper."""
        if not self.transcriber:
            raise AudioProcessingError("Transcriber not initialized")

        if self.transcriber_type == "faster_whisper":
            return await self._transcribe_with_faster_whisper(file_path)
        elif self.transcriber_type == "openai_whisper":
            return await self._transcribe_with_openai_whisper(file_path)
        else:
            raise AudioProcessingError(f"Unknown transcriber type: {self.transcriber_type}")

    async def _transcribe_with_faster_whisper(self, file_path: Path) -> tuple[List[AudioSegment], str]:
        """Transcribe audio file using faster-whisper."""
        # Run transcription in a separate thread to avoid blocking
        loop = asyncio.get_event_loop()
        segments_data, info = await loop.run_in_executor(
            None,
            lambda: self.transcriber.transcribe(
                str(file_path),
                beam_size=self.config.beam_size,
                language=self.config.language,
                vad_filter=self.config.vad_filter,
                vad_parameters=self.config.vad_parameters,
                word_timestamps=self.config.word_timestamps
            )
        )

        # Store language information in metadata
        if hasattr(info, 'language'):
            self.metadata["language"] = info.language
        elif hasattr(info, 'detected_language'):
            self.metadata["language"] = info.detected_language
        else:
            self.metadata["language"] = self.config.language or "unknown"

        # Convert to our segment format
        segments = []
        full_transcript = ""

        for segment_data in segments_data:
            segment = AudioSegment(
                start=segment_data.start,
                end=segment_data.end,
                text=segment_data.text,
                confidence=segment_data.avg_logprob
            )
            segments.append(segment)
            full_transcript += segment_data.text + " "

        return segments, full_transcript.strip()

    async def _transcribe_with_openai_whisper(self, file_path: Path) -> tuple[List[AudioSegment], str]:
        """Transcribe audio file using OpenAI whisper."""
        # Run transcription in a separate thread to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.transcriber.transcribe(
                str(file_path),
                language=self.config.language,
                word_timestamps=self.config.word_timestamps
            )
        )

        # Store language information in metadata
        self.metadata["language"] = result.get("language", self.config.language or "unknown")

        # Convert to our segment format
        segments = []
        full_transcript = result["text"]

        for segment_data in result["segments"]:
            segment = AudioSegment(
                start=segment_data["start"],
                end=segment_data["end"],
                text=segment_data["text"],
                confidence=segment_data.get("avg_logprob", 0.0)
            )
            segments.append(segment)

        return segments, full_transcript.strip()
    
    async def _apply_diarization(self, file_path: Union[str, Path], segments: List[AudioSegment]) -> List[AudioSegment]:
        """Apply speaker diarization to the audio segments.
        
        Args:
            file_path: Path to the audio file
            segments: List of audio segments from transcription
            
        Returns:
            Updated list of audio segments with speaker information
        """
        if not hasattr(self, 'diarization_service') or not self.diarization_service:
            logger.warning("Diarization requested but service not available")
            return segments
        
        try:
            # Call the async diarization service directly
            diarization_result = await self.diarization_service.diarize_audio(
                str(file_path),
                min_speakers=self.config.min_speakers,
                max_speakers=self.config.max_speakers
            )
            
            # Map diarization results to segments based on overlap
            for segment in segments:
                # Find the speaker with the most overlap for this segment
                max_overlap = 0
                assigned_speaker = None
                
                for speaker_segment in diarization_result.segments:
                    # Calculate overlap between transcription segment and speaker segment
                    overlap_start = max(segment.start, speaker_segment.start_time)
                    overlap_end = min(segment.end, speaker_segment.end_time)
                    overlap = max(0, overlap_end - overlap_start)

                    if overlap > max_overlap:
                        max_overlap = overlap
                        assigned_speaker = speaker_segment.speaker_id
                
                # Assign speaker to segment if found
                if assigned_speaker:
                    segment.speaker = assigned_speaker
            
            # Add speaker metadata
            self.metadata["num_speakers"] = len(diarization_result.speakers)
            self.metadata["speakers"] = [
                {"id": s.speaker_id, "name": f"Speaker {s.speaker_id}"}
                for s in diarization_result.speakers
            ]
            
            return segments
            
        except Exception as e:
            logger.error(f"Error in speaker diarization: {str(e)}", exc_info=True)
            return segments
    
    async def _apply_topic_segmentation(self, segments: List[AudioSegment]) -> List[AudioSegment]:
        """Apply topic segmentation to segments using the topic segmentation service."""
        if not hasattr(self, 'topic_segmentation_service') or not self.topic_segmentation_service or len(segments) < 3:
            return segments
        
        try:
            # Prepare transcript and segments for the service
            full_transcript = " ".join([segment.text for segment in segments])
            
            # Convert our segments to the format expected by the service
            transcript_segments = [
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "speaker": segment.speaker
                } for segment in segments
            ]
            
            # Run topic segmentation using the service
            segmentation_result = await self.topic_segmentation_service.segment_transcript(
                transcript=full_transcript,
                transcript_segments=transcript_segments,
                max_segments=5  # Reasonable default
            )
            
            # Create a mapping of topic segments to our audio segments
            # by finding which topic segment each audio segment belongs to
            for segment in segments:
                segment.topic_id = None
                segment.topic_label = None
                
                # Find the topic segment that contains this audio segment
                for topic in segmentation_result.segments:
                    # Check if the segment falls within this topic's time range
                    if segment.start >= topic.start_time and segment.end <= topic.end_time:
                        segment.topic_id = int(topic.topic_id.split('_')[1])  # Convert TOPIC_XX to integer
                        segment.topic_label = topic.title
                        break
                    
                    # If no exact match, find the topic with the most overlap
                    elif max(segment.start, topic.start_time) < min(segment.end, topic.end_time):
                        overlap = min(segment.end, topic.end_time) - max(segment.start, topic.start_time)
                        segment_duration = segment.end - segment.start
                        
                        # If more than 50% of the segment is in this topic, assign it
                        if overlap > (segment_duration * 0.5):
                            segment.topic_id = int(topic.topic_id.split('_')[1])
                            segment.topic_label = topic.title
                            break
            
            # Add topic metadata
            self.metadata["topics"] = [
                {
                    "id": topic.topic_id,
                    "title": topic.title,
                    "summary": topic.summary,
                    "start_time": topic.start_time,
                    "end_time": topic.end_time,
                    "duration": topic.duration,
                    "keywords": topic.keywords,
                    "speaker_distribution": topic.speaker_distribution
                } for topic in segmentation_result.segments
            ]
            
            return segments
            
        except Exception as e:
            logger.warning("Topic segmentation failed", error=str(e))
            return segments