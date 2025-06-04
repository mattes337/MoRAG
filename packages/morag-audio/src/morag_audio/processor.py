"""Audio processor module for MoRAG."""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
import asyncio
import structlog

from morag_core.errors import ProcessingError
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
    model_size: str = "medium"  # tiny, base, small, medium, large-v2
    language: Optional[str] = None  # Auto-detect if None
    enable_diarization: bool = False
    enable_topic_segmentation: bool = False
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
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize the required components based on configuration."""
        self.transcriber = None
        self.diarization_pipeline = None
        self.topic_segmenter = None
        
        # Initialize transcriber
        try:
            from faster_whisper import WhisperModel
            
            device = get_safe_device(self.config.device)
            logger.info("Initializing Whisper model", 
                       model_size=self.config.model_size, 
                       device=device,
                       compute_type=self.config.compute_type)
            
            self.transcriber = WhisperModel(
                model_size_or_path=self.config.model_size,
                device=device,
                compute_type=self.config.compute_type
            )
            logger.info("Whisper model initialized successfully")
        except ImportError:
            logger.error("faster-whisper not installed. Please install with 'pip install faster-whisper'")
            raise AudioProcessingError("Required dependency 'faster-whisper' not installed")
        except Exception as e:
            logger.error("Failed to initialize Whisper model", error=str(e))
            raise AudioProcessingError(f"Failed to initialize Whisper model: {str(e)}")
        
        # Initialize diarization if enabled
        if self.config.enable_diarization:
            try:
                from pyannote.audio import Pipeline
                
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1"
                    # Users need to configure HF_TOKEN in their environment
                )
                logger.info("Speaker diarization pipeline initialized successfully")
            except ImportError:
                logger.warning("pyannote.audio not available, speaker diarization disabled")
                self.config.enable_diarization = False
            except Exception as e:
                logger.warning(f"Failed to initialize speaker diarization: {e}")
                self.config.enable_diarization = False
        
        # Initialize topic segmentation if enabled
        if self.config.enable_topic_segmentation:
            try:
                from sentence_transformers import SentenceTransformer
                from sklearn.cluster import AgglomerativeClustering
                
                device = get_safe_device(self.config.device)
                logger.info("Initializing topic segmentation model", device=device)
                
                try:
                    self.topic_segmenter = SentenceTransformer('all-MiniLM-L6-v2', device=device)
                    logger.info("Topic segmentation model initialized successfully")
                except Exception as device_error:
                    if device != "cpu":
                        logger.warning("GPU initialization failed, trying CPU", error=str(device_error))
                        self.topic_segmenter = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")
                        logger.info("Topic segmentation model initialized on CPU fallback")
                    else:
                        raise
            except ImportError:
                logger.warning("Topic segmentation dependencies not available")
                self.config.enable_topic_segmentation = False
            except Exception as e:
                logger.warning(f"Failed to initialize topic segmentation: {e}")
                self.config.enable_topic_segmentation = False
    
    async def process(self, file_path: Union[str, Path]) -> AudioProcessingResult:
        """Process an audio file to extract transcription and metadata.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            AudioProcessingResult containing transcript, segments, and metadata
            
        Raises:
            AudioProcessingError: If processing fails
        """
        start_time = time.time()
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise AudioProcessingError(f"File not found: {file_path}")
        
        logger.info("Processing audio file", 
                   file_path=str(file_path),
                   enable_diarization=self.config.enable_diarization,
                   enable_topic_segmentation=self.config.enable_topic_segmentation)
        
        try:
            # Extract audio metadata
            metadata = await self._extract_metadata(file_path)
            
            # Transcribe audio
            segments, transcript = await self._transcribe_audio(file_path)
            
            # Apply speaker diarization if enabled
            if self.config.enable_diarization and self.diarization_pipeline:
                segments = await self._apply_diarization(file_path, segments)
            
            # Apply topic segmentation if enabled
            if self.config.enable_topic_segmentation and self.topic_segmenter:
                segments = await self._apply_topic_segmentation(segments)
            
            processing_time = time.time() - start_time
            
            # Update metadata with processing info
            metadata.update({
                "processing_time": processing_time,
                "word_count": len(transcript.split()),
                "segment_count": len(segments),
                "has_speaker_info": self.config.enable_diarization,
                "has_topic_info": self.config.enable_topic_segmentation
            })
            
            if self.config.enable_diarization:
                speakers = set(segment.speaker for segment in segments if segment.speaker)
                metadata["num_speakers"] = len(speakers)
                metadata["speakers"] = list(speakers)
            
            if self.config.enable_topic_segmentation:
                topics = set(segment.topic_id for segment in segments if segment.topic_id is not None)
                metadata["num_topics"] = len(topics)
            
            logger.info("Audio processing completed",
                       file_path=str(file_path),
                       processing_time=processing_time,
                       word_count=metadata["word_count"],
                       segment_count=metadata["segment_count"],
                       num_speakers=metadata.get("num_speakers", 0))
            
            return AudioProcessingResult(
                transcript=transcript,
                segments=segments,
                metadata=metadata,
                file_path=str(file_path),
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error("Audio processing failed", 
                        file_path=str(file_path),
                        error=str(e))
            
            processing_time = time.time() - start_time
            return AudioProcessingResult(
                transcript="",
                segments=[],
                metadata={"error": str(e)},
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
            from pydub import AudioSegment as PydubSegment
            
            # Basic file info
            metadata["filename"] = file_path.name
            metadata["file_size"] = file_path.stat().st_size
            metadata["file_extension"] = file_path.suffix.lower()[1:]
            
            # Extract audio properties
            audio = PydubSegment.from_file(file_path)
            metadata["duration"] = len(audio) / 1000.0  # Convert ms to seconds
            metadata["channels"] = audio.channels
            metadata["sample_rate"] = audio.frame_rate
            metadata["bit_depth"] = audio.sample_width * 8
            
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
    
    async def _apply_diarization(self, file_path: Path, segments: List[AudioSegment]) -> List[AudioSegment]:
        """Apply speaker diarization to segments."""
        if not self.diarization_pipeline:
            logger.warning("Diarization requested but pipeline not available")
            return segments
        
        try:
            # Run diarization in a separate thread
            loop = asyncio.get_event_loop()
            diarization = await loop.run_in_executor(
                None,
                lambda: self.diarization_pipeline(
                    str(file_path),
                    num_speakers=self.config.max_speakers
                )
            )
            
            # Map diarization results to segments
            for segment in segments:
                # Find overlapping speaker segments
                speaker_times = []
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    if max(segment.start, turn.start) < min(segment.end, turn.end):
                        overlap = min(segment.end, turn.end) - max(segment.start, turn.start)
                        speaker_times.append((speaker, overlap))
                
                # Assign the speaker with the most overlap
                if speaker_times:
                    segment.speaker = max(speaker_times, key=lambda x: x[1])[0]
            
            return segments
            
        except Exception as e:
            logger.warning("Speaker diarization failed", error=str(e))
            return segments
    
    async def _apply_topic_segmentation(self, segments: List[AudioSegment]) -> List[AudioSegment]:
        """Apply topic segmentation to segments."""
        if not self.topic_segmenter or len(segments) < 3:
            return segments
        
        try:
            from sklearn.cluster import AgglomerativeClustering
            
            # Get text from segments
            texts = [segment.text for segment in segments]
            
            # Generate embeddings
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: self.topic_segmenter.encode(texts)
            )
            
            # Determine optimal number of topics (between 1 and 5)
            max_topics = min(5, max(1, len(segments) // 5))
            
            # Cluster the embeddings
            clustering_model = AgglomerativeClustering(
                n_clusters=max_topics,
                affinity='cosine',
                linkage='average'
            )
            
            cluster_assignments = await loop.run_in_executor(
                None,
                lambda: clustering_model.fit_predict(embeddings)
            )
            
            # Assign topic IDs to segments
            for i, segment in enumerate(segments):
                segment.topic_id = int(cluster_assignments[i])
            
            return segments
            
        except Exception as e:
            logger.warning("Topic segmentation failed", error=str(e))
            return segments