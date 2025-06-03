"""Audio processing with Faster Whisper for speech-to-text conversion."""

import os
import tempfile
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
import asyncio
import structlog
import time

from faster_whisper import WhisperModel
from pydub import AudioSegment as PydubAudioSegment
import librosa
import mutagen
from mutagen.id3 import ID3NoHeaderError

from morag.core.config import settings
from morag.core.exceptions import ProcessingError, ExternalServiceError
from morag.services.speaker_diarization import speaker_diarization_service, DiarizationResult
from morag.services.topic_segmentation import topic_segmentation_service, TopicSegmentationResult

logger = structlog.get_logger()

@dataclass
class AudioConfig:
    """Configuration for audio processing."""
    model_size: str = "base"  # tiny, base, small, medium, large
    language: Optional[str] = None  # Auto-detect if None
    enable_diarization: bool = False
    chunk_duration: int = 300  # 5 minutes in seconds
    overlap_duration: int = 30  # 30 seconds overlap
    quality_threshold: float = 0.7
    max_file_size: int = 500 * 1024 * 1024  # 500MB
    supported_formats: List[str] = field(default_factory=lambda: [
        "mp3", "wav", "m4a", "flac", "aac", "ogg", "wma"
    ])
    device: str = "cpu"  # cpu or cuda
    compute_type: str = "int8"  # int8, int16, float16, float32

@dataclass
class AudioTranscriptSegment:
    """Represents a segment of processed audio."""
    text: str
    start_time: float
    end_time: float
    confidence: float
    speaker_id: Optional[str] = None
    language: Optional[str] = None

@dataclass
class AudioProcessingResult:
    """Result of audio processing."""
    text: str
    language: str
    confidence: float
    duration: float
    segments: List[AudioTranscriptSegment]
    metadata: Dict[str, Any]
    processing_time: float
    model_used: str
    # Enhanced features
    speaker_diarization: Optional[DiarizationResult] = None
    topic_segmentation: Optional[TopicSegmentationResult] = None

class AudioProcessor:
    """Audio processor using Faster Whisper for speech-to-text."""
    
    def __init__(self, config: Optional[AudioConfig] = None):
        """Initialize audio processor."""
        self.config = config or AudioConfig()
        self._model: Optional[WhisperModel] = None
        self._model_loaded = False
        self._ffmpeg_available = None  # Lazy check on first use

        logger.info("Initialized AudioProcessor",
                   model_size=self.config.model_size,
                   device=self.config.device)

    def _check_ffmpeg_availability(self) -> bool:
        """Check if FFmpeg is available on the system."""
        try:
            import subprocess
            import shutil

            # First try to find ffmpeg in PATH using shutil.which (faster)
            if shutil.which('ffmpeg') is None:
                return False

            # If found in PATH, try to run ffmpeg -version with a short timeout
            result = subprocess.run(['ffmpeg', '-version'],
                                  capture_output=True,
                                  text=True,
                                  timeout=2,
                                  creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, OSError):
            return False
        except Exception:
            # Catch any other unexpected errors and default to False
            return False
    
    def _load_model(self) -> WhisperModel:
        """Load Whisper model lazily."""
        if not self._model_loaded:
            try:
                logger.info("Loading Whisper model", 
                           model_size=self.config.model_size,
                           device=self.config.device)
                
                self._model = WhisperModel(
                    self.config.model_size,
                    device=self.config.device,
                    compute_type=self.config.compute_type
                )
                self._model_loaded = True
                
                logger.info("Whisper model loaded successfully")
                
            except Exception as e:
                logger.error("Failed to load Whisper model", error=str(e))
                raise ExternalServiceError(f"Failed to load Whisper model: {str(e)}", "whisper")
        
        return self._model
    
    async def process_audio_file(
        self,
        file_path: Union[str, Path],
        config: Optional[AudioConfig] = None,
        enable_diarization: Optional[bool] = None,
        enable_topic_segmentation: Optional[bool] = None
    ) -> AudioProcessingResult:
        """Process audio file and extract text with optional enhanced features."""
        start_time = time.time()
        file_path = Path(file_path)
        config = config or self.config

        # Use settings defaults if not provided
        enable_diarization = enable_diarization if enable_diarization is not None else settings.enable_speaker_diarization
        enable_topic_segmentation = enable_topic_segmentation if enable_topic_segmentation is not None else settings.enable_topic_segmentation

        try:
            # Validate file first
            self._validate_audio_file(file_path, config)

            logger.info("Processing audio file",
                       file_path=str(file_path),
                       file_size=file_path.stat().st_size,
                       enable_diarization=enable_diarization,
                       enable_topic_segmentation=enable_topic_segmentation)

            # Extract metadata
            metadata = await self._extract_metadata(file_path)

            # Convert to WAV if needed
            wav_path = await self._convert_to_wav(file_path)

            try:
                # Process with Whisper
                result = await self._transcribe_audio(wav_path, config)

                # Calculate overall confidence
                overall_confidence = sum(seg.confidence for seg in result.segments) / len(result.segments) if result.segments else 0.0

                # Enhanced processing
                speaker_diarization_result = None
                topic_segmentation_result = None

                # Perform speaker diarization if enabled
                if enable_diarization:
                    try:
                        speaker_diarization_result = await speaker_diarization_service.diarize_audio(wav_path)
                        logger.info("Speaker diarization completed",
                                   speakers_detected=speaker_diarization_result.total_speakers)
                    except Exception as e:
                        logger.warning("Speaker diarization failed", error=str(e))

                # Perform topic segmentation if enabled
                if enable_topic_segmentation and result.text:
                    try:
                        # Pass speaker segments for better topic boundaries
                        speaker_segments = speaker_diarization_result.segments if speaker_diarization_result else None
                        topic_segmentation_result = await topic_segmentation_service.segment_topics(
                            result.text,
                            speaker_segments=speaker_segments,
                            transcript_segments=result.segments
                        )
                        logger.info("Topic segmentation completed",
                                   topics_detected=topic_segmentation_result.total_topics)
                    except Exception as e:
                        logger.warning("Topic segmentation failed", error=str(e))

                processing_time = time.time() - start_time

                return AudioProcessingResult(
                    text=result.text,
                    language=result.language,
                    confidence=overall_confidence,
                    duration=metadata.get("duration", 0.0),
                    segments=result.segments,
                    metadata=metadata,
                    processing_time=processing_time,
                    model_used=config.model_size,
                    speaker_diarization=speaker_diarization_result,
                    topic_segmentation=topic_segmentation_result
                )
                
            finally:
                # Clean up temporary WAV file if created
                if wav_path != file_path and wav_path.exists():
                    wav_path.unlink()
                    
        except Exception as e:
            logger.error("Audio processing failed", 
                        file_path=str(file_path),
                        error=str(e))
            raise ProcessingError(f"Audio processing failed: {str(e)}")
    
    def _validate_audio_file(self, file_path: Path, config: AudioConfig) -> None:
        """Validate audio file."""
        if not file_path.exists():
            raise ProcessingError(f"Audio file not found: {file_path}")
        
        if file_path.stat().st_size > config.max_file_size:
            raise ProcessingError(f"Audio file too large: {file_path.stat().st_size} bytes")
        
        file_extension = file_path.suffix.lower().lstrip('.')
        if file_extension not in config.supported_formats:
            raise ProcessingError(f"Unsupported audio format: {file_extension}")
    
    async def _extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from audio file."""
        try:
            # Use mutagen for metadata
            try:
                audio_file = mutagen.File(file_path)
                metadata = {}
                
                if audio_file is not None:
                    # Basic metadata
                    metadata.update({
                        "title": audio_file.get("TIT2", [None])[0] if "TIT2" in audio_file else None,
                        "artist": audio_file.get("TPE1", [None])[0] if "TPE1" in audio_file else None,
                        "album": audio_file.get("TALB", [None])[0] if "TALB" in audio_file else None,
                        "duration": getattr(audio_file.info, "length", 0.0),
                        "bitrate": getattr(audio_file.info, "bitrate", 0),
                        "sample_rate": getattr(audio_file.info, "sample_rate", 0),
                        "channels": getattr(audio_file.info, "channels", 0),
                    })
                
            except (ID3NoHeaderError, Exception):
                # Fallback to librosa for basic info
                try:
                    y, sr = librosa.load(file_path, sr=None)
                    duration = len(y) / sr
                    metadata = {
                        "duration": duration,
                        "sample_rate": sr,
                        "channels": 1 if len(y.shape) == 1 else y.shape[0],
                    }
                except Exception:
                    metadata = {"duration": 0.0}
            
            # Add file info
            metadata.update({
                "file_size": file_path.stat().st_size,
                "file_format": file_path.suffix.lower().lstrip('.'),
                "file_name": file_path.name,
            })
            
            return metadata
            
        except Exception as e:
            logger.warning("Failed to extract audio metadata", 
                          file_path=str(file_path),
                          error=str(e))
            return {"file_name": file_path.name, "duration": 0.0}
    
    async def _convert_to_wav(self, file_path: Path) -> Path:
        """Convert audio file to WAV format if needed."""
        if file_path.suffix.lower() == '.wav':
            return file_path

        try:
            logger.debug("Converting audio to WAV", file_path=str(file_path))

            # Try pydub first (requires FFmpeg for most formats)
            try:
                audio = PydubAudioSegment.from_file(str(file_path))

                # Create temporary WAV file
                temp_dir = Path(tempfile.gettempdir())
                wav_path = temp_dir / f"morag_audio_{int(time.time())}_{file_path.stem}.wav"

                # Export as WAV
                audio.export(str(wav_path), format="wav")

                logger.debug("Audio converted to WAV using pydub",
                            original=str(file_path),
                            converted=str(wav_path))

                return wav_path

            except Exception as pydub_error:
                # Check FFmpeg availability lazily if not already checked
                if self._ffmpeg_available is None:
                    self._ffmpeg_available = self._check_ffmpeg_availability()

                # Check if this is an FFmpeg-related error
                error_str = str(pydub_error).lower()
                if any(keyword in error_str for keyword in ['ffmpeg', 'ffprobe', 'avconv', 'avprobe', 'file specified']):
                    if not self._ffmpeg_available:
                        logger.warning("FFmpeg not available, trying librosa fallback",
                                     file_path=str(file_path),
                                     pydub_error=str(pydub_error))
                    else:
                        logger.warning("FFmpeg error occurred, trying librosa fallback",
                                     file_path=str(file_path),
                                     pydub_error=str(pydub_error))
                else:
                    logger.warning("Pydub conversion failed, trying librosa fallback",
                                 file_path=str(file_path),
                                 pydub_error=str(pydub_error))

                # Fallback to librosa for audio loading
                return await self._convert_to_wav_with_librosa(file_path)

        except Exception as e:
            logger.error("Audio conversion failed",
                        file_path=str(file_path),
                        error=str(e))
            raise ProcessingError(f"Audio conversion failed: {str(e)}")

    async def _convert_to_wav_with_librosa(self, file_path: Path) -> Path:
        """Convert audio file to WAV using librosa as fallback."""
        try:
            import soundfile as sf

            logger.debug("Converting audio to WAV using librosa fallback", file_path=str(file_path))

            # Load audio with librosa
            y, sr = await asyncio.to_thread(librosa.load, str(file_path), sr=None)

            # Create temporary WAV file
            temp_dir = Path(tempfile.gettempdir())
            wav_path = temp_dir / f"morag_audio_{int(time.time())}_{file_path.stem}.wav"

            # Save as WAV using soundfile
            await asyncio.to_thread(sf.write, str(wav_path), y, sr)

            logger.debug("Audio converted to WAV using librosa",
                        original=str(file_path),
                        converted=str(wav_path))

            return wav_path

        except ImportError:
            logger.error("soundfile not available for librosa fallback")
            error_msg = (
                "Audio conversion failed: FFmpeg not available and soundfile not installed. "
                "Please install FFmpeg or run 'pip install soundfile' to enable audio conversion."
            )
            raise ProcessingError(error_msg)
        except Exception as e:
            logger.error("Librosa audio conversion failed",
                        file_path=str(file_path),
                        error=str(e))
            error_msg = (
                f"Audio conversion failed with librosa: {str(e)}. "
                "Consider installing FFmpeg for better audio format support."
            )
            raise ProcessingError(error_msg)
    
    async def _transcribe_audio(
        self,
        audio_path: Path,
        config: AudioConfig
    ) -> AudioProcessingResult:
        """Transcribe audio using Whisper."""
        try:
            # Load model
            model = self._load_model()
            
            # Run transcription in thread pool to avoid blocking
            segments, info = await asyncio.to_thread(
                model.transcribe,
                str(audio_path),
                language=config.language,
                beam_size=5,
                best_of=5,
                temperature=0.0
            )
            
            # Convert segments to our format
            audio_segments = []
            full_text_parts = []

            for segment in segments:
                audio_segment = AudioTranscriptSegment(
                    text=segment.text.strip(),
                    start_time=segment.start,
                    end_time=segment.end,
                    confidence=segment.avg_logprob,  # Use avg_logprob as confidence
                    language=info.language
                )
                audio_segments.append(audio_segment)
                full_text_parts.append(segment.text.strip())
            
            full_text = " ".join(full_text_parts)
            
            return AudioProcessingResult(
                text=full_text,
                language=info.language,
                confidence=0.0,  # Will be calculated later
                duration=info.duration,
                segments=audio_segments,
                metadata={},
                processing_time=0.0,  # Will be set later
                model_used=config.model_size
            )
            
        except Exception as e:
            logger.error("Audio transcription failed", 
                        audio_path=str(audio_path),
                        error=str(e))
            raise ExternalServiceError(f"Audio transcription failed: {str(e)}", "whisper")

# Global instance
audio_processor = AudioProcessor()
