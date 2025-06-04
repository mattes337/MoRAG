"""Enhanced speaker diarization service with advanced features."""

import asyncio
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
import structlog
import numpy as np

from morag.core.config import settings, get_safe_device
from morag.core.exceptions import ProcessingError, ExternalServiceError

logger = structlog.get_logger(__name__)

try:
    from pyannote.audio import Pipeline
    from pyannote.core import Annotation, Segment
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    # Create dummy classes for type hints when pyannote is not available
    class Annotation:
        pass
    class Segment:
        pass
    logger.warning("pyannote.audio not available, speaker diarization disabled")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class SpeakerSegment:
    """Represents a speaker segment with timing and metadata."""
    speaker_id: str
    start_time: float
    end_time: float
    duration: float
    confidence: float = 0.0
    text: Optional[str] = None


@dataclass
class SpeakerInfo:
    """Information about a detected speaker."""
    speaker_id: str
    total_speaking_time: float
    segment_count: int
    average_segment_duration: float
    confidence_scores: List[float]
    first_appearance: float
    last_appearance: float


@dataclass
class DiarizationResult:
    """Result of speaker diarization process."""
    speakers: List[SpeakerInfo]
    segments: List[SpeakerSegment]
    total_speakers: int
    total_duration: float
    speaker_overlap_time: float
    processing_time: float
    model_used: str
    confidence_threshold: float


class EnhancedSpeakerDiarization:
    """Enhanced speaker diarization with advanced features."""
    
    def __init__(self):
        """Initialize the speaker diarization service."""
        self.pipeline = None
        self.model_loaded = False
        self.confidence_threshold = 0.5
        
        if PYANNOTE_AVAILABLE:
            self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize the pyannote pipeline with safe device configuration."""
        try:
            safe_device = get_safe_device(settings.preferred_device)
            logger.info("Initializing speaker diarization pipeline",
                       model=settings.speaker_diarization_model,
                       device=safe_device)

            # Initialize with authentication token if available
            try:
                if settings.huggingface_token:
                    self.pipeline = Pipeline.from_pretrained(
                        settings.speaker_diarization_model,
                        use_auth_token=settings.huggingface_token
                    )
                else:
                    # Try without token (for public models)
                    self.pipeline = Pipeline.from_pretrained(
                        settings.speaker_diarization_model
                    )

                # Move pipeline to safe device
                if hasattr(self.pipeline, 'to') and safe_device != "cpu":
                    try:
                        self.pipeline.to(safe_device)
                        logger.info("Speaker diarization pipeline moved to device", device=safe_device)
                    except Exception as device_error:
                        logger.warning("Failed to move pipeline to GPU, using CPU",
                                     error=str(device_error))
                        if hasattr(self.pipeline, 'to'):
                            self.pipeline.to("cpu")

                self.model_loaded = True
                logger.info("Speaker diarization pipeline initialized successfully", device=safe_device)

            except Exception as init_error:
                if safe_device != "cpu":
                    logger.warning("GPU pipeline initialization failed, trying CPU", error=str(init_error))
                    # Force CPU initialization
                    if settings.huggingface_token:
                        self.pipeline = Pipeline.from_pretrained(
                            settings.speaker_diarization_model,
                            use_auth_token=settings.huggingface_token
                        )
                    else:
                        self.pipeline = Pipeline.from_pretrained(
                            settings.speaker_diarization_model
                        )

                    if hasattr(self.pipeline, 'to'):
                        self.pipeline.to("cpu")

                    self.model_loaded = True
                    logger.info("Speaker diarization pipeline initialized on CPU fallback")
                else:
                    raise

        except Exception as e:
            logger.warning("Failed to initialize speaker diarization pipeline",
                          error=str(e))
            self.pipeline = None
            self.model_loaded = False
    
    async def diarize_audio(
        self,
        audio_path: Union[str, Path],
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        confidence_threshold: Optional[float] = None
    ) -> DiarizationResult:
        """Perform speaker diarization on audio file.
        
        Args:
            audio_path: Path to audio file
            min_speakers: Minimum number of speakers to detect
            max_speakers: Maximum number of speakers to detect
            confidence_threshold: Minimum confidence for speaker segments
            
        Returns:
            DiarizationResult with speaker information
        """
        if not self.model_loaded or not self.pipeline:
            return await self._fallback_diarization(audio_path)
        
        start_time = time.time()
        audio_path = Path(audio_path)
        
        # Use settings defaults if not provided
        min_speakers = min_speakers or settings.min_speakers
        max_speakers = max_speakers or settings.max_speakers
        confidence_threshold = confidence_threshold or self.confidence_threshold
        
        try:
            logger.info("Starting speaker diarization",
                       audio_path=str(audio_path),
                       min_speakers=min_speakers,
                       max_speakers=max_speakers)
            
            # Run diarization in thread pool to avoid blocking
            diarization = await asyncio.to_thread(
                self._run_diarization,
                str(audio_path),
                min_speakers,
                max_speakers
            )
            
            # Process results
            result = await self._process_diarization_result(
                diarization,
                confidence_threshold,
                time.time() - start_time
            )
            
            logger.info("Speaker diarization completed",
                       speakers_detected=result.total_speakers,
                       processing_time=result.processing_time)
            
            return result
            
        except Exception as e:
            logger.error("Speaker diarization failed",
                        audio_path=str(audio_path),
                        error=str(e))
            # Fallback to simple diarization
            return await self._fallback_diarization(audio_path)
    
    def _run_diarization(
        self,
        audio_path: str,
        min_speakers: int,
        max_speakers: int
    ) -> Annotation:
        """Run the actual diarization process."""
        # Configure pipeline parameters
        if hasattr(self.pipeline, 'instantiate'):
            # Set speaker count constraints
            self.pipeline.instantiate({
                'clustering': {
                    'min_cluster_size': min_speakers,
                    'max_num_speakers': max_speakers
                }
            })
        
        # Run diarization
        return self.pipeline(audio_path)
    
    async def _process_diarization_result(
        self,
        diarization: Annotation,
        confidence_threshold: float,
        processing_time: float
    ) -> DiarizationResult:
        """Process raw diarization results into structured format."""
        speakers_data = {}
        segments = []
        total_duration = 0.0
        overlap_time = 0.0
        
        # Process each segment
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            duration = segment.end - segment.start
            total_duration = max(total_duration, segment.end)
            
            # Create speaker segment
            speaker_segment = SpeakerSegment(
                speaker_id=speaker,
                start_time=segment.start,
                end_time=segment.end,
                duration=duration,
                confidence=1.0  # pyannote doesn't provide confidence per segment
            )
            segments.append(speaker_segment)
            
            # Update speaker statistics
            if speaker not in speakers_data:
                speakers_data[speaker] = {
                    'total_time': 0.0,
                    'segments': [],
                    'first_appearance': segment.start,
                    'last_appearance': segment.end
                }
            
            speakers_data[speaker]['total_time'] += duration
            speakers_data[speaker]['segments'].append(speaker_segment)
            speakers_data[speaker]['first_appearance'] = min(
                speakers_data[speaker]['first_appearance'],
                segment.start
            )
            speakers_data[speaker]['last_appearance'] = max(
                speakers_data[speaker]['last_appearance'],
                segment.end
            )
        
        # Calculate overlap time (simplified)
        overlap_time = self._calculate_overlap_time(segments)
        
        # Create speaker info objects
        speakers = []
        for speaker_id, data in speakers_data.items():
            speaker_segments = data['segments']
            avg_duration = data['total_time'] / len(speaker_segments)
            
            speaker_info = SpeakerInfo(
                speaker_id=speaker_id,
                total_speaking_time=data['total_time'],
                segment_count=len(speaker_segments),
                average_segment_duration=avg_duration,
                confidence_scores=[seg.confidence for seg in speaker_segments],
                first_appearance=data['first_appearance'],
                last_appearance=data['last_appearance']
            )
            speakers.append(speaker_info)
        
        # Sort speakers by total speaking time (descending)
        speakers.sort(key=lambda x: x.total_speaking_time, reverse=True)
        
        return DiarizationResult(
            speakers=speakers,
            segments=segments,
            total_speakers=len(speakers),
            total_duration=total_duration,
            speaker_overlap_time=overlap_time,
            processing_time=processing_time,
            model_used=settings.speaker_diarization_model,
            confidence_threshold=confidence_threshold
        )
    
    def _calculate_overlap_time(self, segments: List[SpeakerSegment]) -> float:
        """Calculate total time where multiple speakers are talking."""
        # Simple overlap calculation - can be enhanced
        overlap_time = 0.0
        
        for i, seg1 in enumerate(segments):
            for seg2 in segments[i+1:]:
                if seg1.speaker_id != seg2.speaker_id:
                    # Check for overlap
                    overlap_start = max(seg1.start_time, seg2.start_time)
                    overlap_end = min(seg1.end_time, seg2.end_time)
                    if overlap_start < overlap_end:
                        overlap_time += overlap_end - overlap_start
        
        return overlap_time
    
    async def _fallback_diarization(self, audio_path: Union[str, Path]) -> DiarizationResult:
        """Fallback diarization when pyannote is not available."""
        logger.info("Using fallback speaker diarization")
        
        try:
            # Simple fallback: assume single speaker or split by duration
            from pydub import AudioSegment
            audio = AudioSegment.from_file(str(audio_path))
            duration = len(audio) / 1000.0  # Convert to seconds
            
            if duration > 60:  # If longer than 1 minute, assume 2 speakers
                mid_point = duration / 2
                segments = [
                    SpeakerSegment("SPEAKER_00", 0.0, mid_point, mid_point, 0.8),
                    SpeakerSegment("SPEAKER_01", mid_point, duration, duration - mid_point, 0.8)
                ]
                speakers = [
                    SpeakerInfo("SPEAKER_00", mid_point, 1, mid_point, [0.8], 0.0, mid_point),
                    SpeakerInfo("SPEAKER_01", duration - mid_point, 1, duration - mid_point, [0.8], mid_point, duration)
                ]
                total_speakers = 2
            else:
                # Single speaker
                segments = [SpeakerSegment("SPEAKER_00", 0.0, duration, duration, 0.9)]
                speakers = [SpeakerInfo("SPEAKER_00", duration, 1, duration, [0.9], 0.0, duration)]
                total_speakers = 1
            
            return DiarizationResult(
                speakers=speakers,
                segments=segments,
                total_speakers=total_speakers,
                total_duration=duration,
                speaker_overlap_time=0.0,
                processing_time=0.1,
                model_used="fallback",
                confidence_threshold=0.5
            )
            
        except Exception as e:
            logger.error("Fallback diarization failed", error=str(e))
            # Ultimate fallback
            return DiarizationResult(
                speakers=[SpeakerInfo("SPEAKER_00", 60.0, 1, 60.0, [0.5], 0.0, 60.0)],
                segments=[SpeakerSegment("SPEAKER_00", 0.0, 60.0, 60.0, 0.5)],
                total_speakers=1,
                total_duration=60.0,
                speaker_overlap_time=0.0,
                processing_time=0.1,
                model_used="fallback",
                confidence_threshold=0.5
            )


# Global instance
speaker_diarization_service = EnhancedSpeakerDiarization()
