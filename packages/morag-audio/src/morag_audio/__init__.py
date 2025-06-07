"""MoRAG Audio Processing Package.

This package provides audio processing capabilities for the MoRAG (Modular Retrieval Augmented Generation) system.
"""

from morag_audio.processor import AudioProcessor, AudioConfig, AudioProcessingError
from morag_audio.service import AudioService
from morag_audio.converters import AudioConverter, AudioConversionOptions
from morag_audio.services import (
    SpeakerDiarizationService, SpeakerSegment, SpeakerInfo, DiarizationResult,
    TopicSegmentationService, TopicSegment, TopicSegmentationResult
)

__all__ = [
    "AudioProcessor",
    "AudioConfig",
    "AudioProcessingError",
    "AudioService",
    "AudioConverter",
    "AudioConversionOptions",
    # Speaker diarization
    "SpeakerDiarizationService",
    "SpeakerSegment",
    "SpeakerInfo",
    "DiarizationResult",
    # Topic segmentation
    "TopicSegmentationService",
    "TopicSegment",
    "TopicSegmentationResult",
]

__version__ = "0.1.0"