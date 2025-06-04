"""MoRAG Audio Processing Package.

This package provides audio processing capabilities for the MoRAG (Modular Retrieval Augmented Generation) system.
"""

from morag_audio.processor import AudioProcessor, AudioConfig, AudioProcessingError
from morag_audio.service import AudioService
from morag_audio.converters import AudioConverter, AudioConversionOptions

__all__ = [
    "AudioProcessor",
    "AudioConfig",
    "AudioProcessingError",
    "AudioService",
    "AudioConverter",
    "AudioConversionOptions",
]

__version__ = "0.1.0"