"""MoRAG Video Processing Package.

This package provides video processing capabilities for the MoRAG (Modular Retrieval Augmented Generation) system.
"""

from morag_video.processor import VideoProcessor, VideoConfig, VideoProcessingError
from morag_video.service import VideoService
from morag_video.converters import VideoConverter, VideoConversionOptions

__all__ = [
    "VideoProcessor",
    "VideoConfig",
    "VideoProcessingError",
    "VideoService",
    "VideoConverter",
    "VideoConversionOptions",
]

__version__ = "0.1.0"