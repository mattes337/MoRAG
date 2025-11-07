"""MoRAG Video Processing Package.

This package provides video processing capabilities for the MoRAG (Modular Retrieval Augmented Generation) system.
"""

from morag_video.converters import VideoConversionOptions, VideoConverter
from morag_video.processor import VideoConfig, VideoProcessingError, VideoProcessor
from morag_video.service import VideoService

__all__ = [
    "VideoProcessor",
    "VideoConfig",
    "VideoProcessingError",
    "VideoService",
    "VideoConverter",
    "VideoConversionOptions",
]

__version__ = "0.1.0"
