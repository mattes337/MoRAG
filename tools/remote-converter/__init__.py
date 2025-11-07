"""
MoRAG Remote Converter Package

A standalone remote worker application for MoRAG that processes jobs on remote machines.
"""

__version__ = "1.0.0"
__author__ = "MoRAG Team"
__description__ = "Remote converter for MoRAG multimodal processing"

from .remote_converter import RemoteConverter
from .config import RemoteConverterConfig, setup_logging

__all__ = [
    'RemoteConverter',
    'RemoteConverterConfig',
    'setup_logging'
]
