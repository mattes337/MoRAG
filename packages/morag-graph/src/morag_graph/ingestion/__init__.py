"""Ingestion module for morag-graph.

This module provides functionality for ingesting files and preventing duplicates
using checksums and other metadata.
"""

from .file_ingestion import FileIngestion, FileMetadata

__all__ = ['FileIngestion', 'FileMetadata']