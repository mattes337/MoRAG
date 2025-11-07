"""Utility functions for MoRAG."""

from .deduplication import Deduplicator
from .device import (
    clear_gpu_memory,
    get_device_info,
    get_optimal_batch_size,
    get_safe_device,
)
from .file_handling import (
    cleanup_temp_files,
    ensure_directory,
    get_file_hash,
    get_file_size,
    is_file_readable,
    parse_size_string,
)
from .logging import get_logger, setup_logging
from .processing import (
    ContentType,
    ProcessingMetadata,
    ProcessingMode,
    create_processing_metadata,
    get_output_paths,
)
from .validation import (
    sanitize_filename,
    validate_email,
    validate_file_path,
    validate_url,
)

__all__ = [
    # Device utilities
    "get_safe_device",
    "get_device_info",
    "clear_gpu_memory",
    "get_optimal_batch_size",
    # File handling
    "ensure_directory",
    "get_file_hash",
    "get_file_size",
    "is_file_readable",
    "cleanup_temp_files",
    "parse_size_string",
    # Logging
    "setup_logging",
    "get_logger",
    # Validation
    "validate_file_path",
    "validate_url",
    "validate_email",
    "sanitize_filename",
    # Processing
    "ContentType",
    "ProcessingMode",
    "ProcessingMetadata",
    "create_processing_metadata",
    "get_output_paths",
    # Deduplication
    "Deduplicator",
]
