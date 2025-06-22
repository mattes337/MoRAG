"""Utility functions for MoRAG."""

from .device import (
    get_safe_device,
    get_device_info,
    clear_gpu_memory,
    get_optimal_batch_size,
)
from .file_handling import (
    ensure_directory,
    get_file_hash,
    get_file_size,
    is_file_readable,
    cleanup_temp_files,
    parse_size_string,
)
from .logging import (
    setup_logging,
    get_logger,
)
from .validation import (
    validate_file_path,
    validate_url,
    validate_email,
    sanitize_filename,
)
from .processing import (
    ContentType,
    ProcessingMode,
    ProcessingMetadata,
    create_processing_metadata,
    get_output_paths,
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
]