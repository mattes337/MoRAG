"""MoRAG API module."""

# MoRAGAPI import moved to avoid circular import
# from ..api import MoRAGAPI

from .models import (
    ProcessURLRequest,
    ProcessBatchRequest,
    SearchRequest,
    ProcessingResultResponse,
    IngestFileRequest,
    IngestURLRequest,
    IngestBatchRequest,
    IngestRemoteFileRequest,
    ProcessRemoteFileRequest,
    IngestResponse,
    BatchIngestResponse,
    TaskStatus,
)

from .utils import (
    download_remote_file,
    normalize_content_type,
    normalize_processing_result,
    encode_thumbnails_to_base64,
)

__all__ = [
    # Models
    "ProcessURLRequest",
    "ProcessBatchRequest", 
    "SearchRequest",
    "ProcessingResultResponse",
    "IngestFileRequest",
    "IngestURLRequest",
    "IngestBatchRequest",
    "IngestRemoteFileRequest",
    "ProcessRemoteFileRequest",
    "IngestResponse",
    "BatchIngestResponse",
    "TaskStatus",
    # Utils
    "download_remote_file",
    "normalize_content_type",
    "normalize_processing_result",
    "encode_thumbnails_to_base64",
]
