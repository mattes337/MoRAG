"""OpenAPI schema definitions for MoRAG UI interoperability endpoints."""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


# Enhanced schema definitions for OpenAPI documentation
class OpenAPIMarkdownConversionRequest(BaseModel):
    """OpenAPI schema for markdown conversion request."""
    file: str = Field(description="Binary file content", format="binary")
    include_metadata: bool = Field(default=True, description="Include file metadata in response")
    language: Optional[str] = Field(default=None, description="Language hint for processing")
    document_id: Optional[str] = Field(default=None, description="Optional document identifier for deduplication")

    class Config:
        schema_extra = {
            "example": {
                "file": "(binary file content)",
                "include_metadata": True,
                "language": "en",
                "document_id": "my-document-123"
            }
        }


class OpenAPIMarkdownConversionResponse(BaseModel):
    """OpenAPI schema for markdown conversion response."""
    success: bool = Field(description="Whether conversion was successful")
    markdown: str = Field(description="Converted markdown content")
    metadata: Dict[str, Any] = Field(description="File metadata and conversion info")
    processing_time_ms: float = Field(description="Processing time in milliseconds")
    error_message: Optional[str] = Field(default=None, description="Error message if conversion failed")

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "markdown": "# Document Title\n\nThis is the converted content...",
                "metadata": {
                    "document_id": "my-document-123",
                    "original_format": "pdf",
                    "file_size_bytes": 1048576,
                    "original_filename": "document.pdf",
                    "processing_time_ms": 1250.5,
                    "language": "en"
                },
                "processing_time_ms": 1250.5
            }
        }


class OpenAPIProcessIngestRequest(BaseModel):
    """OpenAPI schema for processing with ingestion request."""
    file: str = Field(description="Binary file content", format="binary")
    webhook_url: str = Field(description="URL for webhook progress notifications", format="uri")
    document_id: Optional[str] = Field(default=None, description="Optional document ID for deduplication")
    webhook_auth_token: Optional[str] = Field(default=None, description="Optional bearer token for webhook authentication")
    collection_name: Optional[str] = Field(default=None, description="Target collection name")
    language: Optional[str] = Field(default=None, description="Language hint for processing")
    chunking_strategy: Optional[str] = Field(default=None, description="Chunking strategy to use")
    chunk_size: Optional[int] = Field(default=None, description="Chunk size for text splitting")
    chunk_overlap: Optional[int] = Field(default=None, description="Overlap between chunks")
    metadata: Optional[str] = Field(default=None, description="Additional metadata as JSON string")

    class Config:
        schema_extra = {
            "example": {
                "file": "(binary file content)",
                "webhook_url": "https://api.example.com/webhooks/processing",
                "document_id": "my-document-456",
                "webhook_auth_token": "bearer-token-123",
                "collection_name": "documents",
                "language": "en",
                "chunking_strategy": "semantic",
                "chunk_size": 4000,
                "chunk_overlap": 200,
                "metadata": "{\"source\": \"upload\", \"category\": \"research\"}"
            }
        }


class OpenAPIProcessIngestResponse(BaseModel):
    """OpenAPI schema for processing with ingestion response."""
    success: bool = Field(description="Whether processing was started successfully")
    task_id: str = Field(description="Background task ID for tracking progress")
    document_id: Optional[str] = Field(description="Document ID (user-provided or generated)")
    estimated_time_seconds: int = Field(description="Estimated processing time in seconds")
    status_url: str = Field(description="URL to check task status")
    message: str = Field(description="Human-readable status message")

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "task_id": "task-abc-123-def",
                "document_id": "my-document-456",
                "estimated_time_seconds": 180,
                "status_url": "/api/v1/status/task-abc-123-def",
                "message": "Processing started for 'document.pdf'. Webhook notifications will be sent to https://api.example.com/webhooks/processing"
            }
        }


class OpenAPIWebhookProgressNotification(BaseModel):
    """OpenAPI schema for webhook progress notifications."""
    task_id: str = Field(description="Background task ID")
    document_id: Optional[str] = Field(description="Document ID if provided")
    step: str = Field(description="Current processing step")
    status: str = Field(description="Step status: started|completed|failed")
    progress_percent: float = Field(description="Overall progress percentage (0-100)")
    timestamp: str = Field(description="ISO8601 timestamp")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Step-specific data")
    error_message: Optional[str] = Field(default=None, description="Error message if status is failed")

    class Config:
        schema_extra = {
            "example": {
                "task_id": "task-abc-123-def",
                "document_id": "my-document-456",
                "step": "markdown_conversion",
                "status": "completed",
                "progress_percent": 25.0,
                "timestamp": "2024-01-15T10:30:00Z",
                "data": {
                    "markdown_file_url": "/api/files/temp/task-abc-123-def/markdown.md",
                    "conversion_metadata": {
                        "original_format": "pdf",
                        "file_size_bytes": 1048576,
                        "content_length": 52428
                    }
                }
            }
        }


class OpenAPISessionFileInfo(BaseModel):
    """OpenAPI schema for session file information."""
    filename: str = Field(description="Name of the file")
    size_bytes: int = Field(description="File size in bytes")
    content_type: str = Field(description="MIME type of the file")
    created_at: str = Field(description="ISO8601 creation timestamp")
    modified_at: Optional[str] = Field(description="ISO8601 modification timestamp")

    class Config:
        schema_extra = {
            "example": {
                "filename": "markdown.md",
                "size_bytes": 52428,
                "content_type": "text/markdown",
                "created_at": "2024-01-15T10:30:00Z",
                "modified_at": "2024-01-15T10:30:00Z"
            }
        }


class OpenAPISessionFilesResponse(BaseModel):
    """OpenAPI schema for session files listing response."""
    session_id: str = Field(description="Session identifier")
    files: List[OpenAPISessionFileInfo] = Field(description="List of files in the session")
    total_size_bytes: int = Field(description="Total size of all files in bytes")
    expires_at: Optional[str] = Field(description="ISO8601 expiration timestamp")

    class Config:
        schema_extra = {
            "example": {
                "session_id": "task-abc-123-def",
                "files": [
                    {
                        "filename": "original.pdf",
                        "size_bytes": 1048576,
                        "content_type": "application/pdf",
                        "created_at": "2024-01-15T10:30:00Z"
                    },
                    {
                        "filename": "markdown.md",
                        "size_bytes": 52428,
                        "content_type": "text/markdown",
                        "created_at": "2024-01-15T10:31:00Z"
                    }
                ],
                "total_size_bytes": 1100004,
                "expires_at": "2024-01-16T10:30:00Z"
            }
        }


class OpenAPIErrorResponse(BaseModel):
    """OpenAPI schema for error responses."""
    detail: str = Field(description="Error message or detailed error information")

    class Config:
        schema_extra = {
            "example": {
                "detail": "Invalid document ID format: invalid<id>"
            }
        }


class OpenAPIDuplicateDocumentError(BaseModel):
    """OpenAPI schema for duplicate document error responses."""
    error: str = Field(description="Error type identifier")
    message: str = Field(description="Human-readable error message")
    existing_document: Dict[str, Any] = Field(description="Information about the existing document")
    options: Dict[str, str] = Field(description="Available options for handling the duplicate")

    class Config:
        schema_extra = {
            "example": {
                "error": "duplicate_document",
                "message": "Document with ID 'my-document-123' already exists",
                "existing_document": {
                    "document_id": "my-document-123",
                    "created_at": "2024-01-15T09:00:00Z",
                    "status": "completed",
                    "facts_count": 25,
                    "keywords_count": 15,
                    "chunks_count": 8
                },
                "options": {
                    "update_url": "/api/process/update/my-document-123",
                    "version_url": "/api/process/version/my-document-123",
                    "delete_url": "/api/process/delete/my-document-123"
                }
            }
        }


class OpenAPICleanupResponse(BaseModel):
    """OpenAPI schema for cleanup operation response."""
    success: bool = Field(description="Whether cleanup was successful")
    cleaned_sessions: int = Field(description="Number of sessions cleaned up")
    message: str = Field(description="Human-readable status message")

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "cleaned_sessions": 3,
                "message": "Cleaned up 3 expired sessions"
            }
        }


class OpenAPIDeleteSessionResponse(BaseModel):
    """OpenAPI schema for session deletion response."""
    success: bool = Field(description="Whether deletion was successful")
    message: str = Field(description="Human-readable status message")

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Session task-abc-123-def deleted successfully"
            }
        }


# OpenAPI tags for endpoint organization
OPENAPI_TAGS = [
    {
        "name": "Conversion",
        "description": "File conversion endpoints for UI preview functionality"
    },
    {
        "name": "Processing",
        "description": "Complete document processing with webhook notifications"
    },
    {
        "name": "Temporary Files",
        "description": "Temporary file management for intermediate processing results"
    },
    {
        "name": "Deduplication",
        "description": "Document deduplication and ID management"
    }
]


# OpenAPI server configuration
OPENAPI_SERVERS = [
    {
        "url": "http://localhost:8000",
        "description": "Development server"
    },
    {
        "url": "https://api.morag.example.com",
        "description": "Production server"
    }
]


# Security schemes for OpenAPI
OPENAPI_SECURITY_SCHEMES = {
    "BearerAuth": {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
        "description": "JWT token for API authentication"
    }
}
