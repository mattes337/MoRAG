"""API models for MoRAG."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class SourceType(str, Enum):
    """Source type enum."""

    DOCUMENT = "document"
    AUDIO = "audio"
    VIDEO = "video"
    IMAGE = "image"
    URL = "url"
    TEXT = "text"


@dataclass
class ErrorResponse:
    """API error response."""

    error: str
    error_type: Optional[str] = None
    status_code: int = 400
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        result = {
            "error": self.error,
            "status_code": self.status_code,
        }

        if self.error_type:
            result["error_type"] = self.error_type

        if self.details:
            result["details"] = self.details

        return result


@dataclass
class TaskStatusResponse:
    """Task status response."""

    task_id: str
    status: str
    progress: float = 0.0  # 0.0 to 1.0
    message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        result = {
            "task_id": self.task_id,
            "status": self.status,
            "progress": self.progress,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

        if self.message:
            result["message"] = self.message

        if self.result:
            result["result"] = self.result

        if self.error:
            result["error"] = self.error

        return result


@dataclass
class IngestionResponse:
    """Ingestion response."""

    task_id: str
    status: str = "pending"
    message: str = "Ingestion task created"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "task_id": self.task_id,
            "status": self.status,
            "message": self.message,
        }


@dataclass
class BatchIngestionResponse:
    """Batch ingestion response."""

    task_ids: List[str]
    status: str = "pending"
    message: str = "Batch ingestion tasks created"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "task_ids": self.task_ids,
            "status": self.status,
            "message": self.message,
        }
