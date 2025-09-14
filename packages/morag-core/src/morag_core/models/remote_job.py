"""Remote job models for MoRAG."""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from datetime import datetime
import uuid
import json


@dataclass
class RemoteJob:
    """Remote conversion job data model."""

    id: str
    ingestion_task_id: str
    source_file_path: str
    content_type: str
    task_options: Dict[str, Any]
    status: str = 'pending'
    worker_id: Optional[str] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    result_data: Optional[Dict[str, Any]] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout_at: Optional[datetime] = None

    def __post_init__(self):
        """Initialize default values after creation."""
        if self.created_at is None:
            self.created_at = datetime.utcnow()

    @classmethod
    def create_new(cls, ingestion_task_id: str, source_file_path: str,
                   content_type: str, task_options: Dict[str, Any]) -> 'RemoteJob':
        """Create a new remote job with generated ID."""
        return cls(
            id=str(uuid.uuid4()),
            ingestion_task_id=ingestion_task_id,
            source_file_path=source_file_path,
            content_type=content_type,
            task_options=task_options,
            created_at=datetime.utcnow()
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for field in ['created_at', 'started_at', 'completed_at', 'timeout_at']:
            if data[field] is not None:
                data[field] = data[field].isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RemoteJob':
        """Create instance from dictionary (JSON deserialization)."""
        # Convert ISO strings back to datetime objects
        for field in ['created_at', 'started_at', 'completed_at', 'timeout_at']:
            if data.get(field) is not None:
                data[field] = datetime.fromisoformat(data[field])
        return cls(**data)

    def can_retry(self) -> bool:
        """Check if job can be retried."""
        return self.retry_count < self.max_retries and self.status in ['failed', 'timeout']

    @property
    def is_expired(self) -> bool:
        """Check if job has expired."""
        if not self.timeout_at:
            return False
        return datetime.utcnow() > self.timeout_at

    @property
    def processing_duration(self) -> float:
        """Get processing duration in seconds."""
        if not self.started_at:
            return 0.0

        end_time = self.completed_at or datetime.utcnow()
        return (end_time - self.started_at).total_seconds()
