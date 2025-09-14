"""Base interfaces for tasks."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass


class TaskStatus(str, Enum):
    """Task status enum."""
    PENDING = "pending"
    STARTED = "started"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskProgress:
    """Task progress information."""
    status: TaskStatus
    progress: float  # 0.0 to 1.0
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class TaskResult:
    """Task result information."""
    task_id: str
    status: TaskStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseTask(ABC):
    """Base class for tasks."""
    
    @abstractmethod
    async def execute(self, *args, **kwargs) -> TaskResult:
        """Execute the task.
        
        Returns:
            Task result
        """
        pass
    
    @abstractmethod
    async def cancel(self) -> bool:
        """Cancel the task.
        
        Returns:
            True if cancellation was successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_status(self) -> TaskProgress:
        """Get task status.
        
        Returns:
            Task progress information
        """
        pass
    
    @abstractmethod
    async def update_progress(self, progress: float, message: Optional[str] = None) -> None:
        """Update task progress.
        
        Args:
            progress: Progress value (0.0 to 1.0)
            message: Optional progress message
        """
        pass


class TaskManager(ABC):
    """Interface for task management."""
    
    @abstractmethod
    async def register_task(self, task_id: str, task_type: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Register a new task.
        
        Args:
            task_id: Unique task ID
            task_type: Type of task
            metadata: Optional task metadata
        """
        pass
    
    @abstractmethod
    async def update_task_status(self, task_id: str, status: TaskStatus, message: Optional[str] = None) -> None:
        """Update task status.
        
        Args:
            task_id: Task ID
            status: New task status
            message: Optional status message
        """
        pass
    
    @abstractmethod
    async def update_task_progress(self, task_id: str, progress: float, message: Optional[str] = None) -> None:
        """Update task progress.
        
        Args:
            task_id: Task ID
            progress: Progress value (0.0 to 1.0)
            message: Optional progress message
        """
        pass
    
    @abstractmethod
    async def get_task_status(self, task_id: str) -> TaskProgress:
        """Get task status.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task progress information
        """
        pass
    
    @abstractmethod
    async def complete_task(self, task_id: str, result: Optional[Dict[str, Any]] = None) -> None:
        """Mark task as completed.
        
        Args:
            task_id: Task ID
            result: Optional task result
        """
        pass
    
    @abstractmethod
    async def fail_task(self, task_id: str, error: str) -> None:
        """Mark task as failed.
        
        Args:
            task_id: Task ID
            error: Error message
        """
        pass