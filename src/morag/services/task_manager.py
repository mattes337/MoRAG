from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import structlog

from celery.result import AsyncResult

# Mock celery app for now
class MockCeleryApp:
    def __init__(self):
        self.control = MockControl()

class MockControl:
    def revoke(self, task_id, terminate=True):
        pass

    def inspect(self):
        return MockInspect()

class MockInspect:
    def active(self):
        return {}

    def scheduled(self):
        return {}

    def reserved(self):
        return {}

celery_app = MockCeleryApp()

logger = structlog.get_logger()

class TaskStatus(Enum):
    """Task status enumeration."""
    PENDING = "pending"
    STARTED = "started"
    PROGRESS = "progress"
    SUCCESS = "success"
    FAILURE = "failure"
    RETRY = "retry"
    REVOKED = "revoked"

@dataclass
class TaskInfo:
    """Task information structure."""
    task_id: str
    status: TaskStatus
    progress: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

class TaskManager:
    """Manages task lifecycle and status tracking."""
    
    def __init__(self):
        self.celery_app = celery_app
    
    def get_task_status(self, task_id: str) -> TaskInfo:
        """Get detailed task status information."""
        try:
            result = AsyncResult(task_id, app=self.celery_app)
            
            # Get basic status
            status = TaskStatus(result.status.lower())
            
            # Initialize task info
            task_info = TaskInfo(
                task_id=task_id,
                status=status
            )
            
            # Handle different status types
            if status == TaskStatus.PENDING:
                task_info.progress = 0.0
                
            elif status == TaskStatus.STARTED:
                task_info.progress = 0.1
                if hasattr(result.info, 'started_at'):
                    task_info.started_at = result.info.get('started_at')
                    
            elif status == TaskStatus.PROGRESS:
                if isinstance(result.info, dict):
                    task_info.progress = result.info.get('progress', 0.0)
                    task_info.metadata = result.info.get('metadata', {})
                    
            elif status == TaskStatus.SUCCESS:
                task_info.progress = 1.0
                task_info.result = result.result
                task_info.completed_at = datetime.utcnow()
                
            elif status == TaskStatus.FAILURE:
                task_info.error = str(result.info)
                task_info.completed_at = datetime.utcnow()
                
            elif status == TaskStatus.RETRY:
                if isinstance(result.info, dict):
                    task_info.error = result.info.get('error')
                    task_info.metadata = result.info.get('metadata', {})
            
            return task_info
            
        except Exception as e:
            logger.error("Failed to get task status", task_id=task_id, error=str(e))
            return TaskInfo(
                task_id=task_id,
                status=TaskStatus.FAILURE,
                error=f"Failed to retrieve task status: {str(e)}"
            )
    
    def update_task_progress(
        self,
        task_id: str,
        progress: float,
        message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update task progress."""
        try:
            # For now, just log the progress update
            # The actual state update should be done from within the task itself
            progress_info = {
                'progress': progress,
                'message': message,
                'metadata': metadata or {},
                'updated_at': datetime.utcnow().isoformat()
            }

            logger.debug(
                "Task progress updated",
                task_id=task_id,
                progress=progress,
                message=message
            )

            # Add to status history
            from src.morag.services.status_history import status_history
            status_history.add_status_event(
                task_id=task_id,
                status='progress',
                progress=progress,
                message=message,
                metadata=metadata
            )

            logger.debug(
                "Task progress updated",
                task_id=task_id,
                progress=progress,
                message=message
            )

        except Exception as e:
            logger.error(
                "Failed to update task progress",
                task_id=task_id,
                error=str(e)
            )

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        try:
            self.celery_app.control.revoke(task_id, terminate=True)
            logger.info("Task cancelled", task_id=task_id)
            return True

        except Exception as e:
            logger.error("Failed to cancel task", task_id=task_id, error=str(e))
            return False

    def get_active_tasks(self) -> List[Dict[str, Any]]:
        """Get list of currently active tasks."""
        try:
            inspect = self.celery_app.control.inspect()
            active_tasks = inspect.active()

            if not active_tasks:
                return []

            # Flatten tasks from all workers
            all_tasks = []
            for worker, tasks in active_tasks.items():
                for task in tasks:
                    task['worker'] = worker
                    all_tasks.append(task)

            return all_tasks

        except Exception as e:
            logger.error("Failed to get active tasks", error=str(e))
            return []

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        try:
            inspect = self.celery_app.control.inspect()

            # Get queue lengths
            active = inspect.active() or {}
            scheduled = inspect.scheduled() or {}
            reserved = inspect.reserved() or {}

            stats = {
                'active_tasks': sum(len(tasks) for tasks in active.values()),
                'scheduled_tasks': sum(len(tasks) for tasks in scheduled.values()),
                'reserved_tasks': sum(len(tasks) for tasks in reserved.values()),
                'workers': list(active.keys()) if active else [],
                'queues': {
                    'document_processing': 0,
                    'audio_processing': 0,
                    'video_processing': 0,
                    'web_processing': 0,
                }
            }

            return stats

        except Exception as e:
            logger.error("Failed to get queue stats", error=str(e))
            return {'error': str(e)}

    async def handle_task_completion(self, task_id: str, result: Dict[str, Any], metadata: Dict[str, Any]):
        """Handle task completion and send webhooks."""
        webhook_url = metadata.get('webhook_url')

        # Add to status history
        from src.morag.services.status_history import status_history
        status_history.add_status_event(
            task_id=task_id,
            status='completed' if result.get('status') != 'failure' else 'failed',
            progress=1.0 if result.get('status') != 'failure' else None,
            message="Task completed successfully" if result.get('status') != 'failure' else f"Task failed: {result.get('error', 'Unknown error')}",
            metadata=metadata
        )

        if webhook_url:
            try:
                from morag.services.webhook import webhook_service
                if result.get('status') != 'failure':
                    await webhook_service.send_task_completed(
                        task_id=task_id,
                        webhook_url=webhook_url,
                        result=result,
                        metadata=metadata
                    )
                else:
                    error = result.get('error', 'Unknown error')
                    await webhook_service.send_task_failed(
                        task_id=task_id,
                        webhook_url=webhook_url,
                        error=error,
                        metadata=metadata
                    )
            except ImportError:
                logger.warning("Webhook service not available")

    async def handle_task_started(self, task_id: str, metadata: Dict[str, Any]):
        """Handle task start and send webhooks."""
        webhook_url = metadata.get('webhook_url')

        # Add to status history
        from src.morag.services.status_history import status_history
        status_history.add_status_event(
            task_id=task_id,
            status='started',
            progress=0.0,
            message="Task started",
            metadata=metadata
        )

        if webhook_url:
            try:
                from morag.services.webhook import webhook_service
                await webhook_service.send_task_started(
                    task_id=task_id,
                    webhook_url=webhook_url,
                    metadata=metadata
                )
            except ImportError:
                logger.warning("Webhook service not available")

    async def handle_task_progress(
        self,
        task_id: str,
        progress: float,
        message: str,
        metadata: Dict[str, Any]
    ):
        """Handle task progress and send webhooks if significant."""
        webhook_url = metadata.get('webhook_url')

        # Only send webhook for significant progress milestones
        if webhook_url and progress in [0.25, 0.5, 0.75]:
            try:
                from morag.services.webhook import webhook_service
                await webhook_service.send_task_progress(
                    task_id=task_id,
                    webhook_url=webhook_url,
                    progress=progress,
                    message=message,
                    metadata=metadata
                )
            except ImportError:
                logger.warning("Webhook service not available")

# Global instance
task_manager = TaskManager()
