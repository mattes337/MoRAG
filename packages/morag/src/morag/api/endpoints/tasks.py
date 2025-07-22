"""Task management endpoints for MoRAG API."""

import structlog
from fastapi import APIRouter, HTTPException

from morag.api.models import TaskStatus
from morag.worker import celery_app

logger = structlog.get_logger(__name__)

tasks_router = APIRouter(prefix="/api/v1/status", tags=["Task Management"])


def setup_task_endpoints(morag_api_getter):
    """Setup task management endpoints with MoRAG API getter function."""
    
    @tasks_router.get("/{task_id}", response_model=TaskStatus)
    async def get_task_status(task_id: str):
        """Get the status of a processing task."""
        try:
            # Get task result from Celery
            task_result = celery_app.AsyncResult(task_id)

            # Map Celery states to our status format
            status_mapping = {
                'PENDING': 'pending',
                'STARTED': 'running',
                'SUCCESS': 'completed',
                'FAILURE': 'failed',
                'RETRY': 'running',
                'REVOKED': 'cancelled'
            }

            status = status_mapping.get(task_result.state, 'unknown')
            
            # Get task info
            task_info = task_result.info or {}
            
            return TaskStatus(
                task_id=task_id,
                status=status,
                progress=task_info.get('progress', 0.0),
                message=task_info.get('message', f"Task is {status}"),
                result=task_result.result if status == 'completed' else None,
                error=str(task_result.info) if status == 'failed' else None,
                created_at=task_info.get('created_at'),
                started_at=task_info.get('started_at'),
                completed_at=task_info.get('completed_at'),
                estimated_time_remaining=task_info.get('estimated_time_remaining')
            )

        except Exception as e:
            logger.error("Failed to get task status", task_id=task_id, error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    @tasks_router.get("/")
    async def list_active_tasks():
        """Get all currently active tasks."""
        try:
            # Get active tasks from Celery
            inspect = celery_app.control.inspect()
            active_tasks = inspect.active()
            
            if not active_tasks:
                return {"active_tasks": [], "total_count": 0}
            
            # Flatten tasks from all workers
            all_tasks = []
            for worker, tasks in active_tasks.items():
                for task in tasks:
                    task['worker'] = worker
                    all_tasks.append(task)
            
            return {"active_tasks": all_tasks, "total_count": len(all_tasks)}

        except Exception as e:
            logger.error("Failed to list active tasks", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    @tasks_router.get("/stats/queues")
    async def get_queue_stats():
        """Get processing queue statistics."""
        try:
            # Get queue stats from Celery
            inspect = celery_app.control.inspect()
            
            # Get active, scheduled, and reserved tasks
            active = inspect.active() or {}
            scheduled = inspect.scheduled() or {}
            reserved = inspect.reserved() or {}
            
            # Count tasks by queue
            stats = {
                "active_tasks": sum(len(tasks) for tasks in active.values()),
                "scheduled_tasks": sum(len(tasks) for tasks in scheduled.values()),
                "reserved_tasks": sum(len(tasks) for tasks in reserved.values()),
                "workers": list(active.keys()) if active else []
            }
            
            return stats

        except Exception as e:
            logger.error("Failed to get queue stats", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    return tasks_router


def setup_task_management_endpoints(morag_api_getter):
    """Setup additional task management endpoints."""
    
    task_mgmt_router = APIRouter(prefix="/api/v1", tags=["Task Management"])
    
    @task_mgmt_router.delete("/ingest/{task_id}")
    async def cancel_task(task_id: str):
        """Cancel a running or pending task."""
        try:
            # Revoke the task
            celery_app.control.revoke(task_id, terminate=True)
            
            logger.info("Task cancelled", task_id=task_id)
            return {"message": f"Task {task_id} has been cancelled", "task_id": task_id}

        except Exception as e:
            logger.error("Failed to cancel task", task_id=task_id, error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    return task_mgmt_router
