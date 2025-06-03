from fastapi import APIRouter, Path, HTTPException, Depends, Query
from typing import List
import structlog

from morag.api.models import TaskStatusResponse
from morag.services.task_manager import task_manager, TaskStatus
from morag.services.status_history import status_history
from morag.api.routes.ingestion import verify_api_key

logger = structlog.get_logger()
router = APIRouter()

@router.get("/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(
    task_id: str = Path(..., description="Task ID to check"),
    api_key: str = Depends(verify_api_key)
):
    """Get the status of an ingestion task."""

    try:
        task_info = task_manager.get_task_status(task_id)

        # Calculate estimated time remaining
        estimated_remaining = None
        if task_info.progress and task_info.progress > 0:
            # Simple estimation based on progress
            if task_info.status == TaskStatus.PROGRESS:
                estimated_remaining = int((1 - task_info.progress) * 300)  # Rough estimate

        return TaskStatusResponse(
            task_id=task_info.task_id,
            status=task_info.status.value,
            progress=task_info.progress,
            result=task_info.result,
            error=task_info.error,
            created_at=task_info.created_at.isoformat() if task_info.created_at else None,
            started_at=task_info.started_at.isoformat() if task_info.started_at else None,
            completed_at=task_info.completed_at.isoformat() if task_info.completed_at else None,
            estimated_time_remaining=estimated_remaining
        )

    except Exception as e:
        logger.error("Failed to get task status", task_id=task_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get task status: {str(e)}")

@router.get("/")
async def list_active_tasks(api_key: str = Depends(verify_api_key)):
    """List all active tasks."""

    try:
        active_tasks = task_manager.get_active_tasks()
        return {
            "active_tasks": active_tasks,
            "count": len(active_tasks)
        }

    except Exception as e:
        logger.error("Failed to list active tasks", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list tasks: {str(e)}")

@router.get("/stats/queues")
async def get_queue_stats(api_key: str = Depends(verify_api_key)):
    """Get queue statistics."""

    try:
        stats = task_manager.get_queue_stats()
        return stats

    except Exception as e:
        logger.error("Failed to get queue stats", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get queue stats: {str(e)}")

@router.get("/{task_id}/history")
async def get_task_history(
    task_id: str = Path(..., description="Task ID to get history for"),
    api_key: str = Depends(verify_api_key)
):
    """Get complete status history for a task."""

    try:
        history = status_history.get_task_history(task_id)

        return {
            "task_id": task_id,
            "history": [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "status": event.status,
                    "progress": event.progress,
                    "message": event.message,
                    "metadata": event.metadata
                }
                for event in history
            ],
            "event_count": len(history)
        }

    except Exception as e:
        logger.error("Failed to get task history", task_id=task_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get task history: {str(e)}")

@router.get("/events/recent")
async def get_recent_events(
    hours: int = Query(24, description="Number of hours to look back"),
    api_key: str = Depends(verify_api_key)
):
    """Get recent status events across all tasks."""

    try:
        events = status_history.get_recent_events(hours)

        return {
            "events": events,
            "count": len(events),
            "hours": hours
        }

    except Exception as e:
        logger.error("Failed to get recent events", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get recent events: {str(e)}")
