from fastapi import APIRouter, Path
from pydantic import BaseModel
from typing import Optional, Dict, Any

router = APIRouter()

class StatusResponse(BaseModel):
    task_id: str
    status: str
    progress: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@router.get("/{task_id}", response_model=StatusResponse)
async def get_task_status(task_id: str = Path(..., description="Task ID to check")):
    """Get the status of an ingestion task. (Placeholder - will be implemented in task 18)"""
    return StatusResponse(
        task_id=task_id,
        status="pending",
        error="Status endpoint not yet implemented"
    )
