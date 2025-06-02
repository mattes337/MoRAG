from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

router = APIRouter()

class IngestionRequest(BaseModel):
    source_type: str
    webhook_url: Optional[str] = None

class IngestionResponse(BaseModel):
    task_id: str
    status: str
    message: str

@router.post("/", response_model=IngestionResponse)
async def create_ingestion_task(request: IngestionRequest):
    """Create a new ingestion task. (Placeholder - will be implemented in task 17)"""
    return IngestionResponse(
        task_id="placeholder-task-id",
        status="pending",
        message="Ingestion endpoint not yet implemented"
    )
