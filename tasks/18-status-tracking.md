# Task 18: Status Tracking and Webhooks

## Overview
Implement comprehensive status tracking with webhook notifications, progress monitoring, and detailed task lifecycle management.

## Prerequisites
- Task 04: Task Queue Setup completed
- Task 17: Ingestion API completed

## Dependencies
- Task 04: Task Queue Setup
- Task 17: Ingestion API

## Implementation Steps

### 1. Webhook Service
Create `src/morag/services/webhook.py`:
```python
from typing import Dict, Any, Optional
import httpx
import structlog
import asyncio
from datetime import datetime

from morag.core.config import settings
from morag.core.exceptions import ExternalServiceError

logger = structlog.get_logger()

class WebhookService:
    """Service for sending webhook notifications."""
    
    def __init__(self):
        self.timeout = settings.webhook_timeout
        self.max_retries = 3
        self.retry_delay = 5  # seconds
    
    async def send_webhook(
        self,
        webhook_url: str,
        payload: Dict[str, Any],
        event_type: str = "task_completed"
    ) -> bool:
        """Send webhook notification with retry logic."""
        
        if not webhook_url:
            return True  # No webhook to send
        
        # Prepare webhook payload
        webhook_payload = {
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "data": payload
        }
        
        logger.info(
            "Sending webhook",
            webhook_url=webhook_url,
            event_type=event_type,
            task_id=payload.get("task_id")
        )
        
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        webhook_url,
                        json=webhook_payload,
                        headers={
                            "Content-Type": "application/json",
                            "User-Agent": "MoRAG-Webhook/1.0"
                        }
                    )
                    
                    if response.status_code < 400:
                        logger.info(
                            "Webhook sent successfully",
                            webhook_url=webhook_url,
                            status_code=response.status_code,
                            attempt=attempt + 1
                        )
                        return True
                    else:
                        logger.warning(
                            "Webhook failed with HTTP error",
                            webhook_url=webhook_url,
                            status_code=response.status_code,
                            response_text=response.text[:200],
                            attempt=attempt + 1
                        )
                        
            except httpx.TimeoutException:
                logger.warning(
                    "Webhook timeout",
                    webhook_url=webhook_url,
                    timeout=self.timeout,
                    attempt=attempt + 1
                )
            except Exception as e:
                logger.warning(
                    "Webhook failed with exception",
                    webhook_url=webhook_url,
                    error=str(e),
                    attempt=attempt + 1
                )
            
            # Wait before retry (except on last attempt)
            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        logger.error(
            "Webhook failed after all retries",
            webhook_url=webhook_url,
            max_retries=self.max_retries
        )
        return False
    
    async def send_task_started(self, task_id: str, webhook_url: str, metadata: Dict[str, Any]) -> bool:
        """Send task started notification."""
        payload = {
            "task_id": task_id,
            "status": "started",
            "message": "Task processing started",
            "metadata": metadata
        }
        return await self.send_webhook(webhook_url, payload, "task_started")
    
    async def send_task_progress(
        self,
        task_id: str,
        webhook_url: str,
        progress: float,
        message: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """Send task progress notification."""
        payload = {
            "task_id": task_id,
            "status": "progress",
            "progress": progress,
            "message": message,
            "metadata": metadata
        }
        return await self.send_webhook(webhook_url, payload, "task_progress")
    
    async def send_task_completed(
        self,
        task_id: str,
        webhook_url: str,
        result: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> bool:
        """Send task completion notification."""
        payload = {
            "task_id": task_id,
            "status": "completed",
            "message": "Task completed successfully",
            "result": result,
            "metadata": metadata
        }
        return await self.send_webhook(webhook_url, payload, "task_completed")
    
    async def send_task_failed(
        self,
        task_id: str,
        webhook_url: str,
        error: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """Send task failure notification."""
        payload = {
            "task_id": task_id,
            "status": "failed",
            "message": "Task failed",
            "error": error,
            "metadata": metadata
        }
        return await self.send_webhook(webhook_url, payload, "task_failed")

# Global instance
webhook_service = WebhookService()
```

### 2. Enhanced Task Manager
Update `src/morag/services/task_manager.py` to include webhook integration:
```python
# Add these imports at the top
from morag.services.webhook import webhook_service

# Add this method to TaskManager class
async def handle_task_completion(self, task_id: str, result: Dict[str, Any], metadata: Dict[str, Any]):
    """Handle task completion and send webhooks."""
    webhook_url = metadata.get('webhook_url')
    
    if webhook_url:
        if result.get('status') == 'success':
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

async def handle_task_started(self, task_id: str, metadata: Dict[str, Any]):
    """Handle task start and send webhooks."""
    webhook_url = metadata.get('webhook_url')
    
    if webhook_url:
        await webhook_service.send_task_started(
            task_id=task_id,
            webhook_url=webhook_url,
            metadata=metadata
        )

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
        await webhook_service.send_task_progress(
            task_id=task_id,
            webhook_url=webhook_url,
            progress=progress,
            message=message,
            metadata=metadata
        )
```

### 3. Enhanced Base Task with Webhook Integration
Update `src/morag/tasks/base.py`:
```python
# Add these imports
from morag.services.webhook import webhook_service

# Update ProcessingTask class
class ProcessingTask(BaseTask):
    """Base class for content processing tasks."""
    
    def __init__(self):
        super().__init__()
        self.start_time = None
        self.webhook_url = None
    
    def on_start(self, task_id, args, kwargs):
        """Called when task starts."""
        self.start_time = datetime.utcnow()
        
        # Extract webhook URL from metadata
        if len(args) > 2 and isinstance(args[2], dict):
            self.webhook_url = args[2].get('webhook_url')
        elif 'metadata' in kwargs and isinstance(kwargs['metadata'], dict):
            self.webhook_url = kwargs['metadata'].get('webhook_url')
        
        # Send webhook notification
        if self.webhook_url:
            asyncio.create_task(webhook_service.send_task_started(
                task_id=task_id,
                webhook_url=self.webhook_url,
                metadata=self._get_metadata(args, kwargs)
            ))
        
        task_manager.update_task_progress(
            task_id,
            progress=0.0,
            message="Task started",
            metadata={'started_at': self.start_time.isoformat()}
        )
    
    def on_success(self, retval, task_id, args, kwargs):
        """Handle task success."""
        super().on_success(retval, task_id, args, kwargs)
        
        # Send webhook notification
        if self.webhook_url:
            asyncio.create_task(webhook_service.send_task_completed(
                task_id=task_id,
                webhook_url=self.webhook_url,
                result=retval,
                metadata=self._get_metadata(args, kwargs)
            ))
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure."""
        super().on_failure(exc, task_id, args, kwargs, einfo)
        
        # Send webhook notification
        if self.webhook_url:
            asyncio.create_task(webhook_service.send_task_failed(
                task_id=task_id,
                webhook_url=self.webhook_url,
                error=str(exc),
                metadata=self._get_metadata(args, kwargs)
            ))
    
    def update_progress(self, progress: float, message: str = None, **metadata):
        """Update task progress with webhook support."""
        task_manager.update_task_progress(
            self.request.id,
            progress=progress,
            message=message,
            metadata=metadata
        )
        
        # Send webhook for significant milestones
        if self.webhook_url and progress in [0.25, 0.5, 0.75]:
            asyncio.create_task(webhook_service.send_task_progress(
                task_id=self.request.id,
                webhook_url=self.webhook_url,
                progress=progress,
                message=message or f"Progress: {progress*100:.0f}%",
                metadata=metadata
            ))
    
    def _get_metadata(self, args, kwargs) -> Dict[str, Any]:
        """Extract metadata from task arguments."""
        if len(args) > 2 and isinstance(args[2], dict):
            return args[2]
        elif 'metadata' in kwargs and isinstance(kwargs['metadata'], dict):
            return kwargs['metadata']
        return {}
```

### 4. Task Status History
Create `src/morag/services/status_history.py`:
```python
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import structlog
from dataclasses import dataclass, asdict
from enum import Enum

import redis
from morag.core.config import settings

logger = structlog.get_logger()

@dataclass
class StatusEvent:
    """Represents a status change event."""
    timestamp: datetime
    status: str
    progress: Optional[float]
    message: Optional[str]
    metadata: Optional[Dict[str, Any]]

class StatusHistory:
    """Manages task status history using Redis."""
    
    def __init__(self):
        self.redis_client = redis.from_url(settings.redis_url)
        self.history_ttl = 86400 * 7  # 7 days
    
    def add_status_event(
        self,
        task_id: str,
        status: str,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a status event to task history."""
        
        event = StatusEvent(
            timestamp=datetime.utcnow(),
            status=status,
            progress=progress,
            message=message,
            metadata=metadata
        )
        
        try:
            # Store as JSON in Redis list
            event_data = {
                'timestamp': event.timestamp.isoformat(),
                'status': event.status,
                'progress': event.progress,
                'message': event.message,
                'metadata': event.metadata
            }
            
            key = f"task_history:{task_id}"
            self.redis_client.lpush(key, json.dumps(event_data))
            self.redis_client.expire(key, self.history_ttl)
            
            # Keep only last 100 events
            self.redis_client.ltrim(key, 0, 99)
            
        except Exception as e:
            logger.error("Failed to add status event", task_id=task_id, error=str(e))
    
    def get_task_history(self, task_id: str) -> List[StatusEvent]:
        """Get complete status history for a task."""
        
        try:
            key = f"task_history:{task_id}"
            events_data = self.redis_client.lrange(key, 0, -1)
            
            events = []
            for event_json in events_data:
                try:
                    event_dict = json.loads(event_json)
                    event = StatusEvent(
                        timestamp=datetime.fromisoformat(event_dict['timestamp']),
                        status=event_dict['status'],
                        progress=event_dict['progress'],
                        message=event_dict['message'],
                        metadata=event_dict['metadata']
                    )
                    events.append(event)
                except Exception as e:
                    logger.warning("Failed to parse status event", error=str(e))
            
            # Sort by timestamp (newest first)
            events.sort(key=lambda x: x.timestamp, reverse=True)
            return events
            
        except Exception as e:
            logger.error("Failed to get task history", task_id=task_id, error=str(e))
            return []
    
    def get_recent_events(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent status events across all tasks."""
        
        try:
            # This is a simplified implementation
            # In production, you might want to use Redis Streams or a proper time-series DB
            
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            all_events = []
            
            # Get all task history keys
            keys = self.redis_client.keys("task_history:*")
            
            for key in keys:
                task_id = key.decode().split(":", 1)[1]
                events = self.get_task_history(task_id)
                
                for event in events:
                    if event.timestamp >= cutoff_time:
                        event_dict = asdict(event)
                        event_dict['task_id'] = task_id
                        event_dict['timestamp'] = event.timestamp.isoformat()
                        all_events.append(event_dict)
            
            # Sort by timestamp
            all_events.sort(key=lambda x: x['timestamp'], reverse=True)
            return all_events[:100]  # Limit to 100 most recent
            
        except Exception as e:
            logger.error("Failed to get recent events", error=str(e))
            return []

# Global instance
status_history = StatusHistory()
```

### 5. Enhanced Status API
Update `src/morag/api/routes/status.py` to include history:
```python
# Add imports
from morag.services.status_history import status_history

# Add new endpoint
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
    hours: int = 24,
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
```

## Testing Instructions

### 1. Test Webhook Notifications
Create a simple webhook receiver for testing:
```python
# webhook_receiver.py
from fastapi import FastAPI, Request
import uvicorn

app = FastAPI()

@app.post("/webhook")
async def receive_webhook(request: Request):
    payload = await request.json()
    print(f"Received webhook: {payload}")
    return {"status": "received"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
```

### 2. Test Status Tracking
```bash
# Start a task with webhook
curl -X POST "http://localhost:8000/api/v1/ingest/file" \
  -H "Authorization: Bearer test-api-key" \
  -F "source_type=document" \
  -F "file=@test.pdf" \
  -F "webhook_url=http://localhost:8001/webhook"

# Get task status
curl -X GET "http://localhost:8000/api/v1/status/{task_id}" \
  -H "Authorization: Bearer test-api-key"

# Get task history
curl -X GET "http://localhost:8000/api/v1/status/{task_id}/history" \
  -H "Authorization: Bearer test-api-key"

# Get recent events
curl -X GET "http://localhost:8000/api/v1/status/events/recent?hours=1" \
  -H "Authorization: Bearer test-api-key"
```

### 3. Integration Test
Create `tests/integration/test_webhooks.py`:
```python
import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from morag.services.webhook import webhook_service

@pytest.mark.asyncio
async def test_webhook_success():
    """Test successful webhook delivery."""
    
    with patch('httpx.AsyncClient.post') as mock_post:
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        result = await webhook_service.send_webhook(
            webhook_url="https://example.com/webhook",
            payload={"task_id": "test-123", "status": "completed"},
            event_type="task_completed"
        )
        
        assert result is True
        mock_post.assert_called_once()

@pytest.mark.asyncio
async def test_webhook_retry():
    """Test webhook retry logic."""
    
    with patch('httpx.AsyncClient.post') as mock_post:
        # Mock failed response
        mock_response = AsyncMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response
        
        result = await webhook_service.send_webhook(
            webhook_url="https://example.com/webhook",
            payload={"task_id": "test-123", "status": "completed"},
            event_type="task_completed"
        )
        
        assert result is False
        assert mock_post.call_count == 3  # Should retry 3 times
```

## Success Criteria
- [ ] Webhook service sends notifications reliably
- [ ] Retry logic works for failed webhooks
- [ ] Task status history is tracked and stored
- [ ] Status API provides comprehensive information
- [ ] Progress milestones trigger webhook notifications
- [ ] Task completion/failure sends appropriate webhooks
- [ ] Recent events endpoint works
- [ ] Integration with existing task system works
- [ ] Error handling prevents webhook failures from affecting tasks
- [ ] Tests pass for webhook functionality

## Next Steps
- Task 19: n8n Workflows (orchestration layer)
- Task 20: Testing Framework (comprehensive testing)
- Task 21: Monitoring and Logging (enhanced observability)
