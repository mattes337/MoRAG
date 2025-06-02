# Task 04: Async Task Queue Setup

## Overview
Set up Celery with Redis for asynchronous task processing, enabling the pipeline to handle long-running ingestion tasks without blocking API responses.

## Prerequisites
- Task 01: Project Setup completed
- Redis server (can use Docker)

## Dependencies
- Task 01: Project Setup

## Implementation Steps

### 1. Redis Docker Setup
Create `docker/docker-compose.redis.yml`:
```yaml
version: '3.8'

services:
  redis:
    image: redis:7.2-alpine
    container_name: morag-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  redis_data:
    driver: local
```

### 2. Celery Configuration
Create `src/morag/core/celery_app.py`:
```python
from celery import Celery
from kombu import Queue
import structlog

from morag.core.config import settings

logger = structlog.get_logger()

# Create Celery app
celery_app = Celery(
    "morag",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "morag.tasks.document_tasks",
        "morag.tasks.audio_tasks", 
        "morag.tasks.video_tasks",
        "morag.tasks.web_tasks",
    ]
)

# Celery configuration
celery_app.conf.update(
    # Task routing
    task_routes={
        "morag.tasks.document_tasks.*": {"queue": "document_processing"},
        "morag.tasks.audio_tasks.*": {"queue": "audio_processing"},
        "morag.tasks.video_tasks.*": {"queue": "video_processing"},
        "morag.tasks.web_tasks.*": {"queue": "web_processing"},
    },
    
    # Queue definitions
    task_queues=(
        Queue("document_processing", routing_key="document_processing"),
        Queue("audio_processing", routing_key="audio_processing"),
        Queue("video_processing", routing_key="video_processing"),
        Queue("web_processing", routing_key="web_processing"),
        Queue("celery", routing_key="celery"),  # Default queue
    ),
    
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Result settings
    result_expires=3600,  # 1 hour
    result_persistent=True,
    
    # Worker settings
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_max_tasks_per_child=1000,
    
    # Retry settings
    task_default_retry_delay=60,
    task_max_retries=3,
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
)

# Task base class with common functionality
class BaseTask(celery_app.Task):
    """Base task class with common functionality."""
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure."""
        logger.error(
            "Task failed",
            task_id=task_id,
            task_name=self.name,
            error=str(exc),
            args=args,
            kwargs=kwargs
        )
    
    def on_success(self, retval, task_id, args, kwargs):
        """Handle task success."""
        logger.info(
            "Task completed successfully",
            task_id=task_id,
            task_name=self.name,
            result_type=type(retval).__name__
        )
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Handle task retry."""
        logger.warning(
            "Task retrying",
            task_id=task_id,
            task_name=self.name,
            error=str(exc),
            retry_count=self.request.retries
        )

# Set base task class
celery_app.Task = BaseTask
```

### 3. Task Status Management
Create `src/morag/services/task_manager.py`:
```python
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import structlog

from celery.result import AsyncResult
from morag.core.celery_app import celery_app

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
            result = AsyncResult(task_id, app=self.celery_app)
            
            progress_info = {
                'progress': progress,
                'message': message,
                'metadata': metadata or {},
                'updated_at': datetime.utcnow().isoformat()
            }
            
            result.update_state(
                state='PROGRESS',
                meta=progress_info
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

# Global instance
task_manager = TaskManager()
```

### 4. Base Task Classes
Create `src/morag/tasks/__init__.py`:
```python
"""Task modules for async processing."""
```

Create `src/morag/tasks/base.py`:
```python
from typing import Dict, Any, Optional
import structlog
from datetime import datetime

from morag.core.celery_app import celery_app, BaseTask
from morag.services.task_manager import task_manager

logger = structlog.get_logger()

class ProcessingTask(BaseTask):
    """Base class for content processing tasks."""
    
    def __init__(self):
        super().__init__()
        self.start_time = None
    
    def on_start(self, task_id, args, kwargs):
        """Called when task starts."""
        self.start_time = datetime.utcnow()
        task_manager.update_task_progress(
            task_id,
            progress=0.0,
            message="Task started",
            metadata={'started_at': self.start_time.isoformat()}
        )
    
    def update_progress(self, progress: float, message: str = None, **metadata):
        """Update task progress."""
        task_manager.update_task_progress(
            self.request.id,
            progress=progress,
            message=message,
            metadata=metadata
        )
    
    def log_step(self, step: str, **kwargs):
        """Log a processing step."""
        logger.info(
            f"Task step: {step}",
            task_id=self.request.id,
            task_name=self.name,
            **kwargs
        )

# Placeholder task implementations (will be implemented in later tasks)
@celery_app.task(bind=True, base=ProcessingTask)
def process_document_task(self, file_path: str, source_type: str, metadata: Dict[str, Any]):
    """Process document file. (Placeholder)"""
    self.update_progress(0.1, "Starting document processing")
    # Implementation will be added in task 05
    return {"status": "placeholder", "message": "Document processing not implemented"}

@celery_app.task(bind=True, base=ProcessingTask)
def process_audio_task(self, file_path: str, metadata: Dict[str, Any]):
    """Process audio file. (Placeholder)"""
    self.update_progress(0.1, "Starting audio processing")
    # Implementation will be added in task 08
    return {"status": "placeholder", "message": "Audio processing not implemented"}

@celery_app.task(bind=True, base=ProcessingTask)
def process_video_task(self, file_path: str, metadata: Dict[str, Any]):
    """Process video file. (Placeholder)"""
    self.update_progress(0.1, "Starting video processing")
    # Implementation will be added in task 09
    return {"status": "placeholder", "message": "Video processing not implemented"}

@celery_app.task(bind=True, base=ProcessingTask)
def process_web_task(self, url: str, metadata: Dict[str, Any]):
    """Process web content. (Placeholder)"""
    self.update_progress(0.1, "Starting web processing")
    # Implementation will be added in task 12
    return {"status": "placeholder", "message": "Web processing not implemented"}
```

### 5. Update Health Check
Update `src/morag/api/routes/health.py`:
```python
# Add imports
from morag.services.task_manager import task_manager
import redis

# Replace Redis check in readiness_check:
    # Check Redis connection
    try:
        r = redis.from_url(settings.redis_url)
        r.ping()
        services["redis"] = "healthy"
        
        # Also check Celery
        stats = task_manager.get_queue_stats()
        if 'error' not in stats:
            services["celery"] = "healthy"
        else:
            services["celery"] = "unhealthy"
            
    except Exception as e:
        logger.error("Redis/Celery health check failed", error=str(e))
        services["redis"] = "unhealthy"
        services["celery"] = "unhealthy"
```

### 6. Worker Startup Script
Create `scripts/start_worker.py`:
```python
#!/usr/bin/env python3
"""Start Celery worker."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morag.core.celery_app import celery_app

if __name__ == "__main__":
    celery_app.start([
        "worker",
        "--loglevel=info",
        "--concurrency=4",
        "--queues=document_processing,audio_processing,video_processing,web_processing,celery"
    ])
```

## Testing Instructions

### 1. Start Redis
```bash
# From project root
docker-compose -f docker/docker-compose.redis.yml up -d

# Test Redis connection
redis-cli ping
```

### 2. Start Celery Worker
```bash
# From project root
python scripts/start_worker.py

# Or use celery command directly:
# cd src && celery -A morag.core.celery_app worker --loglevel=info
```

### 3. Test Task Queue
Create `tests/integration/test_celery.py`:
```python
import pytest
import asyncio
from morag.tasks.base import process_document_task
from morag.services.task_manager import task_manager, TaskStatus

def test_task_submission():
    """Test task submission and status tracking."""
    # Submit a test task
    result = process_document_task.delay(
        file_path="/tmp/test.pdf",
        source_type="document",
        metadata={"test": True}
    )
    
    # Check task was submitted
    assert result.task_id is not None
    
    # Check initial status
    task_info = task_manager.get_task_status(result.task_id)
    assert task_info.task_id == result.task_id
    assert task_info.status in [TaskStatus.PENDING, TaskStatus.STARTED]

def test_queue_stats():
    """Test queue statistics."""
    stats = task_manager.get_queue_stats()
    
    assert 'active_tasks' in stats
    assert 'queues' in stats
    assert isinstance(stats['workers'], list)
```

### 4. Test API Integration
```bash
# Start API with worker running
python src/morag/api/main.py

# Test health check
curl http://localhost:8000/health/ready
```

## Success Criteria
- [ ] Redis container starts successfully
- [ ] Celery worker starts without errors
- [ ] Tasks can be submitted to queues
- [ ] Task status tracking works
- [ ] Progress updates function correctly
- [ ] Health checks report Redis/Celery as healthy
- [ ] Queue statistics are accessible
- [ ] Integration tests pass

## Next Steps
- Task 05: Document Parser (implements document_tasks)
- Task 08: Audio Processing (implements audio_tasks)
- Task 17: Ingestion API (uses task queue for async processing)
