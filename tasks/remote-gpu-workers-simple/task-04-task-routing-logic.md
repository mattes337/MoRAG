# Task 4: Task Routing Logic

## Objective
Implement intelligent task routing logic that handles GPU worker availability, fallback mechanisms, and task distribution.

## Background
The system needs to gracefully handle scenarios where:
1. GPU workers are unavailable (fallback to CPU)
2. GPU workers are overloaded (queue management)
3. Tasks fail on GPU workers (retry on CPU)
4. Mixed worker environments (some GPU, some CPU)

## Implementation Steps

### 4.1 Create Task Router Service

**File**: `packages/morag/src/morag/services/task_router.py`

```python
"""Task routing service for GPU/CPU worker management."""

import time
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

import redis
from celery import Celery
from celery.exceptions import WorkerLostError, Retry

logger = logging.getLogger(__name__)


class WorkerType(Enum):
    CPU = "cpu"
    GPU = "gpu"


@dataclass
class WorkerStatus:
    """Worker status information."""
    worker_id: str
    worker_type: WorkerType
    active_tasks: int
    max_tasks: int
    last_seen: float
    queues: List[str]


class TaskRouter:
    """Intelligent task routing for GPU/CPU workers."""
    
    def __init__(self, celery_app: Celery, redis_client: redis.Redis):
        self.celery_app = celery_app
        self.redis = redis_client
        self.worker_timeout = 60  # seconds
        
    def get_available_workers(self, worker_type: Optional[WorkerType] = None) -> Dict[str, WorkerStatus]:
        """Get currently available workers."""
        try:
            inspect = self.celery_app.control.inspect()
            
            # Get active workers
            active_workers = inspect.active()
            if not active_workers:
                return {}
            
            # Get worker stats
            stats = inspect.stats() or {}
            
            workers = {}
            current_time = time.time()
            
            for worker_name in active_workers.keys():
                # Determine worker type from queues
                worker_queues = self._get_worker_queues(worker_name)
                if 'gpu-tasks' in worker_queues:
                    wtype = WorkerType.GPU
                else:
                    wtype = WorkerType.CPU
                
                # Filter by requested worker type
                if worker_type and wtype != worker_type:
                    continue
                
                # Get worker stats
                worker_stats = stats.get(worker_name, {})
                active_tasks = len(active_workers.get(worker_name, []))
                
                workers[worker_name] = WorkerStatus(
                    worker_id=worker_name,
                    worker_type=wtype,
                    active_tasks=active_tasks,
                    max_tasks=worker_stats.get('pool', {}).get('max-concurrency', 1),
                    last_seen=current_time,
                    queues=worker_queues
                )
            
            return workers
            
        except Exception as e:
            logger.error(f"Failed to get worker status: {e}")
            return {}
    
    def _get_worker_queues(self, worker_name: str) -> List[str]:
        """Get queues for a specific worker."""
        try:
            inspect = self.celery_app.control.inspect([worker_name])
            active_queues = inspect.active_queues()
            if active_queues and worker_name in active_queues:
                return [q['name'] for q in active_queues[worker_name]]
            return []
        except Exception:
            return []
    
    def has_gpu_workers_available(self) -> bool:
        """Check if GPU workers are available and not overloaded."""
        gpu_workers = self.get_available_workers(WorkerType.GPU)
        
        for worker in gpu_workers.values():
            if worker.active_tasks < worker.max_tasks:
                return True
        
        return False
    
    def get_queue_length(self, queue_name: str) -> int:
        """Get current queue length."""
        try:
            return self.redis.llen(queue_name)
        except Exception as e:
            logger.error(f"Failed to get queue length for {queue_name}: {e}")
            return 0
    
    def should_use_gpu_worker(self, requested_gpu: bool, content_type: str) -> bool:
        """Determine if GPU worker should be used based on availability and content type."""
        if not requested_gpu:
            return False
        
        # Check if content type benefits from GPU
        gpu_beneficial_types = ['audio', 'video', 'image']
        if content_type not in gpu_beneficial_types:
            logger.info(f"Content type '{content_type}' doesn't benefit from GPU, using CPU")
            return False
        
        # Check GPU worker availability
        if not self.has_gpu_workers_available():
            logger.warning("GPU requested but no GPU workers available, falling back to CPU")
            return False
        
        # Check GPU queue length
        gpu_queue_length = self.get_queue_length('gpu-tasks')
        cpu_queue_length = self.get_queue_length('celery')
        
        # If GPU queue is significantly longer, consider CPU fallback
        if gpu_queue_length > cpu_queue_length + 5:
            logger.info(f"GPU queue overloaded ({gpu_queue_length} vs {cpu_queue_length}), using CPU")
            return False
        
        return True
    
    def log_task_routing(self, task_name: str, use_gpu: bool, content_type: str, task_id: str):
        """Log task routing decision."""
        worker_type = "GPU" if use_gpu else "CPU"
        queue = "gpu-tasks" if use_gpu else "celery"
        
        logger.info(f"Task routing: {task_name} -> {worker_type} worker",
                   extra={
                       'task_id': task_id,
                       'task_name': task_name,
                       'worker_type': worker_type,
                       'queue': queue,
                       'content_type': content_type
                   })


# Global task router instance
_task_router: Optional[TaskRouter] = None


def get_task_router() -> TaskRouter:
    """Get or create task router instance."""
    global _task_router
    if _task_router is None:
        from morag.worker import celery_app
        import redis
        import os
        
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        redis_client = redis.from_url(redis_url)
        _task_router = TaskRouter(celery_app, redis_client)
    
    return _task_router
```

### 4.2 Update Server to Use Task Router

**File**: `packages/morag/src/morag/server.py`

Add task router integration:

```python
# Add import
from morag.services.task_router import get_task_router

# Update process_file endpoint
@app.post("/process/file", response_model=ProcessingResult, tags=["Processing"])
async def process_file(
    source_type: Optional[str] = Form(None),
    file: UploadFile = File(...),
    gpu: Optional[bool] = Form(False),
    metadata: Optional[str] = Form(None)
):
    """Process uploaded file and return results immediately."""
    temp_path = None
    try:
        # ... existing file upload code ...
        
        # Auto-detect source type if not provided
        if not source_type:
            source_type = get_morag_api()._detect_content_type_from_file(Path(temp_path))
        
        # Determine if GPU should be used
        task_router = get_task_router()
        use_gpu = task_router.should_use_gpu_worker(gpu, source_type)
        
        # Select appropriate task
        if use_gpu:
            task = process_file_task_gpu.delay(temp_path, source_type, options)
        else:
            task = process_file_task.delay(temp_path, source_type, options)
        
        # Log routing decision
        task_router.log_task_routing('process_file', use_gpu, source_type, task.id)
        
        # ... rest of existing code ...
```

### 4.3 Add Fallback Retry Logic

**File**: `packages/morag/src/morag/worker.py`

Add retry logic for GPU task failures:

```python
from morag.services.task_router import get_task_router

@celery_app.task(bind=True, autoretry_for=(WorkerLostError,), retry_kwargs={'max_retries': 1})
def process_file_task_gpu(self, file_path: str, content_type: Optional[str] = None, task_options: Optional[Dict[str, Any]] = None):
    """GPU variant of process_file_task with CPU fallback on failure."""
    try:
        return process_file_task(self, file_path, content_type, task_options)
    except WorkerLostError as e:
        logger.warning(f"GPU worker lost, retrying on CPU: {e}")
        # Retry on CPU queue
        if self.request.retries < 1:
            # Submit to CPU queue instead
            cpu_task = process_file_task.delay(file_path, content_type, task_options)
            return cpu_task.get()
        raise
    except Exception as e:
        logger.error(f"GPU task failed: {e}")
        # For other errors, try CPU fallback
        if self.request.retries < 1:
            logger.info("Attempting CPU fallback for failed GPU task")
            cpu_task = process_file_task.delay(file_path, content_type, task_options)
            return cpu_task.get()
        raise
```

### 4.4 Add Queue Monitoring Endpoint

**File**: `packages/morag/src/morag/server.py`

Add endpoint to monitor queue status:

```python
@app.get("/api/v1/status/workers", tags=["Task Management"])
async def get_worker_status():
    """Get current worker and queue status."""
    try:
        task_router = get_task_router()
        
        # Get worker information
        all_workers = task_router.get_available_workers()
        gpu_workers = task_router.get_available_workers(WorkerType.GPU)
        cpu_workers = task_router.get_available_workers(WorkerType.CPU)
        
        # Get queue lengths
        gpu_queue_length = task_router.get_queue_length('gpu-tasks')
        cpu_queue_length = task_router.get_queue_length('celery')
        
        return {
            "workers": {
                "total": len(all_workers),
                "gpu": len(gpu_workers),
                "cpu": len(cpu_workers),
                "details": {
                    worker_id: {
                        "type": worker.worker_type.value,
                        "active_tasks": worker.active_tasks,
                        "max_tasks": worker.max_tasks,
                        "queues": worker.queues
                    }
                    for worker_id, worker in all_workers.items()
                }
            },
            "queues": {
                "gpu-tasks": {
                    "length": gpu_queue_length,
                    "workers": len(gpu_workers)
                },
                "celery": {
                    "length": cpu_queue_length,
                    "workers": len(cpu_workers)
                }
            },
            "gpu_available": task_router.has_gpu_workers_available()
        }
        
    except Exception as e:
        logger.error("Failed to get worker status", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
```

## Testing

### 4.1 Test Task Routing Logic
```python
# Test routing decisions
from morag.services.task_router import get_task_router

router = get_task_router()

# Test GPU availability check
print("GPU workers available:", router.has_gpu_workers_available())

# Test routing decision
should_use_gpu = router.should_use_gpu_worker(True, 'audio')
print("Should use GPU for audio:", should_use_gpu)

# Test queue monitoring
gpu_queue_len = router.get_queue_length('gpu-tasks')
cpu_queue_len = router.get_queue_length('celery')
print(f"Queue lengths - GPU: {gpu_queue_len}, CPU: {cpu_queue_len}")
```

### 4.2 Test Worker Status Endpoint
```bash
# Check worker status
curl http://localhost:8000/api/v1/status/workers | jq

# Monitor during task processing
curl http://localhost:8000/api/v1/status/workers | jq '.queues'
```

### 4.3 Test Fallback Behavior
```bash
# Submit GPU task when no GPU workers available
curl -X POST "http://localhost:8000/process/file" \
  -F "file=@test.mp3" \
  -F "gpu=true"

# Check logs for fallback messages
```

## Acceptance Criteria

- [ ] Task router service correctly identifies available workers
- [ ] GPU availability check works accurately
- [ ] Queue length monitoring works
- [ ] Intelligent routing considers worker availability and queue load
- [ ] Fallback to CPU works when GPU unavailable
- [ ] Retry logic handles GPU worker failures
- [ ] Worker status endpoint provides comprehensive information
- [ ] Routing decisions are logged for debugging
- [ ] Content type consideration works (GPU beneficial vs not)

## Files Modified/Created

- `packages/morag/src/morag/services/task_router.py` (new)
- `packages/morag/src/morag/server.py` (modified)
- `packages/morag/src/morag/worker.py` (modified)

## Next Steps

After completing this task:
1. Proceed to Task 5: Network Configuration
2. Test routing logic with actual GPU workers
3. Monitor fallback behavior under various conditions

## Notes

- Router makes intelligent decisions based on worker availability
- Fallback mechanisms ensure system reliability
- Queue monitoring prevents overload situations
- Comprehensive logging aids in debugging and monitoring
- Content type awareness optimizes resource usage
