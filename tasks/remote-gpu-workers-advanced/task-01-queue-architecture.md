# Task 1: Queue Architecture Implementation

## Objective
Implement a priority queue system that routes GPU-intensive tasks to GPU workers and CPU-suitable tasks to CPU workers, with automatic failover capabilities.

## Current State Analysis

### Existing Queue System
- Single queue: `celery` (default)
- All workers consume from the same queue
- No task prioritization or routing based on worker capabilities
- Queue configuration in `packages/morag/src/morag/worker.py`:
  ```python
  celery_app.conf.update(
      broker_url=redis_url,
      result_backend=redis_url,
      # ... other config
  )
  ```

### Current Task Types
- **GPU-Intensive**: Audio transcription (Whisper), Video processing, Image analysis
- **CPU-Suitable**: Document processing, Web scraping, Text processing
- **Mixed**: YouTube processing (download + transcription)

## Implementation Plan

### Step 1: Define Queue Structure
Create three specialized queues:

1. **`gpu_priority`** - High priority for GPU-intensive tasks
2. **`cpu_standard`** - Standard priority for CPU-suitable tasks  
3. **`gpu_fallback`** - GPU tasks that can fallback to CPU if no GPU workers available

### Step 2: Update Celery Configuration

#### 2.1 Modify Worker Configuration
**File**: `packages/morag/src/morag/worker.py`

Add queue routing configuration:
```python
# Queue routing configuration
QUEUE_ROUTES = {
    'gpu_priority': {
        'exchange': 'gpu_priority',
        'exchange_type': 'direct',
        'routing_key': 'gpu_priority',
    },
    'cpu_standard': {
        'exchange': 'cpu_standard', 
        'exchange_type': 'direct',
        'routing_key': 'cpu_standard',
    },
    'gpu_fallback': {
        'exchange': 'gpu_fallback',
        'exchange_type': 'direct', 
        'routing_key': 'gpu_fallback',
    }
}

celery_app.conf.update(
    # ... existing config ...
    task_routes={
        # GPU-intensive tasks
        'morag.worker.process_audio_task': {'queue': 'gpu_priority'},
        'morag.worker.process_video_task': {'queue': 'gpu_priority'},
        'morag.worker.process_image_task': {'queue': 'gpu_priority'},
        'morag.ingest_tasks.ingest_audio_task': {'queue': 'gpu_priority'},
        'morag.ingest_tasks.ingest_video_task': {'queue': 'gpu_priority'},
        
        # CPU-suitable tasks
        'morag.worker.process_document_task': {'queue': 'cpu_standard'},
        'morag.worker.process_web_task': {'queue': 'cpu_standard'},
        'morag.ingest_tasks.ingest_document_task': {'queue': 'cpu_standard'},
        
        # Mixed tasks (can use either)
        'morag.worker.process_youtube_task': {'queue': 'gpu_fallback'},
    },
    task_default_queue='cpu_standard',
    task_default_exchange='cpu_standard',
    task_default_exchange_type='direct',
    task_default_routing_key='cpu_standard',
)
```

#### 2.2 Add Queue Priority Configuration
```python
# Queue priority settings
QUEUE_PRIORITIES = {
    'gpu_priority': 10,      # Highest priority
    'gpu_fallback': 5,       # Medium priority  
    'cpu_standard': 1,       # Standard priority
}

celery_app.conf.update(
    worker_direct=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    broker_transport_options={
        'priority_steps': list(range(11)),  # 0-10 priority levels
        'sep': ':',
        'queue_order_strategy': 'priority',
    }
)
```

### Step 3: Create Task Classification System

#### 3.1 Task Classifier Module
**File**: `packages/morag/src/morag/task_classifier.py`

```python
"""Task classification system for routing to appropriate workers."""

from enum import Enum
from typing import Dict, Optional
import structlog

logger = structlog.get_logger(__name__)

class TaskType(Enum):
    GPU_INTENSIVE = "gpu_intensive"
    CPU_SUITABLE = "cpu_suitable" 
    MIXED = "mixed"

class WorkerCapability(Enum):
    GPU = "gpu"
    CPU = "cpu"
    HYBRID = "hybrid"

class TaskClassifier:
    """Classifies tasks and determines appropriate worker routing."""
    
    # Content type to task type mapping
    CONTENT_TYPE_MAPPING = {
        'audio': TaskType.GPU_INTENSIVE,
        'video': TaskType.GPU_INTENSIVE,
        'image': TaskType.GPU_INTENSIVE,
        'document': TaskType.CPU_SUITABLE,
        'web': TaskType.CPU_SUITABLE,
        'youtube': TaskType.MIXED,  # Download (CPU) + Transcription (GPU)
    }
    
    # Task type to queue mapping
    QUEUE_MAPPING = {
        TaskType.GPU_INTENSIVE: 'gpu_priority',
        TaskType.CPU_SUITABLE: 'cpu_standard',
        TaskType.MIXED: 'gpu_fallback',
    }
    
    @classmethod
    def classify_content_type(cls, content_type: str) -> TaskType:
        """Classify content type into task type."""
        task_type = cls.CONTENT_TYPE_MAPPING.get(content_type, TaskType.CPU_SUITABLE)
        logger.info("Content type classified", 
                   content_type=content_type, 
                   task_type=task_type.value)
        return task_type
    
    @classmethod
    def get_queue_for_content_type(cls, content_type: str) -> str:
        """Get appropriate queue for content type."""
        task_type = cls.classify_content_type(content_type)
        queue = cls.QUEUE_MAPPING[task_type]
        logger.info("Queue selected for content type",
                   content_type=content_type,
                   task_type=task_type.value,
                   queue=queue)
        return queue
    
    @classmethod
    def can_worker_handle_task(cls, worker_capability: WorkerCapability, 
                              task_type: TaskType) -> bool:
        """Check if worker can handle specific task type."""
        if worker_capability == WorkerCapability.HYBRID:
            return True
        elif worker_capability == WorkerCapability.GPU:
            return task_type in [TaskType.GPU_INTENSIVE, TaskType.MIXED]
        elif worker_capability == WorkerCapability.CPU:
            return task_type in [TaskType.CPU_SUITABLE, TaskType.MIXED]
        return False
```

### Step 4: Update Task Submission

#### 4.1 Modify Server Task Submission
**File**: `packages/morag/src/morag/server.py`

Update task submission to use dynamic queue routing:

```python
from morag.task_classifier import TaskClassifier

# In file processing endpoint
@app.post("/api/v1/process/file")
async def process_file_endpoint(file: UploadFile, ...):
    # ... existing validation ...
    
    # Determine content type and queue
    content_type = detect_content_type(file.filename, file.content_type)
    queue = TaskClassifier.get_queue_for_content_type(content_type)
    
    # Submit to appropriate queue
    task = process_file_task.apply_async(
        args=[str(temp_path), content_type, options],
        queue=queue,
        priority=TaskClassifier.get_priority_for_queue(queue)
    )
```

### Step 5: Worker Queue Subscription

#### 5.1 Update Worker Startup
**File**: `packages/morag/src/morag/worker.py`

```python
def main():
    """Main entry point for the worker."""
    parser = argparse.ArgumentParser(description="MoRAG Background Worker")
    parser.add_argument("--worker-type", 
                       choices=['cpu', 'gpu', 'hybrid'], 
                       default='cpu',
                       help="Worker capability type")
    parser.add_argument("--queues", 
                       help="Comma-separated list of queues (auto-detected if not specified)")
    
    args = parser.parse_args()
    
    # Auto-detect queues based on worker type
    if not args.queues:
        if args.worker_type == 'gpu':
            queues = ['gpu_priority', 'gpu_fallback']
        elif args.worker_type == 'hybrid':
            queues = ['gpu_priority', 'gpu_fallback', 'cpu_standard']
        else:  # cpu
            queues = ['cpu_standard', 'gpu_fallback']  # CPU can handle fallback
        args.queues = ','.join(queues)
    
    logger.info("Starting worker", 
               worker_type=args.worker_type,
               queues=args.queues)
```

## Testing Requirements

### Unit Tests
1. **Task Classification Tests**
   - Test content type to task type mapping
   - Test queue selection logic
   - Test worker capability matching

2. **Queue Configuration Tests**
   - Test Celery queue setup
   - Test task routing configuration
   - Test priority assignment

### Integration Tests
1. **Queue Routing Tests**
   - Submit tasks and verify they go to correct queues
   - Test priority ordering within queues
   - Test fallback behavior

2. **Worker Subscription Tests**
   - Test workers subscribe to correct queues based on type
   - Test task consumption from multiple queues
   - Test queue priority handling

### Test Files to Create
- `tests/test_task_classifier.py`
- `tests/test_queue_architecture.py`
- `tests/integration/test_queue_routing.py`

## Configuration Changes

### Environment Variables
Add to `.env.example`:
```bash
# Worker Configuration
MORAG_WORKER_TYPE=cpu  # cpu, gpu, hybrid
MORAG_WORKER_QUEUES=   # auto-detected if empty

# Queue Configuration  
MORAG_ENABLE_QUEUE_PRIORITIES=true
MORAG_GPU_QUEUE_PRIORITY=10
MORAG_CPU_QUEUE_PRIORITY=1
```

## Dependencies
- No new external dependencies required
- Uses existing Celery and Redis infrastructure

## Success Criteria
1. Tasks are correctly routed to appropriate queues based on content type
2. Workers can subscribe to queues based on their capabilities
3. GPU tasks have higher priority than CPU tasks
4. System maintains backward compatibility with existing single-queue setup
5. Comprehensive test coverage for all routing scenarios

## Next Steps
After completing this task:
1. Proceed to Task 2: Worker Registration System
2. Test queue routing with simulated GPU/CPU workers
3. Validate priority handling and task distribution

---

**Dependencies**: None (foundational task)
**Estimated Time**: 2-3 days
**Risk Level**: Low (builds on existing Celery infrastructure)
