# Task 1: Queue Architecture Setup

## Objective
Configure Celery to support separate GPU and CPU task queues with simple routing logic.

## Background
Currently, MoRAG uses a single Celery queue ("celery") for all tasks. We need to add a second queue ("gpu-tasks") for GPU-intensive processing while maintaining backward compatibility.

## Implementation Steps

### 1.1 Update Celery Configuration

**File**: `packages/morag/src/morag/worker.py`

Add queue configuration to the existing Celery setup:

```python
# Add after existing celery_app.conf.update() call
celery_app.conf.update(
    # ... existing configuration ...
    
    # Queue routing configuration
    task_routes={
        # GPU-intensive tasks (when gpu=True)
        'morag.worker.process_file_task_gpu': {'queue': 'gpu-tasks'},
        'morag.worker.process_url_task_gpu': {'queue': 'gpu-tasks'},
        'morag.ingest_tasks.ingest_file_task_gpu': {'queue': 'gpu-tasks'},
        'morag.ingest_tasks.ingest_url_task_gpu': {'queue': 'gpu-tasks'},
        'morag.ingest_tasks.ingest_batch_task_gpu': {'queue': 'gpu-tasks'},
        
        # Default CPU tasks (existing behavior)
        'morag.worker.process_file_task': {'queue': 'celery'},
        'morag.worker.process_url_task': {'queue': 'celery'},
        'morag.ingest_tasks.ingest_file_task': {'queue': 'celery'},
        'morag.ingest_tasks.ingest_url_task': {'queue': 'celery'},
        'morag.ingest_tasks.ingest_batch_task': {'queue': 'celery'},
    },
    
    # Default queue remains 'celery' for backward compatibility
    task_default_queue='celery',
    task_default_exchange='celery',
    task_default_exchange_type='direct',
    task_default_routing_key='celery',
)
```

### 1.2 Create GPU Task Variants

**File**: `packages/morag/src/morag/worker.py`

Add GPU-specific task variants that use the same logic but route to GPU queue:

```python
@celery_app.task(bind=True)
def process_file_task_gpu(self, file_path: str, content_type: Optional[str] = None, task_options: Optional[Dict[str, Any]] = None):
    """GPU variant of process_file_task - routes to gpu-tasks queue."""
    return process_file_task(self, file_path, content_type, task_options)

@celery_app.task(bind=True)
def process_url_task_gpu(self, url: str, content_type: Optional[str] = None, task_options: Optional[Dict[str, Any]] = None):
    """GPU variant of process_url_task - routes to gpu-tasks queue."""
    return process_url_task(self, url, content_type, task_options)

# Add similar variants for other tasks...
```

**File**: `packages/morag/src/morag/ingest_tasks.py`

```python
@celery_app.task(bind=True)
def ingest_file_task_gpu(self, file_path: str, content_type: Optional[str] = None, task_options: Optional[Dict[str, Any]] = None):
    """GPU variant of ingest_file_task - routes to gpu-tasks queue."""
    return ingest_file_task(self, file_path, content_type, task_options)

@celery_app.task(bind=True)
def ingest_url_task_gpu(self, url: str, content_type: Optional[str] = None, task_options: Optional[Dict[str, Any]] = None):
    """GPU variant of ingest_url_task - routes to gpu-tasks queue."""
    return ingest_url_task(self, url, content_type, task_options)

@celery_app.task(bind=True)
def ingest_batch_task_gpu(self, items: List[Dict[str, Any]], task_options: Optional[Dict[str, Any]] = None):
    """GPU variant of ingest_batch_task - routes to gpu-tasks queue."""
    return ingest_batch_task(self, items, task_options)
```

### 1.3 Add Queue Selection Helper

**File**: `packages/morag/src/morag/worker.py`

Add helper function to select appropriate task based on GPU flag:

```python
def get_task_for_queue(base_task_name: str, use_gpu: bool = False):
    """Get the appropriate task function based on GPU requirement."""
    if use_gpu:
        gpu_task_name = f"{base_task_name}_gpu"
        return globals().get(gpu_task_name, globals()[base_task_name])
    return globals()[base_task_name]
```

## Testing

### 1.1 Test Queue Configuration
```bash
# Start CPU worker (default queue)
celery -A morag.worker worker --loglevel=info --queues=celery

# Start GPU worker (GPU queue)
celery -A morag.worker worker --loglevel=info --queues=gpu-tasks

# Test queue routing
python -c "
from morag.worker import process_file_task, process_file_task_gpu
print('CPU task queue:', process_file_task.routing_key)
print('GPU task queue:', process_file_task_gpu.routing_key)
"
```

### 1.2 Test Task Routing
```bash
# Test that tasks route to correct queues
python -c "
from morag.worker import celery_app
from celery import current_app

# Check routing configuration
routes = current_app.conf.task_routes
for task, config in routes.items():
    print(f'{task} -> {config}')
"
```

## Acceptance Criteria

- [ ] Celery configuration includes both 'celery' and 'gpu-tasks' queues
- [ ] GPU task variants are created for all main processing tasks
- [ ] Task routing correctly directs GPU tasks to 'gpu-tasks' queue
- [ ] CPU tasks continue using 'celery' queue (backward compatibility)
- [ ] Queue selection helper function works correctly
- [ ] Workers can be started for specific queues
- [ ] No breaking changes to existing functionality

## Files Modified

- `packages/morag/src/morag/worker.py`
- `packages/morag/src/morag/ingest_tasks.py`

## Next Steps

After completing this task:
1. Proceed to Task 2: API Parameter Addition
2. Test queue routing with simple tasks
3. Verify backward compatibility with existing deployments

## Notes

- This approach maintains full backward compatibility
- GPU tasks are identical to CPU tasks but route to different queues
- Workers can consume from either queue based on their capabilities
- No changes to existing task logic or processing code required
