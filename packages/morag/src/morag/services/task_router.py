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
        gpu_beneficial_types = ['audio', 'video', 'image', 'mixed']  # mixed for batch processing
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
