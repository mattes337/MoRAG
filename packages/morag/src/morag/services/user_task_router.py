"""User-specific task routing service for remote worker management."""

import time
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

import redis
from celery import Celery
from celery.exceptions import WorkerLostError, Retry

from morag.services.auth_service import APIKeyService

logger = logging.getLogger(__name__)


class WorkerType(Enum):
    LOCAL = "local"
    REMOTE_GPU = "remote_gpu"
    REMOTE_CPU = "remote_cpu"


@dataclass
class UserWorkerStatus:
    """User worker status information."""
    worker_id: str
    user_id: str
    worker_type: WorkerType
    active_tasks: int
    max_tasks: int
    last_seen: float
    queues: List[str]


class UserTaskRouter:
    """User-specific task routing for remote workers."""

    def __init__(self, celery_app: Celery, redis_client: redis.Redis, api_key_service: APIKeyService):
        self.celery_app = celery_app
        self.redis = redis_client
        self.api_key_service = api_key_service
        self.worker_timeout = 60  # seconds

    def get_user_workers(self, user_id: str) -> Dict[str, UserWorkerStatus]:
        """Get currently available workers for a specific user."""
        try:
            inspect = self.celery_app.control.inspect()

            # Get active workers
            active_workers = inspect.active()
            if not active_workers:
                return {}

            # Get worker stats
            stats = inspect.stats() or {}

            user_workers = {}
            current_time = time.time()
            user_queue_prefix = f"gpu-tasks-{user_id}"

            for worker_name in active_workers.keys():
                # Get worker queues
                worker_queues = self._get_worker_queues(worker_name)

                # Check if this worker serves the user's queue
                user_queue_found = False
                worker_type = WorkerType.LOCAL

                for queue in worker_queues:
                    if queue == user_queue_prefix:
                        worker_type = WorkerType.REMOTE_GPU
                        user_queue_found = True
                        break
                    elif queue == f"cpu-tasks-{user_id}":
                        worker_type = WorkerType.REMOTE_CPU
                        user_queue_found = True
                        break

                if not user_queue_found:
                    continue  # This worker doesn't serve this user

                # Get worker stats
                worker_stats = stats.get(worker_name, {})
                active_tasks = len(active_workers.get(worker_name, []))

                user_workers[worker_name] = UserWorkerStatus(
                    worker_id=worker_name,
                    user_id=user_id,
                    worker_type=worker_type,
                    active_tasks=active_tasks,
                    max_tasks=worker_stats.get('pool', {}).get('max-concurrency', 1),
                    last_seen=current_time,
                    queues=worker_queues
                )

            return user_workers

        except Exception as e:
            logger.error(f"Failed to get user worker status for {user_id}: {e}")
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

    def has_user_workers_available(self, user_id: str) -> bool:
        """Check if user has remote workers available and not overloaded."""
        user_workers = self.get_user_workers(user_id)

        for worker in user_workers.values():
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

    def should_use_remote_worker(self, user_id: Optional[str], use_remote: bool, content_type: str) -> bool:
        """Determine if remote worker should be used based on user and availability."""
        if not use_remote or not user_id:
            return False

        # Check if user has remote workers available
        if not self.has_user_workers_available(user_id):
            logger.warning(f"Remote processing requested for user {user_id} but no workers available, using local")
            return False

        # Check user queue length vs default queue
        user_queue = self.api_key_service.get_user_queue_name(user_id, "gpu")
        user_queue_length = self.get_queue_length(user_queue)
        default_queue_length = self.get_queue_length('celery')

        # If user queue is significantly longer, consider local fallback
        if user_queue_length > default_queue_length + 10:
            logger.info(f"User {user_id} queue overloaded ({user_queue_length} vs {default_queue_length}), using local")
            return False

        return True

    def get_user_queue_name(self, user_id: str, worker_type: str = "gpu") -> str:
        """Get queue name for user and worker type."""
        return self.api_key_service.get_user_queue_name(user_id, worker_type)
    
    def log_task_routing(self, task_name: str, use_gpu: bool, content_type: str, task_id: str, user_id: Optional[str] = None):
        """Log task routing decision."""
        worker_type = "GPU" if use_gpu else "CPU"
        queue = f"gpu-tasks-{user_id}" if use_gpu and user_id else "celery"
        
        logger.info(f"Task routing: {task_name} -> {worker_type} worker",
                   extra={
                       'task_id': task_id,
                       'task_name': task_name,
                       'worker_type': worker_type,
                       'queue': queue,
                       'content_type': content_type,
                       'user_id': user_id
                   })


# Global task router instance
_task_router: Optional[UserTaskRouter] = None


def get_task_router() -> UserTaskRouter:
    """Get or create task router instance."""
    global _task_router
    if _task_router is None:
        from morag.worker import celery_app
        from morag.services.auth_service import APIKeyService
        import redis
        import os
        
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        redis_client = redis.from_url(redis_url)
        api_key_service = APIKeyService(redis_client)
        _task_router = UserTaskRouter(celery_app, redis_client, api_key_service)
    
    return _task_router
