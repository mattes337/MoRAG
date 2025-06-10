"""Standalone test for HTTP task queue functionality."""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum


class TaskStatus(str, Enum):
    """Task status enum."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Task data structure."""
    task_id: str
    task_type: str
    user_id: Optional[str]
    parameters: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    assigned_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    worker_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class Worker:
    """Worker registration data."""
    worker_id: str
    worker_type: str
    api_key: str
    user_id: Optional[str]
    last_seen: datetime = field(default_factory=datetime.now)
    active_tasks: Set[str] = field(default_factory=set)
    max_concurrent_tasks: int = 1


class HTTPTaskQueue:
    """HTTP-based task queue manager."""

    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.workers: Dict[str, Worker] = {}
        self.user_queues: Dict[str, List[str]] = {}
        self.general_queue: List[str] = []
        self._lock = None  # Will be initialized when first used
        self._initialized = False

    async def _ensure_initialized(self):
        """Ensure the task queue is properly initialized with async components."""
        if not self._initialized:
            self._lock = asyncio.Lock()
            self._initialized = True
    
    async def register_worker(self, worker_id: str, worker_type: str, api_key: str,
                            user_id: Optional[str] = None, max_concurrent_tasks: int = 1) -> bool:
        """Register a worker."""
        await self._ensure_initialized()
        async with self._lock:
            self.workers[worker_id] = Worker(
                worker_id=worker_id,
                worker_type=worker_type,
                api_key=api_key,
                user_id=user_id,
                max_concurrent_tasks=max_concurrent_tasks
            )
            return True
    
    async def submit_task(self, task_type: str, parameters: Dict[str, Any],
                         user_id: Optional[str] = None) -> str:
        """Submit a new task."""
        await self._ensure_initialized()
        task_id = str(uuid.uuid4())

        async with self._lock:
            task = Task(
                task_id=task_id,
                task_type=task_type,
                user_id=user_id,
                parameters=parameters
            )
            
            self.tasks[task_id] = task
            await self._add_to_queue(task)
            
            return task_id
    
    async def _add_to_queue(self, task: Task):
        """Add task to appropriate queue."""
        if task.user_id:
            if task.user_id not in self.user_queues:
                self.user_queues[task.user_id] = []
            self.user_queues[task.user_id].append(task.task_id)
        else:
            self.general_queue.append(task.task_id)
    
    async def get_next_task(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """Get next available task for a worker."""
        await self._ensure_initialized()
        async with self._lock:
            if worker_id not in self.workers:
                return None
            
            worker = self.workers[worker_id]
            worker.last_seen = datetime.now()
            
            if len(worker.active_tasks) >= worker.max_concurrent_tasks:
                return None
            
            task = None
            
            # Check user-specific queue first
            if worker.user_id and worker.user_id in self.user_queues:
                queue = self.user_queues[worker.user_id]
                for task_id in queue[:]:
                    if task_id in self.tasks and self.tasks[task_id].status == TaskStatus.PENDING:
                        task = self.tasks[task_id]
                        queue.remove(task_id)
                        break
            
            # Check general queue
            if not task:
                for task_id in self.general_queue[:]:
                    if task_id in self.tasks and self.tasks[task_id].status == TaskStatus.PENDING:
                        task = self.tasks[task_id]
                        self.general_queue.remove(task_id)
                        break
            
            if task:
                task.status = TaskStatus.ASSIGNED
                task.worker_id = worker_id
                task.assigned_at = datetime.now()
                worker.active_tasks.add(task.task_id)
                
                return {
                    "task_id": task.task_id,
                    "task_type": task.task_type,
                    "parameters": task.parameters,
                    "user_id": task.user_id
                }
            
            return None
    
    async def update_task_status(self, task_id: str, status: TaskStatus,
                               result: Optional[Dict[str, Any]] = None,
                               error_message: Optional[str] = None) -> bool:
        """Update task status."""
        await self._ensure_initialized()
        async with self._lock:
            if task_id not in self.tasks:
                return False
            
            task = self.tasks[task_id]
            task.status = status
            
            if status == TaskStatus.PROCESSING and not task.started_at:
                task.started_at = datetime.now()
            elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                task.completed_at = datetime.now()
                if task.worker_id and task.worker_id in self.workers:
                    self.workers[task.worker_id].active_tasks.discard(task_id)
            
            if result:
                task.result = result
            if error_message:
                task.error_message = error_message
            
            return True
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status."""
        await self._ensure_initialized()
        async with self._lock:
            if task_id not in self.tasks:
                return None
            
            task = self.tasks[task_id]
            return {
                "task_id": task.task_id,
                "status": task.status.value,
                "progress": 1.0 if task.status == TaskStatus.COMPLETED else 0.0,
                "message": f"Task {task.status.value}",
                "result": task.result,
                "error": task.error_message,
                "created_at": task.created_at.isoformat(),
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None
            }


async def test_basic_functionality():
    """Test basic task queue operations."""
    print("Testing basic functionality...")
    queue = HTTPTaskQueue()
    
    # Test worker registration
    success = await queue.register_worker("test-worker-1", "cpu", "test-key", "user1")
    assert success, "Worker registration failed"
    
    # Test task submission
    task_id = await queue.submit_task("process_url", {"url": "https://example.com"}, "user1")
    assert task_id is not None, "Task submission failed"
    
    # Test getting next task
    task = await queue.get_next_task("test-worker-1")
    assert task is not None, "Failed to get next task"
    assert task["task_id"] == task_id, "Task ID mismatch"
    assert task["task_type"] == "process_url", "Task type mismatch"
    
    # Test task status update
    success = await queue.update_task_status(task_id, TaskStatus.COMPLETED, {"content": "test result"})
    assert success, "Task status update failed"
    
    # Test getting task status
    status = await queue.get_task_status(task_id)
    assert status is not None, "Failed to get task status"
    assert status["status"] == TaskStatus.COMPLETED.value, "Status mismatch"
    assert status["result"]["content"] == "test result", "Result mismatch"
    
    print("✓ Basic functionality test passed")


async def test_user_routing():
    """Test user-specific task routing."""
    print("Testing user-specific routing...")
    queue = HTTPTaskQueue()
    
    # Register workers for different users
    await queue.register_worker("worker-user1", "gpu", "key1", "user1")
    await queue.register_worker("worker-user2", "gpu", "key2", "user2")
    await queue.register_worker("worker-general", "cpu", "key3", None)
    
    # Submit tasks for specific users
    task1_id = await queue.submit_task("process_url", {"url": "test1"}, "user1")
    task2_id = await queue.submit_task("process_url", {"url": "test2"}, "user2")
    task3_id = await queue.submit_task("process_url", {"url": "test3"}, None)
    
    # Worker for user1 should get user1's task
    task = await queue.get_next_task("worker-user1")
    assert task["task_id"] == task1_id, "User1 worker didn't get user1 task"
    
    # Worker for user2 should get user2's task
    task = await queue.get_next_task("worker-user2")
    assert task["task_id"] == task2_id, "User2 worker didn't get user2 task"
    
    # General worker should get general task
    task = await queue.get_next_task("worker-general")
    assert task["task_id"] == task3_id, "General worker didn't get general task"
    
    print("✓ User-specific routing test passed")


async def test_concurrency_limits():
    """Test worker concurrency limits."""
    print("Testing concurrency limits...")
    queue = HTTPTaskQueue()
    
    # Register worker with max 1 concurrent task
    await queue.register_worker("worker-1", "cpu", "key1", None, max_concurrent_tasks=1)
    
    # Submit multiple tasks
    task1_id = await queue.submit_task("process_url", {"url": "test1"})
    task2_id = await queue.submit_task("process_url", {"url": "test2"})
    
    # Worker should get first task
    task1 = await queue.get_next_task("worker-1")
    assert task1["task_id"] == task1_id, "Worker didn't get first task"
    
    # Worker should not get second task (at limit)
    task2 = await queue.get_next_task("worker-1")
    assert task2 is None, "Worker got second task despite being at limit"
    
    # Complete first task
    await queue.update_task_status(task1_id, TaskStatus.COMPLETED)
    
    # Now worker should get second task
    task2 = await queue.get_next_task("worker-1")
    assert task2["task_id"] == task2_id, "Worker didn't get second task after completing first"
    
    print("✓ Concurrency limits test passed")


async def main():
    """Run all tests."""
    print("Running HTTP Task Queue Tests...")
    print("=" * 40)
    
    await test_basic_functionality()
    await test_user_routing()
    await test_concurrency_limits()
    
    print("=" * 40)
    print("All tests passed! ✓")


if __name__ == "__main__":
    asyncio.run(main())
