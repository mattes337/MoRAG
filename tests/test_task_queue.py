"""Test HTTP task queue functionality."""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch
import sys
from pathlib import Path

# Add the source directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "morag" / "src"))

from morag.task_queue import HTTPTaskQueue, TaskStatus


@pytest.mark.asyncio
async def test_task_queue_basic_functionality():
    """Test basic task queue operations."""
    queue = HTTPTaskQueue()
    
    # Test worker registration
    success = await queue.register_worker(
        worker_id="test-worker-1",
        worker_type="cpu",
        api_key="test-key",
        user_id="user1"
    )
    assert success
    
    # Test task submission
    task_id = await queue.submit_task(
        task_type="process_url",
        parameters={"url": "https://example.com"},
        user_id="user1"
    )
    assert task_id is not None
    
    # Test getting next task
    task = await queue.get_next_task("test-worker-1")
    assert task is not None
    assert task["task_id"] == task_id
    assert task["task_type"] == "process_url"
    
    # Test task status update
    success = await queue.update_task_status(
        task_id=task_id,
        status=TaskStatus.COMPLETED,
        result={"content": "test result"}
    )
    assert success
    
    # Test getting task status
    status = await queue.get_task_status(task_id)
    assert status is not None
    assert status["status"] == TaskStatus.COMPLETED.value
    assert status["result"]["content"] == "test result"


@pytest.mark.asyncio
async def test_user_specific_routing():
    """Test that tasks are routed to user-specific workers."""
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
    assert task["task_id"] == task1_id
    
    # Worker for user2 should get user2's task
    task = await queue.get_next_task("worker-user2")
    assert task["task_id"] == task2_id
    
    # General worker should get general task
    task = await queue.get_next_task("worker-general")
    assert task["task_id"] == task3_id


@pytest.mark.asyncio
async def test_worker_concurrency_limits():
    """Test that workers respect concurrency limits."""
    queue = HTTPTaskQueue()
    
    # Register worker with max 1 concurrent task
    await queue.register_worker("worker-1", "cpu", "key1", None, max_concurrent_tasks=1)
    
    # Submit multiple tasks
    task1_id = await queue.submit_task("process_url", {"url": "test1"})
    task2_id = await queue.submit_task("process_url", {"url": "test2"})
    
    # Worker should get first task
    task1 = await queue.get_next_task("worker-1")
    assert task1["task_id"] == task1_id
    
    # Worker should not get second task (at limit)
    task2 = await queue.get_next_task("worker-1")
    assert task2 is None
    
    # Complete first task
    await queue.update_task_status(task1_id, TaskStatus.COMPLETED)
    
    # Now worker should get second task
    task2 = await queue.get_next_task("worker-1")
    assert task2["task_id"] == task2_id


@pytest.mark.asyncio
async def test_task_reassignment_on_worker_failure():
    """Test that tasks are reassigned when workers fail."""
    queue = HTTPTaskQueue()
    
    # Register worker and submit task
    await queue.register_worker("worker-1", "cpu", "key1", None)
    task_id = await queue.submit_task("process_url", {"url": "test"})
    
    # Worker gets task
    task = await queue.get_next_task("worker-1")
    assert task["task_id"] == task_id
    
    # Simulate worker failure by unregistering
    await queue.unregister_worker("worker-1")
    
    # Register new worker
    await queue.register_worker("worker-2", "cpu", "key2", None)
    
    # New worker should get the reassigned task
    task = await queue.get_next_task("worker-2")
    assert task["task_id"] == task_id


if __name__ == "__main__":
    # Run basic test
    async def run_basic_test():
        print("Testing HTTP Task Queue...")
        await test_task_queue_basic_functionality()
        print("✓ Basic functionality test passed")
        
        await test_user_specific_routing()
        print("✓ User-specific routing test passed")
        
        await test_worker_concurrency_limits()
        print("✓ Worker concurrency limits test passed")
        
        await test_task_reassignment_on_worker_failure()
        print("✓ Task reassignment test passed")
        
        print("All tests passed!")
    
    asyncio.run(run_basic_test())
