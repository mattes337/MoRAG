import asyncio
import time

import pytest
from morag_services.tasks import process_document_task
from src.morag.services.task_manager import TaskStatus, task_manager


def test_task_submission():
    """Test task submission and status tracking."""
    # Submit a test task
    result = process_document_task.delay(
        file_path="/tmp/test.pdf", source_type="document", metadata={"test": True}
    )

    # Check task was submitted
    assert result.task_id is not None

    # Check initial status
    task_info = task_manager.get_task_status(result.task_id)
    assert task_info.task_id == result.task_id
    assert task_info.status in [
        TaskStatus.PENDING,
        TaskStatus.STARTED,
        TaskStatus.SUCCESS,
    ]


def test_queue_stats():
    """Test queue statistics."""
    stats = task_manager.get_queue_stats()

    assert "active_tasks" in stats
    assert "queues" in stats
    assert isinstance(stats["workers"], list)


def test_task_progress_tracking():
    """Test task progress tracking functionality."""
    # Submit a test task
    result = process_document_task.delay(
        file_path="/tmp/test.pdf", source_type="document", metadata={"test": True}
    )

    # Wait a moment for task to start
    time.sleep(1)

    # Check task status
    task_info = task_manager.get_task_status(result.task_id)
    assert task_info.task_id == result.task_id

    # Progress should be set (either 0.0 for pending or 0.1 for started)
    assert task_info.progress is not None
    assert 0.0 <= task_info.progress <= 1.0


def test_task_cancellation():
    """Test task cancellation functionality."""
    # Submit a test task
    result = process_document_task.delay(
        file_path="/tmp/test.pdf", source_type="document", metadata={"test": True}
    )

    # Cancel the task
    cancelled = task_manager.cancel_task(result.task_id)
    assert cancelled is True


def test_active_tasks_listing():
    """Test listing active tasks."""
    # Get active tasks
    active_tasks = task_manager.get_active_tasks()

    # Should return a list (may be empty if no workers running)
    assert isinstance(active_tasks, list)
