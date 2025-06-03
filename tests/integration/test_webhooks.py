"""Integration tests for webhook functionality."""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime

from morag.services.webhook import webhook_service
from morag.services.status_history import status_history, StatusEvent
from morag.services.task_manager import task_manager


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
        
        # Check the payload structure
        call_args = mock_post.call_args
        assert call_args[1]['json']['event_type'] == 'task_completed'
        assert call_args[1]['json']['data']['task_id'] == 'test-123'
        assert 'timestamp' in call_args[1]['json']


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


@pytest.mark.asyncio
async def test_webhook_timeout():
    """Test webhook timeout handling."""
    
    with patch('httpx.AsyncClient.post') as mock_post:
        # Mock timeout exception
        import httpx
        mock_post.side_effect = httpx.TimeoutException("Request timeout")
        
        result = await webhook_service.send_webhook(
            webhook_url="https://example.com/webhook",
            payload={"task_id": "test-123", "status": "completed"},
            event_type="task_completed"
        )
        
        assert result is False
        assert mock_post.call_count == 3  # Should retry 3 times


@pytest.mark.asyncio
async def test_webhook_no_url():
    """Test webhook with no URL provided."""
    
    result = await webhook_service.send_webhook(
        webhook_url="",
        payload={"task_id": "test-123", "status": "completed"},
        event_type="task_completed"
    )
    
    assert result is True  # Should return True when no webhook to send


@pytest.mark.asyncio
async def test_task_started_webhook():
    """Test task started webhook."""
    
    with patch('httpx.AsyncClient.post') as mock_post:
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        result = await webhook_service.send_task_started(
            task_id="test-123",
            webhook_url="https://example.com/webhook",
            metadata={"source_type": "document"}
        )
        
        assert result is True
        call_args = mock_post.call_args
        assert call_args[1]['json']['event_type'] == 'task_started'
        assert call_args[1]['json']['data']['status'] == 'started'


@pytest.mark.asyncio
async def test_task_progress_webhook():
    """Test task progress webhook."""
    
    with patch('httpx.AsyncClient.post') as mock_post:
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        result = await webhook_service.send_task_progress(
            task_id="test-123",
            webhook_url="https://example.com/webhook",
            progress=0.5,
            message="Processing...",
            metadata={"stage": "embedding_generation"}
        )
        
        assert result is True
        call_args = mock_post.call_args
        assert call_args[1]['json']['event_type'] == 'task_progress'
        assert call_args[1]['json']['data']['progress'] == 0.5


@pytest.mark.asyncio
async def test_task_completed_webhook():
    """Test task completed webhook."""
    
    with patch('httpx.AsyncClient.post') as mock_post:
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        result = await webhook_service.send_task_completed(
            task_id="test-123",
            webhook_url="https://example.com/webhook",
            result={"embeddings_stored": 5, "chunks_created": 10},
            metadata={"source_type": "document"}
        )
        
        assert result is True
        call_args = mock_post.call_args
        assert call_args[1]['json']['event_type'] == 'task_completed'
        assert call_args[1]['json']['data']['status'] == 'completed'


@pytest.mark.asyncio
async def test_task_failed_webhook():
    """Test task failed webhook."""
    
    with patch('httpx.AsyncClient.post') as mock_post:
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        result = await webhook_service.send_task_failed(
            task_id="test-123",
            webhook_url="https://example.com/webhook",
            error="File not found",
            metadata={"source_type": "document"}
        )
        
        assert result is True
        call_args = mock_post.call_args
        assert call_args[1]['json']['event_type'] == 'task_failed'
        assert call_args[1]['json']['data']['status'] == 'failed'
        assert call_args[1]['json']['data']['error'] == 'File not found'


def test_status_history_add_event():
    """Test adding status events to history."""
    
    with patch.object(status_history, 'redis_client') as mock_redis:
        status_history.add_status_event(
            task_id="test-123",
            status="started",
            progress=0.0,
            message="Task started",
            metadata={"source_type": "document"}
        )
        
        # Verify Redis operations
        mock_redis.lpush.assert_called_once()
        mock_redis.expire.assert_called_once()
        mock_redis.ltrim.assert_called_once()


def test_status_history_get_task_history():
    """Test getting task history."""
    
    # Mock Redis data
    mock_events = [
        json.dumps({
            'timestamp': '2024-01-01T10:00:00',
            'status': 'started',
            'progress': 0.0,
            'message': 'Task started',
            'metadata': {'source_type': 'document'}
        }),
        json.dumps({
            'timestamp': '2024-01-01T10:01:00',
            'status': 'progress',
            'progress': 0.5,
            'message': 'Processing...',
            'metadata': {'stage': 'embedding_generation'}
        })
    ]
    
    with patch.object(status_history, 'redis_client') as mock_redis:
        mock_redis.lrange.return_value = mock_events
        
        history = status_history.get_task_history("test-123")
        
        assert len(history) == 2
        assert history[0].status == 'progress'  # Should be sorted newest first
        assert history[1].status == 'started'


def test_status_history_get_recent_events():
    """Test getting recent events across all tasks."""
    
    with patch.object(status_history, 'redis_client') as mock_redis:
        # Mock keys and events
        mock_redis.keys.return_value = [b'task_history:test-123', b'task_history:test-456']
        
        with patch.object(status_history, 'get_task_history') as mock_get_history:
            # Mock recent events
            mock_get_history.return_value = [
                StatusEvent(
                    timestamp=datetime.utcnow(),
                    status='completed',
                    progress=1.0,
                    message='Task completed',
                    metadata={'source_type': 'document'}
                )
            ]
            
            events = status_history.get_recent_events(24)
            
            assert len(events) > 0
            assert 'task_id' in events[0]
            assert events[0]['status'] == 'completed'


@pytest.mark.asyncio
async def test_task_manager_webhook_integration():
    """Test task manager webhook integration."""
    
    with patch('morag.services.webhook.webhook_service.send_task_started') as mock_webhook:
        mock_webhook.return_value = True
        
        await task_manager.handle_task_started(
            task_id="test-123",
            metadata={"webhook_url": "https://example.com/webhook", "source_type": "document"}
        )
        
        mock_webhook.assert_called_once_with(
            task_id="test-123",
            webhook_url="https://example.com/webhook",
            metadata={"webhook_url": "https://example.com/webhook", "source_type": "document"}
        )


@pytest.mark.asyncio
async def test_task_manager_completion_webhook():
    """Test task manager completion webhook."""
    
    with patch('morag.services.webhook.webhook_service.send_task_completed') as mock_webhook:
        mock_webhook.return_value = True
        
        await task_manager.handle_task_completion(
            task_id="test-123",
            result={"status": "success", "embeddings_stored": 5},
            metadata={"webhook_url": "https://example.com/webhook", "source_type": "document"}
        )
        
        mock_webhook.assert_called_once()


@pytest.mark.asyncio
async def test_task_manager_failure_webhook():
    """Test task manager failure webhook."""
    
    with patch('morag.services.webhook.webhook_service.send_task_failed') as mock_webhook:
        mock_webhook.return_value = True
        
        await task_manager.handle_task_completion(
            task_id="test-123",
            result={"status": "failure", "error": "File not found"},
            metadata={"webhook_url": "https://example.com/webhook", "source_type": "document"}
        )
        
        mock_webhook.assert_called_once_with(
            task_id="test-123",
            webhook_url="https://example.com/webhook",
            error="File not found",
            metadata={"webhook_url": "https://example.com/webhook", "source_type": "document"}
        )
