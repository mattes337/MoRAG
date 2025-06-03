# MoRAG Webhook Integration Guide

## Overview

MoRAG now supports comprehensive webhook notifications for task lifecycle events, allowing you to receive real-time updates about document processing, status changes, and completion events.

## Features

### ✅ Webhook Notifications
- **Task Started**: Notified when a task begins processing
- **Task Progress**: Updates at 25%, 50%, and 75% completion milestones
- **Task Completed**: Notification when task finishes successfully
- **Task Failed**: Notification when task encounters errors

### ✅ Status History
- Complete event history for each task
- Redis-based storage with 7-day retention
- Detailed progress tracking with timestamps
- Metadata preservation for debugging

### ✅ Retry Logic
- Automatic retry on webhook delivery failures
- Configurable retry attempts (default: 3)
- Exponential backoff delay
- Comprehensive error logging

## Quick Start

### 1. Start the Webhook Receiver (for testing)

```bash
python webhook_receiver.py
```

This starts a simple webhook receiver on `http://localhost:8001/webhook`

### 2. Submit a Task with Webhook

```bash
curl -X POST "http://localhost:8000/api/v1/ingest/file" \
  -H "Authorization: Bearer test-api-key" \
  -F "source_type=document" \
  -F "file=@your-document.pdf" \
  -F "webhook_url=http://localhost:8001/webhook"
```

### 3. Monitor Webhooks

Visit `http://localhost:8001/webhooks` to see all received webhook notifications.

## API Endpoints

### Status Tracking

#### Get Task Status
```http
GET /api/v1/status/{task_id}
Authorization: Bearer {api_key}
```

#### Get Task History
```http
GET /api/v1/status/{task_id}/history
Authorization: Bearer {api_key}
```

#### Get Recent Events
```http
GET /api/v1/status/events/recent?hours=24
Authorization: Bearer {api_key}
```

## Webhook Payload Format

### Task Started
```json
{
  "event_type": "task_started",
  "timestamp": "2024-01-01T10:00:00Z",
  "data": {
    "task_id": "abc123",
    "status": "started",
    "message": "Task processing started",
    "metadata": {
      "source_type": "document",
      "webhook_url": "https://your-app.com/webhook"
    }
  }
}
```

### Task Progress
```json
{
  "event_type": "task_progress",
  "timestamp": "2024-01-01T10:01:00Z",
  "data": {
    "task_id": "abc123",
    "status": "progress",
    "progress": 0.5,
    "message": "Processing...",
    "metadata": {
      "stage": "embedding_generation"
    }
  }
}
```

### Task Completed
```json
{
  "event_type": "task_completed",
  "timestamp": "2024-01-01T10:05:00Z",
  "data": {
    "task_id": "abc123",
    "status": "completed",
    "message": "Task completed successfully",
    "result": {
      "embeddings_stored": 15,
      "chunks_created": 25,
      "processing_time": 45.2
    },
    "metadata": {
      "source_type": "document"
    }
  }
}
```

### Task Failed
```json
{
  "event_type": "task_failed",
  "timestamp": "2024-01-01T10:02:00Z",
  "data": {
    "task_id": "abc123",
    "status": "failed",
    "message": "Task failed",
    "error": "File format not supported",
    "metadata": {
      "source_type": "document"
    }
  }
}
```

## Configuration

### Environment Variables

```bash
# Webhook settings
WEBHOOK_TIMEOUT=30          # Timeout in seconds
WEBHOOK_MAX_RETRIES=3       # Number of retry attempts
WEBHOOK_RETRY_DELAY=5       # Base delay between retries
```

### Webhook URL Requirements

- Must be a valid HTTP/HTTPS URL
- Should respond with 2xx status code for success
- Timeout: 30 seconds (configurable)
- Content-Type: application/json

## Testing

### Run Integration Tests
```bash
pytest tests/integration/test_webhooks.py -v
```

### Run Demo Script
```bash
# Terminal 1: Start webhook receiver
python webhook_receiver.py

# Terminal 2: Start MoRAG API
python -m uvicorn src.morag.main:app --reload

# Terminal 3: Run demo
python test_webhook_demo.py
```

## Security Considerations

### Webhook Security
- Use HTTPS URLs in production
- Implement webhook signature verification
- Validate webhook payloads
- Rate limit webhook endpoints

### API Security
- Always use API keys for authentication
- Restrict webhook URLs to trusted domains
- Monitor webhook delivery failures

## Troubleshooting

### Common Issues

#### Webhook Not Received
1. Check if webhook URL is accessible
2. Verify webhook endpoint returns 2xx status
3. Check MoRAG logs for delivery errors
4. Ensure firewall allows outbound connections

#### Task Status Not Updating
1. Verify Redis is running and accessible
2. Check Celery worker status
3. Review task logs for errors

#### History Not Available
1. Confirm Redis connection
2. Check if task ID is correct
3. Verify history retention settings

### Debug Commands

```bash
# Check webhook delivery logs
docker logs morag-api | grep webhook

# Check Redis status history
redis-cli keys "task_history:*"

# Check Celery task status
celery -A morag.core.celery_app inspect active
```

## Production Deployment

### Recommended Setup
- Use HTTPS for webhook URLs
- Implement webhook signature verification
- Set up monitoring for webhook delivery failures
- Configure appropriate retry policies
- Use Redis Cluster for high availability

### Monitoring
- Track webhook delivery success rates
- Monitor task completion times
- Alert on high failure rates
- Log webhook response times

## Examples

### n8n Integration
```javascript
// n8n webhook node configuration
{
  "httpMethod": "POST",
  "path": "morag-webhook",
  "responseMode": "responseNode",
  "options": {}
}
```

### Zapier Integration
```javascript
// Zapier webhook trigger
{
  "url": "https://hooks.zapier.com/hooks/catch/123456/abcdef/",
  "method": "POST",
  "headers": {
    "Content-Type": "application/json"
  }
}
```

### Custom Application
```python
from flask import Flask, request

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def handle_webhook():
    data = request.json
    event_type = data.get('event_type')
    task_data = data.get('data', {})
    
    if event_type == 'task_completed':
        # Handle successful completion
        process_completed_task(task_data)
    elif event_type == 'task_failed':
        # Handle failure
        handle_task_failure(task_data)
    
    return {'status': 'received'}
```

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the integration tests
3. Check MoRAG logs for detailed error messages
4. Create an issue with reproduction steps
