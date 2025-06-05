"""Webhook service for sending notifications."""

from typing import Dict, Any, Optional
import httpx
import structlog
import asyncio
from datetime import datetime

from morag_core.config import settings
from morag_core.exceptions import ExternalServiceError

logger = structlog.get_logger()

class WebhookService:
    """Service for sending webhook notifications."""
    
    def __init__(self):
        self.timeout = settings.webhook_timeout
        self.max_retries = settings.webhook_max_retries
        self.retry_delay = settings.webhook_retry_delay
    
    async def send_webhook(
        self,
        webhook_url: str,
        payload: Dict[str, Any],
        event_type: str = "task_completed"
    ) -> bool:
        """Send webhook notification with retry logic."""
        
        if not webhook_url:
            return True  # No webhook to send
        
        # Prepare webhook payload
        webhook_payload = {
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "data": payload
        }
        
        logger.info(
            "Sending webhook",
            webhook_url=webhook_url,
            event_type=event_type,
            task_id=payload.get("task_id")
        )
        
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        webhook_url,
                        json=webhook_payload,
                        headers={
                            "Content-Type": "application/json",
                            "User-Agent": "MoRAG-Webhook/1.0"
                        }
                    )
                    
                    if response.status_code < 400:
                        logger.info(
                            "Webhook sent successfully",
                            webhook_url=webhook_url,
                            status_code=response.status_code,
                            attempt=attempt + 1
                        )
                        return True
                    else:
                        logger.warning(
                            "Webhook failed with HTTP error",
                            webhook_url=webhook_url,
                            status_code=response.status_code,
                            response_text=response.text[:200],
                            attempt=attempt + 1
                        )
                        
            except httpx.TimeoutException:
                logger.warning(
                    "Webhook timeout",
                    webhook_url=webhook_url,
                    timeout=self.timeout,
                    attempt=attempt + 1
                )
            except Exception as e:
                logger.warning(
                    "Webhook failed with exception",
                    webhook_url=webhook_url,
                    error=str(e),
                    attempt=attempt + 1
                )
            
            # Wait before retry (except on last attempt)
            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        logger.error(
            "Webhook failed after all retries",
            webhook_url=webhook_url,
            max_retries=self.max_retries
        )
        return False
    
    async def send_task_started(self, task_id: str, webhook_url: str, metadata: Dict[str, Any]) -> bool:
        """Send task started notification."""
        payload = {
            "task_id": task_id,
            "status": "started",
            "message": "Task processing started",
            "metadata": metadata
        }
        return await self.send_webhook(webhook_url, payload, "task_started")
    
    async def send_task_progress(
        self,
        task_id: str,
        webhook_url: str,
        progress: float,
        message: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """Send task progress notification."""
        payload = {
            "task_id": task_id,
            "status": "progress",
            "progress": progress,
            "message": message,
            "metadata": metadata
        }
        return await self.send_webhook(webhook_url, payload, "task_progress")
    
    async def send_task_completed(
        self,
        task_id: str,
        webhook_url: str,
        result: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> bool:
        """Send task completion notification."""
        payload = {
            "task_id": task_id,
            "status": "completed",
            "message": "Task completed successfully",
            "result": result,
            "metadata": metadata
        }
        return await self.send_webhook(webhook_url, payload, "task_completed")
    
    async def send_task_failed(
        self,
        task_id: str,
        webhook_url: str,
        error: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """Send task failure notification."""
        payload = {
            "task_id": task_id,
            "status": "failed",
            "message": "Task failed",
            "error": error,
            "metadata": metadata
        }
        return await self.send_webhook(webhook_url, payload, "task_failed")

# Global instance
webhook_service = WebhookService()
