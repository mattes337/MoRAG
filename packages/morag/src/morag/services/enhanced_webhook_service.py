"""Enhanced webhook service for detailed progress notifications."""

import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
import structlog
import httpx
from urllib.parse import urlparse

from morag.api_models.models import WebhookProgressNotification

logger = structlog.get_logger(__name__)


class EnhancedWebhookService:
    """Service for sending detailed webhook progress notifications."""
    
    def __init__(self, max_retries: int = 3, timeout_seconds: int = 10):
        """Initialize webhook service.
        
        Args:
            max_retries: Maximum number of retry attempts
            timeout_seconds: Request timeout in seconds
        """
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        
    def validate_webhook_url(self, url: str, allow_localhost: bool = False) -> bool:
        """Validate webhook URL for security.
        
        Args:
            url: Webhook URL to validate
            allow_localhost: Whether to allow localhost URLs (for development)
            
        Returns:
            True if URL is valid and safe
        """
        try:
            parsed = urlparse(url)
            
            # Must be HTTP or HTTPS
            if parsed.scheme not in ['http', 'https']:
                logger.warning("Invalid webhook URL scheme", url=url, scheme=parsed.scheme)
                return False
            
            # Check for localhost/internal IPs unless explicitly allowed
            if not allow_localhost:
                hostname = parsed.hostname
                if hostname in ['localhost', '127.0.0.1', '0.0.0.0'] or hostname.startswith('192.168.') or hostname.startswith('10.'):
                    logger.warning("Webhook URL points to internal/localhost address", url=url)
                    return False
            
            return True
            
        except Exception as e:
            logger.error("Failed to parse webhook URL", url=url, error=str(e))
            return False
    
    async def send_progress_notification(
        self,
        webhook_url: str,
        task_id: str,
        document_id: Optional[str],
        step: str,
        status: str,
        progress_percent: float,
        data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        auth_token: Optional[str] = None
    ) -> bool:
        """Send progress notification webhook.
        
        Args:
            webhook_url: Target webhook URL
            task_id: Background task ID
            document_id: Document ID if provided
            step: Processing step name
            status: Step status (started|completed|failed)
            progress_percent: Overall progress (0-100)
            data: Step-specific data
            error_message: Error message if status is failed
            auth_token: Optional bearer token for authentication
            
        Returns:
            True if notification was sent successfully
        """
        if not webhook_url:
            return False
            
        # Create notification payload
        notification = WebhookProgressNotification(
            task_id=task_id,
            document_id=document_id,
            step=step,
            status=status,
            progress_percent=progress_percent,
            timestamp=datetime.now(timezone.utc).isoformat(),
            data=data,
            error_message=error_message
        )
        
        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "MoRAG-Webhook-Service/1.0"
        }
        
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        
        # Send with retries
        for attempt in range(self.max_retries + 1):
            try:
                logger.info("Sending webhook notification",
                           webhook_url=webhook_url,
                           task_id=task_id,
                           step=step,
                           status=status,
                           attempt=attempt + 1)
                
                # Use asyncio to run requests in thread pool
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: httpx.post(
                        webhook_url,
                        json=notification.dict(),
                        headers=headers,
                        timeout=self.timeout_seconds
                    )
                )
                
                if response.status_code in [200, 201, 202]:
                    logger.info("Webhook notification sent successfully",
                               webhook_url=webhook_url,
                               task_id=task_id,
                               status_code=response.status_code)
                    return True
                else:
                    logger.warning("Webhook notification failed",
                                  webhook_url=webhook_url,
                                  task_id=task_id,
                                  status_code=response.status_code,
                                  response_text=response.text[:200])
                    
            except httpx.TimeoutException:
                logger.warning("Webhook notification timeout",
                              webhook_url=webhook_url,
                              task_id=task_id,
                              attempt=attempt + 1)
            except httpx.ConnectError:
                logger.warning("Webhook notification connection error",
                              webhook_url=webhook_url,
                              task_id=task_id,
                              attempt=attempt + 1)
            except Exception as e:
                logger.error("Webhook notification error",
                            webhook_url=webhook_url,
                            task_id=task_id,
                            error=str(e),
                            attempt=attempt + 1)
            
            # Wait before retry (exponential backoff)
            if attempt < self.max_retries:
                wait_time = 2 ** attempt
                logger.info("Retrying webhook notification",
                           webhook_url=webhook_url,
                           task_id=task_id,
                           wait_seconds=wait_time)
                await asyncio.sleep(wait_time)
        
        logger.error("Webhook notification failed after all retries",
                    webhook_url=webhook_url,
                    task_id=task_id,
                    max_retries=self.max_retries)
        return False
    
    async def send_step_started(
        self,
        webhook_url: str,
        task_id: str,
        document_id: Optional[str],
        step: str,
        progress_percent: float,
        auth_token: Optional[str] = None
    ) -> bool:
        """Send step started notification."""
        return await self.send_progress_notification(
            webhook_url=webhook_url,
            task_id=task_id,
            document_id=document_id,
            step=step,
            status="started",
            progress_percent=progress_percent,
            auth_token=auth_token
        )
    
    async def send_step_completed(
        self,
        webhook_url: str,
        task_id: str,
        document_id: Optional[str],
        step: str,
        progress_percent: float,
        data: Optional[Dict[str, Any]] = None,
        auth_token: Optional[str] = None
    ) -> bool:
        """Send step completed notification."""
        return await self.send_progress_notification(
            webhook_url=webhook_url,
            task_id=task_id,
            document_id=document_id,
            step=step,
            status="completed",
            progress_percent=progress_percent,
            data=data,
            auth_token=auth_token
        )
    
    async def send_step_failed(
        self,
        webhook_url: str,
        task_id: str,
        document_id: Optional[str],
        step: str,
        progress_percent: float,
        error_message: str,
        auth_token: Optional[str] = None
    ) -> bool:
        """Send step failed notification."""
        return await self.send_progress_notification(
            webhook_url=webhook_url,
            task_id=task_id,
            document_id=document_id,
            step=step,
            status="failed",
            progress_percent=progress_percent,
            error_message=error_message,
            auth_token=auth_token
        )


# Global webhook service instance
_webhook_service = None


def get_webhook_service() -> EnhancedWebhookService:
    """Get global webhook service instance."""
    global _webhook_service
    if _webhook_service is None:
        _webhook_service = EnhancedWebhookService()
    return _webhook_service
