"""Webhook notification system for stage completions."""

import asyncio
import json
from typing import Optional, Dict, Any
from datetime import datetime
import structlog
import httpx

from .models import StageResult, StageContext, StageType, StageStatus

logger = structlog.get_logger(__name__)


class WebhookNotifier:
    """Handles webhook notifications for stage completion events."""

    def __init__(self,
                 timeout_seconds: float = 30.0,
                 max_retries: int = 3,
                 retry_delay: float = 1.0):
        """Initialize webhook notifier.

        Args:
            timeout_seconds: HTTP request timeout
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    async def notify_stage_completion(self,
                                    webhook_url: str,
                                    stage_result: StageResult,
                                    context: StageContext) -> bool:
        """Send webhook notification for stage completion.

        Args:
            webhook_url: URL to send webhook to
            stage_result: Result of the completed stage
            context: Stage execution context

        Returns:
            True if notification was sent successfully, False otherwise
        """
        if not webhook_url:
            return False

        payload = self._create_webhook_payload(stage_result, context)

        logger.info("Sending webhook notification",
                   webhook_url=webhook_url,
                   stage_type=stage_result.stage_type.value,
                   status=stage_result.status.value)

        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    response = await client.post(
                        webhook_url,
                        json=payload,
                        headers={
                            "Content-Type": "application/json",
                            "User-Agent": "MoRAG-Stages/0.1.0"
                        }
                    )

                    if response.status_code < 400:
                        logger.info("Webhook notification sent successfully",
                                   webhook_url=webhook_url,
                                   status_code=response.status_code,
                                   attempt=attempt + 1)
                        return True
                    else:
                        logger.warning("Webhook notification failed",
                                     webhook_url=webhook_url,
                                     status_code=response.status_code,
                                     response_text=response.text,
                                     attempt=attempt + 1)

            except Exception as e:
                logger.error("Webhook notification error",
                           webhook_url=webhook_url,
                           error=str(e),
                           attempt=attempt + 1)

            # Wait before retry (except on last attempt)
            if attempt < self.max_retries:
                await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff

        logger.error("Webhook notification failed after all retries",
                    webhook_url=webhook_url,
                    max_retries=self.max_retries)
        return False

    def _create_webhook_payload(self,
                               stage_result: StageResult,
                               context: StageContext) -> Dict[str, Any]:
        """Create webhook payload for stage completion.

        Args:
            stage_result: Stage execution result
            context: Stage execution context

        Returns:
            Webhook payload dictionary
        """
        return {
            "event": "stage_completed",
            "timestamp": datetime.now().isoformat(),
            "stage": {
                "type": stage_result.stage_type.value,
                "status": stage_result.status.value,
                "execution_time": stage_result.metadata.execution_time,
                "start_time": stage_result.metadata.start_time.isoformat() if stage_result.metadata.start_time else None,
                "end_time": stage_result.metadata.end_time.isoformat() if stage_result.metadata.end_time else None,
                "error_message": stage_result.error_message
            },
            "files": {
                "input_files": stage_result.metadata.input_files,
                "output_files": [str(f) for f in stage_result.output_files]
            },
            "context": {
                "source_path": str(context.source_path),
                "output_dir": str(context.output_dir),
                "total_stages_completed": len([r for r in context.stage_results.values() if r.success]),
                "total_stages_failed": len([r for r in context.stage_results.values() if r.failed])
            },
            "metadata": {
                "config_used": stage_result.metadata.config_used,
                "metrics": stage_result.metadata.metrics,
                "warnings": stage_result.metadata.warnings
            }
        }

    async def notify_pipeline_completion(self,
                                       webhook_url: str,
                                       context: StageContext,
                                       success: bool,
                                       error_message: Optional[str] = None) -> bool:
        """Send webhook notification for complete pipeline completion.

        Args:
            webhook_url: URL to send webhook to
            context: Stage execution context
            success: Whether pipeline completed successfully
            error_message: Error message if pipeline failed

        Returns:
            True if notification was sent successfully, False otherwise
        """
        if not webhook_url:
            return False

        payload = self._create_pipeline_webhook_payload(context, success, error_message)

        logger.info("Sending pipeline completion webhook",
                   webhook_url=webhook_url,
                   success=success)

        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    response = await client.post(
                        webhook_url,
                        json=payload,
                        headers={
                            "Content-Type": "application/json",
                            "User-Agent": "MoRAG-Stages/0.1.0"
                        }
                    )

                    if response.status_code < 400:
                        logger.info("Pipeline webhook notification sent successfully",
                                   webhook_url=webhook_url,
                                   status_code=response.status_code)
                        return True
                    else:
                        logger.warning("Pipeline webhook notification failed",
                                     webhook_url=webhook_url,
                                     status_code=response.status_code,
                                     response_text=response.text)

            except Exception as e:
                logger.error("Pipeline webhook notification error",
                           webhook_url=webhook_url,
                           error=str(e),
                           attempt=attempt + 1)

            # Wait before retry (except on last attempt)
            if attempt < self.max_retries:
                await asyncio.sleep(self.retry_delay * (2 ** attempt))

        return False

    def _create_pipeline_webhook_payload(self,
                                       context: StageContext,
                                       success: bool,
                                       error_message: Optional[str] = None) -> Dict[str, Any]:
        """Create webhook payload for pipeline completion.

        Args:
            context: Stage execution context
            success: Whether pipeline completed successfully
            error_message: Error message if pipeline failed

        Returns:
            Webhook payload dictionary
        """
        completed_stages = [r for r in context.stage_results.values() if r.success]
        failed_stages = [r for r in context.stage_results.values() if r.failed]
        skipped_stages = [r for r in context.stage_results.values() if r.skipped]

        total_execution_time = sum(r.metadata.execution_time for r in context.stage_results.values())

        return {
            "event": "pipeline_completed",
            "timestamp": datetime.now().isoformat(),
            "pipeline": {
                "success": success,
                "error_message": error_message,
                "total_execution_time": total_execution_time,
                "stages_completed": len(completed_stages),
                "stages_failed": len(failed_stages),
                "stages_skipped": len(skipped_stages)
            },
            "context": {
                "source_path": str(context.source_path),
                "output_dir": str(context.output_dir),
                "intermediate_files": [str(f) for f in context.intermediate_files]
            },
            "stages": [
                {
                    "type": result.stage_type.value,
                    "status": result.status.value,
                    "execution_time": result.metadata.execution_time,
                    "output_files": [str(f) for f in result.output_files],
                    "error_message": result.error_message
                }
                for result in context.stage_results.values()
            ]
        }
