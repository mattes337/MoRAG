from typing import Dict, Any, Optional
import structlog
import asyncio
from datetime import datetime

from morag_services.celery_app import celery_app, BaseTask
from src.morag.services.task_manager import task_manager

logger = structlog.get_logger()

class ProcessingTask(BaseTask):
    """Base class for content processing tasks."""

    def __init__(self):
        super().__init__()
        self.start_time = None
        self.webhook_url = None

    def on_start(self, task_id, args, kwargs):
        """Called when task starts."""
        self.start_time = datetime.utcnow()

        # Extract webhook URL from metadata
        metadata = self._get_metadata(args, kwargs)
        self.webhook_url = metadata.get('webhook_url')

        # Send webhook notification (run async code synchronously)
        if self.webhook_url:
            try:
                # Try to run the async function synchronously
                asyncio.run(task_manager.handle_task_started(
                    task_id=task_id,
                    metadata=metadata
                ))
            except RuntimeError:
                # If there's already an event loop running, skip webhook for now
                logger.warning("Could not send webhook notification - event loop already running", task_id=task_id)

        task_manager.update_task_progress(
            task_id,
            progress=0.0,
            message="Task started",
            metadata={'started_at': self.start_time.isoformat()}
        )

    def on_success(self, retval, task_id, args, kwargs):
        """Handle task success."""
        super().on_success(retval, task_id, args, kwargs)

        # Send webhook notification (run async code synchronously)
        if self.webhook_url:
            metadata = self._get_metadata(args, kwargs)
            try:
                asyncio.run(task_manager.handle_task_completion(
                    task_id=task_id,
                    result=retval,
                    metadata=metadata
                ))
            except RuntimeError:
                logger.warning("Could not send webhook notification - event loop already running", task_id=task_id)

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure."""
        super().on_failure(exc, task_id, args, kwargs, einfo)

        # Send webhook notification (run async code synchronously)
        if self.webhook_url:
            metadata = self._get_metadata(args, kwargs)
            result = {
                'status': 'failure',
                'error': str(exc)
            }
            try:
                asyncio.run(task_manager.handle_task_completion(
                    task_id=task_id,
                    result=result,
                    metadata=metadata
                ))
            except RuntimeError:
                logger.warning("Could not send webhook notification - event loop already running", task_id=task_id)

    def update_progress(self, progress: float, message: str = None, **metadata):
        """Update task progress with webhook support."""
        task_manager.update_task_progress(
            self.request.id,
            progress=progress,
            message=message,
            metadata=metadata
        )

        # Send webhook for significant milestones (run async code synchronously)
        if self.webhook_url and progress in [0.25, 0.5, 0.75]:
            try:
                asyncio.run(task_manager.handle_task_progress(
                    task_id=self.request.id,
                    progress=progress,
                    message=message or f"Progress: {progress*100:.0f}%",
                    metadata=metadata
                ))
            except RuntimeError:
                logger.warning("Could not send webhook notification - event loop already running", task_id=self.request.id)

    def _get_metadata(self, args, kwargs) -> Dict[str, Any]:
        """Extract metadata from task arguments."""
        # Check for metadata in various positions
        if len(args) > 2 and isinstance(args[2], dict):
            return args[2]
        elif 'metadata' in kwargs and isinstance(kwargs['metadata'], dict):
            return kwargs['metadata']
        elif len(args) > 3 and isinstance(args[3], dict):
            return args[3]
        return {}

    def update_status(self, status: str, metadata: Dict[str, Any] = None):
        """Update task status."""
        if status == "PROCESSING":
            progress = 0.5  # Default progress for processing
            if metadata and "stage" in metadata:
                stage = metadata["stage"]
                if stage == "audio_transcription":
                    progress = 0.2
                elif stage == "text_chunking":
                    progress = 0.4
                elif stage == "embedding_generation":
                    progress = 0.6
                elif stage == "language_detection":
                    progress = 0.3
                elif stage == "segment_transcription":
                    progress = 0.5

            # Create a copy of metadata without 'message' key to avoid conflicts
            safe_metadata = {k: v for k, v in (metadata or {}).items() if k != 'message'}
            self.update_progress(progress, f"Status: {status}", **safe_metadata)
        elif status == "SUCCESS":
            safe_metadata = {k: v for k, v in (metadata or {}).items() if k != 'message'}
            self.update_progress(1.0, "Task completed successfully", **safe_metadata)
        elif status == "FAILURE":
            safe_metadata = {k: v for k, v in (metadata or {}).items() if k != 'message'}
            self.update_progress(0.0, "Task failed", **safe_metadata)

    def log_step(self, step: str, **kwargs):
        """Log a processing step."""
        logger.info(
            f"Task step: {step}",
            task_id=self.request.id,
            task_name=self.name,
            **kwargs
        )

# Test task to verify Celery is working
@celery_app.task(bind=True, base=ProcessingTask)
def test_task(self, message: str = "Hello from Celery!"):
    """Simple test task to verify Celery is working."""
    logger.info("Test task started", message=message)

    self.update_status("PROCESSING", {"stage": "testing"})
    self.update_progress(0.5, "Processing test task")

    result = {
        "status": "success",
        "message": message,
        "task_id": self.request.id
    }

    self.update_status("SUCCESS", result)
    logger.info("Test task completed", result=result)

    return result

# Placeholder task implementations (will be implemented in later tasks)
@celery_app.task(bind=True)
def process_document_task(self, file_path: str, source_type: str, metadata: Dict[str, Any]):
    """Process document file. (Placeholder)"""
    logger.info("Starting document processing", file_path=file_path, source_type=source_type)
    # Implementation will be added in task 05
    return {"status": "placeholder", "message": "Document processing not implemented"}

@celery_app.task(bind=True)
def process_audio_task(self, file_path: str, metadata: Dict[str, Any]):
    """Process audio file. (Placeholder)"""
    logger.info("Starting audio processing", file_path=file_path)
    # Implementation will be added in task 08
    return {"status": "placeholder", "message": "Audio processing not implemented"}

@celery_app.task(bind=True)
def process_video_task(self, file_path: str, metadata: Dict[str, Any]):
    """Process video file. (Placeholder)"""
    logger.info("Starting video processing", file_path=file_path)
    # Implementation will be added in task 09
    return {"status": "placeholder", "message": "Video processing not implemented"}

@celery_app.task(bind=True)
def process_web_task(self, url: str, metadata: Dict[str, Any]):
    """Process web content. (Redirects to web_tasks module)"""
    logger.info("Starting web processing", url=url)
    # Import here to avoid circular imports
    from .web_tasks import process_web_url

    # Delegate to the actual implementation
    return process_web_url.delay(url, metadata, task_id=self.request.id)
