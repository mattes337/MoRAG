from celery import Celery
from kombu import Queue
import structlog

from morag.core.config import settings

logger = structlog.get_logger()

# Create Celery app
celery_app = Celery(
    "morag",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "morag.tasks.base",
        "morag.tasks.document_tasks",
        "morag.tasks.audio_tasks",
        "morag.tasks.video_tasks",
        "morag.tasks.web_tasks",
    ]
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    # Result settings
    result_expires=3600,  # 1 hour
    result_persistent=True,
)

# Task base class with common functionality
class BaseTask(celery_app.Task):
    """Base task class with common functionality."""
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure."""
        logger.error(
            "Task failed",
            task_id=task_id,
            task_name=self.name,
            error=str(exc),
            args=args,
            kwargs=kwargs
        )
    
    def on_success(self, retval, task_id, args, kwargs):
        """Handle task success."""
        logger.info(
            "Task completed successfully",
            task_id=task_id,
            task_name=self.name,
            result_type=type(retval).__name__
        )
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Handle task retry."""
        logger.warning(
            "Task retrying",
            task_id=task_id,
            task_name=self.name,
            error=str(exc),
            retry_count=self.request.retries
        )

# Set base task class
celery_app.Task = BaseTask
