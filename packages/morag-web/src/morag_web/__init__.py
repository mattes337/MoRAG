"""MoRAG Web - Web scraping and processing capabilities for the MoRAG system."""

from morag_web.converter import WebConverter
from morag_web.processor import (
    WebContent,
    WebProcessor,
    WebScrapingConfig,
    WebScrapingResult,
)
from morag_web.service import WebService
from morag_web.tasks import (
    ProcessWebUrlsBatchTask,
    ProcessWebUrlTask,
    create_celery_tasks,
    process_web_url,
    process_web_urls_batch,
)

__all__ = [
    "WebProcessor",
    "WebScrapingConfig",
    "WebContent",
    "WebScrapingResult",
    "WebService",
    "WebConverter",
    "ProcessWebUrlTask",
    "ProcessWebUrlsBatchTask",
    "process_web_url",
    "process_web_urls_batch",
    "create_celery_tasks",
]

__version__ = "0.1.0"
