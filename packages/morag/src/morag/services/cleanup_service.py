"""Periodic cleanup service for temporary files."""

import asyncio
import threading
import time
from typing import Optional
import structlog

from morag.utils.file_upload import get_upload_handler

logger = structlog.get_logger(__name__)


class PeriodicCleanupService:
    """Service that periodically cleans up old temporary files."""
    
    def __init__(
        self,
        cleanup_interval_hours: int = 1,
        max_file_age_hours: int = 24,
        max_disk_usage_mb: int = 10000
    ):
        """Initialize the cleanup service.
        
        Args:
            cleanup_interval_hours: How often to run cleanup (default: 1 hour)
            max_file_age_hours: Maximum age before files are eligible for cleanup (default: 24 hours)
            max_disk_usage_mb: Maximum disk usage before aggressive cleanup (default: 10000 MB)
        """
        self.cleanup_interval_hours = cleanup_interval_hours
        self.max_file_age_hours = max_file_age_hours
        self.max_disk_usage_mb = max_disk_usage_mb
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        
        logger.info("PeriodicCleanupService initialized",
                   cleanup_interval_hours=cleanup_interval_hours,
                   max_file_age_hours=max_file_age_hours,
                   max_disk_usage_mb=max_disk_usage_mb)
    
    def start(self) -> None:
        """Start the periodic cleanup service."""
        if self._running:
            logger.warning("Cleanup service is already running")
            return
        
        self._stop_event.clear()
        self._running = True
        
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="PeriodicCleanupService"
        )
        self._cleanup_thread.start()
        
        logger.info("Periodic cleanup service started")
    
    def stop(self) -> None:
        """Stop the periodic cleanup service."""
        if not self._running:
            return
        
        self._running = False
        self._stop_event.set()
        
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            # Give it a moment to stop gracefully
            self._cleanup_thread.join(timeout=5.0)
        
        logger.info("Periodic cleanup service stopped")
    
    def _cleanup_loop(self) -> None:
        """Main cleanup loop that runs in background thread."""
        logger.info("Cleanup loop started")
        
        while not self._stop_event.is_set():
            try:
                # Perform cleanup
                self._perform_cleanup()
                
                # Wait for next cleanup interval
                interval_seconds = self.cleanup_interval_hours * 3600
                if self._stop_event.wait(timeout=interval_seconds):
                    # Stop event was set, exit loop
                    break
                    
            except Exception as e:
                logger.error("Error in cleanup loop", error=str(e))
                # Wait a bit before retrying to avoid tight error loops
                if self._stop_event.wait(timeout=300):  # 5 minutes
                    break
        
        logger.info("Cleanup loop stopped")
    
    def _perform_cleanup(self) -> None:
        """Perform the actual cleanup operation."""
        try:
            upload_handler = get_upload_handler()
            
            deleted_count = upload_handler.cleanup_old_files(
                max_age_hours=self.max_file_age_hours,
                max_disk_usage_mb=self.max_disk_usage_mb
            )
            
            if deleted_count > 0:
                logger.info("Periodic cleanup completed",
                           files_deleted=deleted_count,
                           max_age_hours=self.max_file_age_hours,
                           max_disk_usage_mb=self.max_disk_usage_mb)
            else:
                logger.debug("Periodic cleanup completed - no files to delete")
                
        except Exception as e:
            logger.error("Failed to perform cleanup", error=str(e))
    
    def force_cleanup(self) -> int:
        """Force an immediate cleanup and return number of files deleted.
        
        Returns:
            Number of files deleted
        """
        try:
            upload_handler = get_upload_handler()
            deleted_count = upload_handler.cleanup_old_files(
                max_age_hours=self.max_file_age_hours,
                max_disk_usage_mb=self.max_disk_usage_mb
            )
            
            logger.info("Forced cleanup completed", files_deleted=deleted_count)
            return deleted_count
            
        except Exception as e:
            logger.error("Failed to perform forced cleanup", error=str(e))
            return 0
    
    @property
    def is_running(self) -> bool:
        """Check if the cleanup service is running."""
        return self._running and self._cleanup_thread and self._cleanup_thread.is_alive()


# Global cleanup service instance
_cleanup_service: Optional[PeriodicCleanupService] = None


def get_cleanup_service() -> PeriodicCleanupService:
    """Get the global cleanup service instance."""
    global _cleanup_service
    if _cleanup_service is None:
        _cleanup_service = PeriodicCleanupService()
    return _cleanup_service


def start_cleanup_service(
    cleanup_interval_hours: int = 1,
    max_file_age_hours: int = 24,
    max_disk_usage_mb: int = 10000
) -> None:
    """Start the global cleanup service with specified parameters."""
    global _cleanup_service
    
    if _cleanup_service is not None and _cleanup_service.is_running:
        logger.warning("Cleanup service is already running")
        return
    
    _cleanup_service = PeriodicCleanupService(
        cleanup_interval_hours=cleanup_interval_hours,
        max_file_age_hours=max_file_age_hours,
        max_disk_usage_mb=max_disk_usage_mb
    )
    _cleanup_service.start()


def stop_cleanup_service() -> None:
    """Stop the global cleanup service."""
    global _cleanup_service
    if _cleanup_service is not None:
        _cleanup_service.stop()


def force_cleanup() -> int:
    """Force an immediate cleanup and return number of files deleted."""
    service = get_cleanup_service()
    return service.force_cleanup()
