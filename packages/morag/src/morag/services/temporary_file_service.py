"""Temporary file management service for UI interoperability."""

import asyncio
import mimetypes
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiofiles
import structlog

logger = structlog.get_logger(__name__)


class TemporaryFileService:
    """Service for managing temporary files during processing."""

    def __init__(
        self,
        base_dir: str = "temp_storage",
        retention_hours: int = 24,
        max_session_size_mb: int = 1024,
        cleanup_interval_minutes: int = 60,
    ):
        """Initialize temporary file service.

        Args:
            base_dir: Base directory for temporary file storage
            retention_hours: Hours to retain files before cleanup
            max_session_size_mb: Maximum storage per session in MB
            cleanup_interval_minutes: Interval between cleanup runs
        """
        self.base_dir = Path(base_dir)
        self.retention_hours = retention_hours
        self.max_session_size_mb = max_session_size_mb
        self.cleanup_interval_minutes = cleanup_interval_minutes

        # Ensure base directory exists
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Background cleanup task (started on demand)
        self._cleanup_task = None

    def _start_cleanup_task(self):
        """Start background cleanup task if event loop is running."""
        try:
            if self._cleanup_task is None or self._cleanup_task.done():
                self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        except RuntimeError:
            # No event loop running, cleanup task will be started later
            pass

    async def _periodic_cleanup(self):
        """Periodic cleanup of expired files."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval_minutes * 60)
                await self.cleanup_expired_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in periodic cleanup", error=str(e))

    def get_session_dir(self, session_id: str) -> Path:
        """Get directory path for a session.

        Args:
            session_id: Session identifier

        Returns:
            Path to session directory
        """
        # Validate session ID to prevent path traversal
        if not session_id or not session_id.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"Invalid session ID: {session_id}")

        return self.base_dir / session_id

    async def create_session_dir(self, session_id: str) -> Path:
        """Create directory for a session.

        Args:
            session_id: Session identifier

        Returns:
            Path to created session directory
        """
        # Ensure cleanup task is running
        self._start_cleanup_task()

        session_dir = self.get_session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (session_dir / "artifacts").mkdir(exist_ok=True)
        (session_dir / "artifacts" / "thumbnails").mkdir(exist_ok=True)
        (session_dir / "artifacts" / "chunks").mkdir(exist_ok=True)
        (session_dir / "artifacts" / "analysis").mkdir(exist_ok=True)

        logger.info(
            "Created session directory", session_id=session_id, path=str(session_dir)
        )
        return session_dir

    async def store_file(
        self,
        session_id: str,
        filename: str,
        content: bytes,
        content_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Store a file in the session directory.

        Args:
            session_id: Session identifier
            filename: Name of the file
            content: File content as bytes
            content_type: MIME type of the content

        Returns:
            File metadata dictionary
        """
        # Check session size limit
        current_size = await self.get_session_size(session_id)
        new_size = current_size + len(content)
        max_size_bytes = self.max_session_size_mb * 1024 * 1024

        if new_size > max_size_bytes:
            raise ValueError(
                f"Session size limit exceeded: {new_size} > {max_size_bytes}"
            )

        session_dir = await self.create_session_dir(session_id)
        file_path = session_dir / filename

        # Validate filename to prevent path traversal
        if not filename or ".." in filename or "/" in filename or "\\" in filename:
            raise ValueError(f"Invalid filename: {filename}")

        # Write file content
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)

        # Detect content type if not provided
        if not content_type:
            content_type, _ = mimetypes.guess_type(filename)
            if not content_type:
                content_type = "application/octet-stream"

        metadata = {
            "filename": filename,
            "size_bytes": len(content),
            "content_type": content_type,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
        }

        logger.info(
            "Stored file", session_id=session_id, filename=filename, size=len(content)
        )
        return metadata

    async def store_text_file(
        self, session_id: str, filename: str, content: str, encoding: str = "utf-8"
    ) -> Dict[str, Any]:
        """Store a text file in the session directory.

        Args:
            session_id: Session identifier
            filename: Name of the file
            content: Text content
            encoding: Text encoding

        Returns:
            File metadata dictionary
        """
        content_bytes = content.encode(encoding)
        content_type = "text/plain"

        if filename.endswith(".md"):
            content_type = "text/markdown"
        elif filename.endswith(".json"):
            content_type = "application/json"
        elif filename.endswith(".log"):
            content_type = "text/plain"

        return await self.store_file(session_id, filename, content_bytes, content_type)

    async def get_file(
        self, session_id: str, filename: str
    ) -> Tuple[bytes, Dict[str, Any]]:
        """Retrieve a file from the session directory.

        Args:
            session_id: Session identifier
            filename: Name of the file

        Returns:
            Tuple of (file_content, metadata)
        """
        session_dir = self.get_session_dir(session_id)
        file_path = session_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {filename}")

        # Read file content
        async with aiofiles.open(file_path, "rb") as f:
            content = await f.read()

        # Get file stats
        stat = file_path.stat()
        content_type, _ = mimetypes.guess_type(filename)
        if not content_type:
            content_type = "application/octet-stream"

        metadata = {
            "filename": filename,
            "size_bytes": stat.st_size,
            "content_type": content_type,
            "created_at": datetime.fromtimestamp(
                stat.st_ctime, timezone.utc
            ).isoformat(),
            "modified_at": datetime.fromtimestamp(
                stat.st_mtime, timezone.utc
            ).isoformat(),
            "session_id": session_id,
        }

        return content, metadata

    async def get_file_path(self, session_id: str, filename: str) -> Path:
        """Get the file path for streaming.

        Args:
            session_id: Session identifier
            filename: Name of the file

        Returns:
            Path to the file
        """
        session_dir = self.get_session_dir(session_id)
        file_path = session_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {filename}")

        return file_path

    async def list_session_files(self, session_id: str) -> Dict[str, Any]:
        """List all files in a session.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary with file list and session metadata
        """
        session_dir = self.get_session_dir(session_id)

        if not session_dir.exists():
            return {
                "session_id": session_id,
                "files": [],
                "total_size_bytes": 0,
                "expires_at": None,
            }

        files = []
        total_size = 0

        # Recursively find all files
        for file_path in session_dir.rglob("*"):
            if file_path.is_file():
                stat = file_path.stat()
                relative_path = file_path.relative_to(session_dir)

                content_type, _ = mimetypes.guess_type(file_path.name)
                if not content_type:
                    content_type = "application/octet-stream"

                files.append(
                    {
                        "filename": str(relative_path),
                        "size_bytes": stat.st_size,
                        "content_type": content_type,
                        "created_at": datetime.fromtimestamp(
                            stat.st_ctime, timezone.utc
                        ).isoformat(),
                        "modified_at": datetime.fromtimestamp(
                            stat.st_mtime, timezone.utc
                        ).isoformat(),
                    }
                )
                total_size += stat.st_size

        # Calculate expiration time
        expires_at = None
        if files:
            # Use the newest file's creation time + retention period
            newest_time = max(
                datetime.fromisoformat(f["created_at"].replace("Z", "+00:00"))
                for f in files
            )
            expires_at = (
                newest_time + timedelta(hours=self.retention_hours)
            ).isoformat()

        return {
            "session_id": session_id,
            "files": sorted(files, key=lambda x: x["created_at"]),
            "total_size_bytes": total_size,
            "expires_at": expires_at,
        }

    async def get_session_size(self, session_id: str) -> int:
        """Get total size of files in a session.

        Args:
            session_id: Session identifier

        Returns:
            Total size in bytes
        """
        session_info = await self.list_session_files(session_id)
        return session_info["total_size_bytes"]

    async def delete_session(self, session_id: str) -> bool:
        """Delete all files for a session.

        Args:
            session_id: Session identifier

        Returns:
            True if successful, False otherwise
        """
        try:
            session_dir = self.get_session_dir(session_id)

            if session_dir.exists():
                shutil.rmtree(session_dir)
                logger.info("Deleted session directory", session_id=session_id)

            return True

        except Exception as e:
            logger.error(
                "Failed to delete session", session_id=session_id, error=str(e)
            )
            return False

    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions.

        Returns:
            Number of sessions cleaned up
        """
        cleaned_count = 0
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.retention_hours)

        try:
            for session_dir in self.base_dir.iterdir():
                if not session_dir.is_dir():
                    continue

                # Check if session has expired
                session_expired = True
                for file_path in session_dir.rglob("*"):
                    if file_path.is_file():
                        file_time = datetime.fromtimestamp(
                            file_path.stat().st_mtime, timezone.utc
                        )
                        if file_time > cutoff_time:
                            session_expired = False
                            break

                if session_expired:
                    session_id = session_dir.name
                    if await self.delete_session(session_id):
                        cleaned_count += 1

            if cleaned_count > 0:
                logger.info("Cleaned up expired sessions", count=cleaned_count)

        except Exception as e:
            logger.error("Error during session cleanup", error=str(e))

        return cleaned_count


# Global service instance
_temp_file_service = None


def get_temp_file_service() -> TemporaryFileService:
    """Get global temporary file service instance."""
    global _temp_file_service
    if _temp_file_service is None:
        # Get configuration from environment with proper defaults
        base_dir = os.getenv("MORAG_TEMP_DIR", "temp_storage")
        retention_hours = int(os.getenv("MORAG_TEMP_RETENTION_HOURS", "24"))
        max_session_size_mb = int(os.getenv("MORAG_TEMP_MAX_SESSION_SIZE_MB", "1024"))
        cleanup_interval_minutes = int(
            os.getenv("MORAG_TEMP_CLEANUP_INTERVAL_MINUTES", "60")
        )

        _temp_file_service = TemporaryFileService(
            base_dir=base_dir,
            retention_hours=retention_hours,
            max_session_size_mb=max_session_size_mb,
            cleanup_interval_minutes=cleanup_interval_minutes,
        )

    return _temp_file_service
