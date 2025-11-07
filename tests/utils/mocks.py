"""Mock utilities for testing MoRAG components."""

import asyncio
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from unittest.mock import Mock, AsyncMock
import tempfile
import uuid
from datetime import datetime


class MockStorage:
    """Mock storage for testing."""

    def __init__(self):
        self._data = {}
        self._vectors = {}
        self._connected = False
        self._health_status = "healthy"

    async def connect(self):
        """Mock connection."""
        self._connected = True
        return True

    async def disconnect(self):
        """Mock disconnection."""
        self._connected = False
        return True

    async def store(self, key: str, value: Any) -> bool:
        """Store data."""
        if not self._connected:
            raise ConnectionError("Not connected")
        self._data[key] = value
        return True

    async def retrieve(self, key: str) -> Any:
        """Retrieve data."""
        if not self._connected:
            raise ConnectionError("Not connected")
        return self._data.get(key)

    async def delete(self, key: str) -> bool:
        """Delete data."""
        if not self._connected:
            raise ConnectionError("Not connected")
        if key in self._data:
            del self._data[key]
            return True
        return False

    async def search_vectors(self, query_vector: List[float], limit: int = 10):
        """Simple mock vector search."""
        results = []
        for i in range(min(limit, len(self._vectors))):
            results.append({
                "id": f"result_{i}",
                "score": 0.9 - (i * 0.1),
                "payload": {"text": f"Mock result {i}"}
            })
        return results

    async def store_vector(self, vector_id: str, vector: List[float], payload: Dict[str, Any]):
        """Store vector."""
        self._vectors[vector_id] = {
            "vector": vector,
            "payload": payload
        }
        return True

    async def health_check(self) -> Dict[str, Any]:
        """Health check."""
        return {
            "status": self._health_status,
            "connected": self._connected,
            "data_count": len(self._data),
            "vector_count": len(self._vectors)
        }

    def clear(self):
        """Clear all data."""
        self._data.clear()
        self._vectors.clear()


class MockEmbeddingService:
    """Mock embedding service for testing."""

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self._call_count = 0
        self._rate_limited = False
        self._health_status = "healthy"

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate deterministic embedding based on text hash."""
        self._call_count += 1

        if self._rate_limited:
            raise Exception("Rate limited")

        if not text.strip():
            raise ValueError("Empty text")

        # Generate deterministic embedding
        embedding = []
        for i in range(self.embedding_dim):
            hash_val = hash(text + str(i)) % 2000
            embedding.append((hash_val - 1000) / 1000.0)

        return embedding

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = []
        for text in texts:
            embedding = await self.generate_embedding(text)
            embeddings.append(embedding)
        return embeddings

    def set_rate_limited(self, rate_limited: bool):
        """Simulate rate limiting."""
        self._rate_limited = rate_limited

    def get_call_count(self) -> int:
        """Get number of API calls made."""
        return self._call_count

    async def health_check(self) -> Dict[str, Any]:
        """Health check."""
        return {
            "status": self._health_status,
            "embedding_dimension": self.embedding_dim,
            "call_count": self._call_count,
            "rate_limited": self._rate_limited
        }


class MockProcessor:
    """Mock processor for testing."""

    def __init__(self, supported_formats: List[str] = None):
        self.supported_formats = supported_formats or [".txt", ".md", ".pdf"]
        self._processing_time = 1.0
        self._should_fail = False
        self._processed_files = []

    def set_processing_time(self, time_seconds: float):
        """Set processing time for testing."""
        self._processing_time = time_seconds

    def set_should_fail(self, should_fail: bool):
        """Control whether processing should fail."""
        self._should_fail = should_fail

    def is_supported_format(self, file_path: Path) -> bool:
        """Check if format is supported."""
        return file_path.suffix.lower() in self.supported_formats

    async def process(self, file_path: Path, **kwargs) -> Dict[str, Any]:
        """Mock processing."""
        await asyncio.sleep(0.01)  # Simulate processing

        if self._should_fail:
            return {
                "success": False,
                "error": "Mock processing failed",
                "file_path": str(file_path)
            }

        if not self.is_supported_format(file_path):
            return {
                "success": False,
                "error": f"Unsupported format: {file_path.suffix}",
                "file_path": str(file_path)
            }

        self._processed_files.append(str(file_path))

        return {
            "success": True,
            "content": f"Processed content from {file_path.name}",
            "file_path": str(file_path),
            "processing_time": self._processing_time,
            "metadata": {"format": file_path.suffix, "size": 1024}
        }

    def get_processed_files(self) -> List[str]:
        """Get list of processed files."""
        return self._processed_files.copy()


class MockTaskManager:
    """Mock task manager for testing."""

    def __init__(self):
        self.tasks = {}
        self._task_counter = 0

    async def create_task(self, task_type: str, data: Any, **kwargs) -> str:
        """Create a new task."""
        self._task_counter += 1
        task_id = f"task_{self._task_counter}"

        self.tasks[task_id] = {
            "id": task_id,
            "type": task_type,
            "status": "pending",
            "data": data,
            "created_at": datetime.now(),
            "result": None,
            "error": None,
            **kwargs
        }

        return task_id

    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task by ID."""
        return self.tasks.get(task_id)

    async def update_task_status(self, task_id: str, status: str, result: Any = None, error: str = None):
        """Update task status."""
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = status
            if result is not None:
                self.tasks[task_id]["result"] = result
            if error is not None:
                self.tasks[task_id]["error"] = error

    async def list_tasks(self, task_type: str = None, status: str = None) -> List[Dict[str, Any]]:
        """List tasks with optional filters."""
        tasks = list(self.tasks.values())

        if task_type:
            tasks = [t for t in tasks if t["type"] == task_type]

        if status:
            tasks = [t for t in tasks if t["status"] == status]

        return tasks

    def clear_tasks(self):
        """Clear all tasks."""
        self.tasks.clear()
        self._task_counter = 0


class MockFileSystem:
    """Mock file system for testing file operations."""

    def __init__(self):
        self._files = {}
        self._directories = set()

    def create_file(self, path: Union[str, Path], content: str = "", binary: bool = False):
        """Create a mock file."""
        path = Path(path)
        self._files[str(path)] = {
            "content": content.encode() if binary else content,
            "binary": binary,
            "size": len(content.encode() if binary else content.encode()),
            "created": datetime.now()
        }

        # Ensure parent directories exist
        for parent in path.parents:
            self._directories.add(str(parent))

    def create_directory(self, path: Union[str, Path]):
        """Create a mock directory."""
        self._directories.add(str(Path(path)))

    def exists(self, path: Union[str, Path]) -> bool:
        """Check if file or directory exists."""
        path_str = str(Path(path))
        return path_str in self._files or path_str in self._directories

    def is_file(self, path: Union[str, Path]) -> bool:
        """Check if path is a file."""
        return str(Path(path)) in self._files

    def is_directory(self, path: Union[str, Path]) -> bool:
        """Check if path is a directory."""
        return str(Path(path)) in self._directories

    def read_file(self, path: Union[str, Path]) -> str:
        """Read file content."""
        path_str = str(Path(path))
        if path_str not in self._files:
            raise FileNotFoundError(f"File not found: {path}")

        file_info = self._files[path_str]
        if file_info["binary"]:
            return file_info["content"].decode()
        return file_info["content"]

    def write_file(self, path: Union[str, Path], content: str, binary: bool = False):
        """Write file content."""
        self.create_file(path, content, binary)

    def delete_file(self, path: Union[str, Path]) -> bool:
        """Delete a file."""
        path_str = str(Path(path))
        if path_str in self._files:
            del self._files[path_str]
            return True
        return False

    def list_files(self, directory: Union[str, Path], pattern: str = "*") -> List[str]:
        """List files in directory."""
        directory_str = str(Path(directory))
        files = []

        for file_path in self._files.keys():
            if file_path.startswith(directory_str):
                relative_path = file_path[len(directory_str):].lstrip("/\\")
                if "/" not in relative_path and "\\" not in relative_path:  # Direct child
                    files.append(file_path)

        return files

    def get_file_size(self, path: Union[str, Path]) -> int:
        """Get file size."""
        path_str = str(Path(path))
        if path_str not in self._files:
            raise FileNotFoundError(f"File not found: {path}")
        return self._files[path_str]["size"]

    def clear(self):
        """Clear all files and directories."""
        self._files.clear()
        self._directories.clear()


class MockConfiguration:
    """Mock configuration for testing."""

    def __init__(self, **kwargs):
        self._config = {
            "api_key": "test-api-key",
            "embedding_model": "test-model",
            "chunk_size": 1000,
            "batch_size": 50,
            "timeout": 30,
            "retry_attempts": 3,
            "debug": True,
            **kwargs
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)

    def set(self, key: str, value: Any):
        """Set configuration value."""
        self._config[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Get all configuration as dictionary."""
        return self._config.copy()

    def update(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        self._config.update(updates)


def create_temp_file(content: str = "test content", suffix: str = ".txt") -> Path:
    """Create a temporary file for testing."""
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False)
    temp_file.write(content)
    temp_file.close()
    return Path(temp_file.name)


def create_temp_directory() -> Path:
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    return Path(temp_dir)


def generate_test_data(count: int = 10) -> List[Dict[str, Any]]:
    """Generate test data for various testing scenarios."""
    test_data = []

    for i in range(count):
        test_data.append({
            "id": f"test_{i}",
            "name": f"Test Item {i}",
            "value": i * 10,
            "category": "A" if i % 2 == 0 else "B",
            "active": i % 3 == 0,
            "metadata": {
                "created": datetime.now().isoformat(),
                "index": i
            }
        })

    return test_data


class MockLogger:
    """Mock logger for testing."""

    def __init__(self):
        self.logs = []

    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logs.append({"level": "info", "message": message, "extra": kwargs})

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logs.append({"level": "warning", "message": message, "extra": kwargs})

    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logs.append({"level": "error", "message": message, "extra": kwargs})

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logs.append({"level": "debug", "message": message, "extra": kwargs})

    def get_logs(self, level: str = None) -> List[Dict[str, Any]]:
        """Get logged messages."""
        if level:
            return [log for log in self.logs if log["level"] == level]
        return self.logs.copy()

    def clear_logs(self):
        """Clear all logs."""
        self.logs.clear()

    def has_log(self, message: str, level: str = None) -> bool:
        """Check if a specific message was logged."""
        for log in self.logs:
            if level and log["level"] != level:
                continue
            if message in log["message"]:
                return True
        return False
