"""Test configuration and fixtures for MoRAG tests."""

import pytest
import asyncio
import tempfile
import shutil
import sys
from pathlib import Path
from typing import Generator, AsyncGenerator
from unittest.mock import MagicMock, AsyncMock, patch
import redis
from fastapi.testclient import TestClient

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morag_core.config import settings

# Define result classes for testing
class EmbeddingResult:
    def __init__(self, embedding, token_count, model):
        self.embedding = embedding
        self.token_count = token_count
        self.model = model

class SummaryResult:
    def __init__(self, summary, token_count, model):
        self.summary = summary
        self.token_count = token_count
        self.model = model


class TestSettings:
    """Test-specific settings."""
    redis_url = "redis://localhost:6379/15"  # Use different DB for tests
    qdrant_collection_name = "test_morag_documents"
    upload_dir = "./test_uploads"
    temp_dir = "./test_temp"
    log_level = "DEBUG"


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
def temp_files():
    """Create temporary directory for test files."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def sample_documents(temp_files):
    """Create sample documents for testing."""
    docs = {}
    
    # Sample text document
    docs['sample.txt'] = temp_files / "sample.txt"
    docs['sample.txt'].write_text("""
# Sample Document

This is a sample document for testing the MoRAG pipeline.

## Introduction

The document contains multiple sections to test chunking and processing.

## Content

Here is some content that should be processed and embedded.
The system should handle this text appropriately.

## Conclusion

This concludes the sample document.
    """)
    
    # Sample markdown
    docs['sample.md'] = temp_files / "sample.md"
    docs['sample.md'].write_text("""
# Machine Learning Guide

Machine learning is a subset of artificial intelligence.

## Types of Learning

- Supervised Learning
- Unsupervised Learning  
- Reinforcement Learning

## Applications

Machine learning has many applications in various fields.
    """)
    
    return docs


@pytest.fixture
def mock_gemini_service():
    """Mock Gemini service for testing."""
    
    class MockGeminiService:
        async def generate_embedding(self, text, task_type="retrieval_document"):
            # Return mock embedding
            return EmbeddingResult(
                embedding=[0.1] * 768,
                token_count=len(text.split()),
                model='mock-embedding-model'
            )
        
        async def generate_embeddings_batch(self, texts, **kwargs):
            results = []
            for text in texts:
                result = await self.generate_embedding(text)
                results.append(result)
            return results
        
        async def generate_summary(self, text, max_length=150, style="concise"):
            # Return mock summary
            summary = text[:max_length] + "..." if len(text) > max_length else text
            return SummaryResult(
                summary=summary,
                token_count=len(summary.split()),
                model='mock-text-model'
            )
        
        async def health_check(self):
            return {
                "status": "healthy",
                "embedding_model": "mock-embedding-model",
                "text_model": "mock-text-model",
                "embedding_dimension": 768
            }
    
    return MockGeminiService()


@pytest.fixture
def mock_qdrant_service():
    """Mock Qdrant service for testing."""
    
    class MockQdrantService:
        def __init__(self):
            self.connected = False
            self.points = []
        
        async def connect(self):
            self.connected = True
        
        async def create_collection(self, vector_size=768, force_recreate=False):
            return True
        
        async def store_embedding(self, embedding, text, metadata, collection_name="test"):
            point_id = len(self.points)
            self.points.append({
                "id": point_id,
                "embedding": embedding,
                "text": text,
                "metadata": metadata
            })
            return point_id
        
        async def search_similar(self, query_embedding, limit=5, score_threshold=0.5):
            # Return mock search results
            return [
                {
                    "id": 0,
                    "score": 0.9,
                    "text": "Mock search result",
                    "metadata": {"test": True}
                }
            ]
        
        async def get_collection_info(self):
            return {
                "name": TestSettings.qdrant_collection_name,
                "vectors_count": len(self.points),
                "points_count": len(self.points)
            }
    
    return MockQdrantService()


@pytest.fixture
def mock_task_manager():
    """Mock task manager for testing."""
    
    class MockTaskManager:
        def __init__(self):
            self.tasks = {}
        
        async def create_task(self, task_type, source_data, metadata=None):
            task_id = f"test_task_{len(self.tasks)}"
            self.tasks[task_id] = {
                "id": task_id,
                "type": task_type,
                "status": "pending",
                "source_data": source_data,
                "metadata": metadata or {},
                "result": None
            }
            return task_id
        
        async def get_task_status(self, task_id):
            return self.tasks.get(task_id, {"status": "not_found"})
        
        async def update_task_status(self, task_id, status, result=None):
            if task_id in self.tasks:
                self.tasks[task_id]["status"] = status
                if result:
                    self.tasks[task_id]["result"] = result
    
    return MockTaskManager()


@pytest.fixture
def auth_headers():
    """Authentication headers for API tests."""
    return {"Authorization": "Bearer test-api-key"}


@pytest.fixture
def mock_youtube_metadata():
    """Mock YouTube metadata for testing."""

    class YouTubeMetadata:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    return YouTubeMetadata(
        id="test_video_123",
        title="Test Video Title",
        description="This is a test video description for testing purposes.",
        uploader="Test Channel",
        upload_date="2024-01-15",
        duration=300,
        view_count=1000,
        like_count=50,
        comment_count=10,
        tags=["test", "video", "sample"],
        categories=["Education"],
        thumbnail_url="https://example.com/thumbnail.jpg",
        webpage_url="https://youtube.com/watch?v=test_video_123",
        channel_id="UC_test_channel",
        channel_url="https://youtube.com/channel/UC_test_channel",
        playlist_id=None,
        playlist_title=None,
        playlist_index=None
    )


@pytest.fixture
def mock_celery_task():
    """Mock Celery task for testing."""
    
    class MockCeleryTask:
        def __init__(self):
            self.request = MagicMock()
            self.request.id = "test_task_id"
            self.status_updates = []
        
        async def update_status(self, status, result=None):
            self.status_updates.append({"status": status, "result": result})
    
    return MockCeleryTask()
