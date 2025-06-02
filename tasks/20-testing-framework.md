# Task 20: Testing Framework Implementation

## Overview
Implement comprehensive testing framework with unit tests, integration tests, and end-to-end tests for the MoRAG pipeline.

## Prerequisites
- All core tasks completed (01-19)
- Testing dependencies installed

## Dependencies
- Task 01: Project Setup
- Task 02: API Framework
- Task 04: Task Queue Setup
- Task 17: Ingestion API

## Implementation Steps

### 1. Test Configuration
Create `tests/conftest.py`:
```python
import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Generator, AsyncGenerator
import redis
from fastapi.testclient import TestClient

# Test imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morag.api.main import create_app
from morag.core.config import settings
from morag.services.storage import qdrant_service
from morag.services.embedding import gemini_service
from morag.services.task_manager import task_manager

# Test settings override
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

@pytest.fixture(scope="session")
def test_app():
    """Create test FastAPI application."""
    app = create_app()
    return app

@pytest.fixture(scope="session")
def client(test_app):
    """Create test client."""
    return TestClient(test_app)

@pytest.fixture(scope="function")
async def clean_database():
    """Clean test database before and after each test."""
    # Setup: Clean before test
    try:
        await qdrant_service.connect()
        # Delete test collection if exists
        try:
            await qdrant_service.client.delete_collection(
                collection_name=TestSettings.qdrant_collection_name
            )
        except:
            pass  # Collection might not exist
        
        # Create fresh test collection
        await qdrant_service.create_collection(vector_size=768, force_recreate=True)
        
    except Exception as e:
        pytest.skip(f"Database setup failed: {e}")
    
    yield
    
    # Teardown: Clean after test
    try:
        await qdrant_service.client.delete_collection(
            collection_name=TestSettings.qdrant_collection_name
        )
    except:
        pass

@pytest.fixture(scope="function")
def clean_redis():
    """Clean Redis test database."""
    r = redis.from_url(TestSettings.redis_url)
    r.flushdb()
    yield r
    r.flushdb()

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
    
    # Sample PDF content (as text for testing)
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
def mock_gemini_service(monkeypatch):
    """Mock Gemini service for testing."""
    
    class MockGeminiService:
        async def generate_embedding(self, text, task_type="retrieval_document"):
            # Return mock embedding
            return type('EmbeddingResult', (), {
                'embedding': [0.1] * 768,
                'token_count': len(text.split()),
                'model': 'mock-embedding-model'
            })()
        
        async def generate_embeddings_batch(self, texts, **kwargs):
            results = []
            for text in texts:
                result = await self.generate_embedding(text)
                results.append(result)
            return results
        
        async def generate_summary(self, text, max_length=150, style="concise"):
            # Return mock summary
            summary = text[:max_length] + "..." if len(text) > max_length else text
            return type('SummaryResult', (), {
                'summary': summary,
                'token_count': len(summary.split()),
                'model': 'mock-text-model'
            })()
        
        async def health_check(self):
            return {
                "status": "healthy",
                "embedding_model": "mock-embedding-model",
                "text_model": "mock-text-model",
                "embedding_dimension": 768
            }
    
    mock_service = MockGeminiService()
    monkeypatch.setattr("morag.services.embedding.gemini_service", mock_service)
    return mock_service

@pytest.fixture
def auth_headers():
    """Authentication headers for API tests."""
    return {"Authorization": "Bearer test-api-key"}
```

### 2. Unit Tests for Core Services
Create `tests/unit/test_services.py`:
```python
import pytest
from unittest.mock import AsyncMock, patch
from morag.services.embedding import GeminiService
from morag.services.storage import QdrantService
from morag.services.chunking import ChunkingService

class TestGeminiService:
    """Test Gemini service functionality."""
    
    @pytest.mark.asyncio
    async def test_embedding_generation(self, mock_gemini_service):
        """Test embedding generation."""
        text = "This is a test document."
        
        result = await mock_gemini_service.generate_embedding(text)
        
        assert result.embedding is not None
        assert len(result.embedding) == 768
        assert result.token_count > 0
        assert result.model == "mock-embedding-model"
    
    @pytest.mark.asyncio
    async def test_batch_embedding_generation(self, mock_gemini_service):
        """Test batch embedding generation."""
        texts = ["First text", "Second text", "Third text"]
        
        results = await mock_gemini_service.generate_embeddings_batch(texts)
        
        assert len(results) == len(texts)
        for result in results:
            assert len(result.embedding) == 768
    
    @pytest.mark.asyncio
    async def test_summary_generation(self, mock_gemini_service):
        """Test text summarization."""
        text = "This is a long text that needs to be summarized. " * 10
        
        result = await mock_gemini_service.generate_summary(text, max_length=50)
        
        assert result.summary is not None
        assert len(result.summary) <= 60  # Allow some flexibility
        assert result.model == "mock-text-model"

class TestQdrantService:
    """Test Qdrant service functionality."""
    
    @pytest.mark.asyncio
    async def test_connection(self, clean_database):
        """Test Qdrant connection."""
        await qdrant_service.connect()
        
        info = await qdrant_service.get_collection_info()
        assert info["name"] == "test_morag_documents"
        assert info["vectors_count"] >= 0
    
    @pytest.mark.asyncio
    async def test_store_and_search(self, clean_database, mock_gemini_service):
        """Test storing and searching chunks."""
        await qdrant_service.connect()
        
        # Test data
        chunks = [
            {
                "text": "Machine learning is a subset of AI.",
                "summary": "ML and AI",
                "source": "test.txt",
                "source_type": "document",
                "chunk_index": 0,
                "metadata": {"test": True}
            }
        ]
        
        embeddings = [[0.1] * 768]
        
        # Store chunks
        point_ids = await qdrant_service.store_chunks(chunks, embeddings)
        assert len(point_ids) == 1
        
        # Search
        results = await qdrant_service.search_similar(
            query_embedding=[0.1] * 768,
            limit=5,
            score_threshold=0.5
        )
        
        assert len(results) >= 1
        assert results[0]["text"] == chunks[0]["text"]

class TestChunkingService:
    """Test chunking service functionality."""
    
    @pytest.mark.asyncio
    async def test_simple_chunking(self):
        """Test simple text chunking."""
        from morag.services.chunking import chunking_service
        
        text = "This is a test sentence. " * 50
        
        chunks = await chunking_service.semantic_chunk(
            text=text,
            chunk_size=100,
            strategy="simple"
        )
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 120 for chunk in chunks)  # Allow some flexibility
    
    @pytest.mark.asyncio
    async def test_chunking_with_metadata(self):
        """Test chunking with metadata extraction."""
        from morag.services.chunking import chunking_service
        
        text = "Apple Inc. is a technology company. Tim Cook is the CEO."
        
        chunks = await chunking_service.chunk_with_metadata(
            text=text,
            strategy="semantic"
        )
        
        assert len(chunks) >= 1
        chunk = chunks[0]
        
        assert "text" in chunk
        assert "word_count" in chunk
        assert chunk["word_count"] > 0
```

### 3. Integration Tests
Create `tests/integration/test_document_processing.py`:
```python
import pytest
import tempfile
from pathlib import Path
from morag.processors.document import document_processor
from morag.tasks.document_tasks import process_document_task

class TestDocumentProcessing:
    """Test complete document processing pipeline."""
    
    @pytest.mark.asyncio
    async def test_markdown_processing(self, sample_documents, mock_gemini_service, clean_database):
        """Test processing of markdown documents."""
        md_file = sample_documents['sample.md']
        
        # Test document parsing
        result = await document_processor.parse_document(md_file)
        
        assert len(result.chunks) > 0
        assert result.word_count > 0
        assert result.metadata["parser"] == "unstructured"
        
        # Check chunk content
        chunk_texts = [chunk.text for chunk in result.chunks]
        combined_text = " ".join(chunk_texts)
        assert "Machine learning" in combined_text
        assert "Applications" in combined_text
    
    @pytest.mark.asyncio
    async def test_text_processing(self, sample_documents, mock_gemini_service, clean_database):
        """Test processing of text documents."""
        txt_file = sample_documents['sample.txt']
        
        result = await document_processor.parse_document(txt_file)
        
        assert len(result.chunks) > 0
        assert result.word_count > 0
        
        # Verify content is preserved
        chunk_texts = [chunk.text for chunk in result.chunks]
        combined_text = " ".join(chunk_texts)
        assert "Sample Document" in combined_text
        assert "Introduction" in combined_text
    
    @pytest.mark.asyncio
    async def test_document_task_integration(self, sample_documents, mock_gemini_service, clean_database):
        """Test complete document processing task."""
        txt_file = sample_documents['sample.txt']
        
        # This would normally be run through Celery, but we'll call directly for testing
        # In a real test, you'd use Celery's test utilities
        
        metadata = {
            "test": True,
            "source": str(txt_file)
        }
        
        # Note: This is a simplified test - in practice you'd need to handle async Celery tasks
        # For now, we'll test the document processor directly
        result = await document_processor.parse_document(txt_file)
        
        assert result is not None
        assert len(result.chunks) > 0
```

### 4. API Integration Tests
Create `tests/integration/test_api.py`:
```python
import pytest
import json
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient

class TestIngestionAPI:
    """Test ingestion API endpoints."""
    
    def test_health_endpoints(self, client):
        """Test health check endpoints."""
        # Basic health check
        response = client.get("/health/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        
        # Readiness check
        response = client.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "services" in data
    
    def test_file_upload_validation(self, client, auth_headers):
        """Test file upload validation."""
        # Test without authentication
        response = client.post("/api/v1/ingest/file")
        assert response.status_code == 403  # Unauthorized
        
        # Test with invalid file type
        with tempfile.NamedTemporaryFile(suffix=".xyz") as f:
            f.write(b"invalid content")
            f.seek(0)
            
            response = client.post(
                "/api/v1/ingest/file",
                headers=auth_headers,
                data={"source_type": "document"},
                files={"file": ("test.xyz", f, "application/octet-stream")}
            )
            assert response.status_code == 400  # Validation error
    
    def test_url_ingestion(self, client, auth_headers):
        """Test URL ingestion endpoint."""
        payload = {
            "source_type": "web",
            "url": "https://example.com",
            "metadata": {"test": True}
        }
        
        response = client.post(
            "/api/v1/ingest/url",
            headers=auth_headers,
            json=payload
        )
        
        # Should create task successfully
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "pending"
    
    def test_status_endpoints(self, client, auth_headers):
        """Test status checking endpoints."""
        # Test with non-existent task
        response = client.get(
            "/api/v1/status/non-existent-task",
            headers=auth_headers
        )
        # Should still return status (might be pending/not found)
        assert response.status_code in [200, 404]
        
        # Test queue stats
        response = client.get(
            "/api/v1/status/stats/queues",
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "active_tasks" in data

class TestBatchIngestion:
    """Test batch ingestion functionality."""
    
    def test_batch_url_ingestion(self, client, auth_headers):
        """Test batch URL ingestion."""
        payload = {
            "items": [
                {
                    "source_type": "web",
                    "url": "https://example.com/page1",
                    "metadata": {"page": 1}
                },
                {
                    "source_type": "web", 
                    "url": "https://example.com/page2",
                    "metadata": {"page": 2}
                }
            ],
            "webhook_url": "https://example.com/webhook"
        }
        
        response = client.post(
            "/api/v1/ingest/batch",
            headers=auth_headers,
            json=payload
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "batch_id" in data
        assert "task_ids" in data
        assert data["total_items"] >= 0
```

### 5. End-to-End Tests
Create `tests/e2e/test_pipeline.py`:
```python
import pytest
import tempfile
import time
from pathlib import Path

class TestCompletePipeline:
    """Test complete end-to-end pipeline."""
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_document_to_vector_pipeline(
        self,
        sample_documents,
        mock_gemini_service,
        clean_database,
        client,
        auth_headers
    ):
        """Test complete pipeline from document upload to vector storage."""
        
        # Step 1: Upload document
        txt_file = sample_documents['sample.txt']
        
        with open(txt_file, 'rb') as f:
            response = client.post(
                "/api/v1/ingest/file",
                headers=auth_headers,
                data={"source_type": "document"},
                files={"file": ("sample.txt", f, "text/plain")}
            )
        
        assert response.status_code == 200
        task_data = response.json()
        task_id = task_data["task_id"]
        
        # Step 2: Monitor task progress
        max_wait = 30  # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            response = client.get(
                f"/api/v1/status/{task_id}",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            status_data = response.json()
            
            if status_data["status"] in ["success", "failure"]:
                break
                
            time.sleep(1)
        
        # Step 3: Verify task completion
        assert status_data["status"] == "success"
        assert "result" in status_data
        
        # Step 4: Verify data in vector database
        from morag.services.storage import qdrant_service
        
        await qdrant_service.connect()
        info = await qdrant_service.get_collection_info()
        assert info["points_count"] > 0
        
        # Step 5: Test search functionality
        search_results = await qdrant_service.search_similar(
            query_embedding=[0.1] * 768,
            limit=5,
            score_threshold=0.0
        )
        
        assert len(search_results) > 0
        assert any("Sample Document" in result["text"] for result in search_results)
```

### 6. Performance Tests
Create `tests/performance/test_performance.py`:
```python
import pytest
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

class TestPerformance:
    """Test performance characteristics."""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_chunking_performance(self):
        """Test chunking performance with large text."""
        from morag.services.chunking import chunking_service
        
        # Generate large text
        large_text = "This is a test sentence. " * 1000  # ~5000 words
        
        start_time = time.time()
        
        chunks = await chunking_service.semantic_chunk(
            text=large_text,
            chunk_size=500,
            strategy="semantic"
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        assert len(chunks) > 0
        assert processing_time < 10.0  # Should complete within 10 seconds
        
        print(f"Chunked {len(large_text)} characters into {len(chunks)} chunks in {processing_time:.2f}s")
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_embedding_batch_performance(self, mock_gemini_service):
        """Test batch embedding performance."""
        texts = [f"Test document number {i}" for i in range(50)]
        
        start_time = time.time()
        
        results = await mock_gemini_service.generate_embeddings_batch(
            texts,
            batch_size=10
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        assert len(results) == len(texts)
        assert processing_time < 30.0  # Should complete within 30 seconds
        
        print(f"Generated {len(results)} embeddings in {processing_time:.2f}s")
```

### 7. Test Runner Script
Create `scripts/run_tests.py`:
```python
#!/usr/bin/env python3
"""Test runner script with different test categories."""

import subprocess
import sys
import argparse
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*50}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.stdout:
        print("STDOUT:")
        print(result.stdout)
    
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    if result.returncode != 0:
        print(f"❌ {description} failed with return code {result.returncode}")
        return False
    else:
        print(f"✅ {description} passed")
        return True

def main():
    parser = argparse.ArgumentParser(description="Run MoRAG tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--e2e", action="store_true", help="Run end-to-end tests")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    
    args = parser.parse_args()
    
    if not any([args.unit, args.integration, args.e2e, args.performance, args.all]):
        args.all = True  # Default to all tests
    
    success = True
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    
    if args.all or args.unit:
        cmd = ["python", "-m", "pytest", "tests/unit/", "-v"]
        if args.coverage:
            cmd.extend(["--cov=src/morag", "--cov-report=html"])
        success &= run_command(cmd, "Unit Tests")
    
    if args.all or args.integration:
        cmd = ["python", "-m", "pytest", "tests/integration/", "-v"]
        success &= run_command(cmd, "Integration Tests")
    
    if args.all or args.e2e:
        cmd = ["python", "-m", "pytest", "tests/e2e/", "-v", "-m", "not slow"]
        success &= run_command(cmd, "End-to-End Tests")
    
    if args.performance:
        cmd = ["python", "-m", "pytest", "tests/performance/", "-v", "-m", "performance"]
        success &= run_command(cmd, "Performance Tests")
    
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## Testing Instructions

### 1. Install Test Dependencies
```bash
pip install pytest pytest-asyncio pytest-cov pytest-mock
```

### 2. Run Tests
```bash
# Run all tests
python scripts/run_tests.py --all

# Run specific test categories
python scripts/run_tests.py --unit
python scripts/run_tests.py --integration
python scripts/run_tests.py --e2e

# Run with coverage
python scripts/run_tests.py --unit --coverage

# Run specific test file
pytest tests/unit/test_services.py -v

# Run with markers
pytest -m "not slow" -v
pytest -m "performance" -v
```

### 3. Continuous Integration
Create `.github/workflows/tests.yml`:
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      redis:
        image: redis:7.2-alpine
        ports:
          - 6379:6379
      
      qdrant:
        image: qdrant/qdrant:v1.7.4
        ports:
          - 6333:6333
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -e .[dev]
        python -m spacy download en_core_web_sm
    
    - name: Run tests
      run: |
        python scripts/run_tests.py --unit --integration --coverage
      env:
        GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

## Success Criteria
- [ ] Test framework is properly configured
- [ ] Unit tests cover core functionality
- [ ] Integration tests verify component interaction
- [ ] End-to-end tests validate complete workflows
- [ ] Performance tests ensure acceptable speed
- [ ] Test fixtures provide reliable test data
- [ ] Mocking works for external services
- [ ] Coverage reporting is functional
- [ ] CI/CD pipeline runs tests automatically
- [ ] All tests pass consistently

## Next Steps
- Task 21: Monitoring and Logging (observability)
- Task 22: Deployment Configuration (production setup)
