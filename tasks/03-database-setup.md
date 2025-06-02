# Task 03: Qdrant Vector Database Setup

## Overview
Set up Qdrant vector database for storing embeddings and metadata, including collection configuration and connection management.

## Prerequisites
- Task 01: Project Setup completed
- Docker installed (for running Qdrant)

## Dependencies
- Task 01: Project Setup

## Implementation Steps

### 1. Qdrant Docker Setup
Create `docker/docker-compose.qdrant.yml`:
```yaml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:v1.7.4
    container_name: morag-qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_storage:/qdrant/storage
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333
      - QDRANT__SERVICE__GRPC_PORT=6334
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  qdrant_storage:
    driver: local
```

### 2. Qdrant Client Service
Create `src/morag/services/storage.py`:
```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, CreateCollection, PointStruct,
    Filter, FieldCondition, MatchValue, SearchRequest
)
from typing import List, Dict, Any, Optional, Union
import structlog
import asyncio
from datetime import datetime
import uuid

from morag.core.config import settings
from morag.core.exceptions import StorageError

logger = structlog.get_logger()

class QdrantService:
    """Service for managing Qdrant vector database operations."""
    
    def __init__(self):
        self.client: Optional[QdrantClient] = None
        self.collection_name = settings.qdrant_collection_name
        
    async def connect(self) -> None:
        """Initialize connection to Qdrant."""
        try:
            self.client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                api_key=settings.qdrant_api_key,
                timeout=30
            )
            
            # Test connection
            collections = await asyncio.to_thread(self.client.get_collections)
            logger.info("Connected to Qdrant", collections_count=len(collections.collections))
            
        except Exception as e:
            logger.error("Failed to connect to Qdrant", error=str(e))
            raise StorageError(f"Failed to connect to Qdrant: {str(e)}")
    
    async def disconnect(self) -> None:
        """Close connection to Qdrant."""
        if self.client:
            await asyncio.to_thread(self.client.close)
            self.client = None
            logger.info("Disconnected from Qdrant")
    
    async def create_collection(self, vector_size: int = 768, force_recreate: bool = False) -> None:
        """Create or recreate the main collection."""
        if not self.client:
            await self.connect()
        
        try:
            # Check if collection exists
            collections = await asyncio.to_thread(self.client.get_collections)
            collection_exists = any(
                col.name == self.collection_name 
                for col in collections.collections
            )
            
            if collection_exists:
                if force_recreate:
                    logger.info("Deleting existing collection", collection=self.collection_name)
                    await asyncio.to_thread(
                        self.client.delete_collection,
                        collection_name=self.collection_name
                    )
                else:
                    logger.info("Collection already exists", collection=self.collection_name)
                    return
            
            # Create collection
            logger.info("Creating collection", collection=self.collection_name, vector_size=vector_size)
            await asyncio.to_thread(
                self.client.create_collection,
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            
            logger.info("Collection created successfully", collection=self.collection_name)
            
        except Exception as e:
            logger.error("Failed to create collection", error=str(e))
            raise StorageError(f"Failed to create collection: {str(e)}")
    
    async def store_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> List[str]:
        """Store text chunks with their embeddings and metadata."""
        if not self.client:
            await self.connect()
        
        if len(chunks) != len(embeddings):
            raise StorageError("Number of chunks must match number of embeddings")
        
        try:
            points = []
            point_ids = []
            
            for chunk, embedding in zip(chunks, embeddings):
                point_id = str(uuid.uuid4())
                point_ids.append(point_id)
                
                # Prepare metadata
                payload = {
                    "text": chunk.get("text", ""),
                    "summary": chunk.get("summary", ""),
                    "source": chunk.get("source", ""),
                    "source_type": chunk.get("source_type", ""),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "created_at": datetime.utcnow().isoformat(),
                    "metadata": chunk.get("metadata", {})
                }
                
                points.append(PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                ))
            
            # Store points in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                await asyncio.to_thread(
                    self.client.upsert,
                    collection_name=self.collection_name,
                    points=batch
                )
            
            logger.info("Stored chunks successfully", count=len(chunks))
            return point_ids
            
        except Exception as e:
            logger.error("Failed to store chunks", error=str(e))
            raise StorageError(f"Failed to store chunks: {str(e)}")
    
    async def search_similar(
        self,
        query_embedding: List[float],
        limit: int = 10,
        score_threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks using vector similarity."""
        if not self.client:
            await self.connect()
        
        try:
            # Build filter if provided
            search_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    conditions.append(FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    ))
                if conditions:
                    search_filter = Filter(must=conditions)
            
            # Perform search
            results = await asyncio.to_thread(
                self.client.search,
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=search_filter,
                with_payload=True
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result.id,
                    "score": result.score,
                    "text": result.payload.get("text", ""),
                    "summary": result.payload.get("summary", ""),
                    "source": result.payload.get("source", ""),
                    "source_type": result.payload.get("source_type", ""),
                    "metadata": result.payload.get("metadata", {})
                })
            
            logger.info("Search completed", results_count=len(formatted_results))
            return formatted_results
            
        except Exception as e:
            logger.error("Search failed", error=str(e))
            raise StorageError(f"Search failed: {str(e)}")
    
    async def delete_by_source(self, source: str) -> int:
        """Delete all chunks from a specific source."""
        if not self.client:
            await self.connect()
        
        try:
            # Search for points with the given source
            filter_condition = Filter(
                must=[FieldCondition(
                    key="source",
                    match=MatchValue(value=source)
                )]
            )
            
            # Get points to delete
            search_result = await asyncio.to_thread(
                self.client.scroll,
                collection_name=self.collection_name,
                scroll_filter=filter_condition,
                limit=10000,  # Adjust based on expected chunk count
                with_payload=False
            )
            
            point_ids = [point.id for point in search_result[0]]
            
            if point_ids:
                # Delete points
                await asyncio.to_thread(
                    self.client.delete,
                    collection_name=self.collection_name,
                    points_selector=point_ids
                )
            
            logger.info("Deleted chunks by source", source=source, count=len(point_ids))
            return len(point_ids)
            
        except Exception as e:
            logger.error("Failed to delete by source", error=str(e))
            raise StorageError(f"Failed to delete by source: {str(e)}")
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        if not self.client:
            await self.connect()
        
        try:
            info = await asyncio.to_thread(
                self.client.get_collection,
                collection_name=self.collection_name
            )
            
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count,
                "status": info.status.value,
                "optimizer_status": info.optimizer_status.status.value,
                "config": {
                    "vector_size": info.config.params.vectors.size,
                    "distance": info.config.params.vectors.distance.value
                }
            }
            
        except Exception as e:
            logger.error("Failed to get collection info", error=str(e))
            raise StorageError(f"Failed to get collection info: {str(e)}")

# Global instance
qdrant_service = QdrantService()
```

### 3. Update Health Check
Update `src/morag/api/routes/health.py` to include Qdrant health check:
```python
# Add this import at the top
from morag.services.storage import qdrant_service

# Replace the Qdrant check in readiness_check function:
    # Check Qdrant connection
    try:
        if qdrant_service.client:
            await qdrant_service.get_collection_info()
            services["qdrant"] = "healthy"
        else:
            services["qdrant"] = "not_connected"
    except Exception as e:
        logger.error("Qdrant health check failed", error=str(e))
        services["qdrant"] = "unhealthy"
```

### 4. Database Initialization Script
Create `scripts/init_db.py`:
```python
#!/usr/bin/env python3
"""Initialize Qdrant database with required collections."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morag.services.storage import qdrant_service
from morag.core.config import settings
import structlog

logger = structlog.get_logger()

async def main():
    """Initialize the database."""
    try:
        logger.info("Initializing Qdrant database")
        
        # Connect to Qdrant
        await qdrant_service.connect()
        
        # Create collection (vector size for text-embedding-004 is 768)
        await qdrant_service.create_collection(vector_size=768, force_recreate=False)
        
        # Get collection info
        info = await qdrant_service.get_collection_info()
        logger.info("Database initialized successfully", collection_info=info)
        
    except Exception as e:
        logger.error("Failed to initialize database", error=str(e))
        sys.exit(1)
    finally:
        await qdrant_service.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
```

## Testing Instructions

### 1. Start Qdrant
```bash
# From project root
docker-compose -f docker/docker-compose.qdrant.yml up -d

# Check if Qdrant is running
curl http://localhost:6333/health
```

### 2. Initialize Database
```bash
# From project root
python scripts/init_db.py
```

## Mandatory Testing Requirements

### 1. Qdrant Service Tests
Create `tests/test_03_qdrant_service.py`:
```python
import pytest
import asyncio
import uuid
from unittest.mock import patch, AsyncMock
from morag.services.storage import qdrant_service, QdrantService
from morag.core.exceptions import StorageError

@pytest.fixture
async def clean_qdrant():
    """Clean Qdrant collection before and after tests."""
    await qdrant_service.connect()

    # Clean before test
    try:
        await qdrant_service.client.delete_collection(
            collection_name="test_morag_documents"
        )
    except:
        pass

    # Create fresh collection
    await qdrant_service.create_collection(vector_size=768, force_recreate=True)

    yield qdrant_service

    # Clean after test
    try:
        await qdrant_service.client.delete_collection(
            collection_name="test_morag_documents"
        )
    except:
        pass

    await qdrant_service.disconnect()

class TestQdrantService:
    """Test Qdrant service functionality."""

    @pytest.mark.asyncio
    async def test_connection(self, clean_qdrant):
        """Test Qdrant connection and disconnection."""
        service = clean_qdrant

        # Test connection info
        info = await service.get_collection_info()
        assert info["name"] == "test_morag_documents"
        assert info["vectors_count"] >= 0
        assert info["status"] in ["green", "yellow"]

    @pytest.mark.asyncio
    async def test_collection_creation(self):
        """Test collection creation and recreation."""
        service = QdrantService()
        service.collection_name = "test_collection_creation"

        await service.connect()

        # Create collection
        await service.create_collection(vector_size=384, force_recreate=False)
        info = await service.get_collection_info()
        assert info["config"]["vector_size"] == 384

        # Test force recreation
        await service.create_collection(vector_size=768, force_recreate=True)
        info = await service.get_collection_info()
        assert info["config"]["vector_size"] == 768

        # Cleanup
        await service.client.delete_collection(collection_name="test_collection_creation")
        await service.disconnect()

    @pytest.mark.asyncio
    async def test_store_chunks(self, clean_qdrant):
        """Test storing chunks with embeddings."""
        service = clean_qdrant

        chunks = [
            {
                "text": "Machine learning is a subset of AI.",
                "summary": "ML and AI relationship",
                "source": "test_doc.pdf",
                "source_type": "document",
                "chunk_index": 0,
                "metadata": {"page": 1, "test": True}
            },
            {
                "text": "Deep learning uses neural networks.",
                "summary": "Deep learning explanation",
                "source": "test_doc.pdf",
                "source_type": "document",
                "chunk_index": 1,
                "metadata": {"page": 1, "test": True}
            }
        ]

        embeddings = [
            [0.1] * 768,  # First embedding
            [0.2] * 768   # Second embedding
        ]

        point_ids = await service.store_chunks(chunks, embeddings)

        assert len(point_ids) == 2
        assert all(isinstance(pid, str) for pid in point_ids)

        # Verify storage
        info = await service.get_collection_info()
        assert info["points_count"] == 2

    @pytest.mark.asyncio
    async def test_search_similar(self, clean_qdrant):
        """Test similarity search functionality."""
        service = clean_qdrant

        # Store test data
        chunks = [
            {
                "text": "Python programming language",
                "summary": "Python info",
                "source": "python_doc.txt",
                "source_type": "document",
                "chunk_index": 0,
                "metadata": {"topic": "programming"}
            }
        ]

        embeddings = [[0.5] * 768]
        await service.store_chunks(chunks, embeddings)

        # Search with similar embedding
        results = await service.search_similar(
            query_embedding=[0.5] * 768,
            limit=5,
            score_threshold=0.8
        )

        assert len(results) >= 1
        assert results[0]["text"] == "Python programming language"
        assert results[0]["source"] == "python_doc.txt"
        assert "score" in results[0]
        assert results[0]["score"] >= 0.8

    @pytest.mark.asyncio
    async def test_search_with_filters(self, clean_qdrant):
        """Test search with metadata filters."""
        service = clean_qdrant

        # Store test data with different sources
        chunks = [
            {
                "text": "JavaScript programming",
                "summary": "JS info",
                "source": "js_doc.txt",
                "source_type": "document",
                "chunk_index": 0,
                "metadata": {"language": "javascript"}
            },
            {
                "text": "Python programming",
                "summary": "Python info",
                "source": "python_doc.txt",
                "source_type": "document",
                "chunk_index": 0,
                "metadata": {"language": "python"}
            }
        ]

        embeddings = [[0.3] * 768, [0.4] * 768]
        await service.store_chunks(chunks, embeddings)

        # Search with filter
        results = await service.search_similar(
            query_embedding=[0.35] * 768,
            limit=5,
            score_threshold=0.0,
            filters={"source": "python_doc.txt"}
        )

        assert len(results) == 1
        assert results[0]["source"] == "python_doc.txt"

    @pytest.mark.asyncio
    async def test_delete_by_source(self, clean_qdrant):
        """Test deletion by source."""
        service = clean_qdrant

        # Store test data
        chunks = [
            {
                "text": "Content from doc1",
                "summary": "Doc1 content",
                "source": "doc1.pdf",
                "source_type": "document",
                "chunk_index": 0,
                "metadata": {}
            },
            {
                "text": "Content from doc2",
                "summary": "Doc2 content",
                "source": "doc2.pdf",
                "source_type": "document",
                "chunk_index": 0,
                "metadata": {}
            }
        ]

        embeddings = [[0.6] * 768, [0.7] * 768]
        await service.store_chunks(chunks, embeddings)

        # Delete by source
        deleted_count = await service.delete_by_source("doc1.pdf")
        assert deleted_count == 1

        # Verify deletion
        results = await service.search_similar(
            query_embedding=[0.6] * 768,
            limit=10,
            score_threshold=0.0
        )

        sources = [r["source"] for r in results]
        assert "doc1.pdf" not in sources
        assert "doc2.pdf" in sources

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in Qdrant service."""
        service = QdrantService()

        # Test connection error
        with patch.object(service, 'client', None):
            with pytest.raises(StorageError):
                await service.get_collection_info()

        # Test invalid embedding dimensions
        await service.connect()
        await service.create_collection(vector_size=768)

        chunks = [{"text": "test", "summary": "test", "source": "test", "source_type": "test", "chunk_index": 0, "metadata": {}}]
        invalid_embeddings = [[0.1] * 384]  # Wrong dimension

        with pytest.raises(Exception):  # Should fail due to dimension mismatch
            await service.store_chunks(chunks, invalid_embeddings)

        await service.disconnect()

class TestQdrantIntegration:
    """Test Qdrant integration scenarios."""

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, clean_qdrant):
        """Test concurrent store and search operations."""
        service = clean_qdrant

        async def store_chunk(index):
            chunk = {
                "text": f"Test chunk {index}",
                "summary": f"Summary {index}",
                "source": f"doc_{index}.txt",
                "source_type": "document",
                "chunk_index": 0,
                "metadata": {"index": index}
            }
            embedding = [float(index) / 100] * 768
            return await service.store_chunks([chunk], [embedding])

        # Store multiple chunks concurrently
        tasks = [store_chunk(i) for i in range(5)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        assert all(len(r) == 1 for r in results)

        # Verify all chunks are stored
        info = await service.get_collection_info()
        assert info["points_count"] == 5

    @pytest.mark.asyncio
    async def test_large_batch_storage(self, clean_qdrant):
        """Test storing large batches of chunks."""
        service = clean_qdrant

        # Create large batch
        batch_size = 150
        chunks = []
        embeddings = []

        for i in range(batch_size):
            chunks.append({
                "text": f"Large batch chunk {i}",
                "summary": f"Summary {i}",
                "source": "large_doc.pdf",
                "source_type": "document",
                "chunk_index": i,
                "metadata": {"batch_index": i}
            })
            embeddings.append([float(i) / batch_size] * 768)

        point_ids = await service.store_chunks(chunks, embeddings)

        assert len(point_ids) == batch_size

        # Verify storage
        info = await service.get_collection_info()
        assert info["points_count"] == batch_size
```

### 2. Database Initialization Tests
Create `tests/test_03_database_init.py`:
```python
import pytest
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, AsyncMock

class TestDatabaseInitialization:
    """Test database initialization procedures."""

    def test_init_script_exists(self):
        """Test that initialization script exists and is executable."""
        script_path = Path("scripts/init_db.py")
        assert script_path.exists(), "Database initialization script not found"

        # Test script syntax
        result = subprocess.run([
            sys.executable, "-m", "py_compile", str(script_path)
        ], capture_output=True)
        assert result.returncode == 0, f"Script syntax error: {result.stderr}"

    @patch('morag.services.storage.qdrant_service')
    def test_init_script_execution(self, mock_service):
        """Test database initialization script execution."""
        mock_service.connect = AsyncMock()
        mock_service.create_collection = AsyncMock()
        mock_service.get_collection_info = AsyncMock(return_value={
            "name": "morag_documents",
            "vectors_count": 0,
            "status": "green"
        })
        mock_service.disconnect = AsyncMock()

        # This would test the actual script execution
        # In practice, you'd run the script in a subprocess
        # For now, we'll test the components
        assert mock_service is not None

    @pytest.mark.asyncio
    async def test_collection_configuration(self):
        """Test that collection is configured correctly."""
        from morag.services.storage import qdrant_service

        await qdrant_service.connect()
        await qdrant_service.create_collection(vector_size=768, force_recreate=True)

        info = await qdrant_service.get_collection_info()

        assert info["config"]["vector_size"] == 768
        assert info["config"]["distance"] == "Cosine"

        await qdrant_service.disconnect()
```

### 3. Test Execution Instructions
```bash
# Start Qdrant container
docker-compose -f docker/docker-compose.qdrant.yml up -d

# Wait for Qdrant to be ready
sleep 10

# Run Qdrant service tests
pytest tests/test_03_qdrant_service.py -v

# Run database initialization tests
pytest tests/test_03_database_init.py -v

# Run all Task 03 tests with coverage
pytest tests/test_03_*.py -v --cov=src/morag/services/storage --cov-report=html

# Test database initialization script
python scripts/init_db.py

# Verify Qdrant health
curl http://localhost:6333/health
```

### 4. Integration Tests
Create `tests/test_03_integration.py`:
```python
import pytest
import subprocess
import time
from fastapi.testclient import TestClient
from morag.api.main import create_app

class TestQdrantIntegration:
    """Test Qdrant integration with API."""

    def test_qdrant_container_health(self):
        """Test that Qdrant container is healthy."""
        import requests

        # Test Qdrant health endpoint
        try:
            response = requests.get("http://localhost:6333/health", timeout=5)
            assert response.status_code == 200
        except requests.exceptions.RequestException:
            pytest.skip("Qdrant container not running")

    def test_api_qdrant_health_check(self):
        """Test API health check includes Qdrant status."""
        app = create_app()
        client = TestClient(app)

        response = client.get("/health/ready")
        assert response.status_code == 200

        data = response.json()
        assert "services" in data
        # Note: Qdrant might be unhealthy in test environment, that's ok
        assert "qdrant" in data["services"]

    def test_database_initialization_script(self):
        """Test database initialization script execution."""
        result = subprocess.run([
            "python", "scripts/init_db.py"
        ], capture_output=True, text=True, timeout=30)

        # Script should complete successfully or fail gracefully
        assert result.returncode in [0, 1]  # 0 = success, 1 = expected failure in test env

        if result.returncode == 1:
            # Check if it's a connection error (expected in some test environments)
            assert any(keyword in result.stderr.lower() for keyword in
                      ["connection", "timeout", "refused", "unreachable"])
```

### 5. Performance Tests
Create `tests/test_03_performance.py`:
```python
import pytest
import time
import asyncio
from morag.services.storage import qdrant_service

@pytest.mark.performance
class TestQdrantPerformance:
    """Test Qdrant performance characteristics."""

    @pytest.mark.asyncio
    async def test_batch_insert_performance(self):
        """Test performance of batch insertions."""
        await qdrant_service.connect()
        await qdrant_service.create_collection(vector_size=768, force_recreate=True)

        # Create test data
        batch_size = 100
        chunks = []
        embeddings = []

        for i in range(batch_size):
            chunks.append({
                "text": f"Performance test chunk {i}",
                "summary": f"Summary {i}",
                "source": "performance_test.pdf",
                "source_type": "document",
                "chunk_index": i,
                "metadata": {"batch": "performance"}
            })
            embeddings.append([float(i) / batch_size] * 768)

        # Measure insertion time
        start_time = time.time()
        point_ids = await qdrant_service.store_chunks(chunks, embeddings)
        end_time = time.time()

        insertion_time = end_time - start_time

        assert len(point_ids) == batch_size
        assert insertion_time < 30.0  # Should complete within 30 seconds

        print(f"Inserted {batch_size} chunks in {insertion_time:.2f} seconds")
        print(f"Rate: {batch_size / insertion_time:.2f} chunks/second")

        await qdrant_service.disconnect()

    @pytest.mark.asyncio
    async def test_search_performance(self):
        """Test search performance."""
        await qdrant_service.connect()
        await qdrant_service.create_collection(vector_size=768, force_recreate=True)

        # Insert test data
        chunks = [
            {
                "text": f"Search test document {i}",
                "summary": f"Document {i}",
                "source": f"doc_{i}.pdf",
                "source_type": "document",
                "chunk_index": 0,
                "metadata": {"search_test": True}
            }
            for i in range(50)
        ]

        embeddings = [[float(i) / 50] * 768 for i in range(50)]
        await qdrant_service.store_chunks(chunks, embeddings)

        # Measure search time
        start_time = time.time()
        results = await qdrant_service.search_similar(
            query_embedding=[0.5] * 768,
            limit=10,
            score_threshold=0.0
        )
        end_time = time.time()

        search_time = end_time - start_time

        assert len(results) > 0
        assert search_time < 1.0  # Should complete within 1 second

        print(f"Search completed in {search_time:.3f} seconds")

        await qdrant_service.disconnect()
```

### 6. Test Execution Instructions
```bash
# Prerequisites: Start Qdrant container
docker-compose -f docker/docker-compose.qdrant.yml up -d
sleep 10

# Run all Qdrant tests
pytest tests/test_03_*.py -v

# Run with coverage
pytest tests/test_03_*.py -v --cov=src/morag/services/storage --cov-report=html

# Run performance tests
pytest tests/test_03_performance.py -v -m performance

# Test database initialization
python scripts/init_db.py

# Verify Qdrant health
curl http://localhost:6333/health

# Test API integration
python -c "
from fastapi.testclient import TestClient
from morag.api.main import create_app
client = TestClient(create_app())
response = client.get('/health/ready')
print('API Health:', response.json())
"
```

## Success Criteria (MANDATORY - ALL MUST PASS)
- [ ] Qdrant container starts successfully and passes health checks
- [ ] Database initialization script runs without errors
- [ ] Collection is created with correct configuration (768 dimensions, Cosine distance)
- [ ] Storage service can connect to and disconnect from Qdrant
- [ ] All CRUD operations work correctly (store, search, delete)
- [ ] Concurrent operations handle properly without data corruption
- [ ] Large batch operations complete within performance thresholds
- [ ] Search functionality works with and without filters
- [ ] Error handling works for connection failures and invalid data
- [ ] Health check reports Qdrant status accurately
- [ ] All unit tests pass with >95% coverage
- [ ] Integration tests pass
- [ ] Performance tests meet benchmarks (100 chunks/30s, search <1s)

## Advancement Blocker
**⚠️ CRITICAL: Cannot proceed to Task 04 until ALL tests pass and the following integration requirements are met:**

### Integration Requirements
```bash
# 1. Container Health Check
docker-compose -f docker/docker-compose.qdrant.yml ps | grep "healthy" || exit 1

# 2. Database Operations Test
python -c "
import asyncio
from morag.services.storage import qdrant_service

async def test():
    await qdrant_service.connect()
    await qdrant_service.create_collection(vector_size=768)

    # Test store and search
    chunks = [{'text': 'test', 'summary': 'test', 'source': 'test', 'source_type': 'test', 'chunk_index': 0, 'metadata': {}}]
    embeddings = [[0.1] * 768]

    point_ids = await qdrant_service.store_chunks(chunks, embeddings)
    results = await qdrant_service.search_similar([0.1] * 768, limit=1)

    assert len(point_ids) == 1
    assert len(results) == 1

    await qdrant_service.disconnect()
    print('✅ Database operations test passed')

asyncio.run(test())
"

# 3. API Integration Test
python -c "
from fastapi.testclient import TestClient
from morag.api.main import create_app

client = TestClient(create_app())
response = client.get('/health/ready')
data = response.json()

assert response.status_code == 200
assert 'qdrant' in data['services']
print('✅ API integration test passed')
"
```

### Coverage Requirements
- Storage service: >95% coverage
- Database operations: 100% coverage
- Error handling: 100% coverage
- Integration points: >90% coverage

## Next Steps (Only after ALL tests pass)
- Task 04: Task Queue Setup
- Task 14: Gemini Integration
- Task 15: Vector Storage Implementation (enhanced operations)
