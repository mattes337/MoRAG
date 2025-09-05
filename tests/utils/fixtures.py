"""Test fixtures for MoRAG testing."""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Generator
import asyncio

from .mocks import (
    MockStorage,
    MockEmbeddingService,
    MockProcessor,
    MockTaskManager,
    MockFileSystem,
    MockConfiguration,
    MockLogger
)


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_directory() -> Generator[Path, None, None]:
    """Create temporary directory for tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_files(temp_directory: Path) -> Dict[str, Path]:
    """Create sample files for testing."""
    files = {}
    
    # Text file
    files['text'] = temp_directory / "sample.txt"
    files['text'].write_text("""
This is a sample text document for testing.
It contains multiple lines and paragraphs.

This is the second paragraph with more content
to test text processing capabilities.
    """.strip())
    
    # Markdown file
    files['markdown'] = temp_directory / "sample.md"
    files['markdown'].write_text("""
# Sample Document

This is a **sample** markdown document.

## Section 1

- Item 1
- Item 2
- Item 3

## Section 2

Some more content with `code` and [links](https://example.com).
    """.strip())
    
    # JSON file
    files['json'] = temp_directory / "sample.json"
    files['json'].write_text("""
{
    "name": "Test Document",
    "type": "sample",
    "content": "This is sample JSON content",
    "metadata": {
        "created": "2024-01-01",
        "version": "1.0"
    }
}
    """.strip())
    
    # Binary file (fake)
    files['binary'] = temp_directory / "sample.bin"
    files['binary'].write_bytes(b'\x00\x01\x02\x03\x04\x05')
    
    # Large text file
    files['large'] = temp_directory / "large.txt"
    large_content = "This is a large file. " * 1000
    files['large'].write_text(large_content)
    
    return files


@pytest.fixture
def sample_documents() -> List[Dict[str, Any]]:
    """Create sample document data."""
    return [
        {
            "id": "doc_1",
            "title": "Introduction to Machine Learning",
            "content": "Machine learning is a subset of artificial intelligence...",
            "type": "article",
            "metadata": {"author": "John Doe", "date": "2024-01-01"}
        },
        {
            "id": "doc_2",
            "title": "Deep Learning Fundamentals",
            "content": "Deep learning uses neural networks with multiple layers...",
            "type": "tutorial",
            "metadata": {"author": "Jane Smith", "date": "2024-01-02"}
        },
        {
            "id": "doc_3",
            "title": "Natural Language Processing",
            "content": "NLP combines computational linguistics with machine learning...",
            "type": "guide",
            "metadata": {"author": "Bob Johnson", "date": "2024-01-03"}
        }
    ]


@pytest.fixture
def sample_entities() -> List[Dict[str, Any]]:
    """Create sample entity data."""
    return [
        {
            "id": "entity_1",
            "name": "Machine Learning",
            "type": "Technology",
            "attributes": {"field": "AI", "year_established": "1950s"}
        },
        {
            "id": "entity_2", 
            "name": "John Doe",
            "type": "Person",
            "attributes": {"role": "Researcher", "expertise": "ML"}
        },
        {
            "id": "entity_3",
            "name": "Stanford University",
            "type": "Organization",
            "attributes": {"type": "University", "location": "California"}
        }
    ]


@pytest.fixture
def sample_relations() -> List[Dict[str, Any]]:
    """Create sample relation data."""
    return [
        {
            "id": "rel_1",
            "source_id": "entity_2",
            "target_id": "entity_1",
            "type": "RESEARCHES",
            "attributes": {"since": "2020", "expertise_level": "expert"}
        },
        {
            "id": "rel_2",
            "source_id": "entity_2",
            "target_id": "entity_3", 
            "type": "AFFILIATED_WITH",
            "attributes": {"role": "Professor", "department": "Computer Science"}
        }
    ]


@pytest.fixture
def mock_storage():
    """Create mock storage instance."""
    return MockStorage()


@pytest.fixture
async def connected_mock_storage():
    """Create connected mock storage instance."""
    storage = MockStorage()
    await storage.connect()
    yield storage
    await storage.disconnect()


@pytest.fixture
def mock_embedding_service():
    """Create mock embedding service."""
    return MockEmbeddingService(embedding_dim=384)


@pytest.fixture
def mock_processor():
    """Create mock processor."""
    return MockProcessor(supported_formats=[".txt", ".md", ".pdf"])


@pytest.fixture
def mock_task_manager():
    """Create mock task manager."""
    return MockTaskManager()


@pytest.fixture
def mock_file_system():
    """Create mock file system."""
    return MockFileSystem()


@pytest.fixture
def mock_configuration():
    """Create mock configuration."""
    return MockConfiguration()


@pytest.fixture
def mock_logger():
    """Create mock logger."""
    return MockLogger()


@pytest.fixture
def api_headers():
    """Standard API headers for testing."""
    return {
        "Content-Type": "application/json",
        "Authorization": "Bearer test-token",
        "User-Agent": "MoRAG-Test/1.0"
    }


@pytest.fixture
def sample_embeddings() -> List[List[float]]:
    """Create sample embedding vectors."""
    return [
        [0.1, 0.2, 0.3, 0.4] * 96,  # 384-dim vector
        [0.5, 0.6, 0.7, 0.8] * 96,
        [0.9, 0.1, 0.2, 0.3] * 96
    ]


@pytest.fixture
def sample_chunks() -> List[Dict[str, Any]]:
    """Create sample text chunks."""
    return [
        {
            "id": "chunk_1",
            "text": "This is the first chunk of text for testing purposes.",
            "start_index": 0,
            "end_index": 53,
            "metadata": {"section": "introduction"}
        },
        {
            "id": "chunk_2", 
            "text": "This is the second chunk with different content to test chunking.",
            "start_index": 54,
            "end_index": 119,
            "metadata": {"section": "body"}
        },
        {
            "id": "chunk_3",
            "text": "This is the final chunk to complete the test content.",
            "start_index": 120,
            "end_index": 173,
            "metadata": {"section": "conclusion"}
        }
    ]


@pytest.fixture
def database_config():
    """Database configuration for testing."""
    return {
        "qdrant": {
            "host": "localhost",
            "port": 6333,
            "collection_name": "test_collection"
        },
        "neo4j": {
            "uri": "bolt://localhost:7687",
            "user": "test",
            "password": "test",
            "database": "test_db"
        },
        "redis": {
            "host": "localhost",
            "port": 6379,
            "db": 15  # Use separate DB for tests
        }
    }


@pytest.fixture
def ai_service_config():
    """AI service configuration for testing."""
    return {
        "gemini": {
            "api_key": "test-gemini-key",
            "model": "gemini-pro",
            "embedding_model": "embedding-001"
        },
        "openai": {
            "api_key": "test-openai-key", 
            "model": "gpt-4",
            "embedding_model": "text-embedding-ada-002"
        }
    }


@pytest.fixture
def processing_config():
    """Processing configuration for testing."""
    return {
        "chunk_size": 1000,
        "overlap_size": 200,
        "batch_size": 50,
        "max_workers": 4,
        "timeout": 30,
        "retry_attempts": 3
    }


@pytest.fixture
def cleanup_temp_files():
    """Fixture to track and cleanup temporary files."""
    temp_files = []
    
    def add_temp_file(file_path: Path):
        temp_files.append(file_path)
        return file_path
    
    yield add_temp_file
    
    # Cleanup
    for file_path in temp_files:
        try:
            if file_path.exists():
                if file_path.is_file():
                    file_path.unlink()
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
        except Exception:
            pass  # Ignore cleanup errors


@pytest.fixture
async def async_mock_services():
    """Create a collection of async mock services."""
    services = {
        "storage": MockStorage(),
        "embedding": MockEmbeddingService(),
        "processor": MockProcessor(),
        "task_manager": MockTaskManager()
    }
    
    # Connect storage
    await services["storage"].connect()
    
    yield services
    
    # Cleanup
    await services["storage"].disconnect()


class TestDataGenerator:
    """Helper class to generate test data."""
    
    @staticmethod
    def create_test_entities(count: int = 5) -> List[Dict[str, Any]]:
        """Create test entities."""
        entities = []
        entity_types = ["Person", "Organization", "Technology", "Location", "Concept"]
        
        for i in range(count):
            entities.append({
                "id": f"entity_{i}",
                "name": f"Test Entity {i}",
                "type": entity_types[i % len(entity_types)],
                "attributes": {
                    "test_attribute": f"value_{i}",
                    "created_at": "2024-01-01",
                    "index": i
                }
            })
        
        return entities
    
    @staticmethod
    def create_test_relations(entity_count: int = 5) -> List[Dict[str, Any]]:
        """Create test relations between entities."""
        relations = []
        relation_types = ["RELATED_TO", "PART_OF", "SIMILAR_TO", "DEPENDS_ON"]
        
        for i in range(entity_count - 1):
            relations.append({
                "id": f"relation_{i}",
                "source_id": f"entity_{i}",
                "target_id": f"entity_{i + 1}",
                "type": relation_types[i % len(relation_types)],
                "attributes": {
                    "strength": 0.8,
                    "created_at": "2024-01-01"
                }
            })
        
        return relations
    
    @staticmethod
    def create_test_documents(count: int = 5) -> List[Dict[str, Any]]:
        """Create test documents."""
        documents = []
        document_types = ["article", "paper", "tutorial", "guide", "reference"]
        
        for i in range(count):
            documents.append({
                "id": f"doc_{i}",
                "title": f"Test Document {i}",
                "content": f"This is test document {i} with sample content for testing purposes.",
                "type": document_types[i % len(document_types)],
                "metadata": {
                    "author": f"Author {i}",
                    "created_at": "2024-01-01",
                    "word_count": 15 + i * 5
                }
            })
        
        return documents


@pytest.fixture
def test_data_generator():
    """Provide test data generator instance."""
    return TestDataGenerator()