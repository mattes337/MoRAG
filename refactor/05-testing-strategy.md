# MoRAG Refactoring Task 5: Comprehensive Testing Strategy

## Priority: HIGH (Critical for reliability after refactoring)
## Estimated Time: 10-15 hours
## Impact: Comprehensive test coverage enabling confident refactoring and rapid bug detection

## Overview
This task addresses the critical testing gap in MoRAG. Current coverage is inadequate: 66 test files for 321 production files (20.6% coverage). This task creates a comprehensive testing strategy to achieve >80% test coverage with fast, reliable tests that enable rapid development and deployment.

## Current Testing Analysis

### Existing Test Coverage
```bash
Total Production Files: 321
Current Test Files: 66  
Coverage Ratio: 20.6%

Test Distribution:
- Unit tests: ~40 files
- Integration tests: ~20 files  
- Manual tests: ~6 files
```

### Critical Gaps (Files without tests):
- **Core interfaces**: `morag_core/interfaces/*.py` (0% tested)
- **Storage classes**: `*storage.py` files (30% tested)  
- **Service classes**: `*service.py` files (25% tested)
- **Processor classes**: `*processor.py` files (40% tested)
- **Base classes**: After deduplication task, new base classes need tests

### Test Quality Issues:
- Many tests are integration-heavy (slow, brittle)
- Missing mocking for external dependencies
- No performance/load testing
- Limited error scenario coverage
- No property-based testing for complex logic

## Comprehensive Testing Strategy

### 1. Test Pyramid Structure

```
                    ┌─────────────────┐
                    │   E2E Tests     │  <-- 5% (CLI, full workflows)
                    │   (5-10 tests)  │
                    └─────────────────┘
                  ┌─────────────────────┐
                  │  Integration Tests  │  <-- 15% (component interaction)
                  │   (50-80 tests)     │
                  └─────────────────────┘
              ┌─────────────────────────────┐
              │      Unit Tests             │  <-- 80% (isolated components)
              │     (300+ tests)            │
              └─────────────────────────────┘
```

### 2. Unit Test Strategy (Target: 300+ tests)

#### Core Components (Priority 1)
**Base Classes** (After deduplication):
```python
# tests/unit/core/test_base_storage.py
class TestBaseStorage:
    """Test unified storage base class."""
    
    @pytest.fixture
    def mock_storage(self):
        class MockStorage(BaseStorage):
            async def _establish_connection(self):
                return "mock_connection"
            async def _validate_connection(self):
                pass
        return MockStorage({"host": "test"})
    
    async def test_connection_success(self, mock_storage):
        result = await mock_storage.connect()
        assert result is True
        assert mock_storage._connection == "mock_connection"
    
    async def test_connection_failure(self, mock_storage):
        # Test error handling
        pass
    
    async def test_health_check_healthy(self, mock_storage):
        await mock_storage.connect()
        health = await mock_storage.health_check()
        assert health["status"] == "healthy"
        assert health["connection"] is True
    
    async def test_health_check_disconnected(self, mock_storage):
        health = await mock_storage.health_check() 
        assert health["status"] == "disconnected"
        assert health["connection"] is False

# Similar structure for:
# tests/unit/core/test_base_processor.py
# tests/unit/core/test_base_service.py
```

**Storage Classes**:
```python
# tests/unit/storage/test_qdrant_storage.py
class TestQdrantStorage:
    """Test Qdrant storage implementation."""
    
    @pytest.fixture 
    def qdrant_storage(self):
        return QdrantStorage({"host": "localhost", "port": 6333})
    
    @pytest.fixture
    def mock_qdrant_client(self):
        with mock.patch('qdrant_client.QdrantClient') as mock_client:
            yield mock_client
    
    async def test_vector_insert(self, qdrant_storage, mock_qdrant_client):
        vectors = [{"id": "1", "vector": [0.1, 0.2]}]
        result = await qdrant_storage.insert_vectors(vectors)
        assert result.success is True
        mock_qdrant_client.return_value.upsert.assert_called_once()
    
    async def test_vector_search(self, qdrant_storage, mock_qdrant_client):
        query_vector = [0.1, 0.2, 0.3]
        mock_qdrant_client.return_value.search.return_value = [
            mock.Mock(id="1", score=0.9, payload={"text": "test"})
        ]
        
        results = await qdrant_storage.search_vectors(query_vector, limit=5)
        assert len(results) == 1
        assert results[0].score == 0.9
    
    async def test_error_handling(self, qdrant_storage, mock_qdrant_client):
        mock_qdrant_client.return_value.search.side_effect = Exception("Connection failed")
        
        with pytest.raises(StorageError, match="Connection failed"):
            await qdrant_storage.search_vectors([0.1, 0.2])

# Similar comprehensive tests for:
# tests/unit/storage/test_neo4j_storage.py  
# tests/unit/storage/test_json_storage.py
```

**Service Classes**:
```python
# tests/unit/services/test_embedding_service.py
class TestGeminiEmbeddingService:
    """Test Gemini embedding service."""
    
    @pytest.fixture
    def embedding_service(self):
        return GeminiEmbeddingService(api_key="test-key")
    
    @pytest.fixture  
    def mock_gemini_client(self):
        with mock.patch('google.genai.Client') as mock_client:
            yield mock_client
    
    async def test_single_embedding(self, embedding_service, mock_gemini_client):
        mock_gemini_client.return_value.embed.return_value = [0.1, 0.2, 0.3]
        
        result = await embedding_service.generate_embedding("test text")
        assert len(result) == 3
        assert result == [0.1, 0.2, 0.3]
    
    async def test_batch_embeddings(self, embedding_service, mock_gemini_client):
        texts = ["text1", "text2", "text3"]
        mock_gemini_client.return_value.embed_batch.return_value = [
            [0.1, 0.2], [0.3, 0.4], [0.5, 0.6]
        ]
        
        results = await embedding_service.generate_embeddings(texts)
        assert len(results) == 3
        assert all(len(emb) == 2 for emb in results)
    
    async def test_rate_limiting(self, embedding_service, mock_gemini_client):
        # Test rate limit handling
        pass
    
    async def test_api_error_handling(self, embedding_service, mock_gemini_client):
        mock_gemini_client.return_value.embed.side_effect = Exception("API Error")
        
        with pytest.raises(EmbeddingError):
            await embedding_service.generate_embedding("test")
```

**Processor Classes**:
```python  
# tests/unit/processors/test_audio_processor.py
class TestAudioProcessor:
    """Test audio processing functionality."""
    
    @pytest.fixture
    def audio_processor(self):
        return AudioProcessor({"model": "whisper-base"})
    
    @pytest.fixture
    def sample_audio_file(self, tmp_path):
        # Create a minimal audio file for testing
        audio_path = tmp_path / "test.wav"
        # Generate simple sine wave or use test asset
        return audio_path
    
    async def test_audio_transcription(self, audio_processor, sample_audio_file):
        with mock.patch.object(audio_processor, '_transcribe') as mock_transcribe:
            mock_transcribe.return_value = {
                "text": "Hello world",
                "segments": [{"start": 0, "end": 2, "text": "Hello world"}]
            }
            
            result = await audio_processor.process(sample_audio_file)
            assert result.text == "Hello world"
            assert len(result.segments) == 1
    
    async def test_unsupported_format(self, audio_processor):
        fake_file = Path("test.unsupported")
        
        with pytest.raises(UnsupportedFormatError):
            await audio_processor.process(fake_file)
    
    @pytest.mark.parametrize("file_format", [".mp3", ".wav", ".m4a", ".flac"])
    async def test_supported_formats(self, audio_processor, file_format):
        # Test all supported audio formats
        pass
```

#### Property-Based Testing for Complex Logic
```python
# tests/unit/test_fact_extraction.py
from hypothesis import given, strategies as st

class TestFactExtraction:
    """Property-based tests for fact extraction logic."""
    
    @given(st.text(min_size=10, max_size=1000))
    async def test_fact_extraction_properties(self, text):
        """Test fact extraction invariants."""
        extractor = FactExtractor()
        facts = await extractor.extract_facts(text)
        
        # Properties that should always hold:
        assert isinstance(facts, list)
        assert all(isinstance(fact, Fact) for fact in facts)
        assert all(fact.confidence >= 0.0 and fact.confidence <= 1.0 for fact in facts)
        assert all(len(fact.subject) > 0 for fact in facts if facts)
    
    @given(
        st.lists(st.text(min_size=5), min_size=1, max_size=10),
        st.integers(min_value=1, max_value=50)
    )
    async def test_chunking_properties(self, texts, chunk_size):
        """Test chunking logic properties.""" 
        chunker = SemanticChunker(chunk_size=chunk_size)
        chunks = await chunker.chunk_texts(texts)
        
        # Chunking invariants:
        total_input = sum(len(text) for text in texts)
        total_output = sum(len(chunk.text) for chunk in chunks)
        assert total_output <= total_input * 1.1  # Allow some overlap
        assert all(len(chunk.text) <= chunk_size * 1.2 for chunk in chunks)  # Size limits
```

### 3. Integration Test Strategy (Target: 50-80 tests)

#### Component Interaction Tests
```python
# tests/integration/test_ingestion_pipeline.py
class TestIngestionPipeline:
    """Test complete ingestion workflow."""
    
    @pytest.fixture
    async def test_environment(self):
        """Set up test databases and services."""
        # Start test Qdrant instance
        # Start test Redis instance  
        # Mock external APIs
        yield
        # Cleanup
    
    async def test_document_to_vector_workflow(self, test_environment):
        """Test complete document processing pipeline."""
        # 1. Document ingestion
        coordinator = IngestionCoordinator()
        test_doc = "test_document.pdf"
        
        # 2. Processing stages
        result = await coordinator.process_document(test_doc)
        
        # 3. Verify results in vector store
        assert result.success is True
        assert result.chunks_created > 0
        assert result.facts_extracted > 0
        
        # 4. Test retrieval works
        retrieval_result = await coordinator.search_similar("test query")
        assert len(retrieval_result.chunks) > 0

# tests/integration/test_stage_pipeline.py
class TestStagePipeline:
    """Test stage-based processing pipeline."""
    
    async def test_markdown_conversion_to_facts(self):
        """Test full stage pipeline execution."""
        stages = ["markdown-conversion", "chunker", "fact-generator"]
        pipeline = StagePipeline(stages)
        
        input_file = "test_input.pdf"
        result = await pipeline.execute(input_file)
        
        # Verify each stage output
        assert result.stages_completed == 3
        assert result.markdown_file.exists()
        assert result.chunks_file.exists()
        assert result.facts_file.exists()
```

### 4. End-to-End Test Strategy (Target: 5-10 tests)

```python
# tests/e2e/test_cli_workflows.py
class TestCLIWorkflows:
    """Test complete CLI workflows end-to-end."""
    
    async def test_full_document_processing_cli(self):
        """Test complete document processing via CLI."""
        # Use subprocess to test actual CLI
        result = subprocess.run([
            "python", "cli/morag-stages.py", "process", 
            "test_document.pdf", "--output-dir", "test_output"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "Processing complete" in result.stdout
        
        # Verify output files created
        assert Path("test_output/test_document.md").exists()
        assert Path("test_output/test_document.chunks.json").exists()
    
    async def test_api_full_workflow(self):
        """Test complete workflow via REST API."""
        async with httpx.AsyncClient() as client:
            # Upload file
            with open("test_document.pdf", "rb") as f:
                response = await client.post(
                    "http://localhost:8000/api/v1/stages/process",
                    files={"file": f}
                )
            
            assert response.status_code == 200
            result = response.json()
            assert result["status"] == "completed"
```

### 5. Performance and Load Testing

```python
# tests/performance/test_processing_performance.py  
class TestProcessingPerformance:
    """Performance and load testing."""
    
    @pytest.mark.performance
    async def test_embedding_batch_performance(self):
        """Test batch embedding performance."""
        service = GeminiEmbeddingService()
        texts = ["test text"] * 100
        
        start_time = time.time()
        results = await service.generate_embeddings(texts)
        duration = time.time() - start_time
        
        # Performance assertions
        assert duration < 30  # Should complete in under 30 seconds
        assert len(results) == 100
        
        # Throughput testing
        throughput = len(texts) / duration
        assert throughput > 3  # At least 3 embeddings per second
    
    @pytest.mark.load  
    async def test_concurrent_processing(self):
        """Test concurrent processing load."""
        coordinator = IngestionCoordinator()
        tasks = []
        
        # Create 10 concurrent processing tasks
        for i in range(10):
            task = asyncio.create_task(
                coordinator.process_document(f"test_doc_{i}.pdf")
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        assert all(result.success for result in results)
```

## Test Infrastructure and Utilities

### 1. Test Fixtures and Factories
```python
# tests/conftest.py
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from pathlib import Path

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_gemini_client():
    """Mock Gemini AI client.""" 
    client = Mock()
    client.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
    client.embed_batch = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
    return client

@pytest.fixture
def sample_documents(tmp_path):
    """Create sample documents for testing."""
    doc_dir = tmp_path / "documents"
    doc_dir.mkdir()
    
    # Create various test documents
    (doc_dir / "simple.txt").write_text("This is a simple test document.")
    (doc_dir / "complex.md").write_text("# Complex Document\n\nWith multiple sections.")
    
    return doc_dir

@pytest.fixture
def test_database_config():
    """Test database configuration."""
    return {
        "qdrant": {"host": "localhost", "port": 6333},
        "neo4j": {"uri": "bolt://localhost:7687", "user": "test", "password": "test"}
    }
```

### 2. Mock Strategies
```python
# tests/utils/mocks.py
class MockStorage:
    """Mock storage for testing."""
    def __init__(self):
        self._data = {}
        self._vectors = {}
    
    async def store(self, key: str, value: Any) -> bool:
        self._data[key] = value
        return True
    
    async def retrieve(self, key: str) -> Any:
        return self._data.get(key)
    
    async def search_vectors(self, query_vector: List[float], limit: int = 10):
        # Simple mock vector search
        return [{"id": "1", "score": 0.9, "payload": {"text": "mock result"}}]

class MockEmbeddingService:
    """Mock embedding service for testing."""
    async def generate_embedding(self, text: str) -> List[float]:
        # Generate deterministic embedding based on text hash
        return [hash(text) % 1000 / 1000.0 for _ in range(384)]
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [await self.generate_embedding(text) for text in texts]
```

## Implementation Timeline

### Phase 1: Core Unit Tests (5 hours)
**Week 1:**
1. **Base classes tests** (1.5 hours):
   - `test_base_storage.py`
   - `test_base_processor.py`  
   - `test_base_service.py`

2. **Critical storage tests** (1.5 hours):
   - `test_qdrant_storage.py`
   - `test_neo4j_storage.py`

3. **Essential service tests** (1 hour):
   - `test_embedding_service.py`
   - `test_graph_processor.py`

4. **Core processor tests** (1 hour):
   - `test_audio_processor.py`
   - `test_document_processor.py`

### Phase 2: Comprehensive Unit Coverage (4 hours)
**Week 2:**
1. **Interface tests** (1 hour):
   - All `morag_core/interfaces/*.py` files

2. **Utility tests** (1 hour): 
   - Configuration loading, logging, error handling

3. **Model tests** (1 hour):
   - Data models, validation logic

4. **Property-based tests** (1 hour):
   - Complex algorithms (fact extraction, chunking)

### Phase 3: Integration and E2E Tests (3 hours)
**Week 2:**
1. **Integration tests** (2 hours):
   - Pipeline workflows
   - Component interactions

2. **E2E tests** (1 hour):
   - CLI workflows
   - API workflows

### Phase 4: Performance and Specialized Tests (2 hours)
**Week 3:**
1. **Performance tests** (1 hour):
   - Load testing, throughput testing

2. **Error scenario tests** (1 hour):
   - Network failures, API errors, invalid inputs

## Success Criteria

### Quantitative Goals
- [ ] **Unit test coverage**: ≥80% (from current ~20%)
- [ ] **Total tests**: 300+ unit tests (from current ~66 total)
- [ ] **Test execution time**: <5 minutes for full suite  
- [ ] **Fast feedback**: Unit tests complete in <30 seconds

### Qualitative Goals
- [ ] **Reliable tests**: <1% flaky test rate
- [ ] **Clear test names**: All tests clearly describe what they test
- [ ] **Good mocking**: External dependencies properly mocked
- [ ] **Comprehensive error coverage**: All error scenarios tested

### Coverage Targets by Component
- **Core interfaces**: 100% test coverage
- **Storage classes**: 90% test coverage  
- **Service classes**: 85% test coverage
- **Processor classes**: 80% test coverage
- **Utilities**: 90% test coverage

## Testing Best Practices

### Test Naming Convention
```python
def test_<method>_<scenario>_<expected_result>():
    """
    Examples:
    test_connect_with_valid_config_returns_true()
    test_search_vectors_with_empty_query_raises_error()  
    test_process_document_with_invalid_format_returns_error()
    """
```

### Test Structure (AAA Pattern)
```python
async def test_example():
    # Arrange
    service = MyService(config)
    mock_data = create_mock_data()
    
    # Act  
    result = await service.process(mock_data)
    
    # Assert
    assert result.success is True
    assert len(result.items) == 5
```

### Parameterized Testing
```python
@pytest.mark.parametrize("input,expected", [
    ("simple text", 2),
    ("complex multi-sentence text. With punctuation!", 3),
    ("", 0),
])
def test_text_processing(input, expected):
    result = process_text(input)
    assert len(result.sentences) == expected
```

## Continuous Integration Integration

### GitHub Actions Configuration
```yaml
# .github/workflows/test.yml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
        
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -e '.[dev]'
        
    - name: Run unit tests
      run: pytest tests/unit/ -v --cov=packages/ --cov-report=xml
      
    - name: Run integration tests  
      run: pytest tests/integration/ -v
      
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

### Local Development Workflow
```bash
# Run tests during development:
pytest tests/unit/ -v --tb=short           # Fast unit tests
pytest tests/integration/ -v               # Integration tests
pytest tests/performance/ -m "not slow"    # Quick performance tests

# Before committing:
pytest tests/ --cov=packages/ --cov-report=html
open htmlcov/index.html                    # Review coverage

# Performance testing:
pytest tests/performance/ -m performance -v
```

This comprehensive testing strategy ensures that after completing the refactoring tasks, MoRAG will have robust, fast, and reliable tests that enable confident development and deployment. The tests serve as both quality assurance and documentation of expected behavior.