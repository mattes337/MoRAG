# Task 10: Update Tests and Documentation

## Overview
**COMPLETELY REPLACE** all tests with new stage-based architecture tests and create comprehensive documentation for the new system only.

## Objectives
- **REMOVE ALL EXISTING TESTS** and create new ones for stage-based architecture
- Create comprehensive test coverage for all named stages
- **REMOVE OLD API DOCUMENTATION** and create new documentation for stage-based API only
- Create deployment and configuration guides for new system only
- Add performance benchmarks and monitoring
- **NO DOCUMENTATION FOR OLD SYSTEM** - clean slate approach

## Deliverables

### 1. Stage-Based Test Framework
```python
# packages/morag-stages/tests/test_framework.py
import pytest
import asyncio
import tempfile
from pathlib import Path
from typing import Dict, Any, List
import json

from morag_stages import StageManager, StageType
from morag_stages.models import StageContext, StageStatus
from morag_stages.file_manager import FileManager

class StageTestFramework:
    """Test framework for stage-based processing."""

    def __init__(self):
        self.temp_dir = None
        self.stage_manager = None
        self.file_manager = None

    async def setup(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.stage_manager = StageManager()
        self.file_manager = FileManager(self.temp_dir / "storage")

        # Create test data directory
        self.test_data_dir = self.temp_dir / "test_data"
        self.test_data_dir.mkdir()

        # Create output directory
        self.output_dir = self.temp_dir / "output"
        self.output_dir.mkdir()

    async def teardown(self):
        """Clean up test environment."""
        import shutil
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def create_test_file(self, filename: str, content: str) -> Path:
        """Create a test file with given content."""
        file_path = self.test_data_dir / filename
        file_path.write_text(content, encoding='utf-8')
        return file_path

    def create_test_markdown(self, filename: str = "test.md") -> Path:
        """Create a test markdown file."""
        content = """---
title: Test Document
content_type: document
processed_at: 2024-01-01T00:00:00
---

# Test Document

This is a test document for stage processing.

## Section 1

Some content here with important information.

## Section 2

More content with different topics and concepts.
"""
        return self.create_test_file(filename, content)

    def create_test_chunks_json(self, filename: str = "test.chunks.json") -> Path:
        """Create a test chunks JSON file."""
        chunks_data = {
            "document_metadata": {
                "title": "Test Document",
                "content_type": "document"
            },
            "summary": "This is a test document summary.",
            "chunk_count": 2,
            "chunks": [
                {
                    "index": 0,
                    "content": "This is the first chunk of content.",
                    "token_count": 8,
                    "embedding": [0.1] * 1536,
                    "source_metadata": {"title": "Test Document"}
                },
                {
                    "index": 1,
                    "content": "This is the second chunk of content.",
                    "token_count": 9,
                    "embedding": [0.2] * 1536,
                    "source_metadata": {"title": "Test Document"}
                }
            ]
        }

        file_path = self.test_data_dir / filename
        file_path.write_text(json.dumps(chunks_data, indent=2), encoding='utf-8')
        return file_path

    def create_test_facts_json(self, filename: str = "test.facts.json") -> Path:
        """Create a test facts JSON file."""
        facts_data = {
            "facts": [
                {
                    "statement": "Test documents contain information",
                    "subject": "Test documents",
                    "predicate": "contain",
                    "object": "information",
                    "confidence": 0.9,
                    "source_chunk_index": 0
                }
            ],
            "entities": [
                {
                    "name": "Test documents",
                    "normalized_name": "test_documents",
                    "entity_type": "Concept",
                    "confidence": 0.9
                },
                {
                    "name": "information",
                    "normalized_name": "information",
                    "entity_type": "Concept",
                    "confidence": 0.8
                }
            ],
            "relations": [
                {
                    "subject": "Test documents",
                    "predicate": "CONTAIN",
                    "object": "information",
                    "confidence": 0.9
                }
            ]
        }

        file_path = self.test_data_dir / filename
        file_path.write_text(json.dumps(facts_data, indent=2), encoding='utf-8')
        return file_path

    async def execute_stage_test(self,
                                stage_type: StageType,
                                input_files: List[Path],
                                config: Dict[str, Any] = None) -> 'StageResult':
        """Execute a stage for testing."""

        context = StageContext(
            source_path=input_files[0],
            output_dir=self.output_dir,
            config=config or {}
        )

        return await self.stage_manager.execute_stage(stage_type, input_files, context)

    def assert_stage_success(self, result: 'StageResult'):
        """Assert that stage execution was successful."""
        assert result.status == StageStatus.COMPLETED
        assert result.error_message is None
        assert len(result.output_files) > 0
        assert all(f.exists() for f in result.output_files)

    def assert_file_content(self, file_path: Path, expected_content: str = None, min_length: int = None):
        """Assert file content meets expectations."""
        assert file_path.exists(), f"File {file_path} does not exist"

        content = file_path.read_text(encoding='utf-8')

        if expected_content:
            assert expected_content in content

        if min_length:
            assert len(content) >= min_length

@pytest.fixture
async def stage_test_framework():
    """Pytest fixture for stage test framework."""
    framework = StageTestFramework()
    await framework.setup()
    yield framework
    await framework.teardown()
```

### 2. Individual Stage Tests
```python
# packages/morag-stages/tests/test_stage1.py
import pytest
from pathlib import Path

from morag_stages import StageType
from morag_stages.tests.test_framework import StageTestFramework

@pytest.mark.asyncio
async def test_stage1_markdown_input(stage_test_framework: StageTestFramework):
    """Test Stage 1 with markdown input."""

    # Create test markdown file
    test_file = stage_test_framework.create_test_markdown("input.md")

    # Execute markdown-conversion
    result = await stage_test_framework.execute_stage_test(
        StageType.MARKDOWN_CONVERSION,
        [test_file]
    )

    # Assert success
    stage_test_framework.assert_stage_success(result)

    # Check output file
    output_file = result.output_files[0]
    stage_test_framework.assert_file_content(output_file, "# Test Document")

@pytest.mark.asyncio
async def test_stage1_pdf_input(stage_test_framework: StageTestFramework):
    """Test Stage 1 with PDF input (mock)."""

    # This would require actual PDF processing
    # For now, test with markdown as placeholder
    test_file = stage_test_framework.create_test_markdown("input.md")

    result = await stage_test_framework.execute_stage_test(
        StageType.MARKDOWN_CONVERSION,
        [test_file],
        config={'markdown_conversion': {'preserve_formatting': True}}
    )

    stage_test_framework.assert_stage_success(result)

# packages/morag-stages/tests/test_stage3.py
@pytest.mark.asyncio
async def test_stage3_chunking(stage_test_framework: StageTestFramework):
    """Test Stage 3 chunking functionality."""

    # Create test markdown file
    test_file = stage_test_framework.create_test_markdown("input.md")

    # Execute Stage 3
    result = await stage_test_framework.execute_stage_test(
        StageType.CHUNKING,
        [test_file],
        config={
            'stage3': {
                'chunk_strategy': 'semantic',
                'chunk_size': 1000,
                'generate_summary': True
            }
        }
    )

    # Assert success
    stage_test_framework.assert_stage_success(result)

    # Check chunks file
    chunks_file = result.output_files[0]
    assert chunks_file.name.endswith('.chunks.json')

    # Validate chunks content
    import json
    with open(chunks_file, 'r') as f:
        chunks_data = json.load(f)

    assert 'chunks' in chunks_data
    assert 'summary' in chunks_data
    assert len(chunks_data['chunks']) > 0

    # Check chunk structure
    chunk = chunks_data['chunks'][0]
    assert 'content' in chunk
    assert 'embedding' in chunk
    assert 'index' in chunk

# packages/morag-stages/tests/test_stage_chain.py
@pytest.mark.asyncio
async def test_full_stage_chain(stage_test_framework: StageTestFramework):
    """Test complete stage chain execution."""

    # Create test input
    test_file = stage_test_framework.create_test_markdown("input.md")

    # Define stage chain using canonical names
    stages = [
        StageType.MARKDOWN_CONVERSION,
        StageType.CHUNKER,
        StageType.FACT_GENERATOR
    ]

    # Execute stage chain
    results = await stage_test_framework.stage_manager.execute_stage_chain(
        stages,
        [test_file],
        stage_test_framework.create_context(test_file)
    )

    # Assert all stages completed
    assert len(results) == len(stages)
    for stage_type, result in results.items():
        stage_test_framework.assert_stage_success(result)

    # Check final output
    final_result = results[StageType.FACT_GENERATOR]
    facts_file = final_result.output_files[0]
    assert facts_file.name.endswith('.facts.json')
```

### 3. API Integration Tests
```python
# packages/morag/tests/test_stage_api.py
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import json

from morag.server import app

client = TestClient(app)

def test_stage1_api_endpoint():
    """Test Stage 1 API endpoint."""

    # Create test file
    test_content = "# Test Document\n\nThis is test content."

    with open("test_input.md", "w") as f:
        f.write(test_content)

    # Test API call
    with open("test_input.md", "rb") as f:
        response = client.post(
            "/api/v1/stages/1/execute",
            files={"file": ("test.md", f, "text/markdown")},
            data={"config": json.dumps({"include_timestamps": False})}
        )

    assert response.status_code == 200
    data = response.json()

    assert data["stage"] == 1
    assert data["status"] == "completed"
    assert len(data["output_files"]) > 0

def test_stage_chain_api():
    """Test stage chain API endpoint."""

    request_data = {
        "stages": [1, 3, 4],
        "input_file": "test_input.md",
        "config": {
            "stage3": {"chunk_size": 1000},
            "stage4": {"extract_entities": True}
        }
    }

    response = client.post("/api/v1/stages/chain", json=request_data)

    assert response.status_code == 200
    data = response.json()

    assert data["overall_success"] == True
    assert len(data["results"]) == 3

def test_file_download_api():
    """Test file download API."""

    # This would require a valid file_id from previous stage execution
    # For now, test the endpoint structure
    response = client.get("/api/v1/files/download/invalid_id")

    # Should return 404 for invalid ID
    assert response.status_code == 404
```

### 4. Performance Tests
```python
# packages/morag-stages/tests/test_performance.py
import pytest
import time
import asyncio
from pathlib import Path

from morag_stages.tests.test_framework import StageTestFramework

@pytest.mark.asyncio
async def test_stage_performance_benchmarks(stage_test_framework: StageTestFramework):
    """Test performance benchmarks for stages."""

    # Create larger test content
    large_content = "# Large Document\n\n" + "This is test content. " * 1000
    test_file = stage_test_framework.create_test_file("large.md", large_content)

    # Benchmark markdown-conversion
    start_time = time.time()
    result1 = await stage_test_framework.execute_stage_test(
        StageType.MARKDOWN_CONVERSION,
        [test_file]
    )
    stage1_time = time.time() - start_time

    # Benchmark chunker
    start_time = time.time()
    result3 = await stage_test_framework.execute_stage_test(
        StageType.CHUNKER,
        result1.output_files
    )
    stage3_time = time.time() - start_time

    # Assert performance expectations
    assert stage1_time < 10.0  # Should complete within 10 seconds
    assert stage3_time < 30.0  # Should complete within 30 seconds

    print(f"Stage 1 time: {stage1_time:.2f}s")
    print(f"Stage 3 time: {stage3_time:.2f}s")

@pytest.mark.asyncio
async def test_concurrent_stage_execution():
    """Test concurrent execution of multiple stages."""

    frameworks = [StageTestFramework() for _ in range(3)]

    try:
        # Setup all frameworks
        await asyncio.gather(*[f.setup() for f in frameworks])

        # Create test files
        test_files = []
        for i, framework in enumerate(frameworks):
            test_file = framework.create_test_markdown(f"test_{i}.md")
            test_files.append((framework, test_file))

        # Execute stages concurrently
        start_time = time.time()

        tasks = []
        for framework, test_file in test_files:
            task = framework.execute_stage_test(StageType.MARKDOWN_CONVERSION, [test_file])
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        concurrent_time = time.time() - start_time

        # All should succeed
        for result in results:
            assert result.status.value == "completed"

        print(f"Concurrent execution time: {concurrent_time:.2f}s")

    finally:
        # Cleanup
        await asyncio.gather(*[f.teardown() for f in frameworks])
```

### 5. Documentation Updates
```markdown
# docs/STAGE_BASED_ARCHITECTURE.md

# MoRAG Stage-Based Processing Architecture

## Overview

MoRAG now uses a stage-based processing architecture that allows for modular, reusable, and flexible content processing. Each stage is independent and can be executed separately or as part of a chain.

## Stages

### Stage 1: Input-to-Markdown
Converts input files to unified markdown format.

**Input**: Video/Audio/Document files, URLs
**Output**: `.md` file with metadata header
**CLI**: `morag stage 1 input.mp4`
**API**: `POST /api/v1/stages/1/execute`

### Stage 2: Markdown Optimizer (Optional)
Improves and fixes transcription errors using LLM.

**Input**: `.md` file from Stage 1
**Output**: `.opt.md` file (optimized markdown)
**CLI**: `morag stage 2 input.md`
**API**: `POST /api/v1/stages/2/execute`

### Stage 3: Chunking
Creates summary, chunks, and embeddings.

**Input**: `.md` or `.opt.md` file
**Output**: `.chunks.json` file
**CLI**: `morag stage 3 input.md`
**API**: `POST /api/v1/stages/3/execute`

### Stage 4: Fact Generation
Extracts facts, entities, and relations.

**Input**: `.chunks.json` file
**Output**: `.facts.json` file
**CLI**: `morag stage 4 input.chunks.json`
**API**: `POST /api/v1/stages/4/execute`

### Stage 5: Ingestion
Performs database ingestion.

**Input**: `.chunks.json` and `.facts.json` files
**Output**: `.ingestion.json` file
**CLI**: `morag stage 5 input.chunks.json input.facts.json`
**API**: `POST /api/v1/stages/5/execute`

## Usage Examples

### CLI Usage
```bash
# Execute single stage
morag stage 1 video.mp4 --output-dir ./output

# Execute stage chain
morag stages 1-3 video.mp4

# Full pipeline with optimization
morag process video.mp4 --optimize

# Batch processing
python cli/batch-process-stages.py ./input_folder --stages 1,3,5
```

### API Usage
```python
import requests

# Execute Stage 1
with open('video.mp4', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/stages/1/execute',
        files={'file': f},
        data={'config': '{"include_timestamps": true}'}
    )

# Execute stage chain
response = requests.post(
    'http://localhost:8000/api/v1/stages/chain',
    json={
        'stages': [1, 3, 4, 5],
        'input_file': 'video.mp4',
        'webhook_url': 'https://my-app.com/webhook'
    }
)
```

## Configuration

Each stage can be configured independently:

```yaml
# config.yaml
stage1:
  include_timestamps: true
  transcription_model: "whisper-large"

stage2:
  llm_model: "gemini-pro"
  fix_transcription_errors: true

stage3:
  chunk_strategy: "semantic"
  chunk_size: 4000
  generate_summary: true

stage4:
  extract_entities: true
  extract_relations: true
  domain: "medical"

stage5:
  databases: ["qdrant", "neo4j"]
  collection_name: "my_collection"
```

## File Management

All stage outputs are stored with unique file IDs and can be downloaded:

```bash
# Download file by ID
curl http://localhost:8000/api/v1/files/download/{file_id}

# List files by source
curl http://localhost:8000/api/v1/files/list/by-source?source_file=video.mp4

# Cleanup old files
curl -X DELETE http://localhost:8000/api/v1/files/cleanup?max_age_hours=24
```

## Webhook Integration

Configure webhook URLs to receive notifications when stages complete:

```json
{
  "event_type": "stage_completed",
  "stage": {
    "number": 1,
    "name": "INPUT_TO_MARKDOWN",
    "status": "completed"
  },
  "output_files": [
    {
      "filename": "video.md",
      "file_id": "abc123",
      "download_url": "/api/v1/files/download/abc123"
    }
  ]
}
```
```

## Implementation Steps

1. **REMOVE ALL EXISTING TESTS COMPLETELY**
2. **Create comprehensive test framework for stage-based architecture**
3. **Create new tests for all named stages**
4. **Add performance benchmarks and monitoring**
5. **Create API integration tests for new endpoints only**
6. **REMOVE OLD DOCUMENTATION and create new documentation with examples**
7. **Add deployment guides for new system only**
8. **Create troubleshooting documentation for new architecture**
9. **Add monitoring and logging guides**
10. **Add security and best practices documentation for new system**

## Testing Requirements

- Unit tests for all stage implementations
- Integration tests for stage chains
- API endpoint tests
- Performance benchmarks
- Error handling validation
- Webhook notification tests
- File management tests
- Security and access control tests

## Files to Create/Update

- `packages/morag-stages/tests/test_framework.py`
- `packages/morag-stages/tests/test_stage*.py` (for each stage)
- `packages/morag/tests/test_stage_api.py`
- `packages/morag-stages/tests/test_performance.py`
- `docs/STAGE_BASED_ARCHITECTURE.md`
- `docs/API_REFERENCE.md`
- `docs/DEPLOYMENT_GUIDE.md`
- `docs/MIGRATION_GUIDE.md`

## Success Criteria

- All new tests pass with good coverage (>90%) for stage-based architecture
- Performance benchmarks meet expectations for named stages
- Documentation is comprehensive and accurate for new system only
- API reference is complete with examples for new endpoints only
- **ALL OLD DOCUMENTATION IS REMOVED** - no migration guide needed
- Deployment guide covers new system scenarios only
- **NO LEGACY TEST CODE REMAINS**
