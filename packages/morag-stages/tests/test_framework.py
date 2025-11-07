"""Test framework for stage-based processing."""

import asyncio
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import structlog

# Add path for stage imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morag_stages import StageManager, StageType
from morag_stages.models import StageContext, StageStatus

logger = structlog.get_logger(__name__)


class StageTestFramework:
    """Test framework for stage-based processing."""

    def __init__(self):
        self.temp_dir = None
        self.stage_manager = None
        self.test_data_dir = None
        self.output_dir = None

    async def setup(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.stage_manager = StageManager()

        # Create test data directory
        self.test_data_dir = self.temp_dir / "test_data"
        self.test_data_dir.mkdir()

        # Create output directory
        self.output_dir = self.temp_dir / "output"
        self.output_dir.mkdir()

        logger.info("Test framework setup complete", temp_dir=str(self.temp_dir))

    async def teardown(self):
        """Clean up test environment."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            logger.info("Test framework cleanup complete")

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
source_file: test_input.pdf
---

# Test Document

This is a test document for stage processing.

## Section 1

Some content here with important information about artificial intelligence and machine learning.

## Section 2

More content with different topics and concepts related to data science and natural language processing.

## Conclusion

This document contains various topics that can be used for testing fact extraction and entity recognition.
"""
        return self.create_test_file(filename, content)

    def create_test_chunks_json(self, filename: str = "test.chunks.json") -> Path:
        """Create a test chunks JSON file."""
        chunks_data = {
            "document_metadata": {
                "title": "Test Document",
                "content_type": "document",
                "source_file": "test_input.pdf",
                "processed_at": "2024-01-01T00:00:00"
            },
            "summary": "This is a test document that covers artificial intelligence, machine learning, data science, and natural language processing topics.",
            "chunk_count": 3,
            "chunks": [
                {
                    "index": 0,
                    "content": "This is a test document for stage processing. Some content here with important information about artificial intelligence and machine learning.",
                    "token_count": 25,
                    "embedding": [0.1] * 1536,
                    "source_metadata": {
                        "title": "Test Document",
                        "section": "Section 1"
                    }
                },
                {
                    "index": 1,
                    "content": "More content with different topics and concepts related to data science and natural language processing.",
                    "token_count": 18,
                    "embedding": [0.2] * 1536,
                    "source_metadata": {
                        "title": "Test Document",
                        "section": "Section 2"
                    }
                },
                {
                    "index": 2,
                    "content": "This document contains various topics that can be used for testing fact extraction and entity recognition.",
                    "token_count": 19,
                    "embedding": [0.3] * 1536,
                    "source_metadata": {
                        "title": "Test Document",
                        "section": "Conclusion"
                    }
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
                    "statement": "Test documents contain information about artificial intelligence",
                    "subject": "Test documents",
                    "predicate": "contain information about",
                    "object": "artificial intelligence",
                    "confidence": 0.9,
                    "source_chunk_index": 0,
                    "domain": "technology"
                },
                {
                    "statement": "Machine learning is related to data science",
                    "subject": "Machine learning",
                    "predicate": "is related to",
                    "object": "data science",
                    "confidence": 0.85,
                    "source_chunk_index": 1,
                    "domain": "technology"
                }
            ],
            "entities": [
                {
                    "name": "artificial intelligence",
                    "normalized_name": "artificial_intelligence",
                    "entity_type": "Technology",
                    "confidence": 0.95,
                    "mentions": 2
                },
                {
                    "name": "machine learning",
                    "normalized_name": "machine_learning",
                    "entity_type": "Technology",
                    "confidence": 0.9,
                    "mentions": 1
                },
                {
                    "name": "data science",
                    "normalized_name": "data_science",
                    "entity_type": "Field",
                    "confidence": 0.88,
                    "mentions": 1
                }
            ],
            "relations": [
                {
                    "subject": "machine learning",
                    "predicate": "IS_PART_OF",
                    "object": "artificial intelligence",
                    "confidence": 0.9,
                    "relation_type": "hierarchical"
                },
                {
                    "subject": "data science",
                    "predicate": "USES",
                    "object": "machine learning",
                    "confidence": 0.85,
                    "relation_type": "functional"
                }
            ],
            "keywords": [
                "artificial intelligence",
                "machine learning",
                "data science",
                "natural language processing",
                "testing",
                "fact extraction"
            ]
        }

        file_path = self.test_data_dir / filename
        file_path.write_text(json.dumps(facts_data, indent=2), encoding='utf-8')
        return file_path

    def create_context(self, source_file: Path, config: Optional[Dict[str, Any]] = None) -> StageContext:
        """Create a stage context for testing."""
        return StageContext(
            source_path=source_file,
            output_dir=self.output_dir,
            config=config or {}
        )

    async def execute_stage_test(self,
                                stage_type: StageType,
                                input_files: List[Path],
                                config: Optional[Dict[str, Any]] = None):
        """Execute a stage for testing."""

        context = self.create_context(input_files[0], config)

        return await self.stage_manager.execute_stage(stage_type, input_files, context)

    def assert_stage_success(self, result):
        """Assert that stage execution was successful."""
        assert result.status == StageStatus.COMPLETED, f"Stage failed with status: {result.status}, error: {result.error_message}"
        assert result.error_message is None, f"Stage had error: {result.error_message}"
        assert len(result.output_files) > 0, "Stage produced no output files"
        assert all(f.exists() for f in result.output_files), "Some output files do not exist"

    def assert_stage_skipped(self, result):
        """Assert that stage execution was skipped."""
        assert result.status == StageStatus.SKIPPED, f"Expected stage to be skipped, got: {result.status}"
        assert len(result.output_files) > 0, "Skipped stage should still have output files"

    def assert_file_content(self, file_path: Path, expected_content: Optional[str] = None, min_length: Optional[int] = None):
        """Assert file content meets expectations."""
        assert file_path.exists(), f"File {file_path} does not exist"

        content = file_path.read_text(encoding='utf-8')

        if expected_content:
            assert expected_content in content, f"Expected content '{expected_content}' not found in file"

        if min_length:
            assert len(content) >= min_length, f"File content too short: {len(content)} < {min_length}"

    def assert_json_structure(self, file_path: Path, required_keys: List[str]):
        """Assert JSON file has required structure."""
        assert file_path.exists(), f"JSON file {file_path} does not exist"

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for key in required_keys:
            assert key in data, f"Required key '{key}' not found in JSON file"

    def get_file_by_extension(self, files: List[Path], extension: str) -> Path:
        """Get file with specific extension from list."""
        for file_path in files:
            if file_path.name.endswith(extension):
                return file_path
        raise AssertionError(f"No file with extension '{extension}' found in {[f.name for f in files]}")


@pytest.fixture
async def stage_test_framework():
    """Pytest fixture for stage test framework."""
    framework = StageTestFramework()
    await framework.setup()
    yield framework
    await framework.teardown()


@pytest.fixture
def sample_markdown_file(stage_test_framework):
    """Fixture for sample markdown file."""
    return stage_test_framework.create_test_markdown()


@pytest.fixture
def sample_chunks_file(stage_test_framework):
    """Fixture for sample chunks JSON file."""
    return stage_test_framework.create_test_chunks_json()


@pytest.fixture
def sample_facts_file(stage_test_framework):
    """Fixture for sample facts JSON file."""
    return stage_test_framework.create_test_facts_json()
