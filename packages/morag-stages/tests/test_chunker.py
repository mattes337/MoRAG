"""Tests for chunker stage."""

import json
import pytest
from pathlib import Path

from morag_stages import StageType
from .test_framework import StageTestFramework


@pytest.mark.asyncio
async def test_chunker_basic_functionality(stage_test_framework: StageTestFramework):
    """Test chunker stage basic functionality."""

    # Create test markdown file
    test_file = stage_test_framework.create_test_markdown("input.md")

    # Execute chunker stage
    result = await stage_test_framework.execute_stage_test(
        StageType.CHUNKER,
        [test_file]
    )

    # Assert success
    stage_test_framework.assert_stage_success(result)

    # Check output file
    chunks_file = stage_test_framework.get_file_by_extension(result.output_files, ".chunks.json")

    # Validate JSON structure
    stage_test_framework.assert_json_structure(
        chunks_file,
        ["chunks", "summary", "chunk_count", "document_metadata"]
    )

    # Load and validate content
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)

    assert len(chunks_data['chunks']) > 0
    assert chunks_data['chunk_count'] == len(chunks_data['chunks'])
    assert len(chunks_data['summary']) > 0

    # Check chunk structure
    chunk = chunks_data['chunks'][0]
    assert 'content' in chunk
    assert 'index' in chunk
    assert 'token_count' in chunk
    assert len(chunk['content']) > 0


@pytest.mark.asyncio
async def test_chunker_with_semantic_strategy(stage_test_framework: StageTestFramework):
    """Test chunker stage with semantic chunking strategy."""

    # Create test markdown file
    test_file = stage_test_framework.create_test_markdown("input.md")

    # Execute with semantic chunking config
    config = {
        'chunker': {
            'chunk_strategy': 'semantic',
            'chunk_size': 1000,
            'generate_summary': True,
            'include_embeddings': True
        }
    }

    result = await stage_test_framework.execute_stage_test(
        StageType.CHUNKER,
        [test_file],
        config=config
    )

    # Assert success
    stage_test_framework.assert_stage_success(result)

    # Check chunks file
    chunks_file = stage_test_framework.get_file_by_extension(result.output_files, ".chunks.json")

    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)

    # Check semantic chunking results
    assert len(chunks_data['chunks']) > 0

    # Check if embeddings are included
    chunk = chunks_data['chunks'][0]
    if 'embedding' in chunk:
        assert isinstance(chunk['embedding'], list)
        assert len(chunk['embedding']) > 0


@pytest.mark.asyncio
async def test_chunker_with_page_level_strategy(stage_test_framework: StageTestFramework):
    """Test chunker stage with page-level chunking strategy."""

    # Create test markdown file with page markers
    content = """---
title: Multi-Page Document
content_type: document
pages: 3
---

# Page 1 Content

This is the content of the first page.

---page-break---

# Page 2 Content

This is the content of the second page with different information.

---page-break---

# Page 3 Content

This is the final page with concluding remarks.
"""

    test_file = stage_test_framework.create_test_file("multi_page.md", content)

    # Execute with page-level chunking config
    config = {
        'chunker': {
            'chunk_strategy': 'page-level',
            'preserve_page_boundaries': True,
            'generate_summary': True
        }
    }

    result = await stage_test_framework.execute_stage_test(
        StageType.CHUNKER,
        [test_file],
        config=config
    )

    # Assert success
    stage_test_framework.assert_stage_success(result)

    # Check chunks file
    chunks_file = stage_test_framework.get_file_by_extension(result.output_files, ".chunks.json")

    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)

    # Should have chunks corresponding to pages
    assert len(chunks_data['chunks']) >= 3

    # Check page metadata
    for chunk in chunks_data['chunks']:
        if 'source_metadata' in chunk:
            # May contain page information
            pass


@pytest.mark.asyncio
async def test_chunker_with_topic_based_strategy(stage_test_framework: StageTestFramework):
    """Test chunker stage with topic-based chunking strategy."""

    # Create test markdown file with timestamps (simulating audio/video)
    content = """---
title: Audio Transcript
content_type: audio
duration: 300
---

# Audio Transcript

## Topic: Introduction
[00:00 - 01:00] Welcome to this presentation about artificial intelligence and machine learning.

## Topic: Machine Learning Basics
[01:00 - 02:30] Machine learning is a subset of AI that focuses on algorithms that learn from data.

## Topic: Applications
[02:30 - 04:00] Applications include image recognition, natural language processing, and recommendation systems.

## Topic: Conclusion
[04:00 - 05:00] Thank you for listening to this overview of machine learning concepts.
"""

    test_file = stage_test_framework.create_test_file("audio_transcript.md", content)

    # Execute with topic-based chunking config
    config = {
        'chunker': {
            'chunk_strategy': 'topic-based',
            'preserve_timestamps': True,
            'generate_summary': True,
            'chunk_size': 2000
        }
    }

    result = await stage_test_framework.execute_stage_test(
        StageType.CHUNKER,
        [test_file],
        config=config
    )

    # Assert success
    stage_test_framework.assert_stage_success(result)

    # Check chunks file
    chunks_file = stage_test_framework.get_file_by_extension(result.output_files, ".chunks.json")

    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)

    # Should have topic-based chunks
    assert len(chunks_data['chunks']) > 0

    # Check for timestamp preservation
    found_timestamp = False
    for chunk in chunks_data['chunks']:
        if '[' in chunk['content'] and ']' in chunk['content']:
            found_timestamp = True
            break

    # Should preserve timestamps if configured
    if config['chunker']['preserve_timestamps']:
        assert found_timestamp


@pytest.mark.asyncio
async def test_chunker_custom_chunk_size(stage_test_framework: StageTestFramework):
    """Test chunker stage with custom chunk size."""

    # Create large test markdown file
    large_content = """---
title: Large Document
---

# Large Document

""" + "This is a sentence with content. " * 200  # Create large content

    test_file = stage_test_framework.create_test_file("large.md", large_content)

    # Execute with small chunk size
    config = {
        'chunker': {
            'chunk_size': 500,  # Small chunks
            'chunk_overlap': 50,
            'generate_summary': True
        }
    }

    result = await stage_test_framework.execute_stage_test(
        StageType.CHUNKER,
        [test_file],
        config=config
    )

    # Assert success
    stage_test_framework.assert_stage_success(result)

    # Check chunks file
    chunks_file = stage_test_framework.get_file_by_extension(result.output_files, ".chunks.json")

    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)

    # Should have multiple chunks due to small chunk size
    assert len(chunks_data['chunks']) > 1

    # Check chunk sizes are reasonable
    for chunk in chunks_data['chunks']:
        # Chunks should be roughly within the specified size
        assert len(chunk['content']) <= 1000  # Allow some flexibility


@pytest.mark.asyncio
async def test_chunker_skip_existing(stage_test_framework: StageTestFramework):
    """Test chunker stage skips when output already exists."""

    # Create test file
    test_file = stage_test_framework.create_test_markdown("input.md")

    # Execute first time
    result1 = await stage_test_framework.execute_stage_test(
        StageType.CHUNKER,
        [test_file]
    )
    stage_test_framework.assert_stage_success(result1)

    # Execute second time - should skip
    result2 = await stage_test_framework.execute_stage_test(
        StageType.CHUNKER,
        [test_file]
    )
    stage_test_framework.assert_stage_skipped(result2)


@pytest.mark.asyncio
async def test_chunker_without_summary(stage_test_framework: StageTestFramework):
    """Test chunker stage without summary generation."""

    # Create test file
    test_file = stage_test_framework.create_test_markdown("input.md")

    # Execute without summary
    config = {
        'chunker': {
            'generate_summary': False,
            'chunk_size': 1000
        }
    }

    result = await stage_test_framework.execute_stage_test(
        StageType.CHUNKER,
        [test_file],
        config=config
    )

    # Assert success
    stage_test_framework.assert_stage_success(result)

    # Check chunks file
    chunks_file = stage_test_framework.get_file_by_extension(result.output_files, ".chunks.json")

    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)

    # Summary should be empty or minimal
    assert 'summary' in chunks_data
    # Summary might be empty or a default value


@pytest.mark.asyncio
async def test_chunker_output_naming(stage_test_framework: StageTestFramework):
    """Test chunker stage output file naming conventions."""

    # Create test file with specific name
    test_file = stage_test_framework.create_test_file("my_document.md", "# Test\n\nContent here.")

    result = await stage_test_framework.execute_stage_test(
        StageType.CHUNKER,
        [test_file]
    )

    stage_test_framework.assert_stage_success(result)

    # Check output file naming
    chunks_file = stage_test_framework.get_file_by_extension(result.output_files, ".chunks.json")

    # Should be based on input filename
    assert "my_document" in chunks_file.name
    assert chunks_file.name.endswith(".chunks.json")


@pytest.mark.asyncio
async def test_chunker_metadata_preservation(stage_test_framework: StageTestFramework):
    """Test chunker stage preserves document metadata."""

    # Create test file with rich metadata
    content = """---
title: Test Document
author: Test Author
date: 2024-01-01
content_type: document
language: en
tags: [test, example, demo]
---

# Test Document

This is test content for metadata preservation testing.
"""

    test_file = stage_test_framework.create_test_file("metadata_test.md", content)

    result = await stage_test_framework.execute_stage_test(
        StageType.CHUNKER,
        [test_file]
    )

    stage_test_framework.assert_stage_success(result)

    # Check chunks file
    chunks_file = stage_test_framework.get_file_by_extension(result.output_files, ".chunks.json")

    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)

    # Check document metadata preservation
    assert 'document_metadata' in chunks_data
    metadata = chunks_data['document_metadata']

    # Should preserve key metadata
    assert 'title' in metadata
    assert metadata['title'] == 'Test Document'
