"""Tests for markdown-conversion stage."""

import pytest
from pathlib import Path

from morag_stages import StageType
from .test_framework import StageTestFramework


@pytest.mark.asyncio
async def test_markdown_conversion_with_markdown_input(stage_test_framework: StageTestFramework):
    """Test markdown-conversion stage with markdown input."""
    
    # Create test markdown file
    test_file = stage_test_framework.create_test_markdown("input.md")
    
    # Execute markdown-conversion stage
    result = await stage_test_framework.execute_stage_test(
        StageType.MARKDOWN_CONVERSION,
        [test_file]
    )
    
    # Assert success
    stage_test_framework.assert_stage_success(result)
    
    # Check output file
    output_file = stage_test_framework.get_file_by_extension(result.output_files, ".md")
    stage_test_framework.assert_file_content(output_file, "# Test Document")
    stage_test_framework.assert_file_content(output_file, "artificial intelligence")


@pytest.mark.asyncio
async def test_markdown_conversion_with_text_input(stage_test_framework: StageTestFramework):
    """Test markdown-conversion stage with plain text input."""
    
    # Create test text file
    text_content = """This is a plain text document.
    
It contains multiple paragraphs with information about various topics.

The document discusses technology, science, and other subjects.
"""
    test_file = stage_test_framework.create_test_file("input.txt", text_content)
    
    # Execute markdown-conversion stage
    result = await stage_test_framework.execute_stage_test(
        StageType.MARKDOWN_CONVERSION,
        [test_file]
    )
    
    # Assert success
    stage_test_framework.assert_stage_success(result)
    
    # Check output file
    output_file = stage_test_framework.get_file_by_extension(result.output_files, ".md")
    stage_test_framework.assert_file_content(output_file, "plain text document")


@pytest.mark.asyncio
async def test_markdown_conversion_with_config(stage_test_framework: StageTestFramework):
    """Test markdown-conversion stage with custom configuration."""
    
    # Create test file
    test_file = stage_test_framework.create_test_markdown("input.md")
    
    # Execute with custom config
    config = {
        'markdown-conversion': {
            'preserve_formatting': True,
            'include_metadata': True,
            'add_source_attribution': True
        }
    }
    
    result = await stage_test_framework.execute_stage_test(
        StageType.MARKDOWN_CONVERSION,
        [test_file],
        config=config
    )
    
    # Assert success
    stage_test_framework.assert_stage_success(result)
    
    # Check output file has metadata
    output_file = stage_test_framework.get_file_by_extension(result.output_files, ".md")
    content = output_file.read_text(encoding='utf-8')
    
    # Should preserve original metadata
    assert "---" in content
    assert "title:" in content


@pytest.mark.asyncio
async def test_markdown_conversion_skip_existing(stage_test_framework: StageTestFramework):
    """Test markdown-conversion stage skips when output already exists."""
    
    # Create test file
    test_file = stage_test_framework.create_test_markdown("input.md")
    
    # Execute first time
    result1 = await stage_test_framework.execute_stage_test(
        StageType.MARKDOWN_CONVERSION,
        [test_file]
    )
    stage_test_framework.assert_stage_success(result1)
    
    # Execute second time - should skip
    result2 = await stage_test_framework.execute_stage_test(
        StageType.MARKDOWN_CONVERSION,
        [test_file]
    )
    stage_test_framework.assert_stage_skipped(result2)


@pytest.mark.asyncio
async def test_markdown_conversion_audio_simulation(stage_test_framework: StageTestFramework):
    """Test markdown-conversion stage with audio file simulation."""
    
    # Create a file that simulates audio processing output
    audio_content = """---
title: Audio Transcript
content_type: audio
duration: 120
processed_at: 2024-01-01T00:00:00
---

# Audio Transcript

[00:00 - 00:15] Speaker 1: Welcome to this presentation about artificial intelligence.

[00:15 - 00:30] Speaker 1: Today we'll discuss machine learning algorithms and their applications.

[00:30 - 00:45] Speaker 2: Thank you for the introduction. Let's start with the basics.

[00:45 - 01:00] Speaker 2: Machine learning is a subset of artificial intelligence.

[01:00 - 01:15] Speaker 1: That's correct. It involves training algorithms on data.

[01:15 - 02:00] Speaker 2: The applications are vast, from image recognition to natural language processing.
"""
    
    test_file = stage_test_framework.create_test_file("audio.mp3", audio_content)
    
    # Execute with audio-specific config
    config = {
        'markdown-conversion': {
            'include_timestamps': True,
            'speaker_diarization': True,
            'content_type': 'audio'
        }
    }
    
    result = await stage_test_framework.execute_stage_test(
        StageType.MARKDOWN_CONVERSION,
        [test_file],
        config=config
    )
    
    # Assert success
    stage_test_framework.assert_stage_success(result)
    
    # Check output contains timestamps and speakers
    output_file = stage_test_framework.get_file_by_extension(result.output_files, ".md")
    content = output_file.read_text(encoding='utf-8')
    
    assert "[00:00 - 00:15]" in content
    assert "Speaker 1:" in content
    assert "artificial intelligence" in content


@pytest.mark.asyncio
async def test_markdown_conversion_video_simulation(stage_test_framework: StageTestFramework):
    """Test markdown-conversion stage with video file simulation."""
    
    # Create a file that simulates video processing output
    video_content = """---
title: Video Transcript
content_type: video
duration: 180
resolution: 1920x1080
processed_at: 2024-01-01T00:00:00
---

# Video Transcript

## Topic: Introduction to Data Science

[00:00 - 00:20] The video begins with an overview of data science fundamentals.

[00:20 - 00:45] Key concepts include statistics, programming, and domain expertise.

[00:45 - 01:10] The presenter demonstrates data visualization techniques.

[01:10 - 01:35] Machine learning models are introduced with practical examples.

[01:35 - 02:00] The session covers data preprocessing and feature engineering.

[02:00 - 03:00] Final section discusses real-world applications and case studies.
"""
    
    test_file = stage_test_framework.create_test_file("video.mp4", video_content)
    
    # Execute with video-specific config
    config = {
        'markdown-conversion': {
            'include_timestamps': True,
            'topic_segmentation': True,
            'content_type': 'video'
        }
    }
    
    result = await stage_test_framework.execute_stage_test(
        StageType.MARKDOWN_CONVERSION,
        [test_file],
        config=config
    )
    
    # Assert success
    stage_test_framework.assert_stage_success(result)
    
    # Check output contains video-specific elements
    output_file = stage_test_framework.get_file_by_extension(result.output_files, ".md")
    content = output_file.read_text(encoding='utf-8')
    
    assert "Video Transcript" in content
    assert "[00:00 - 00:20]" in content
    assert "data science" in content
    assert "Topic:" in content


@pytest.mark.asyncio
async def test_markdown_conversion_error_handling(stage_test_framework: StageTestFramework):
    """Test markdown-conversion stage error handling."""
    
    # Create an empty file to trigger potential errors
    empty_file = stage_test_framework.create_test_file("empty.txt", "")
    
    # Execute stage - should handle gracefully
    result = await stage_test_framework.execute_stage_test(
        StageType.MARKDOWN_CONVERSION,
        [empty_file]
    )
    
    # Should either succeed with minimal output or fail gracefully
    if result.status.value == "failed":
        assert result.error_message is not None
        assert len(result.error_message) > 0
    else:
        # If it succeeds, should have some output
        stage_test_framework.assert_stage_success(result)


@pytest.mark.asyncio
async def test_markdown_conversion_multiple_files(stage_test_framework: StageTestFramework):
    """Test markdown-conversion stage with multiple input files."""
    
    # Create multiple test files
    file1 = stage_test_framework.create_test_file("doc1.txt", "First document content.")
    file2 = stage_test_framework.create_test_file("doc2.txt", "Second document content.")
    
    # Execute with multiple files
    result = await stage_test_framework.execute_stage_test(
        StageType.MARKDOWN_CONVERSION,
        [file1, file2]
    )
    
    # Should process the first file (or handle multiple files appropriately)
    if result.status == StageStatus.COMPLETED:
        stage_test_framework.assert_stage_success(result)
    else:
        # If multiple files aren't supported, should fail gracefully
        assert result.error_message is not None


@pytest.mark.asyncio
async def test_markdown_conversion_output_naming(stage_test_framework: StageTestFramework):
    """Test markdown-conversion stage output file naming conventions."""
    
    # Create test file with specific name
    test_file = stage_test_framework.create_test_file("my_document.txt", "Test content")
    
    result = await stage_test_framework.execute_stage_test(
        StageType.MARKDOWN_CONVERSION,
        [test_file]
    )
    
    stage_test_framework.assert_stage_success(result)
    
    # Check output file naming
    output_file = stage_test_framework.get_file_by_extension(result.output_files, ".md")
    
    # Should be based on input filename
    assert "my_document" in output_file.name
    assert output_file.name.endswith(".md")
