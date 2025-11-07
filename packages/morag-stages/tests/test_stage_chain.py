"""Tests for stage chain execution."""

import json
import pytest
from pathlib import Path

from morag_stages import StageType
from morag_stages.models import StageStatus
from .test_framework import StageTestFramework


@pytest.mark.asyncio
async def test_basic_stage_chain(stage_test_framework: StageTestFramework):
    """Test basic stage chain execution."""

    # Create test input
    test_file = stage_test_framework.create_test_markdown("input.md")

    # Define stage chain using canonical names
    stages = [
        StageType.MARKDOWN_CONVERSION,
        StageType.CHUNKER,
        StageType.FACT_GENERATOR
    ]

    # Create context
    context = stage_test_framework.create_context(test_file)

    # Execute stage chain
    results = await stage_test_framework.stage_manager.execute_stage_chain(
        stages,
        [test_file],
        context
    )

    # Assert all stages completed
    assert len(results) == len(stages)

    for result in results:
        stage_test_framework.assert_stage_success(result)

    # Check that outputs flow correctly between stages
    # Stage 1 should produce .md file
    md_result = results[0]
    md_file = stage_test_framework.get_file_by_extension(md_result.output_files, ".md")
    assert md_file.exists()

    # Stage 2 should produce .chunks.json file
    chunks_result = results[1]
    chunks_file = stage_test_framework.get_file_by_extension(chunks_result.output_files, ".chunks.json")
    assert chunks_file.exists()

    # Stage 3 should produce .facts.json file
    facts_result = results[2]
    facts_file = stage_test_framework.get_file_by_extension(facts_result.output_files, ".facts.json")
    assert facts_file.exists()


@pytest.mark.asyncio
async def test_full_pipeline_with_optimization(stage_test_framework: StageTestFramework):
    """Test full pipeline including markdown optimization."""

    # Create test input
    test_file = stage_test_framework.create_test_markdown("input.md")

    # Define full stage chain including optimization
    stages = [
        StageType.MARKDOWN_CONVERSION,
        StageType.MARKDOWN_OPTIMIZER,
        StageType.CHUNKER,
        StageType.FACT_GENERATOR,
        StageType.INGESTOR
    ]

    # Create context with configuration
    config = {
        'markdown-optimizer': {
            'fix_transcription_errors': True,
            'improve_readability': True,
            'preserve_timestamps': True
        },
        'chunker': {
            'chunk_strategy': 'semantic',
            'chunk_size': 2000,
            'generate_summary': True
        },
        'fact-generator': {
            'extract_entities': True,
            'extract_relations': True,
            'domain': 'general'
        },
        'ingestor': {
            'databases': ['qdrant'],
            'collection_name': 'test_collection'
        }
    }

    context = stage_test_framework.create_context(test_file, config)

    # Execute full pipeline
    results = await stage_test_framework.stage_manager.execute_stage_chain(
        stages,
        [test_file],
        context
    )

    # Assert all stages completed
    assert len(results) == len(stages)

    for result in results:
        stage_test_framework.assert_stage_success(result)

    # Check specific outputs
    # Should have optimized markdown
    opt_result = results[1]
    opt_file = stage_test_framework.get_file_by_extension(opt_result.output_files, ".opt.md")
    assert opt_file.exists()

    # Should have ingestion results
    ingest_result = results[4]
    ingest_file = stage_test_framework.get_file_by_extension(ingest_result.output_files, ".ingestion.json")
    assert ingest_file.exists()


@pytest.mark.asyncio
async def test_stage_chain_with_failure(stage_test_framework: StageTestFramework):
    """Test stage chain behavior when a stage fails."""

    # Create test input that might cause failure
    empty_file = stage_test_framework.create_test_file("empty.txt", "")

    # Define stage chain
    stages = [
        StageType.MARKDOWN_CONVERSION,
        StageType.CHUNKER,
        StageType.FACT_GENERATOR
    ]

    # Create context
    context = stage_test_framework.create_context(empty_file)

    # Execute stage chain
    results = await stage_test_framework.stage_manager.execute_stage_chain(
        stages,
        [empty_file],
        context
    )

    # Check results - some stages might fail with empty input
    assert len(results) <= len(stages)  # Might stop early on failure

    # At least one stage should have run
    assert len(results) > 0

    # Check if any stage failed
    failed_stages = [r for r in results if r.status == StageStatus.FAILED]

    if failed_stages:
        # If stages failed, they should have error messages
        for failed_result in failed_stages:
            assert failed_result.error_message is not None
            assert len(failed_result.error_message) > 0


@pytest.mark.asyncio
async def test_stage_chain_resume_capability(stage_test_framework: StageTestFramework):
    """Test stage chain resume capability with existing outputs."""

    # Create test input
    test_file = stage_test_framework.create_test_markdown("input.md")

    # Execute first stage only
    context = stage_test_framework.create_context(test_file)

    result1 = await stage_test_framework.stage_manager.execute_stage(
        StageType.MARKDOWN_CONVERSION,
        [test_file],
        context
    )
    stage_test_framework.assert_stage_success(result1)

    # Now execute full chain - first stage should be skipped
    stages = [
        StageType.MARKDOWN_CONVERSION,
        StageType.CHUNKER,
        StageType.FACT_GENERATOR
    ]

    results = await stage_test_framework.stage_manager.execute_stage_chain(
        stages,
        [test_file],
        context
    )

    # First stage should be skipped, others should complete
    assert len(results) == len(stages)

    # First result should be skipped
    stage_test_framework.assert_stage_skipped(results[0])

    # Other stages should complete
    for result in results[1:]:
        stage_test_framework.assert_stage_success(result)


@pytest.mark.asyncio
async def test_stage_chain_partial_execution(stage_test_framework: StageTestFramework):
    """Test executing only part of the stage chain."""

    # Create test input
    test_file = stage_test_framework.create_test_markdown("input.md")

    # Execute only first two stages
    stages = [
        StageType.MARKDOWN_CONVERSION,
        StageType.CHUNKER
    ]

    context = stage_test_framework.create_context(test_file)

    results = await stage_test_framework.stage_manager.execute_stage_chain(
        stages,
        [test_file],
        context
    )

    # Should complete both stages
    assert len(results) == 2

    for result in results:
        stage_test_framework.assert_stage_success(result)

    # Should have markdown and chunks files
    md_result = results[0]
    md_file = stage_test_framework.get_file_by_extension(md_result.output_files, ".md")
    assert md_file.exists()

    chunks_result = results[1]
    chunks_file = stage_test_framework.get_file_by_extension(chunks_result.output_files, ".chunks.json")
    assert chunks_file.exists()


@pytest.mark.asyncio
async def test_stage_chain_with_different_configs(stage_test_framework: StageTestFramework):
    """Test stage chain with different configurations for each stage."""

    # Create test input
    test_file = stage_test_framework.create_test_markdown("input.md")

    # Define stage-specific configurations
    config = {
        'markdown-conversion': {
            'preserve_formatting': True,
            'include_metadata': True
        },
        'chunker': {
            'chunk_strategy': 'semantic',
            'chunk_size': 1500,
            'generate_summary': True
        },
        'fact-generator': {
            'extract_entities': True,
            'extract_relations': True,
            'domain': 'technology',
            'min_confidence': 0.8
        }
    }

    stages = [
        StageType.MARKDOWN_CONVERSION,
        StageType.CHUNKER,
        StageType.FACT_GENERATOR
    ]

    context = stage_test_framework.create_context(test_file, config)

    results = await stage_test_framework.stage_manager.execute_stage_chain(
        stages,
        [test_file],
        context
    )

    # All stages should complete with their specific configs
    assert len(results) == len(stages)

    for result in results:
        stage_test_framework.assert_stage_success(result)


@pytest.mark.asyncio
async def test_stage_chain_output_validation(stage_test_framework: StageTestFramework):
    """Test that stage chain outputs are properly validated."""

    # Create test input
    test_file = stage_test_framework.create_test_markdown("input.md")

    stages = [
        StageType.MARKDOWN_CONVERSION,
        StageType.CHUNKER,
        StageType.FACT_GENERATOR
    ]

    context = stage_test_framework.create_context(test_file)

    results = await stage_test_framework.stage_manager.execute_stage_chain(
        stages,
        [test_file],
        context
    )

    # Validate each stage's output
    for i, result in enumerate(results):
        stage_test_framework.assert_stage_success(result)

        # Check that output files exist and are valid
        for output_file in result.output_files:
            assert output_file.exists()
            assert output_file.stat().st_size > 0

            # Validate JSON files
            if output_file.name.endswith('.json'):
                try:
                    with open(output_file, 'r', encoding='utf-8') as f:
                        json.load(f)  # Should parse without error
                except json.JSONDecodeError:
                    pytest.fail(f"Invalid JSON in {output_file}")

            # Validate markdown files
            elif output_file.name.endswith('.md'):
                content = output_file.read_text(encoding='utf-8')
                assert len(content.strip()) > 0


@pytest.mark.asyncio
async def test_stage_chain_execution_time_tracking(stage_test_framework: StageTestFramework):
    """Test that stage chain tracks execution times properly."""

    # Create test input
    test_file = stage_test_framework.create_test_markdown("input.md")

    stages = [
        StageType.MARKDOWN_CONVERSION,
        StageType.CHUNKER
    ]

    context = stage_test_framework.create_context(test_file)

    results = await stage_test_framework.stage_manager.execute_stage_chain(
        stages,
        [test_file],
        context
    )

    # Check execution time tracking
    for result in results:
        stage_test_framework.assert_stage_success(result)

        # Should have execution time metadata
        if hasattr(result, 'metadata') and hasattr(result.metadata, 'execution_time'):
            assert result.metadata.execution_time >= 0

        # Execution time should be reasonable (not negative, not extremely large)
        if hasattr(result, 'metadata') and hasattr(result.metadata, 'execution_time'):
            assert result.metadata.execution_time < 300  # Less than 5 minutes for test
