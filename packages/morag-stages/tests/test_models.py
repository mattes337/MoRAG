"""Tests for stage models."""

import pytest
from pathlib import Path
from datetime import datetime

from morag_stages.models import (
    StageType, StageStatus, StageResult, StageContext,
    StageMetadata, StageConfig, PipelineConfig
)


class TestStageType:
    """Test StageType enum."""

    def test_stage_types(self):
        """Test all stage types are defined."""
        assert StageType.MARKDOWN_CONVERSION.value == "markdown-conversion"
        assert StageType.MARKDOWN_OPTIMIZER.value == "markdown-optimizer"
        assert StageType.CHUNKER.value == "chunker"
        assert StageType.FACT_GENERATOR.value == "fact-generator"
        assert StageType.INGESTOR.value == "ingestor"


class TestStageStatus:
    """Test StageStatus enum."""

    def test_stage_statuses(self):
        """Test all stage statuses are defined."""
        assert StageStatus.PENDING.value == "pending"
        assert StageStatus.RUNNING.value == "running"
        assert StageStatus.COMPLETED.value == "completed"
        assert StageStatus.FAILED.value == "failed"
        assert StageStatus.SKIPPED.value == "skipped"


class TestStageMetadata:
    """Test StageMetadata model."""

    def test_create_metadata(self):
        """Test creating stage metadata."""
        metadata = StageMetadata(
            execution_time=1.5,
            start_time=datetime.now(),
            input_files=["input.txt"],
            output_files=["output.md"]
        )

        assert metadata.execution_time == 1.5
        assert isinstance(metadata.start_time, datetime)
        assert metadata.input_files == ["input.txt"]
        assert metadata.output_files == ["output.md"]
        assert metadata.config_used == {}
        assert metadata.metrics == {}
        assert metadata.warnings == []


class TestStageResult:
    """Test StageResult model."""

    def test_create_result(self):
        """Test creating stage result."""
        metadata = StageMetadata(
            execution_time=1.0,
            start_time=datetime.now(),
            input_files=["input.txt"],
            output_files=["output.md"]
        )

        result = StageResult(
            stage_type=StageType.MARKDOWN_CONVERSION,
            status=StageStatus.COMPLETED,
            output_files=[Path("output.md")],
            metadata=metadata
        )

        assert result.stage_type == StageType.MARKDOWN_CONVERSION
        assert result.status == StageStatus.COMPLETED
        assert result.output_files == [Path("output.md")]
        assert result.metadata == metadata
        assert result.error_message is None
        assert result.data == {}

    def test_result_properties(self):
        """Test result convenience properties."""
        metadata = StageMetadata(
            execution_time=1.0,
            start_time=datetime.now(),
            input_files=["input.txt"],
            output_files=["output.md"]
        )

        # Test successful result
        success_result = StageResult(
            stage_type=StageType.MARKDOWN_CONVERSION,
            status=StageStatus.COMPLETED,
            output_files=[Path("output.md")],
            metadata=metadata
        )

        assert success_result.success is True
        assert success_result.failed is False
        assert success_result.skipped is False

        # Test failed result
        failed_result = StageResult(
            stage_type=StageType.MARKDOWN_CONVERSION,
            status=StageStatus.FAILED,
            output_files=[],
            metadata=metadata,
            error_message="Test error"
        )

        assert failed_result.success is False
        assert failed_result.failed is True
        assert failed_result.skipped is False

        # Test skipped result
        skipped_result = StageResult(
            stage_type=StageType.MARKDOWN_CONVERSION,
            status=StageStatus.SKIPPED,
            output_files=[Path("output.md")],
            metadata=metadata
        )

        assert skipped_result.success is False
        assert skipped_result.failed is False
        assert skipped_result.skipped is True

    def test_get_output_by_extension(self):
        """Test getting output files by extension."""
        metadata = StageMetadata(
            execution_time=1.0,
            start_time=datetime.now(),
            input_files=["input.txt"],
            output_files=["output.md", "output.json"]
        )

        result = StageResult(
            stage_type=StageType.MARKDOWN_CONVERSION,
            status=StageStatus.COMPLETED,
            output_files=[Path("output.md"), Path("output.json"), Path("data.txt")],
            metadata=metadata
        )

        # Test finding existing extensions
        md_file = result.get_output_by_extension('.md')
        assert md_file == Path("output.md")

        json_file = result.get_output_by_extension('.json')
        assert json_file == Path("output.json")

        # Test case insensitive
        md_file_upper = result.get_output_by_extension('.MD')
        assert md_file_upper == Path("output.md")

        # Test non-existing extension
        xml_file = result.get_output_by_extension('.xml')
        assert xml_file is None

    def test_get_outputs_by_extension(self):
        """Test getting all output files by extension."""
        metadata = StageMetadata(
            execution_time=1.0,
            start_time=datetime.now(),
            input_files=["input.txt"],
            output_files=["file1.md", "file2.md", "output.json"]
        )

        result = StageResult(
            stage_type=StageType.MARKDOWN_CONVERSION,
            status=StageStatus.COMPLETED,
            output_files=[Path("file1.md"), Path("file2.md"), Path("output.json")],
            metadata=metadata
        )

        # Test finding multiple files
        md_files = result.get_outputs_by_extension('.md')
        assert len(md_files) == 2
        assert Path("file1.md") in md_files
        assert Path("file2.md") in md_files

        # Test finding single file
        json_files = result.get_outputs_by_extension('.json')
        assert len(json_files) == 1
        assert json_files[0] == Path("output.json")

        # Test non-existing extension
        xml_files = result.get_outputs_by_extension('.xml')
        assert len(xml_files) == 0


class TestStageContext:
    """Test StageContext model."""

    def test_create_context(self):
        """Test creating stage context."""
        context = StageContext(
            source_path=Path("input.txt"),
            output_dir=Path("output")
        )

        assert context.source_path == Path("input.txt")
        assert context.output_dir == Path("output")
        assert context.webhook_url is None
        assert context.config == {}
        assert context.stage_results == {}
        assert context.resume_from_existing is True
        assert context.cleanup_intermediate is False
        assert context.max_parallel_stages == 1
        assert context.intermediate_files == []

    def test_stage_config_methods(self):
        """Test stage configuration methods."""
        context = StageContext(
            source_path=Path("input.txt"),
            output_dir=Path("output")
        )

        # Test getting empty config
        config = context.get_stage_config(StageType.MARKDOWN_CONVERSION)
        assert config == {}

        # Test setting and getting config
        test_config = {"param1": "value1", "param2": 42}
        context.set_stage_config(StageType.MARKDOWN_CONVERSION, test_config)

        retrieved_config = context.get_stage_config(StageType.MARKDOWN_CONVERSION)
        assert retrieved_config == test_config

        # Test other stage still empty
        other_config = context.get_stage_config(StageType.CHUNKER)
        assert other_config == {}

    def test_stage_result_methods(self):
        """Test stage result management methods."""
        context = StageContext(
            source_path=Path("input.txt"),
            output_dir=Path("output")
        )

        metadata = StageMetadata(
            execution_time=1.0,
            start_time=datetime.now(),
            input_files=["input.txt"],
            output_files=["output.md"]
        )

        result = StageResult(
            stage_type=StageType.MARKDOWN_CONVERSION,
            status=StageStatus.COMPLETED,
            output_files=[Path("output.md")],
            metadata=metadata
        )

        # Test adding result
        context.add_stage_result(result)

        assert len(context.stage_results) == 1
        assert StageType.MARKDOWN_CONVERSION in context.stage_results
        assert context.intermediate_files == [Path("output.md")]

        # Test getting result
        retrieved_result = context.get_stage_result(StageType.MARKDOWN_CONVERSION)
        assert retrieved_result == result

        # Test non-existing result
        missing_result = context.get_stage_result(StageType.CHUNKER)
        assert missing_result is None

        # Test has completed
        assert context.has_stage_completed(StageType.MARKDOWN_CONVERSION) is True
        assert context.has_stage_completed(StageType.CHUNKER) is False
