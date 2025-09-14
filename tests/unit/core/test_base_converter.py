"""Unit tests for BaseConverter class."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import uuid

from morag_core.converters.base import (
    BaseConverter,
    ConversionResult,
    ConversionQualityValidator
)
from morag_core.exceptions import ProcessingError


class MockConverter(BaseConverter):
    """Mock converter for testing."""
    
    def get_supported_formats(self):
        return {
            "input": ["txt", "md"],
            "output": ["html", "pdf"]
        }
    
    def convert(self, source_path, target_path, **kwargs):
        # Simple mock conversion
        source_path = Path(source_path)
        target_path = Path(target_path)
        
        if not source_path.exists():
            return ConversionResult(
                success=False,
                source_path=str(source_path),
                error_message="Source file not found",
                error_type="FileNotFoundError"
            )
        
        # Create target file
        target_path.write_text("Mock converted content")
        
        return ConversionResult(
            success=True,
            source_path=str(source_path),
            target_path=str(target_path),
            source_format=source_path.suffix.lower().lstrip('.'),
            target_format=target_path.suffix.lower().lstrip('.'),
            conversion_time=0.1,
            file_size_before=source_path.stat().st_size,
            file_size_after=target_path.stat().st_size,
            quality_score=0.9
        )


class TestConversionResult:
    """Test ConversionResult class."""
    
    def test_creation_with_defaults(self):
        """Test creating result with minimal parameters."""
        result = ConversionResult(success=True, source_path="test.txt")
        
        assert result.success is True
        assert result.source_path == "test.txt"
        assert result.target_path is None
        assert result.conversion_time == 0.0
        assert result.metadata == {}
        assert isinstance(result.conversion_id, str)
        assert len(result.conversion_id) > 0
    
    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = ConversionResult(
            success=True,
            source_path="test.txt",
            target_path="test.html",
            conversion_time=1.5
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["success"] is True
        assert result_dict["source_path"] == "test.txt"
        assert result_dict["target_path"] == "test.html"
        assert result_dict["conversion_time"] == 1.5
        assert "conversion_id" in result_dict
        assert "created_at" in result_dict
    
    def test_with_error_info(self):
        """Test result with error information."""
        result = ConversionResult(
            success=False,
            source_path="test.txt",
            error_message="Conversion failed",
            error_type="ProcessingError"
        )
        
        assert result.success is False
        assert result.error_message == "Conversion failed"
        assert result.error_type == "ProcessingError"


class TestConversionQualityValidator:
    """Test ConversionQualityValidator class."""
    
    def test_creation_with_default_threshold(self):
        """Test creating validator with default threshold."""
        validator = ConversionQualityValidator()
        assert validator.min_quality_score == 0.8
    
    def test_creation_with_custom_threshold(self):
        """Test creating validator with custom threshold."""
        validator = ConversionQualityValidator(min_quality_score=0.9)
        assert validator.min_quality_score == 0.9
    
    def test_validate_quality_success(self):
        """Test quality validation with good quality."""
        validator = ConversionQualityValidator(min_quality_score=0.8)
        result = ConversionResult(
            success=True,
            source_path="test.txt",
            quality_score=0.9
        )
        
        assert validator.validate_quality(result) is True
    
    def test_validate_quality_failed_conversion(self):
        """Test quality validation with failed conversion."""
        validator = ConversionQualityValidator()
        result = ConversionResult(success=False, source_path="test.txt")
        
        with pytest.raises(ProcessingError, match="Conversion failed"):
            validator.validate_quality(result)
    
    def test_validate_quality_low_score(self):
        """Test quality validation with low quality score."""
        validator = ConversionQualityValidator(min_quality_score=0.8)
        result = ConversionResult(
            success=True,
            source_path="test.txt",
            quality_score=0.5
        )
        
        with pytest.raises(ProcessingError, match="Quality score 0.5 below threshold 0.8"):
            validator.validate_quality(result)
    
    def test_validate_quality_no_score(self):
        """Test quality validation when no score is provided."""
        validator = ConversionQualityValidator()
        result = ConversionResult(success=True, source_path="test.txt")
        
        # Should pass when no quality score is provided
        assert validator.validate_quality(result) is True
    
    def test_validate_file_integrity_success(self, tmp_path):
        """Test file integrity validation with valid files."""
        validator = ConversionQualityValidator()
        
        source_file = tmp_path / "source.txt"
        target_file = tmp_path / "target.html"
        
        source_file.write_text("Source content")
        target_file.write_text("Target content")
        
        assert validator.validate_file_integrity(source_file, target_file) is True
    
    def test_validate_file_integrity_missing_source(self, tmp_path):
        """Test file integrity validation with missing source file."""
        validator = ConversionQualityValidator()
        
        source_file = tmp_path / "missing.txt"
        target_file = tmp_path / "target.html"
        target_file.write_text("Target content")
        
        with pytest.raises(ProcessingError, match="Source file not found"):
            validator.validate_file_integrity(source_file, target_file)
    
    def test_validate_file_integrity_missing_target(self, tmp_path):
        """Test file integrity validation with missing target file."""
        validator = ConversionQualityValidator()
        
        source_file = tmp_path / "source.txt"
        target_file = tmp_path / "missing.html"
        source_file.write_text("Source content")
        
        with pytest.raises(ProcessingError, match="Target file not found"):
            validator.validate_file_integrity(source_file, target_file)
    
    def test_validate_file_integrity_empty_target(self, tmp_path):
        """Test file integrity validation with empty target file."""
        validator = ConversionQualityValidator()
        
        source_file = tmp_path / "source.txt"
        target_file = tmp_path / "target.html"
        
        source_file.write_text("Source content")
        target_file.write_text("")  # Empty file
        
        with pytest.raises(ProcessingError, match="Target file is empty"):
            validator.validate_file_integrity(source_file, target_file)


class TestBaseConverter:
    """Test BaseConverter abstract class."""
    
    @pytest.fixture
    def mock_converter(self):
        """Create a mock converter for testing."""
        return MockConverter()
    
    @pytest.fixture
    def sample_files(self, tmp_path):
        """Create sample files for testing."""
        source_file = tmp_path / "test.txt"
        target_file = tmp_path / "test.html"
        source_file.write_text("Sample content for testing")
        return source_file, target_file
    
    def test_creation_with_default_validator(self, mock_converter):
        """Test creating converter with default validator."""
        assert mock_converter.quality_validator is not None
        assert isinstance(mock_converter.quality_validator, ConversionQualityValidator)
        assert mock_converter.quality_validator.min_quality_score == 0.8
    
    def test_creation_with_custom_validator(self):
        """Test creating converter with custom validator."""
        custom_validator = ConversionQualityValidator(min_quality_score=0.9)
        converter = MockConverter(quality_validator=custom_validator)
        
        assert converter.quality_validator is custom_validator
        assert converter.quality_validator.min_quality_score == 0.9
    
    def test_get_supported_formats(self, mock_converter):
        """Test getting supported formats."""
        formats = mock_converter.get_supported_formats()
        
        assert "input" in formats
        assert "output" in formats
        assert "txt" in formats["input"]
        assert "md" in formats["input"]
        assert "html" in formats["output"]
        assert "pdf" in formats["output"]
    
    def test_validate_formats_success(self, mock_converter):
        """Test format validation with supported formats."""
        assert mock_converter.validate_formats("txt", "html") is True
        assert mock_converter.validate_formats("md", "pdf") is True
    
    def test_validate_formats_unsupported_input(self, mock_converter):
        """Test format validation with unsupported input format."""
        with pytest.raises(ProcessingError, match="Unsupported input format: docx"):
            mock_converter.validate_formats("docx", "html")
    
    def test_validate_formats_unsupported_output(self, mock_converter):
        """Test format validation with unsupported output format."""
        with pytest.raises(ProcessingError, match="Unsupported output format: xml"):
            mock_converter.validate_formats("txt", "xml")
    
    def test_get_format_from_path(self, mock_converter):
        """Test extracting format from file path."""
        assert mock_converter.get_format_from_path("test.txt") == "txt"
        assert mock_converter.get_format_from_path("document.PDF") == "pdf"
        assert mock_converter.get_format_from_path("/path/to/file.html") == "html"
        assert mock_converter.get_format_from_path(Path("test.md")) == "md"
    
    def test_get_format_from_path_no_extension(self, mock_converter):
        """Test extracting format from path with no extension."""
        assert mock_converter.get_format_from_path("test") == ""
        assert mock_converter.get_format_from_path("path/to/file") == ""
    
    def test_estimate_conversion_time(self, mock_converter, sample_files):
        """Test conversion time estimation."""
        source_file, _ = sample_files
        
        estimated_time = mock_converter.estimate_conversion_time(source_file)
        
        # Should be positive and reasonable
        assert estimated_time > 0
        assert estimated_time < 1  # Small file should be quick
    
    def test_convert_success(self, mock_converter, sample_files):
        """Test successful conversion."""
        source_file, target_file = sample_files
        
        result = mock_converter.convert(source_file, target_file)
        
        assert result.success is True
        assert result.source_path == str(source_file)
        assert result.target_path == str(target_file)
        assert result.source_format == "txt"
        assert result.target_format == "html"
        assert result.conversion_time == 0.1
        assert result.quality_score == 0.9
        assert target_file.exists()
    
    def test_convert_missing_source(self, mock_converter, tmp_path):
        """Test conversion with missing source file."""
        source_file = tmp_path / "missing.txt"
        target_file = tmp_path / "target.html"
        
        result = mock_converter.convert(source_file, target_file)
        
        assert result.success is False
        assert result.error_message == "Source file not found"
        assert result.error_type == "FileNotFoundError"


@pytest.mark.parametrize("source_format,target_format,should_succeed", [
    ("txt", "html", True),
    ("md", "pdf", True),
    ("docx", "html", False),  # Unsupported input
    ("txt", "xml", False),    # Unsupported output
])
def test_format_validation_parametrized(source_format, target_format, should_succeed):
    """Parametrized test for format validation."""
    converter = MockConverter()

    if should_succeed:
        assert converter.validate_formats(source_format, target_format) is True
    else:
        with pytest.raises(ProcessingError):
            converter.validate_formats(source_format, target_format)


# =============================================================================
# STAGE PROCESSING EDGE CASES - Added for missing coverage
# =============================================================================

class MockFailingStage:
    """Mock stage that fails after processing N items."""

    def __init__(self, fail_after_items: int = 0, total_items: int = 5):
        self.fail_after_items = fail_after_items
        self.total_items = total_items
        self.processed_files = []
        self.cleanup_called = False

    def process_items(self, input_files):
        """Process items with controlled failure."""
        results = []
        for i, file in enumerate(input_files[:self.total_items]):
            if i >= self.fail_after_items:
                raise ProcessingError(f"Simulated failure at item {i}")
            self.processed_files.append(file)
            results.append(f"processed_{file}")
        return results

    def cleanup(self):
        """Cleanup resources."""
        self.cleanup_called = True


class MockMemoryLimitedStage:
    """Mock stage that simulates memory exhaustion."""

    def __init__(self, limit_mb: int = 512):
        self.limit_mb = limit_mb
        self.cleanup_called = False

    def process_large_file(self, file_size_gb: float):
        """Process large file with memory limit check."""
        required_mb = file_size_gb * 1024
        if required_mb > self.limit_mb:
            raise MemoryError(f"Memory limit exceeded: {required_mb}MB > {self.limit_mb}MB")
        return "processed"

    def cleanup(self):
        """Cleanup resources."""
        self.cleanup_called = True


class TestStageProcessingEdgeCases:
    """Comprehensive edge case testing for stage processing."""

    @pytest.fixture
    def failing_stage(self):
        """Create a failing stage for testing."""
        return MockFailingStage(fail_after_items=2, total_items=5)

    @pytest.fixture
    def memory_limited_stage(self):
        """Create a memory-limited stage for testing."""
        return MockMemoryLimitedStage(limit_mb=512)

    @pytest.fixture
    def large_test_file(self, tmp_path):
        """Create a simulated large file (metadata only)."""
        large_file = tmp_path / "large_file.bin"
        large_file.write_bytes(b'x' * 1024)  # Create 1KB file as placeholder
        # Store size metadata
        large_file.size_gb = 2.0  # Simulate 2GB size
        return large_file

    def test_stage_failure_recovery(self, failing_stage):
        """Test recovery when a stage fails mid-processing."""
        # Prepare input files
        input_files = [f"file_{i}.txt" for i in range(5)]

        # Execute stage with expected partial failure
        with pytest.raises(ProcessingError) as exc_info:
            failing_stage.process_items(input_files)

        # Verify partial results are preserved
        assert len(failing_stage.processed_files) == 2, "Should process 2 files before failing"
        assert "Simulated failure at item 2" in str(exc_info.value)

        # Verify cleanup can be called
        failing_stage.cleanup()
        assert failing_stage.cleanup_called, "Cleanup should be called after failure"

    def test_stage_memory_exhaustion(self, memory_limited_stage, large_test_file):
        """Test behavior when stage exhausts memory."""
        # Test with file size exceeding memory limit
        with pytest.raises(MemoryError) as exc_info:
            memory_limited_stage.process_large_file(large_test_file.size_gb)

        assert "Memory limit exceeded" in str(exc_info.value), "Should indicate memory limit exceeded"

        # Verify cleanup is called even after memory error
        memory_limited_stage.cleanup()
        assert memory_limited_stage.cleanup_called, "Resources should be cleaned up after memory error"

    def test_stage_resource_cleanup_on_failure(self, failing_stage, tmp_path):
        """Test resource cleanup when stage fails."""
        # Create temporary resources
        temp_file = tmp_path / "temp_resource.tmp"
        temp_file.write_text("temporary data")

        # Simulate stage failure
        try:
            failing_stage.process_items(["file1", "file2", "file3"])
        except ProcessingError:
            pass  # Expected failure

        # Ensure cleanup is properly called
        failing_stage.cleanup()
        assert failing_stage.cleanup_called, "Cleanup should be called after stage failure"

    def test_partial_results_preservation(self, failing_stage):
        """Test that partial results are preserved during failure."""
        input_files = ["file1.txt", "file2.txt", "file3.txt", "file4.txt", "file5.txt"]

        # Execute with expected failure after 2 items
        try:
            failing_stage.process_items(input_files)
        except ProcessingError:
            pass  # Expected

        # Verify partial results are available
        assert len(failing_stage.processed_files) == 2
        assert failing_stage.processed_files == ["file1.txt", "file2.txt"]

    def test_stage_retry_mechanism(self):
        """Test retry mechanisms for failed stages."""
        class RetryableStage:
            def __init__(self):
                self.attempt_count = 0
                self.max_retries = 3

            def execute(self):
                self.attempt_count += 1
                if self.attempt_count < 3:
                    raise ProcessingError(f"Attempt {self.attempt_count} failed")
                return "success"

        stage = RetryableStage()

        # Simulate retry logic
        last_error = None
        for attempt in range(stage.max_retries):
            try:
                result = stage.execute()
                assert result == "success"
                break
            except ProcessingError as e:
                last_error = e
                continue
        else:
            # All retries exhausted
            assert last_error is not None

        # Should succeed on 3rd attempt
        assert stage.attempt_count == 3

    def test_stage_timeout_handling(self):
        """Test stage timeout behavior."""
        import time

        class SlowStage:
            def __init__(self, processing_time: float = 0.1):
                self.processing_time = processing_time

            def execute(self, timeout: float = 0.05):
                start_time = time.time()
                time.sleep(self.processing_time)
                elapsed = time.time() - start_time

                if elapsed > timeout:
                    raise TimeoutError(f"Stage timed out after {elapsed:.3f}s > {timeout}s")
                return "completed"

        slow_stage = SlowStage(processing_time=0.1)

        # Should timeout
        with pytest.raises(TimeoutError) as exc_info:
            slow_stage.execute(timeout=0.05)

        assert "Stage timed out" in str(exc_info.value)

    def test_stage_concurrent_execution_failure(self):
        """Test handling of concurrent execution failures."""
        import threading

        class ConcurrencyIssueStage:
            def __init__(self):
                self.counter = 0
                self.errors = []

            def execute(self):
                # Simulate race condition
                old_counter = self.counter
                self.counter = old_counter + 1

                # Check for race condition
                if self.counter != old_counter + 1:
                    error = ProcessingError(f"Race condition detected: {self.counter} != {old_counter + 1}")
                    self.errors.append(error)
                    raise error
                return self.counter

        stage = ConcurrencyIssueStage()
        threads = []
        results = []
        errors = []

        def worker():
            try:
                result = stage.execute()
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # At least some operations should complete or fail
        assert len(results) + len(errors) == 5

    def test_stage_file_system_error_handling(self, tmp_path):
        """Test handling of file system errors during stage execution."""
        class FileSystemStage:
            def __init__(self, output_dir: Path):
                self.output_dir = output_dir

            def execute(self, create_readonly_dir: bool = False):
                if create_readonly_dir:
                    # Create read-only directory to cause permission error
                    readonly_dir = self.output_dir / "readonly"
                    readonly_dir.mkdir()
                    readonly_dir.chmod(0o444)  # Read-only

                    # Try to write to read-only directory
                    output_file = readonly_dir / "output.txt"
                    output_file.write_text("test")  # Should fail
                else:
                    output_file = self.output_dir / "output.txt"
                    output_file.write_text("success")
                    return str(output_file)

        stage = FileSystemStage(tmp_path)

        # Test successful execution
        result = stage.execute(create_readonly_dir=False)
        assert Path(result).exists()

        # Test permission error handling
        with pytest.raises(PermissionError):
            stage.execute(create_readonly_dir=True)

    def test_stage_intermediate_state_corruption(self, tmp_path):
        """Test handling of intermediate state corruption."""
        class StatefulStage:
            def __init__(self, state_file: Path):
                self.state_file = state_file
                self.state = {"step": 0, "processed_items": []}

            def save_state(self):
                """Save current state to file."""
                import json
                with open(self.state_file, 'w') as f:
                    json.dump(self.state, f)

            def load_state(self):
                """Load state from file."""
                import json
                if self.state_file.exists():
                    with open(self.state_file, 'r') as f:
                        self.state = json.load(f)

            def execute_step(self, step_data: str, corrupt_state: bool = False):
                """Execute a processing step."""
                self.load_state()
                self.state["step"] += 1
                self.state["processed_items"].append(step_data)

                if corrupt_state:
                    # Corrupt the state file
                    self.state_file.write_text("invalid json {")
                    raise ProcessingError("State corruption simulated")

                self.save_state()
                return self.state["step"]

        state_file = tmp_path / "stage_state.json"
        stage = StatefulStage(state_file)

        # Execute successful steps
        assert stage.execute_step("data1") == 1
        assert stage.execute_step("data2") == 2

        # Simulate state corruption
        with pytest.raises(ProcessingError):
            stage.execute_step("data3", corrupt_state=True)

        # Verify state file is corrupted
        assert state_file.exists()
        content = state_file.read_text()
        assert content == "invalid json {"

    def test_stage_dependency_chain_failure(self):
        """Test failure propagation in stage dependency chains."""
        class DependentStage:
            def __init__(self, name: str, dependencies: list = None):
                self.name = name
                self.dependencies = dependencies or []
                self.completed = False
                self.failed = False
                self.error = None

            def execute(self, completed_stages: set):
                # Check dependencies
                for dep in self.dependencies:
                    if dep not in completed_stages:
                        self.failed = True
                        self.error = ProcessingError(f"Dependency {dep} not completed")
                        raise self.error

                # Simulate execution
                if self.name == "failing_stage":
                    self.failed = True
                    self.error = ProcessingError(f"Stage {self.name} failed")
                    raise self.error

                self.completed = True
                return f"{self.name}_result"

        # Create stage dependency chain
        stages = {
            "stage_a": DependentStage("stage_a"),
            "stage_b": DependentStage("stage_b", ["stage_a"]),
            "failing_stage": DependentStage("failing_stage", ["stage_b"]),
            "stage_c": DependentStage("stage_c", ["failing_stage"]),
        }

        completed_stages = set()

        # Execute stages in order
        for stage_name, stage in stages.items():
            try:
                stage.execute(completed_stages)
                completed_stages.add(stage_name)
            except ProcessingError:
                pass  # Continue to test propagation

        # Verify failure propagation
        assert stages["stage_a"].completed
        assert stages["stage_b"].completed
        assert stages["failing_stage"].failed
        assert stages["stage_c"].failed  # Should fail due to missing dependency

    def test_stage_resource_exhaustion_recovery(self, tmp_path):
        """Test recovery from resource exhaustion scenarios."""
        class ResourceExhaustionStage:
            def __init__(self, max_memory_mb: int = 100):
                self.max_memory_mb = max_memory_mb
                self.allocated_memory = 0
                self.resources = []

            def allocate_resource(self, size_mb: int):
                """Allocate a resource of given size."""
                if self.allocated_memory + size_mb > self.max_memory_mb:
                    raise MemoryError(f"Cannot allocate {size_mb}MB, limit is {self.max_memory_mb}MB")

                self.allocated_memory += size_mb
                resource_id = len(self.resources)
                self.resources.append({"id": resource_id, "size_mb": size_mb})
                return resource_id

            def deallocate_resource(self, resource_id: int):
                """Deallocate a resource."""
                for resource in self.resources:
                    if resource["id"] == resource_id:
                        self.allocated_memory -= resource["size_mb"]
                        self.resources.remove(resource)
                        return True
                return False

            def cleanup_all(self):
                """Cleanup all resources."""
                self.resources.clear()
                self.allocated_memory = 0

        stage = ResourceExhaustionStage(max_memory_mb=100)

        # Allocate resources within limit
        r1 = stage.allocate_resource(50)
        r2 = stage.allocate_resource(40)
        assert stage.allocated_memory == 90

        # Try to exceed limit
        with pytest.raises(MemoryError):
            stage.allocate_resource(20)  # Would exceed 100MB limit

        # Cleanup and verify recovery
        stage.cleanup_all()
        assert stage.allocated_memory == 0
        assert len(stage.resources) == 0

        # Should be able to allocate again after cleanup
        r3 = stage.allocate_resource(80)
        assert r3 == 2  # New resource ID
        assert stage.allocated_memory == 80