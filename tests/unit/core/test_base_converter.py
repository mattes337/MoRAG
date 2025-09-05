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