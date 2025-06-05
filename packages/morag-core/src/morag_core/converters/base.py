"""Base converter classes for MoRAG."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import uuid

from ..exceptions import ProcessingError


@dataclass
class ConversionResult:
    """Result of a conversion operation."""
    
    # Basic result info
    success: bool
    source_path: str
    target_path: Optional[str] = None
    
    # Conversion details
    source_format: Optional[str] = None
    target_format: Optional[str] = None
    conversion_time: float = 0.0
    file_size_before: Optional[int] = None
    file_size_after: Optional[int] = None
    
    # Quality metrics
    quality_score: Optional[float] = None
    compression_ratio: Optional[float] = None
    
    # Error information
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Processing details
    conversion_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "success": self.success,
            "source_path": self.source_path,
            "target_path": self.target_path,
            "source_format": self.source_format,
            "target_format": self.target_format,
            "conversion_time": self.conversion_time,
            "file_size_before": self.file_size_before,
            "file_size_after": self.file_size_after,
            "quality_score": self.quality_score,
            "compression_ratio": self.compression_ratio,
            "error_message": self.error_message,
            "error_type": self.error_type,
            "metadata": self.metadata,
            "conversion_id": self.conversion_id,
            "created_at": self.created_at.isoformat(),
        }


class ConversionQualityValidator:
    """Validator for conversion quality."""
    
    def __init__(self, min_quality_score: float = 0.8):
        """Initialize validator.
        
        Args:
            min_quality_score: Minimum acceptable quality score
        """
        self.min_quality_score = min_quality_score
    
    def validate_quality(self, result: ConversionResult) -> bool:
        """Validate conversion quality.
        
        Args:
            result: Conversion result to validate
            
        Returns:
            True if quality is acceptable
            
        Raises:
            ProcessingError: If quality is below threshold
        """
        if not result.success:
            raise ProcessingError("Conversion failed")
        
        if result.quality_score is not None:
            if result.quality_score < self.min_quality_score:
                raise ProcessingError(
                    f"Quality score {result.quality_score} below threshold {self.min_quality_score}"
                )
        
        # Additional quality checks can be added here
        return True
    
    def validate_file_integrity(self, source_path: Union[str, Path], target_path: Union[str, Path]) -> bool:
        """Validate file integrity after conversion.
        
        Args:
            source_path: Source file path
            target_path: Target file path
            
        Returns:
            True if files are valid
            
        Raises:
            ProcessingError: If validation fails
        """
        source_path = Path(source_path)
        target_path = Path(target_path)
        
        # Check if files exist
        if not source_path.exists():
            raise ProcessingError(f"Source file not found: {source_path}")
        
        if not target_path.exists():
            raise ProcessingError(f"Target file not found: {target_path}")
        
        # Check if target file is not empty
        if target_path.stat().st_size == 0:
            raise ProcessingError(f"Target file is empty: {target_path}")
        
        return True


class BaseConverter(ABC):
    """Base class for converters."""
    
    def __init__(self, quality_validator: Optional[ConversionQualityValidator] = None):
        """Initialize converter.
        
        Args:
            quality_validator: Quality validator instance
        """
        self.quality_validator = quality_validator or ConversionQualityValidator()
    
    @abstractmethod
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Get supported input and output formats.
        
        Returns:
            Dictionary with 'input' and 'output' format lists
        """
        pass
    
    @abstractmethod
    def convert(
        self,
        source_path: Union[str, Path],
        target_path: Union[str, Path],
        **kwargs
    ) -> ConversionResult:
        """Convert file from source to target format.
        
        Args:
            source_path: Source file path
            target_path: Target file path
            **kwargs: Additional conversion options
            
        Returns:
            Conversion result
            
        Raises:
            ProcessingError: If conversion fails
        """
        pass
    
    def validate_formats(self, source_format: str, target_format: str) -> bool:
        """Validate that conversion between formats is supported.
        
        Args:
            source_format: Source format
            target_format: Target format
            
        Returns:
            True if conversion is supported
            
        Raises:
            ProcessingError: If conversion is not supported
        """
        supported = self.get_supported_formats()
        
        if source_format not in supported.get("input", []):
            raise ProcessingError(f"Unsupported input format: {source_format}")
        
        if target_format not in supported.get("output", []):
            raise ProcessingError(f"Unsupported output format: {target_format}")
        
        return True
    
    def get_format_from_path(self, file_path: Union[str, Path]) -> str:
        """Get format from file path extension.
        
        Args:
            file_path: File path
            
        Returns:
            Format string
        """
        return Path(file_path).suffix.lower().lstrip('.')
    
    def estimate_conversion_time(self, source_path: Union[str, Path]) -> float:
        """Estimate conversion time based on file size.
        
        Args:
            source_path: Source file path
            
        Returns:
            Estimated time in seconds
        """
        # Basic estimation based on file size
        # This can be overridden in subclasses for more accurate estimates
        file_size = Path(source_path).stat().st_size
        # Assume 1MB per second processing speed
        return file_size / (1024 * 1024)
