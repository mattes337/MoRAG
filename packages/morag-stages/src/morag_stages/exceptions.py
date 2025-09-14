"""Exception classes for MoRAG Stages."""

from typing import Optional, List, Any


class StageError(Exception):
    """Base exception for all stage-related errors."""
    
    def __init__(self, message: str, stage_type: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message)
        self.stage_type = stage_type
        self.details = details or {}


class StageValidationError(StageError):
    """Raised when stage input validation fails."""
    
    def __init__(self, message: str, stage_type: Optional[str] = None, 
                 invalid_files: Optional[List[str]] = None, details: Optional[dict] = None):
        super().__init__(message, stage_type, details)
        self.invalid_files = invalid_files or []


class StageExecutionError(StageError):
    """Raised when stage execution fails."""
    
    def __init__(self, message: str, stage_type: Optional[str] = None, 
                 original_error: Optional[Exception] = None, details: Optional[dict] = None):
        super().__init__(message, stage_type, details)
        self.original_error = original_error


class StageDependencyError(StageError):
    """Raised when stage dependencies are not met."""
    
    def __init__(self, message: str, stage_type: Optional[str] = None,
                 missing_dependencies: Optional[List[str]] = None, details: Optional[dict] = None):
        super().__init__(message, stage_type, details)
        self.missing_dependencies = missing_dependencies or []


class StageConfigurationError(StageError):
    """Raised when stage configuration is invalid."""
    
    def __init__(self, message: str, stage_type: Optional[str] = None,
                 config_errors: Optional[List[str]] = None, details: Optional[dict] = None):
        super().__init__(message, stage_type, details)
        self.config_errors = config_errors or []


class StageFileError(StageError):
    """Raised when stage file operations fail."""
    
    def __init__(self, message: str, stage_type: Optional[str] = None,
                 file_path: Optional[str] = None, operation: Optional[str] = None, 
                 details: Optional[dict] = None):
        super().__init__(message, stage_type, details)
        self.file_path = file_path
        self.operation = operation


class StageTimeoutError(StageError):
    """Raised when stage execution times out."""
    
    def __init__(self, message: str, stage_type: Optional[str] = None,
                 timeout_seconds: Optional[float] = None, details: Optional[dict] = None):
        super().__init__(message, stage_type, details)
        self.timeout_seconds = timeout_seconds
