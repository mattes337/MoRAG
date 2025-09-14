"""Stage result models."""

from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field

from .stage import StageType, StageStatus


class StageMetadata(BaseModel):
    """Metadata for stage execution."""
    
    execution_time: float = Field(description="Execution time in seconds")
    start_time: datetime = Field(description="Stage start time")
    end_time: Optional[datetime] = Field(default=None, description="Stage end time")
    input_files: List[str] = Field(description="Input file paths")
    output_files: List[str] = Field(description="Output file paths")
    config_used: Dict[str, Any] = Field(default_factory=dict, description="Configuration used")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Stage-specific metrics")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Path: lambda v: str(v)
        }


class StageResult(BaseModel):
    """Result of stage execution."""
    
    stage_type: StageType = Field(description="Type of stage that was executed")
    status: StageStatus = Field(description="Execution status")
    output_files: List[Path] = Field(description="Generated output files")
    metadata: StageMetadata = Field(description="Execution metadata")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    
    # Additional data that stages can include
    data: Dict[str, Any] = Field(default_factory=dict, description="Stage-specific output data")
    
    class Config:
        json_encoders = {
            Path: lambda v: str(v),
            StageType: lambda v: v.value,
            StageStatus: lambda v: v.value
        }
    
    @property
    def success(self) -> bool:
        """Check if stage execution was successful."""
        return self.status == StageStatus.COMPLETED
    
    @property
    def failed(self) -> bool:
        """Check if stage execution failed."""
        return self.status == StageStatus.FAILED
    
    @property
    def skipped(self) -> bool:
        """Check if stage execution was skipped."""
        return self.status == StageStatus.SKIPPED
    
    def get_output_by_extension(self, extension: str) -> Optional[Path]:
        """Get first output file with specified extension.
        
        Args:
            extension: File extension to search for (e.g., '.md', '.json')
            
        Returns:
            First matching file path or None
        """
        for file_path in self.output_files:
            if file_path.suffix.lower() == extension.lower():
                return file_path
        return None
    
    def get_outputs_by_extension(self, extension: str) -> List[Path]:
        """Get all output files with specified extension.
        
        Args:
            extension: File extension to search for (e.g., '.md', '.json')
            
        Returns:
            List of matching file paths
        """
        return [
            file_path for file_path in self.output_files
            if file_path.suffix.lower() == extension.lower()
        ]
