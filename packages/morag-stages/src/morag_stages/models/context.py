"""Stage context models."""

from typing import Any, Dict, Optional, List, Union
from pathlib import Path
from pydantic import BaseModel, Field, field_validator

from .result import StageResult
from .stage import StageType

# Import URLPath if available
try:
    from morag.utils.url_path import URLPath
    URL_PATH_AVAILABLE = True
except ImportError:
    URL_PATH_AVAILABLE = False
    # Create a dummy URLPath for type hints
    class URLPath:  # type: ignore
        pass


class StageContext(BaseModel):
    """Context for stage execution containing configuration and state."""

    model_config = {
        "arbitrary_types_allowed": True,
        "json_encoders": {
            Path: lambda v: str(v),
            StageType: lambda v: v.value
        }
    }

    source_path: Union[Path, URLPath] = Field(description="Original source file path or URL")
    output_dir: Path = Field(description="Output directory for generated files")
    webhook_url: Optional[str] = Field(default=None, description="Webhook URL for notifications")
    config: Dict[str, Any] = Field(default_factory=dict, description="Stage configurations")
    
    # Runtime state
    stage_results: Dict[StageType, StageResult] = Field(
        default_factory=dict, 
        description="Results from completed stages"
    )
    
    # Global settings
    resume_from_existing: bool = Field(
        default=True,
        description="Skip stages if output files already exist"
    )

    @field_validator('source_path', mode='before')
    @classmethod
    def validate_source_path(cls, v):
        """Validate source_path to accept both Path and URLPath objects."""
        if URL_PATH_AVAILABLE and hasattr(v, 'url_str'):
            # URLPath object - return as is
            return v
        elif isinstance(v, (str, Path)):
            # String or Path - convert to Path
            return Path(v) if not isinstance(v, Path) else v
        else:
            # Unknown type - let Pydantic handle the error
            return v
    cleanup_intermediate: bool = Field(
        default=False, 
        description="Clean up intermediate files after pipeline completion"
    )
    max_parallel_stages: int = Field(
        default=1, 
        description="Maximum number of stages to run in parallel"
    )
    
    # File tracking
    intermediate_files: List[Path] = Field(
        default_factory=list,
        description="List of intermediate files created during processing"
    )
    

    
    def get_stage_config(self, stage_type: StageType) -> Dict[str, Any]:
        """Get configuration for a specific stage.
        
        Args:
            stage_type: Stage type to get config for
            
        Returns:
            Configuration dictionary for the stage
        """
        config = self.config.get(stage_type.value, {})
        return config if isinstance(config, dict) else {}
    
    def set_stage_config(self, stage_type: StageType, config: Dict[str, Any]) -> None:
        """Set configuration for a specific stage.
        
        Args:
            stage_type: Stage type to set config for
            config: Configuration dictionary
        """
        self.config[stage_type.value] = config
    
    def add_stage_result(self, result: StageResult) -> None:
        """Add a stage result to the context.
        
        Args:
            result: Stage result to add
        """
        self.stage_results[result.stage_type] = result
        
        # Track intermediate files
        for output_file in result.output_files:
            if output_file not in self.intermediate_files:
                self.intermediate_files.append(output_file)
    
    def get_stage_result(self, stage_type: StageType) -> Optional[StageResult]:
        """Get result for a specific stage.
        
        Args:
            stage_type: Stage type to get result for
            
        Returns:
            Stage result if available, None otherwise
        """
        return self.stage_results.get(stage_type)
    
    def has_stage_completed(self, stage_type: StageType) -> bool:
        """Check if a stage has completed successfully.
        
        Args:
            stage_type: Stage type to check
            
        Returns:
            True if stage completed successfully, False otherwise
        """
        result = self.get_stage_result(stage_type)
        return result is not None and result.success
    
    def get_latest_output_files(self, extension: Optional[str] = None) -> List[Path]:
        """Get output files from the most recently completed stage.
        
        Args:
            extension: Optional file extension filter
            
        Returns:
            List of output files from latest stage
        """
        if not self.stage_results:
            return []
        
        # Get the most recent successful result
        latest_result = None
        for result in self.stage_results.values():
            if (result.success and result.metadata.end_time is not None and
                (latest_result is None or
                 (latest_result.metadata.end_time is not None and
                  result.metadata.end_time > latest_result.metadata.end_time))):
                latest_result = result
        
        if latest_result is None:
            return []
        
        if extension:
            return latest_result.get_outputs_by_extension(extension)
        else:
            return latest_result.output_files
    
    def get_output_file_for_stage(self, stage_type: StageType, extension: str) -> Optional[Path]:
        """Get output file with specific extension from a stage.
        
        Args:
            stage_type: Stage type to get output from
            extension: File extension to search for
            
        Returns:
            First matching output file or None
        """
        result = self.get_stage_result(stage_type)
        if result is None:
            return None
        return result.get_output_by_extension(extension)
