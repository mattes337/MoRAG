"""Base stage interface and types."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional
from pathlib import Path

# Import types only for type hints to avoid circular imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .result import StageResult
    from .context import StageContext


class StageType(Enum):
    """Canonical stage types for MoRAG processing."""
    MARKDOWN_CONVERSION = "markdown-conversion"
    MARKDOWN_OPTIMIZER = "markdown-optimizer"
    CHUNKER = "chunker"
    FACT_GENERATOR = "fact-generator"
    INGESTOR = "ingestor"


class StageStatus(Enum):
    """Stage execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class Stage(ABC):
    """Abstract base class for all processing stages."""
    
    def __init__(self, stage_type: StageType):
        """Initialize stage with its type.
        
        Args:
            stage_type: The type of this stage
        """
        self.stage_type = stage_type
        
    @abstractmethod
    async def execute(self,
                     input_files: List[Path],
                     context: "StageContext") -> "StageResult":
        """Execute the stage with given input files and context.
        
        Args:
            input_files: List of input file paths
            context: Stage execution context
            
        Returns:
            Stage execution result
            
        Raises:
            StageExecutionError: If stage execution fails
        """
        pass
        
    @abstractmethod
    def validate_inputs(self, input_files: List[Path]) -> bool:
        """Validate that input files are suitable for this stage.
        
        Args:
            input_files: List of input file paths to validate
            
        Returns:
            True if inputs are valid, False otherwise
        """
        pass
        
    @abstractmethod
    def get_dependencies(self) -> List[StageType]:
        """Return list of stages that must complete before this stage.
        
        Returns:
            List of required stage types
        """
        pass
        
    @abstractmethod
    def get_expected_outputs(self, input_files: List[Path], context: "StageContext") -> List[Path]:
        """Get expected output file paths for given inputs.
        
        Args:
            input_files: List of input file paths
            context: Stage execution context
            
        Returns:
            List of expected output file paths
        """
        pass
        
    def get_stage_name(self) -> str:
        """Get the canonical name of this stage.
        
        Returns:
            Stage name as string
        """
        return self.stage_type.value
        
    def is_optional(self) -> bool:
        """Check if this stage is optional in the pipeline.
        
        Returns:
            True if stage is optional, False if required
        """
        # Only markdown-optimizer is optional by default
        return self.stage_type == StageType.MARKDOWN_OPTIMIZER
        
    def get_config_schema(self) -> dict:
        """Get configuration schema for this stage.
        
        Returns:
            JSON schema for stage configuration
        """
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": True
        }
