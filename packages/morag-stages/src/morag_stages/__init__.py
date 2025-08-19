"""MoRAG Stages - Stage-based processing system for MoRAG."""

from .models import (
    StageType,
    StageStatus,
    StageResult,
    StageContext,
    Stage,
)

from .manager import StageManager
from .registry import StageRegistry, register_stage, get_global_registry
from .exceptions import (
    StageError,
    StageValidationError,
    StageExecutionError,
    StageDependencyError,
)

# Import and register all stage implementations
from .stages import (
    MarkdownConversionStage,
    MarkdownOptimizerStage,
    ChunkerStage,
    FactGeneratorStage,
    IngestorStage,
)

# Auto-register all stages
def _register_default_stages():
    """Register all default stage implementations."""
    try:
        register_stage(MarkdownConversionStage)
        register_stage(MarkdownOptimizerStage)
        register_stage(ChunkerStage)
        register_stage(FactGeneratorStage)
        register_stage(IngestorStage)
    except Exception as e:
        import structlog
        logger = structlog.get_logger(__name__)
        logger.warning("Failed to register some stages", error=str(e))

# Register stages on import
_register_default_stages()

__version__ = "0.1.0"

__all__ = [
    # Core models
    "StageType",
    "StageStatus",
    "StageResult",
    "StageContext",
    "Stage",

    # Management
    "StageManager",
    "StageRegistry",
    "register_stage",
    "get_global_registry",

    # Stage implementations
    "MarkdownConversionStage",
    "MarkdownOptimizerStage",
    "ChunkerStage",
    "FactGeneratorStage",
    "IngestorStage",

    # Exceptions
    "StageError",
    "StageValidationError",
    "StageExecutionError",
    "StageDependencyError",
]
