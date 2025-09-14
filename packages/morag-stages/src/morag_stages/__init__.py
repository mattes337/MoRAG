"""MoRAG Stages - Stage-based processing system for MoRAG.

MoRAG Stages provides a modular, resumable pipeline architecture for processing
various content types through standardized stages. This package is the foundation
of MoRAG v2.0's stage-based processing approach.

## Core Concepts

### Stage-Based Processing
The system breaks content processing into discrete, resumable stages:

1. **`markdown-conversion`** - Convert input files to unified markdown format
2. **`markdown-optimizer`** - LLM-based text improvement and optimization
3. **`chunker`** - Create summary, chunks, and contextual embeddings
4. **`fact-generator`** - Extract facts, entities, relations, and keywords
5. **`ingestor`** - Database ingestion and storage

### Resume Capability
Each stage produces standardized output files that enable automatic resume:
- `filename.md` (markdown-conversion)
- `filename.opt.md` (markdown-optimizer)
- `filename.chunks.json` (chunker)
- `filename.facts.json` (fact-generator)
- `filename.ingestion.json` (ingestor)

## Usage Examples

### Basic Stage Execution
```python
from morag_stages import StageManager

# Initialize stage manager
manager = StageManager()

# Execute single stage
result = await manager.execute_stage(
    stage_name="markdown-conversion",
    input_path="document.pdf",
    output_dir="./output"
)

print(f"Stage completed: {result.status}")
print(f"Output file: {result.output_path}")
```

### Stage Chain Execution
```python
from morag_stages import StageManager

# Execute multiple stages in sequence
manager = StageManager()

stages = ["markdown-conversion", "chunker", "fact-generator"]
results = await manager.execute_stage_chain(
    stages=stages,
    input_path="document.pdf",
    output_dir="./output"
)

for stage, result in results.items():
    print(f"{stage}: {result.status}")
```

### Custom Stage Registration
```python
from morag_stages import register_stage, Stage, StageContext, StageResult

class CustomProcessingStage(Stage):
    stage_name = "custom-processing"
    input_types = [".md"]
    output_extension = ".custom.json"

    async def execute(self, context: StageContext) -> StageResult:
        # Custom processing logic
        processed_content = self.process_content(context.input_content)

        return StageResult(
            status="completed",
            output_content=processed_content,
            metadata={"processing_time": "10s"}
        )

# Register custom stage
register_stage(CustomProcessingStage)
```

### CLI Usage
The package provides CLI access through `morag-stages.py`:

```bash
# Execute single stage
python cli/morag-stages.py stage markdown-conversion input.pdf

# Execute stage chain
python cli/morag-stages.py stages "markdown-conversion,chunker,fact-generator" input.pdf

# Full pipeline with optimization
python cli/morag-stages.py process input.pdf --optimize --output-dir ./output

# List available stages
python cli/morag-stages.py list
```

## Available Stages

### MarkdownConversionStage
Converts various input formats to unified markdown:
- Supports: PDF, Word, Excel, PowerPoint, images, audio, video
- Output: `filename.md`
- Features: OCR, transcription, universal format support

### MarkdownOptimizerStage (Optional)
LLM-powered text improvement:
- Input: Markdown files
- Output: `filename.opt.md`
- Features: Clarity improvement, structure optimization, error correction

### ChunkerStage
Semantic content segmentation:
- Input: Markdown files
- Output: `filename.chunks.json`
- Features: Summary generation, intelligent chunking, contextual embeddings

### FactGeneratorStage
Structured knowledge extraction:
- Input: Chunked content
- Output: `filename.facts.json`
- Features: Entity extraction, relationship detection, fact scoring

### IngestorStage
Database storage and indexing:
- Input: Facts and chunks
- Output: `filename.ingestion.json`
- Features: Neo4j storage, Qdrant indexing, deduplication

## Configuration

### Environment Variables
```bash
# LLM Configuration
MORAG_GEMINI_MODEL=gemini-1.5-pro
MORAG_FACT_EXTRACTION_AGENT_MODEL=gemini-2.0-flash

# Processing Configuration
MORAG_CHUNK_SIZE=2000
MORAG_BATCH_SIZE=50
MORAG_MAX_WORKERS=4

# Stage-specific Configuration
MORAG_OPTIMIZE_MARKDOWN=true
MORAG_EXTRACT_ENTITIES=true
MORAG_GENERATE_SUMMARIES=true
```

### Programmatic Configuration
```python
from morag_stages import StageManager, StageContext

# Configure stage context
context = StageContext(
    input_path="document.pdf",
    output_dir="./output",
    config={
        "chunk_size": 1500,
        "llm_model": "gemini-1.5-flash",
        "extract_entities": True
    }
)

manager = StageManager()
result = await manager.execute_stage("chunker", context)
```

## Stage Models

### StageResult
Standard result format for all stages:
```python
@dataclass
class StageResult:
    status: StageStatus  # completed, failed, skipped
    output_path: Optional[Path]
    metadata: Dict[str, Any]
    error: Optional[str]
    execution_time: float
```

### StageContext
Execution context passed between stages:
```python
@dataclass
class StageContext:
    input_path: Path
    output_dir: Path
    config: Dict[str, Any]
    metadata: Dict[str, Any]
```

## Error Handling

The package provides comprehensive error handling:
- `StageError` - Base stage error
- `StageValidationError` - Input validation failures
- `StageExecutionError` - Processing failures
- `StageDependencyError` - Missing dependencies

```python
from morag_stages import StageError

try:
    result = await manager.execute_stage("chunker", context)
except StageValidationError as e:
    print(f"Invalid input: {e}")
except StageExecutionError as e:
    print(f"Processing failed: {e}")
```

## Installation

```bash
pip install morag-stages
```

Or as part of the full MoRAG system:

```bash
pip install packages/morag/
```

## Version

Current version: {__version__}
"""

# Load environment variables from .env file early
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    # Look for .env file in current directory and parent directories
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        # Try parent directories up to 3 levels
        for parent in list(Path.cwd().parents)[:3]:
            env_path = parent / ".env"
            if env_path.exists():
                break
    if env_path.exists():
        load_dotenv(env_path)
        # Only print in debug mode to avoid spam
        if os.getenv('MORAG_DEBUG', '').lower() in ('true', '1', 'yes'):
            print(f"[DEBUG] Loaded environment variables from: {env_path}")
except ImportError:
    # python-dotenv not available, continue without .env loading
    pass

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

from .error_handling import (
    stage_error_handler,
    validation_error_handler,
    standalone_validation_handler,
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

    # Error handling decorators
    "stage_error_handler",
    "validation_error_handler",
    "standalone_validation_handler",
]
