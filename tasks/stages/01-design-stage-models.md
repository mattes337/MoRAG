# Task 1: Design Stage Models and Interfaces

## Overview
Design the foundational models, interfaces, and data structures for the stage-based processing system. This task completely replaces all existing processing interfaces and models.

## Objectives
- Define base `Stage` class and interface using canonical stage names
- Create `StageResult` and `StageContext` models
- Design file naming conventions and metadata structures
- Implement stage dependency management
- Create configuration models for each stage
- **REMOVE ALL LEGACY INTERFACES** - no backwards compatibility

## Deliverables

### 1. Base Stage Interface (Complete Replacement)
```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from enum import Enum

class StageType(Enum):
    MARKDOWN_CONVERSION = "markdown-conversion"
    MARKDOWN_OPTIMIZER = "markdown-optimizer"
    CHUNKER = "chunker"
    FACT_GENERATOR = "fact-generator"
    INGESTOR = "ingestor"

class StageStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class StageResult:
    def __init__(self,
                 stage_type: StageType,
                 status: StageStatus,
                 output_files: List[Path],
                 metadata: Dict[str, Any],
                 error_message: Optional[str] = None,
                 execution_time: Optional[float] = None):
        self.stage_type = stage_type
        self.status = status
        self.output_files = output_files
        self.metadata = metadata
        self.error_message = error_message
        self.execution_time = execution_time

class StageContext:
    def __init__(self,
                 source_path: Path,
                 output_dir: Path,
                 webhook_url: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        self.source_path = source_path
        self.output_dir = output_dir
        self.webhook_url = webhook_url
        self.config = config or {}
        self.stage_results: Dict[StageType, StageResult] = {}

class Stage(ABC):
    def __init__(self, stage_type: StageType):
        self.stage_type = stage_type

    @abstractmethod
    async def execute(self,
                     input_files: List[Path],
                     context: StageContext) -> StageResult:
        """Execute the stage with given input files and context."""
        pass

    @abstractmethod
    def validate_inputs(self, input_files: List[Path]) -> bool:
        """Validate that input files are suitable for this stage."""
        pass

    @abstractmethod
    def get_dependencies(self) -> List[StageType]:
        """Return list of stages that must complete before this stage."""
        pass
```

### 2. File Naming Convention (Complete Replacement)
```python
class FileNamingConvention:
    @staticmethod
    def get_stage_output_filename(source_path: Path, stage_type: StageType) -> str:
        base_name = source_path.stem

        extensions = {
            StageType.MARKDOWN_CONVERSION: ".md",
            StageType.MARKDOWN_OPTIMIZER: ".opt.md",
            StageType.CHUNKER: ".chunks.json",
            StageType.FACT_GENERATOR: ".facts.json",
            StageType.INGESTOR: ".ingestion.json"
        }

        return f"{base_name}{extensions[stage_type]}"

    @staticmethod
    def get_metadata_filename(source_path: Path, stage_type: StageType) -> str:
        base_name = source_path.stem
        return f"{base_name}.{stage_type.value}.meta.json"
```

### 3. Stage Configuration Models (Complete Replacement)
```python
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

class StageConfig(BaseModel):
    enabled: bool = True
    timeout_seconds: Optional[int] = None
    retry_count: int = 0
    config: Dict[str, Any] = {}

class MarkdownConversionConfig(StageConfig):
    # markdown-conversion specific config
    include_timestamps: bool = True
    transcription_model: str = "whisper-large"
    chunk_on_sentences: bool = True

class MarkdownOptimizerConfig(StageConfig):
    # markdown-optimizer specific config
    llm_model: str = "gemini-pro"
    optimization_prompt: Optional[str] = None
    preserve_timestamps: bool = True
    fix_transcription_errors: bool = True
    use_system_user_messages: bool = True  # Use system/user message pattern

class ChunkerConfig(StageConfig):
    # chunker specific config
    chunk_strategy: str = "semantic"
    chunk_size: int = 4000
    chunk_overlap: int = 200
    generate_summary: bool = True
    generate_contextual_summaries: bool = True  # Generate contextual summaries
    embedding_model: str = "text-embedding-004"
    contextual_embedding_window: int = 2  # Include surrounding chunks for context

class FactGeneratorConfig(StageConfig):
    # fact-generator specific config
    extract_entities: bool = True
    extract_relations: bool = True
    extract_keywords: bool = True
    domain: Optional[str] = None
    max_facts_per_chunk: Optional[int] = None

class IngestorConfig(StageConfig):
    # ingestor specific config
    databases: List[str] = ["qdrant"]
    replace_existing: bool = False
    collection_name: Optional[str] = None
    language: Optional[str] = None

class PipelineConfig(BaseModel):
    markdown_conversion: MarkdownConversionConfig = MarkdownConversionConfig()
    markdown_optimizer: MarkdownOptimizerConfig = MarkdownOptimizerConfig()
    chunker: ChunkerConfig = ChunkerConfig()
    fact_generator: FactGeneratorConfig = FactGeneratorConfig()
    ingestor: IngestorConfig = IngestorConfig()

    webhook_url: Optional[str] = None
    output_dir: str = "./output"
    temp_dir: str = "./temp"
    cleanup_temp_files: bool = True
```

### 4. Stage Dependency Manager (Complete Replacement)
```python
class StageDependencyManager:
    DEPENDENCIES = {
        StageType.MARKDOWN_CONVERSION: [],
        StageType.MARKDOWN_OPTIMIZER: [StageType.MARKDOWN_CONVERSION],
        StageType.CHUNKER: [StageType.MARKDOWN_CONVERSION],  # Can use optimized or original
        StageType.FACT_GENERATOR: [StageType.CHUNKER],
        StageType.INGESTOR: [StageType.CHUNKER, StageType.FACT_GENERATOR]
    }

    @classmethod
    def get_execution_order(cls, requested_stages: List[StageType]) -> List[StageType]:
        """Return stages in correct execution order based on dependencies."""
        all_required = set()

        def add_dependencies(stage: StageType):
            if stage not in all_required:
                all_required.add(stage)
                for dep in cls.DEPENDENCIES[stage]:
                    add_dependencies(dep)

        for stage in requested_stages:
            add_dependencies(stage)

        # Sort by stage number (enum value)
        return sorted(all_required, key=lambda x: x.value)

    @classmethod
    def validate_stage_chain(cls, stages: List[StageType]) -> bool:
        """Validate that all dependencies are satisfied."""
        completed = set()

        for stage in stages:
            for dep in cls.DEPENDENCIES[stage]:
                if dep not in completed:
                    return False
            completed.add(stage)

        return True
```

### 5. Webhook Integration
```python
import aiohttp
import json
from typing import Optional

class WebhookNotifier:
    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url

    async def notify_stage_completion(self,
                                    stage_result: StageResult,
                                    context: StageContext):
        """Send webhook notification when stage completes."""
        if not self.webhook_url:
            return

        payload = {
            "stage": stage_result.stage_type.value,
            "status": stage_result.status.value,
            "source_file": str(context.source_path),
            "output_files": [str(f) for f in stage_result.output_files],
            "metadata": stage_result.metadata,
            "execution_time": stage_result.execution_time,
            "error_message": stage_result.error_message
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status != 200:
                        logger.warning(f"Webhook notification failed: {response.status}")
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
```

## Implementation Steps

1. **Create base models package**: `packages/morag-stages/src/morag_stages/models/`
2. **Implement stage interfaces**: Define abstract base classes and protocols
3. **Create configuration system**: Implement Pydantic models for stage configs
4. **Add dependency management**: Implement stage ordering and validation
5. **Implement webhook system**: Add notification capabilities
6. **Add file management**: Implement naming conventions and file tracking
7. **Create stage registry**: System to register and discover available stages
8. **Add validation**: Input/output validation for each stage
9. **Implement logging**: Structured logging for stage execution
10. **Add metrics**: Performance and execution metrics collection

## Testing Requirements

- Unit tests for all model classes
- Integration tests for dependency management
- Webhook notification testing
- File naming convention validation
- Configuration validation tests
- Stage execution order tests

## Files to Create

- `packages/morag-stages/src/morag_stages/models/__init__.py`
- `packages/morag-stages/src/morag_stages/models/stage.py`
- `packages/morag-stages/src/morag_stages/models/config.py`
- `packages/morag-stages/src/morag_stages/models/result.py`
- `packages/morag-stages/src/morag_stages/models/context.py`
- `packages/morag-stages/src/morag_stages/dependency_manager.py`
- `packages/morag-stages/src/morag_stages/webhook.py`
- `packages/morag-stages/src/morag_stages/file_manager.py`
- `packages/morag-stages/tests/test_models.py`
- `packages/morag-stages/tests/test_dependency_manager.py`

## Success Criteria

- All stage models are well-defined and documented
- Dependency management works correctly for all stage combinations
- File naming conventions are consistent and predictable
- Webhook notifications work reliably
- Configuration system is flexible and extensible
- All tests pass and provide good coverage
