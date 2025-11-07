"""Main fact generation stage orchestration."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
from morag_core.config import FactGeneratorConfig
from morag_core.interfaces import IServiceCoordinator

from ..error_handling import stage_error_handler, validation_error_handler
from ..exceptions import StageExecutionError, StageValidationError
from ..models import (
    Stage,
    StageContext,
    StageMetadata,
    StageResult,
    StageStatus,
    StageType,
)

# Import sanitization function
try:
    from morag_core.utils.validation import ValidationError, sanitize_filepath
except ImportError:

    class ValidationError(Exception):
        """Fallback ValidationError class."""

        pass

    def sanitize_filepath(filepath, base_dir=None):
        """Fallback sanitization function with enhanced security."""
        import re
        from pathlib import Path

        if not filepath:
            raise ValidationError("Empty file path provided")

        path = Path(filepath)
        path_str = str(path)

        # Check for null bytes
        if "\x00" in path_str:
            raise ValidationError(f"Null byte detected in path: {filepath}")

        # Check for dangerous patterns
        dangerous_patterns = [
            r"[;&|`$()]",  # Shell metacharacters
            r"\$\(",  # Command substitution
            r"`.*`",  # Backtick command substitution
            r"\.\./",  # Directory traversal
            r"\.\.\\",  # Windows directory traversal
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, path_str):
                raise ValidationError(
                    f"Dangerous characters or patterns detected in path: {filepath}"
                )

        # Resolve and validate path
        try:
            resolved = path.resolve()
        except (OSError, RuntimeError) as e:
            raise ValidationError(f"Failed to resolve path {filepath}: {str(e)}")

        # Basic path traversal protection
        if base_dir is None:
            base_dir = Path.cwd().resolve()
        else:
            base_dir = Path(base_dir).resolve()

        try:
            resolved.relative_to(base_dir)
        except ValueError:
            raise ValidationError(
                f"Path traversal detected - path outside base directory: {filepath}"
            )

        # Additional filename validation
        filename = resolved.name
        if filename and filename.startswith(".."):
            raise ValidationError(f"Filename cannot start with double dots: {filename}")

        return resolved


from .fact_extraction_engine import FactExtractionEngine

logger = structlog.get_logger(__name__)


class FactGeneratorStage(Stage):
    """Stage that extracts facts, entities, relations, and keywords from chunks."""

    def __init__(
        self,
        stage_type: StageType = StageType.FACT_GENERATOR,
        coordinator: Optional[IServiceCoordinator] = None,
    ):
        """Initialize fact generator stage.

        Args:
            stage_type: Type of stage
            coordinator: Service coordinator implementing IServiceCoordinator (None for backward compatibility)
        """
        super().__init__(stage_type)

        # Handle backward compatibility
        if coordinator is None:
            # Create a basic service coordinator for backward compatibility
            self.coordinator = None
            logger.warning(
                "FactGeneratorStage initialized without coordinator - will use fallback services"
            )
        else:
            self.coordinator = coordinator

        # Default configuration
        self.config = {
            "chunk_size_threshold": 100,
            "max_chunks_per_batch": 10,
            "enable_entity_normalization": True,
            "enable_fact_deduplication": True,
            "enable_relation_deduplication": True,
            "enable_keyword_extraction": True,
            "enable_quality_scoring": True,
            "min_fact_confidence": 0.3,
            "min_relation_confidence": 0.2,
            "enable_batch_processing": True,
            "fact_extraction_timeout": 300,
            "use_llm_fallback": True,
            "enable_extraction_caching": True,
            "extraction_cache_ttl": 3600,
        }

        # Initialize services (will be dependency-injected)
        self.fact_extractor = None
        self.entity_normalizer = None
        self.agent = None
        self.extraction_engine = FactExtractionEngine()

    async def _initialize_services(self):
        """Initialize services using dependency injection."""
        if self.coordinator is None:
            logger.warning(
                "No service coordinator available, using fallback initialization"
            )
            await self.extraction_engine.initialize()
            return

        try:
            # Initialize services through coordinator
            await self.coordinator.initialize_services()

            # Get services from coordinator
            try:
                self.fact_extractor = await self.coordinator.get_service(
                    "fact_extractor"
                )
                logger.info("Fact extractor service obtained")
            except Exception as e:
                logger.warning("Fact extractor service not available", error=str(e))
                self.fact_extractor = None

            try:
                self.entity_normalizer = await self.coordinator.get_service(
                    "entity_normalizer"
                )
                logger.info("Entity normalizer service obtained")
            except Exception as e:
                logger.warning("Entity normalizer service not available", error=str(e))
                self.entity_normalizer = None

            try:
                self.agent = await self.coordinator.get_service("fact_extraction_agent")
                logger.info("Fact extraction agent service obtained")
            except Exception as e:
                logger.warning(
                    "Fact extraction agent service not available", error=str(e)
                )
                self.agent = None

            # Initialize extraction engine with obtained services
            await self.extraction_engine.initialize(
                fact_extractor=self.fact_extractor,
                entity_normalizer=self.entity_normalizer,
                agent=self.agent,
            )

        except Exception as e:
            logger.error("Failed to initialize fact generation services", error=str(e))
            raise StageExecutionError(f"Service initialization failed: {str(e)}")

    @stage_error_handler("fact_generation_execute")
    async def execute(
        self,
        input_files: List[Path],
        context: StageContext,
        output_dir: Optional[Path] = None,
    ) -> StageResult:
        """Execute fact generation on chunk files.

        Args:
            input_files: List of chunk.json files to process
            context: Stage execution context
            output_dir: Optional output directory override

        Returns:
            StageResult with fact generation results
        """
        # Get output directory from context if not provided
        if output_dir is None:
            output_dir = context.output_dir or Path.cwd()

        # Get configuration from context
        context_config = context.config.get("fact-generator", {})

        # Merge context config into self.config (preserving defaults)
        self.config.update(context_config)

        # Create FactGeneratorConfig object for extraction engine
        try:
            from morag_core.config import FactGeneratorConfig as CoreFactGeneratorConfig
            # Create config object from merged dict
            fact_gen_config = CoreFactGeneratorConfig.from_env_and_overrides(context_config)
        except ImportError:
            # Fallback: create a simple object with dict attributes
            class SimpleConfig:
                def __init__(self, config_dict):
                    for key, value in config_dict.items():
                        setattr(self, key, value)
                    # Add required attributes with defaults
                    if not hasattr(self, 'enable_batch_processing'):
                        self.enable_batch_processing = True
                    if not hasattr(self, 'max_chunks_per_batch'):
                        self.max_chunks_per_batch = 10
            fact_gen_config = SimpleConfig(self.config)

        start_time = datetime.now()
        output_files_list = []

        try:
            # Initialize services using dependency injection
            await self._initialize_services()

            # Validate inputs
            if not self.validate_inputs(input_files):
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()

                stage_metadata = StageMetadata(
                    execution_time=execution_time,
                    start_time=start_time,
                    end_time=end_time,
                    input_files=[str(f) for f in input_files],
                    output_files=[],
                    config_used=self.config,
                )

                return StageResult(
                    stage_type=self.stage_type,
                    status=StageStatus.FAILED,
                    output_files=[],
                    metadata=stage_metadata,
                    error_message="Input validation failed",
                )

            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)

            # Process each chunk file
            results = []
            errors = []
            total_facts = 0
            total_entities = 0
            total_relations = 0

            for input_file in input_files:
                try:
                    logger.info("Processing chunk file", file=str(input_file))

                    # Sanitize input file path for security
                    try:
                        safe_base_dir = Path.cwd()
                        sanitized_input = sanitize_filepath(
                            input_file, base_dir=safe_base_dir
                        )
                    except (ValidationError, ValueError, Exception) as e:
                        logger.error(
                            "File path sanitization failed",
                            file=str(input_file),
                            error=str(e),
                        )
                        errors.append(
                            {
                                "input_file": str(input_file),
                                "error": f"File path sanitization failed: {str(e)}",
                            }
                        )
                        continue

                    # Load chunks from JSON file
                    with open(sanitized_input, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    chunks = data.get("chunks", [])
                    if not chunks:
                        logger.warning("No chunks found in file", file=str(input_file))
                        continue

                    # Generate output filename
                    output_filename = input_file.stem + ".facts.json"
                    output_file = output_dir / output_filename

                    # Extract facts from chunks
                    extracted_results = (
                        await self.extraction_engine.extract_from_chunks(chunks, fact_gen_config)
                    )

                    # Compile results
                    file_results = {
                        "source_file": str(input_file),
                        "processing_timestamp": datetime.now().isoformat(),
                        "chunk_count": len(chunks),
                        "extraction_results": extracted_results,
                        "statistics": {
                            "total_facts": len(extracted_results.get("facts", [])),
                            "total_entities": len(
                                extracted_results.get("entities", [])
                            ),
                            "total_relations": len(
                                extracted_results.get("relations", [])
                            ),
                            "total_keywords": len(
                                extracted_results.get("keywords", [])
                            ),
                        },
                    }

                    # Write results to file
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(file_results, f, indent=2, ensure_ascii=False)

                    # Track output file
                    output_files_list.append(output_file)

                    # Update totals
                    total_facts += file_results["statistics"]["total_facts"]
                    total_entities += file_results["statistics"]["total_entities"]
                    total_relations += file_results["statistics"]["total_relations"]

                    results.append(
                        {
                            "input_file": str(input_file),
                            "output_file": str(output_file),
                            "statistics": file_results["statistics"],
                        }
                    )

                    logger.info(
                        "Chunk file processed successfully",
                        file=str(input_file),
                        output=str(output_file),
                        facts=file_results["statistics"]["total_facts"],
                    )

                except Exception as e:
                    error_msg = f"Failed to process {input_file}: {str(e)}"
                    errors.append({"input_file": str(input_file), "error": error_msg})
                    logger.error(
                        "Chunk file processing failed",
                        file=str(input_file),
                        error=str(e),
                    )

            # Determine overall status
            total_files = len(input_files)
            successful_files = len(results)

            if successful_files == total_files:
                status = StageStatus.COMPLETED
            elif successful_files > 0:
                status = StageStatus.PARTIAL
            else:
                status = StageStatus.FAILED

            # Create metadata
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            stage_metadata = StageMetadata(
                execution_time=execution_time,
                start_time=start_time,
                end_time=end_time,
                input_files=[str(f) for f in input_files],
                output_files=[str(f) for f in output_files_list],
                config_used=self.config,
                metrics={
                    "total_files": total_files,
                    "successful_files": successful_files,
                    "failed_files": len(errors),
                    "success_rate": successful_files / total_files if total_files > 0 else 0.0,
                    "total_facts_extracted": total_facts,
                    "total_entities_extracted": total_entities,
                    "total_relations_extracted": total_relations,
                }
            )

            return StageResult(
                stage_type=self.stage_type,
                status=status,
                output_files=output_files_list,
                metadata=stage_metadata,
                data={"results": results, "errors": errors}
            )

        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            logger.error("Stage execution failed", error=str(e))

            stage_metadata = StageMetadata(
                execution_time=execution_time,
                start_time=start_time,
                end_time=end_time,
                input_files=[str(f) for f in input_files],
                output_files=[str(f) for f in output_files_list],
                config_used=self.config,
            )

            return StageResult(
                stage_type=self.stage_type,
                status=StageStatus.FAILED,
                output_files=output_files_list,
                metadata=stage_metadata,
                error_message=str(e)
            )

    @validation_error_handler("fact_generation_validate_inputs")
    def validate_inputs(self, input_files: List[Path]) -> bool:
        """Validate input chunk files."""
        if not input_files:
            logger.error("No input files provided")
            return False

        for input_file in input_files:
            if not input_file.exists():
                logger.error("Input file does not exist", file=str(input_file))
                return False

            if not input_file.suffix == ".json":
                logger.error("Input file is not a JSON file", file=str(input_file))
                return False

            # Validate file contains chunk data
            try:
                # Sanitize input file path for security
                try:
                    safe_base_dir = Path.cwd()
                    sanitized_input = sanitize_filepath(
                        input_file, base_dir=safe_base_dir
                    )
                except (ValidationError, ValueError, Exception) as e:
                    logger.error(
                        "File path sanitization failed during validation",
                        file=str(input_file),
                        error=str(e),
                    )
                    return False

                with open(sanitized_input, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if "chunks" not in data:
                    logger.error(
                        "Input file does not contain chunks", file=str(input_file)
                    )
                    return False

            except Exception as e:
                logger.error(
                    "Error reading input file", file=str(input_file), error=str(e)
                )
                return False

        return True

    def get_dependencies(self) -> List[StageType]:
        """Get list of stage dependencies."""
        return [StageType.CHUNKER]  # Fact generation depends on chunker

    def get_expected_outputs(
        self, input_files: List[Path], context: StageContext
    ) -> List[Path]:
        """Get expected output files."""
        output_dir = context.output_dir if context else Path.cwd()
        outputs = []

        for input_file in input_files:
            output_filename = input_file.stem + ".facts.json"
            outputs.append(output_dir / output_filename)

        return outputs


__all__ = ["FactGeneratorStage"]
