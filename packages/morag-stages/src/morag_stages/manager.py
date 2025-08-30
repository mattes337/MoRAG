"""Stage manager for executing processing stages."""

import asyncio
import time
import os
import json
import shutil
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path
import structlog

from .models import (
    Stage, StageType, StageStatus, StageResult, StageContext,
    StageMetadata, PipelineConfig
)
from .registry import StageRegistry, get_global_registry
from .exceptions import (
    StageError, StageExecutionError, StageValidationError,
    StageDependencyError, StageTimeoutError
)
from .webhook import WebhookNotifier

logger = structlog.get_logger(__name__)


class StageManager:
    """Manages execution of processing stages."""

    def __init__(self,
                 registry: Optional[StageRegistry] = None,
                 webhook_notifier: Optional[WebhookNotifier] = None):
        """Initialize stage manager.

        Args:
            registry: Stage registry to use (defaults to global registry)
            webhook_notifier: Webhook notifier for stage completion events
        """
        self.registry = registry or get_global_registry()
        self.webhook_notifier = webhook_notifier or WebhookNotifier()

        # Mock mode configuration
        self.mock_mode = os.getenv('MORAG_MOCK_MODE', 'false').lower() == 'true'
        self.mock_data_dir = Path(os.getenv('MORAG_MOCK_DATA_DIR', './mock'))

        if self.mock_mode:
            logger.info("Mock mode enabled", mock_data_dir=str(self.mock_data_dir))

    def _get_mock_output_file(self, stage_type: StageType, input_files: List[Path], context: StageContext) -> Optional[Path]:
        """Get the mock output file for a given stage type.

        Args:
            stage_type: Type of stage
            input_files: Input files (used to determine base filename)
            context: Stage execution context

        Returns:
            Path to mock output file if it exists, None otherwise
        """
        stage_dir = self.mock_data_dir / stage_type.value

        # Determine content type from context or input files
        content_type = self._detect_content_type(input_files, context)

        # Define expected output file patterns for each stage and content type
        if stage_type == StageType.MARKDOWN_CONVERSION:
            mock_file = stage_dir / f"{content_type}.md"
        elif stage_type == StageType.MARKDOWN_OPTIMIZER:
            mock_file = stage_dir / f"{content_type}.optimized.md"
        elif stage_type == StageType.CHUNKER:
            mock_file = stage_dir / f"{content_type}.chunks.json"
        elif stage_type == StageType.FACT_GENERATOR:
            mock_file = stage_dir / f"{content_type}.facts.json"
        elif stage_type == StageType.INGESTOR:
            mock_file = stage_dir / f"{content_type}.ingestion.json"
        else:
            return None

        # Fallback to sample files if content-type specific file doesn't exist
        if not mock_file.exists():
            if stage_type == StageType.MARKDOWN_CONVERSION:
                mock_file = stage_dir / "sample.md"
            elif stage_type == StageType.MARKDOWN_OPTIMIZER:
                mock_file = stage_dir / "sample.optimized.md"
            elif stage_type == StageType.CHUNKER:
                mock_file = stage_dir / "sample.chunks.json"
            elif stage_type == StageType.FACT_GENERATOR:
                mock_file = stage_dir / "sample.facts.json"
            elif stage_type == StageType.INGESTOR:
                mock_file = stage_dir / "sample.ingestion.json"

        return mock_file if mock_file.exists() else None

    def _detect_content_type(self, input_files: List[Path], context: StageContext) -> str:
        """Detect content type from input files or context.

        Args:
            input_files: Input files to analyze
            context: Stage execution context

        Returns:
            Content type string (document, video, audio, web, youtube)
        """
        # Check context config for explicit content type
        if context.config.get('content_type'):
            return context.config['content_type']

        # Detect from file extensions
        if input_files:
            file_ext = input_files[0].suffix.lower()
            if file_ext in ['.pdf', '.doc', '.docx', '.txt']:
                return 'document'
            elif file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
                return 'video'
            elif file_ext in ['.mp3', '.wav', '.m4a', '.flac']:
                return 'audio'
            elif file_ext in ['.html', '.htm']:
                return 'web'

        # Check for YouTube URLs in context
        source_path = getattr(context, 'source_path', None)
        if source_path and 'youtube.com' in str(source_path):
            return 'youtube'
        elif source_path and any(domain in str(source_path) for domain in ['http://', 'https://']):
            return 'web'

        # Default fallback
        return 'document'

    async def _execute_mock_stage(self,
                                 stage_type: StageType,
                                 input_files: List[Path],
                                 context: StageContext) -> StageResult:
        """Execute a stage in mock mode by copying pre-generated outputs.

        Args:
            stage_type: Type of stage to execute
            input_files: Input files for the stage
            context: Execution context

        Returns:
            Stage execution result with mock data
        """
        logger.info("Executing stage in mock mode", stage_type=stage_type.value)

        # Get mock output file
        mock_file = self._get_mock_output_file(stage_type, input_files, context)
        if not mock_file:
            raise StageExecutionError(
                f"Mock data not found for stage {stage_type.value}",
                stage_type=stage_type.value
            )

        # Generate output filename based on input
        if input_files:
            base_name = input_files[0].stem
        else:
            base_name = "sample"

        # Determine output file extension based on stage type
        if stage_type == StageType.MARKDOWN_CONVERSION:
            output_file = context.output_dir / f"{base_name}.md"
        elif stage_type == StageType.MARKDOWN_OPTIMIZER:
            output_file = context.output_dir / f"{base_name}.optimized.md"
        elif stage_type == StageType.CHUNKER:
            output_file = context.output_dir / f"{base_name}.chunks.json"
        elif stage_type == StageType.FACT_GENERATOR:
            output_file = context.output_dir / f"{base_name}.facts.json"
        elif stage_type == StageType.INGESTOR:
            output_file = context.output_dir / f"{base_name}.ingestion.json"
        else:
            output_file = context.output_dir / f"{base_name}.output"

        # Ensure output directory exists
        context.output_dir.mkdir(parents=True, exist_ok=True)

        # Copy mock file to output location
        shutil.copy2(mock_file, output_file)

        # Load mock data for result
        mock_data = {}
        if mock_file.suffix == '.json':
            try:
                with open(mock_file, 'r', encoding='utf-8') as f:
                    mock_data = json.load(f)
            except Exception as e:
                logger.warning("Failed to load mock JSON data", error=str(e))

        # Create metadata
        metadata = StageMetadata(
            execution_time=0.5,  # Simulate quick execution
            start_time=datetime.now(),
            end_time=datetime.now(),
            input_files=[str(f) for f in input_files],
            output_files=[str(output_file)],
            config_used=context.get_stage_config(stage_type),
            metrics={
                "mock_mode": True,
                "mock_source": str(mock_file),
                "stage_type": stage_type.value
            }
        )

        return StageResult(
            stage_type=stage_type,
            status=StageStatus.COMPLETED,
            output_files=[output_file],
            metadata=metadata,
            data=mock_data
        )

    async def execute_stage(self,
                           stage_type: StageType,
                           input_files: List[Path],
                           context: StageContext) -> StageResult:
        """Execute a single stage.

        Args:
            stage_type: Type of stage to execute
            input_files: Input files for the stage
            context: Execution context

        Returns:
            Stage execution result

        Raises:
            StageError: If stage execution fails
        """
        logger.info("Starting stage execution", stage_type=stage_type.value, mock_mode=self.mock_mode)

        # Check if mock mode is enabled
        if self.mock_mode:
            return await self._execute_mock_stage(stage_type, input_files, context)
        
        # Get stage instance
        try:
            stage = self.registry.get_stage(stage_type)
        except StageError as e:
            logger.error("Stage not found", stage_type=stage_type.value, error=str(e))
            raise
        
        # Check if stage should be skipped due to existing outputs
        if context.resume_from_existing:
            expected_outputs = stage.get_expected_outputs(input_files, context)
            if all(output.exists() for output in expected_outputs):
                logger.info("Skipping stage - outputs already exist", 
                           stage_type=stage_type.value, outputs=[str(f) for f in expected_outputs])
                
                # Create skipped result
                metadata = StageMetadata(
                    execution_time=0.0,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    input_files=[str(f) for f in input_files],
                    output_files=[str(f) for f in expected_outputs],
                    config_used=context.get_stage_config(stage_type)
                )
                
                result = StageResult(
                    stage_type=stage_type,
                    status=StageStatus.SKIPPED,
                    output_files=expected_outputs,
                    metadata=metadata
                )
                
                context.add_stage_result(result)
                return result
        
        # Validate inputs
        try:
            logger.debug("Validating inputs for stage",
                        stage_type=stage_type.value,
                        input_files=[str(f) for f in input_files],
                        input_count=len(input_files))

            if not stage.validate_inputs(input_files):
                # Get more detailed validation info
                validation_details = []
                for i, file in enumerate(input_files):
                    file_info = {
                        "index": i,
                        "path": str(file),
                        "exists": file.exists() if hasattr(file, 'exists') else "unknown",
                        "is_url": str(file).startswith(('http://', 'https://'))
                    }
                    if hasattr(file, 'exists') and file.exists():
                        try:
                            file_info["size"] = file.stat().st_size
                            file_info["suffix"] = file.suffix
                        except:
                            pass
                    validation_details.append(file_info)

                logger.error("Input validation failed - detailed info",
                           stage_type=stage_type.value,
                           validation_details=validation_details)

                # Create more descriptive error message based on stage type
                error_message = self._create_validation_error_message(stage_type, input_files, validation_details)

                raise StageValidationError(
                    error_message,
                    stage_type=stage_type.value,
                    invalid_files=[str(f) for f in input_files]
                )
        except StageValidationError:
            raise
        except Exception as e:
            logger.error("Input validation failed with exception",
                        stage_type=stage_type.value,
                        error=str(e),
                        error_type=e.__class__.__name__,
                        input_files=[str(f) for f in input_files])
            raise StageValidationError(
                f"Input validation error for {stage_type.value}: {e}",
                stage_type=stage_type.value,
                invalid_files=[str(f) for f in input_files]
            )
        
        # Execute stage
        start_time = datetime.now()
        execution_start = time.time()
        
        try:
            logger.info("Executing stage", stage_type=stage_type.value, 
                       input_files=[str(f) for f in input_files])
            
            # Get stage configuration
            stage_config = context.get_stage_config(stage_type)
            timeout = stage_config.get('timeout_seconds')
            
            # Execute with optional timeout
            if timeout:
                result = await asyncio.wait_for(
                    stage.execute(input_files, context),
                    timeout=timeout
                )
            else:
                result = await stage.execute(input_files, context)
            
            execution_time = time.time() - execution_start
            end_time = datetime.now()
            
            # Update result metadata
            result.metadata.execution_time = execution_time
            result.metadata.start_time = start_time
            result.metadata.end_time = end_time
            result.metadata.input_files = [str(f) for f in input_files]
            result.metadata.config_used = stage_config
            
            # Add result to context
            context.add_stage_result(result)
            
            logger.info("Stage completed successfully", 
                       stage_type=stage_type.value,
                       execution_time=execution_time,
                       output_files=[str(f) for f in result.output_files])
            
            # Send webhook notification
            if context.webhook_url:
                await self.webhook_notifier.notify_stage_completion(
                    context.webhook_url, result, context
                )
            
            return result
            
        except asyncio.TimeoutError:
            execution_time = time.time() - execution_start
            error_msg = f"Stage {stage_type.value} timed out after {timeout} seconds"
            logger.error("Stage execution timeout", 
                        stage_type=stage_type.value, 
                        timeout=timeout,
                        execution_time=execution_time)
            
            raise StageTimeoutError(
                error_msg,
                stage_type=stage_type.value,
                timeout_seconds=timeout
            )
            
        except Exception as e:
            execution_time = time.time() - execution_start
            error_msg = f"Stage {stage_type.value} execution failed: {e}"
            logger.error("Stage execution failed", 
                        stage_type=stage_type.value,
                        execution_time=execution_time,
                        error=str(e))
            
            # Create failed result
            metadata = StageMetadata(
                execution_time=execution_time,
                start_time=start_time,
                end_time=datetime.now(),
                input_files=[str(f) for f in input_files],
                output_files=[],
                config_used=context.get_stage_config(stage_type)
            )
            
            failed_result = StageResult(
                stage_type=stage_type,
                status=StageStatus.FAILED,
                output_files=[],
                metadata=metadata,
                error_message=str(e)
            )
            
            context.add_stage_result(failed_result)
            
            # Send webhook notification for failure
            if context.webhook_url:
                await self.webhook_notifier.notify_stage_completion(
                    context.webhook_url, failed_result, context
                )
            
            raise StageExecutionError(
                error_msg,
                stage_type=stage_type.value,
                original_error=e
            )
    
    async def execute_stage_chain(self,
                                 stage_types: List[StageType],
                                 initial_input_files: List[Path],
                                 context: StageContext) -> List[StageResult]:
        """Execute a chain of stages in dependency order.
        
        Args:
            stage_types: List of stage types to execute
            initial_input_files: Initial input files
            context: Execution context
            
        Returns:
            List of stage results in execution order
            
        Raises:
            StageError: If stage chain execution fails
        """
        logger.info("Starting stage chain execution", 
                   stages=[s.value for s in stage_types])
        
        # Validate and order stages by dependencies
        try:
            ordered_stages = self.registry.get_dependency_order(stage_types)
        except StageDependencyError as e:
            logger.error("Stage dependency validation failed", error=str(e))
            raise
        
        results: List[StageResult] = []
        current_input_files = initial_input_files

        for stage_type in ordered_stages:
            try:
                result = await self.execute_stage(stage_type, current_input_files, context)
                results.append(result)

                # Use outputs as inputs for next stage (if successful or skipped)
                if result.success or result.skipped:
                    current_input_files = result.output_files
                elif not self.registry.get_stage(stage_type).is_optional():
                    # Required stage failed, stop execution
                    logger.error("Required stage failed, stopping chain",
                               stage_type=stage_type.value)
                    break
                
            except StageError as e:
                logger.error("Stage chain execution failed", 
                           stage_type=stage_type.value, error=str(e))
                
                # For optional stages, continue with original inputs
                stage = self.registry.get_stage(stage_type)
                if stage.is_optional():
                    logger.warning("Optional stage failed, continuing chain", 
                                 stage_type=stage_type.value)
                    continue
                else:
                    # Required stage failed, re-raise
                    raise
        
        logger.info("Stage chain execution completed", 
                   total_stages=len(results),
                   successful_stages=len([r for r in results if r.success]))
        
        return results

    def _create_validation_error_message(self, stage_type: StageType, input_files: List[Path], validation_details: List[dict]) -> str:
        """Create a descriptive validation error message based on stage type and input files.

        Args:
            stage_type: The stage type that failed validation
            input_files: The input files that failed validation
            validation_details: Detailed validation information

        Returns:
            Descriptive error message
        """
        file_count = len(input_files)

        # Stage-specific validation messages
        if stage_type == StageType.FACT_GENERATOR:
            if file_count != 1:
                return f"Fact generator stage requires exactly 1 input file, got {file_count}. Files: {[str(f) for f in input_files]}"

            file = input_files[0]
            if not file.exists():
                return f"Fact generator stage input file does not exist: {file}"

            if not str(file).endswith('.chunks.json'):
                return f"Fact generator stage requires a .chunks.json file (output from chunker stage), but got: {file}. Make sure to run the chunker stage first."

            return f"Fact generator stage input file {file} exists but does not contain valid chunk data."

        elif stage_type == StageType.CHUNKER:
            if file_count != 1:
                return f"Chunker stage requires exactly 1 input file, got {file_count}. Files: {[str(f) for f in input_files]}"

            file = input_files[0]
            if not file.exists():
                return f"Chunker stage input file does not exist: {file}"

            if not file.suffix.lower() in ['.md', '.markdown']:
                return f"Chunker stage requires a markdown file (.md or .markdown), but got: {file}. Make sure to run the markdown-conversion stage first."

            return f"Chunker stage input file {file} exists but is not valid markdown."

        elif stage_type == StageType.MARKDOWN_CONVERSION:
            if file_count != 1:
                return f"Markdown conversion stage requires exactly 1 input file, got {file_count}. Files: {[str(f) for f in input_files]}"

            file = input_files[0]
            if not file.exists():
                return f"Markdown conversion stage input file does not exist: {file}"

            supported_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.mp3', '.wav', '.flac', '.m4a', '.pdf', '.docx', '.txt', '.md'}
            if file.suffix.lower() not in supported_extensions and not str(file).startswith(('http://', 'https://')):
                return f"Markdown conversion stage requires a supported file type {supported_extensions} or URL, but got: {file}"

            return f"Markdown conversion stage input file {file} exists but could not be processed."

        elif stage_type == StageType.INGESTOR:
            if file_count < 1:
                return f"Ingestor stage requires at least 1 input file, got {file_count}."

            invalid_files = []
            for file in input_files:
                if not file.exists():
                    invalid_files.append(f"{file} (does not exist)")
                elif not str(file).endswith('.json'):
                    invalid_files.append(f"{file} (not a .json file)")

            if invalid_files:
                return f"Ingestor stage requires .json files, but found invalid files: {invalid_files}"

            return f"Ingestor stage input files exist but contain invalid JSON data."

        # Default fallback message with more detailed validation info
        if file_count == 1:
            file = input_files[0]
            if not file.exists():
                return f"Input validation failed for stage {stage_type.value}. File does not exist: {file}"
            elif hasattr(file, 'suffix'):
                return f"Input validation failed for stage {stage_type.value}. File format may not be supported: {file} (extension: {file.suffix})"
            else:
                return f"Input validation failed for stage {stage_type.value}. File validation failed: {file}"
        else:
            return f"Input validation failed for stage {stage_type.value}. Expected: 1 file, got: {file_count}. Files: {[str(f) for f in input_files]}"
