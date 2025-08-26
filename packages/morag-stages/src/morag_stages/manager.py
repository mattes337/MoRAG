"""Stage manager for executing processing stages."""

import asyncio
import time
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
        logger.info("Starting stage execution", stage_type=stage_type.value)
        
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
            if not stage.validate_inputs(input_files):
                raise StageValidationError(
                    f"Input validation failed for stage {stage_type.value}",
                    stage_type=stage_type.value,
                    invalid_files=[str(f) for f in input_files]
                )
        except Exception as e:
            logger.error("Input validation failed", stage_type=stage_type.value, error=str(e))
            raise StageValidationError(
                f"Input validation error: {e}",
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
