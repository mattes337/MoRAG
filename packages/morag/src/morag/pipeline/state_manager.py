"""Pipeline state management for execution tracking and recovery."""

import asyncio
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
from morag_core.config import get_settings
from morag_core.exceptions import ProcessingError

logger = structlog.get_logger(__name__)


class PipelineStatus(Enum):
    """Pipeline execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RECOVERING = "recovering"


class StageStatus(Enum):
    """Stage execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageState:
    """State information for a pipeline stage."""

    stage_name: str
    status: StageStatus
    start_time: Optional[str]
    end_time: Optional[str]
    processing_time: float
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    error_message: Optional[str]
    retry_count: int
    metadata: Dict[str, Any]


@dataclass
class PipelineState:
    """Complete pipeline execution state."""

    pipeline_id: str
    pipeline_type: str
    status: PipelineStatus
    start_time: str
    end_time: Optional[str]
    total_processing_time: float
    current_stage: Optional[str]
    stages: Dict[str, StageState]
    input_parameters: Dict[str, Any]
    output_results: Dict[str, Any]
    error_message: Optional[str]
    retry_count: int
    metadata: Dict[str, Any]


@dataclass
class CheckpointData:
    """Checkpoint data for pipeline recovery."""

    pipeline_state: PipelineState
    checkpoint_time: str
    recovery_data: Dict[str, Any]
    file_references: List[str]


class PipelineStateManager:
    """Manages pipeline execution state and recovery."""

    def __init__(
        self, state_dir: Optional[Path] = None, config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the pipeline state manager.

        Args:
            state_dir: Directory for state files
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.settings = get_settings()

        # Directory configuration
        self.state_dir = state_dir or Path("pipeline_state")
        self.state_dir.mkdir(exist_ok=True)

        # State management settings
        self.auto_checkpoint = self.config.get("auto_checkpoint", True)
        self.checkpoint_interval = self.config.get(
            "checkpoint_interval", 30.0
        )  # seconds
        self.max_retries = self.config.get("max_retries", 3)
        self.retention_days = self.config.get("retention_days", 7)

        # Recovery settings
        self.enable_recovery = self.config.get("enable_recovery", True)
        self.recovery_timeout = self.config.get("recovery_timeout", 300.0)  # seconds

        # Performance settings
        self.async_saves = self.config.get("async_saves", True)
        self.compress_state = self.config.get("compress_state", False)

        # Active pipelines tracking
        self.active_pipelines: Dict[str, PipelineState] = {}
        self.checkpoint_tasks: Dict[str, asyncio.Task] = {}

        logger.info(
            "Pipeline state manager initialized",
            state_dir=str(self.state_dir),
            auto_checkpoint=self.auto_checkpoint,
            checkpoint_interval=self.checkpoint_interval,
            enable_recovery=self.enable_recovery,
        )

    async def create_pipeline(
        self,
        pipeline_id: str,
        pipeline_type: str,
        input_parameters: Dict[str, Any],
        stages: List[str],
    ) -> PipelineState:
        """Create a new pipeline execution state.

        Args:
            pipeline_id: Unique pipeline identifier
            pipeline_type: Type of pipeline (ingestion, resolution, etc.)
            input_parameters: Input parameters for the pipeline
            stages: List of stage names

        Returns:
            Initial pipeline state
        """
        try:
            # Create initial stage states
            stage_states = {}
            for stage_name in stages:
                stage_states[stage_name] = StageState(
                    stage_name=stage_name,
                    status=StageStatus.PENDING,
                    start_time=None,
                    end_time=None,
                    processing_time=0.0,
                    input_data={},
                    output_data={},
                    error_message=None,
                    retry_count=0,
                    metadata={},
                )

            # Create pipeline state
            pipeline_state = PipelineState(
                pipeline_id=pipeline_id,
                pipeline_type=pipeline_type,
                status=PipelineStatus.PENDING,
                start_time=datetime.now(timezone.utc).isoformat(),
                end_time=None,
                total_processing_time=0.0,
                current_stage=None,
                stages=stage_states,
                input_parameters=input_parameters,
                output_results={},
                error_message=None,
                retry_count=0,
                metadata={},
            )

            # Register active pipeline
            self.active_pipelines[pipeline_id] = pipeline_state

            # Save initial state
            await self.save_checkpoint(pipeline_id, "initialization", {})

            # Start auto-checkpoint if enabled
            if self.auto_checkpoint:
                await self._start_auto_checkpoint(pipeline_id)

            logger.info(
                "Pipeline created",
                pipeline_id=pipeline_id,
                pipeline_type=pipeline_type,
                stages=len(stages),
            )

            return pipeline_state

        except Exception as e:
            logger.error(f"Failed to create pipeline {pipeline_id}: {e}")
            raise ProcessingError(f"Failed to create pipeline: {e}")

    async def start_stage(
        self, pipeline_id: str, stage_name: str, input_data: Dict[str, Any]
    ) -> None:
        """Start execution of a pipeline stage.

        Args:
            pipeline_id: Pipeline identifier
            stage_name: Stage name
            input_data: Input data for the stage
        """
        try:
            if pipeline_id not in self.active_pipelines:
                raise ValueError(f"Pipeline {pipeline_id} not found")

            pipeline_state = self.active_pipelines[pipeline_id]

            if stage_name not in pipeline_state.stages:
                raise ValueError(f"Stage {stage_name} not found in pipeline")

            # Update stage state
            stage_state = pipeline_state.stages[stage_name]
            stage_state.status = StageStatus.RUNNING
            stage_state.start_time = datetime.now(timezone.utc).isoformat()
            stage_state.input_data = input_data

            # Update pipeline state
            pipeline_state.current_stage = stage_name
            if pipeline_state.status == PipelineStatus.PENDING:
                pipeline_state.status = PipelineStatus.RUNNING

            logger.info("Stage started", pipeline_id=pipeline_id, stage_name=stage_name)

        except Exception as e:
            logger.error(f"Failed to start stage {stage_name}: {e}")
            raise ProcessingError(f"Failed to start stage: {e}")

    async def complete_stage(
        self,
        pipeline_id: str,
        stage_name: str,
        output_data: Dict[str, Any],
        processing_time: float,
    ) -> None:
        """Complete execution of a pipeline stage.

        Args:
            pipeline_id: Pipeline identifier
            stage_name: Stage name
            output_data: Output data from the stage
            processing_time: Stage processing time
        """
        try:
            if pipeline_id not in self.active_pipelines:
                raise ValueError(f"Pipeline {pipeline_id} not found")

            pipeline_state = self.active_pipelines[pipeline_id]

            if stage_name not in pipeline_state.stages:
                raise ValueError(f"Stage {stage_name} not found in pipeline")

            # Update stage state
            stage_state = pipeline_state.stages[stage_name]
            stage_state.status = StageStatus.COMPLETED
            stage_state.end_time = datetime.now(timezone.utc).isoformat()
            stage_state.output_data = output_data
            stage_state.processing_time = processing_time

            # Update pipeline total time
            pipeline_state.total_processing_time += processing_time

            # Check if all stages are complete
            all_completed = all(
                stage.status in [StageStatus.COMPLETED, StageStatus.SKIPPED]
                for stage in pipeline_state.stages.values()
            )

            if all_completed:
                await self._complete_pipeline(pipeline_id)

            logger.info(
                "Stage completed",
                pipeline_id=pipeline_id,
                stage_name=stage_name,
                processing_time=processing_time,
            )

        except Exception as e:
            logger.error(f"Failed to complete stage {stage_name}: {e}")
            raise ProcessingError(f"Failed to complete stage: {e}")

    async def fail_stage(
        self, pipeline_id: str, stage_name: str, error_message: str, retry: bool = True
    ) -> bool:
        """Mark a stage as failed and optionally retry.

        Args:
            pipeline_id: Pipeline identifier
            stage_name: Stage name
            error_message: Error message
            retry: Whether to retry the stage

        Returns:
            True if stage will be retried, False otherwise
        """
        try:
            if pipeline_id not in self.active_pipelines:
                raise ValueError(f"Pipeline {pipeline_id} not found")

            pipeline_state = self.active_pipelines[pipeline_id]

            if stage_name not in pipeline_state.stages:
                raise ValueError(f"Stage {stage_name} not found in pipeline")

            stage_state = pipeline_state.stages[stage_name]
            stage_state.retry_count += 1
            stage_state.error_message = error_message

            # Check if we should retry
            should_retry = retry and stage_state.retry_count <= self.max_retries

            if should_retry:
                stage_state.status = StageStatus.PENDING
                logger.warning(
                    "Stage failed, will retry",
                    pipeline_id=pipeline_id,
                    stage_name=stage_name,
                    retry_count=stage_state.retry_count,
                    error=error_message,
                )
                return True
            else:
                stage_state.status = StageStatus.FAILED
                stage_state.end_time = datetime.now(timezone.utc).isoformat()

                # Fail the entire pipeline
                await self._fail_pipeline(
                    pipeline_id, f"Stage {stage_name} failed: {error_message}"
                )

                logger.error(
                    "Stage failed permanently",
                    pipeline_id=pipeline_id,
                    stage_name=stage_name,
                    retry_count=stage_state.retry_count,
                    error=error_message,
                )
                return False

        except Exception as e:
            logger.error(f"Failed to handle stage failure: {e}")
            return False

    async def save_checkpoint(
        self, pipeline_id: str, stage: str, state: Dict[str, Any]
    ) -> None:
        """Save pipeline state checkpoint for recovery.

        Args:
            pipeline_id: Pipeline identifier
            stage: Current stage name
            state: Additional state data
        """
        try:
            if pipeline_id not in self.active_pipelines:
                return

            pipeline_state = self.active_pipelines[pipeline_id]

            # Create checkpoint data
            checkpoint = CheckpointData(
                pipeline_state=pipeline_state,
                checkpoint_time=datetime.now(timezone.utc).isoformat(),
                recovery_data=state,
                file_references=[],  # Could be populated with intermediate file paths
            )

            # Save checkpoint to file
            checkpoint_file = self.state_dir / f"{pipeline_id}_checkpoint.json"

            if self.async_saves:
                await self._save_checkpoint_async(checkpoint_file, checkpoint)
            else:
                await self._save_checkpoint_sync(checkpoint_file, checkpoint)

            logger.debug("Checkpoint saved", pipeline_id=pipeline_id, stage=stage)

        except Exception as e:
            logger.warning(f"Failed to save checkpoint for {pipeline_id}: {e}")

    async def resume_pipeline(self, pipeline_id: str) -> Optional[PipelineState]:
        """Resume pipeline from last successful checkpoint.

        Args:
            pipeline_id: Pipeline identifier

        Returns:
            Pipeline state if recovery successful, None otherwise
        """
        try:
            if not self.enable_recovery:
                logger.warning("Pipeline recovery is disabled")
                return None

            # Load checkpoint
            checkpoint_file = self.state_dir / f"{pipeline_id}_checkpoint.json"

            if not checkpoint_file.exists():
                logger.warning(f"No checkpoint found for pipeline {pipeline_id}")
                return None

            checkpoint = await self._load_checkpoint(checkpoint_file)

            if not checkpoint:
                return None

            # Restore pipeline state
            pipeline_state = checkpoint.pipeline_state
            pipeline_state.status = PipelineStatus.RECOVERING

            # Register as active pipeline
            self.active_pipelines[pipeline_id] = pipeline_state

            # Start auto-checkpoint if enabled
            if self.auto_checkpoint:
                await self._start_auto_checkpoint(pipeline_id)

            logger.info(
                "Pipeline resumed from checkpoint",
                pipeline_id=pipeline_id,
                checkpoint_time=checkpoint.checkpoint_time,
            )

            return pipeline_state

        except Exception as e:
            logger.error(f"Failed to resume pipeline {pipeline_id}: {e}")
            return None

    async def get_pipeline_state(self, pipeline_id: str) -> Optional[PipelineState]:
        """Get current pipeline state.

        Args:
            pipeline_id: Pipeline identifier

        Returns:
            Pipeline state if found, None otherwise
        """
        return self.active_pipelines.get(pipeline_id)

    async def list_active_pipelines(self) -> List[str]:
        """List all active pipeline IDs.

        Returns:
            List of active pipeline IDs
        """
        return list(self.active_pipelines.keys())

    async def cleanup_completed_pipelines(self) -> int:
        """Clean up completed pipeline states.

        Returns:
            Number of pipelines cleaned up
        """
        try:
            cleanup_count = 0
            pipelines_to_remove = []

            for pipeline_id, pipeline_state in self.active_pipelines.items():
                if pipeline_state.status in [
                    PipelineStatus.COMPLETED,
                    PipelineStatus.FAILED,
                ]:
                    # Check if pipeline is old enough to clean up
                    if pipeline_state.end_time:
                        end_time = datetime.fromisoformat(pipeline_state.end_time)
                        cutoff_time = datetime.now(timezone.utc).timestamp() - (
                            self.retention_days * 24 * 3600
                        )

                        if end_time.timestamp() < cutoff_time:
                            pipelines_to_remove.append(pipeline_id)

            # Remove old pipelines
            for pipeline_id in pipelines_to_remove:
                await self._cleanup_pipeline(pipeline_id)
                cleanup_count += 1

            logger.info(f"Cleaned up {cleanup_count} completed pipelines")
            return cleanup_count

        except Exception as e:
            logger.error(f"Failed to cleanup completed pipelines: {e}")
            return 0

    async def _complete_pipeline(self, pipeline_id: str) -> None:
        """Mark pipeline as completed."""
        pipeline_state = self.active_pipelines[pipeline_id]
        pipeline_state.status = PipelineStatus.COMPLETED
        pipeline_state.end_time = datetime.now(timezone.utc).isoformat()

        # Stop auto-checkpoint
        await self._stop_auto_checkpoint(pipeline_id)

        # Save final checkpoint
        await self.save_checkpoint(pipeline_id, "completion", {})

        logger.info(f"Pipeline {pipeline_id} completed successfully")

    async def _fail_pipeline(self, pipeline_id: str, error_message: str) -> None:
        """Mark pipeline as failed."""
        pipeline_state = self.active_pipelines[pipeline_id]
        pipeline_state.status = PipelineStatus.FAILED
        pipeline_state.end_time = datetime.now(timezone.utc).isoformat()
        pipeline_state.error_message = error_message

        # Stop auto-checkpoint
        await self._stop_auto_checkpoint(pipeline_id)

        # Save final checkpoint
        await self.save_checkpoint(pipeline_id, "failure", {"error": error_message})

        logger.error(f"Pipeline {pipeline_id} failed: {error_message}")

    async def _start_auto_checkpoint(self, pipeline_id: str) -> None:
        """Start auto-checkpoint task for a pipeline."""
        if pipeline_id in self.checkpoint_tasks:
            return

        async def checkpoint_loop():
            while pipeline_id in self.active_pipelines:
                try:
                    await asyncio.sleep(self.checkpoint_interval)

                    pipeline_state = self.active_pipelines.get(pipeline_id)
                    if (
                        pipeline_state
                        and pipeline_state.status == PipelineStatus.RUNNING
                    ):
                        await self.save_checkpoint(
                            pipeline_id, pipeline_state.current_stage or "unknown", {}
                        )

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.warning(f"Auto-checkpoint failed for {pipeline_id}: {e}")

        task = asyncio.create_task(checkpoint_loop())
        self.checkpoint_tasks[pipeline_id] = task

    async def _stop_auto_checkpoint(self, pipeline_id: str) -> None:
        """Stop auto-checkpoint task for a pipeline."""
        if pipeline_id in self.checkpoint_tasks:
            task = self.checkpoint_tasks[pipeline_id]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            del self.checkpoint_tasks[pipeline_id]

    async def _save_checkpoint_async(
        self, file_path: Path, checkpoint: CheckpointData
    ) -> None:
        """Save checkpoint asynchronously."""
        try:
            import aiofiles

            async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(asdict(checkpoint), indent=2, default=str))
        except ImportError:
            await self._save_checkpoint_sync(file_path, checkpoint)

    async def _save_checkpoint_sync(
        self, file_path: Path, checkpoint: CheckpointData
    ) -> None:
        """Save checkpoint synchronously."""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(asdict(checkpoint), f, indent=2, default=str)

    async def _load_checkpoint(self, file_path: Path) -> Optional[CheckpointData]:
        """Load checkpoint from file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Reconstruct checkpoint data
            pipeline_state_data = data["pipeline_state"]

            # Reconstruct stage states
            stages = {}
            for stage_name, stage_data in pipeline_state_data["stages"].items():
                stages[stage_name] = StageState(**stage_data)

            pipeline_state_data["stages"] = stages
            pipeline_state_data["status"] = PipelineStatus(
                pipeline_state_data["status"]
            )

            pipeline_state = PipelineState(**pipeline_state_data)

            checkpoint = CheckpointData(
                pipeline_state=pipeline_state,
                checkpoint_time=data["checkpoint_time"],
                recovery_data=data["recovery_data"],
                file_references=data["file_references"],
            )

            return checkpoint

        except Exception as e:
            logger.error(f"Failed to load checkpoint from {file_path}: {e}")
            return None

    async def _cleanup_pipeline(self, pipeline_id: str) -> None:
        """Clean up pipeline state and files."""
        # Remove from active pipelines
        if pipeline_id in self.active_pipelines:
            del self.active_pipelines[pipeline_id]

        # Stop auto-checkpoint
        await self._stop_auto_checkpoint(pipeline_id)

        # Remove checkpoint file
        checkpoint_file = self.state_dir / f"{pipeline_id}_checkpoint.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()
