"""Stage-based REST API endpoints for MoRAG."""

import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import structlog
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks, Depends, Query
from fastapi.responses import FileResponse

# Add path for stage imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "morag-stages" / "src"))

from morag_stages import StageManager, StageType, StageStatus
from morag_stages.models import StageContext
from morag.api_models.stage_models import (
    StageTypeEnum, StageStatusEnum, StageExecutionRequest, StageExecutionResponse,
    StageChainRequest, StageChainResponse, StageStatusResponse, StageFileMetadata,
    FileDownloadResponse, FileListResponse, StageInfoResponse, StageListResponse,
    ErrorResponse, HealthCheckResponse, StageExecutionMetadata
)
from morag.utils.file_upload import get_upload_handler

logger = structlog.get_logger(__name__)

# Create router with new prefix
router = APIRouter(prefix="/api/v1/stages", tags=["stages"])

# Global stage manager instance
stage_manager = StageManager()

# Background job tracking
background_jobs: Dict[str, Dict[str, Any]] = {}


def convert_stage_type(stage_enum: StageTypeEnum) -> StageType:
    """Convert API enum to internal stage type."""
    mapping = {
        StageTypeEnum.MARKDOWN_CONVERSION: StageType.MARKDOWN_CONVERSION,
        StageTypeEnum.MARKDOWN_OPTIMIZER: StageType.MARKDOWN_OPTIMIZER,
        StageTypeEnum.CHUNKER: StageType.CHUNKER,
        StageTypeEnum.FACT_GENERATOR: StageType.FACT_GENERATOR,
        StageTypeEnum.INGESTOR: StageType.INGESTOR,
    }
    return mapping[stage_enum]


def convert_stage_status(status: StageStatus) -> StageStatusEnum:
    """Convert internal stage status to API enum."""
    mapping = {
        StageStatus.PENDING: StageStatusEnum.NOT_STARTED,
        StageStatus.RUNNING: StageStatusEnum.IN_PROGRESS,
        StageStatus.COMPLETED: StageStatusEnum.COMPLETED,
        StageStatus.FAILED: StageStatusEnum.FAILED,
        StageStatus.SKIPPED: StageStatusEnum.SKIPPED,
    }
    return mapping[status]


def create_file_metadata(file_path: Path, stage_type: StageTypeEnum) -> StageFileMetadata:
    """Create file metadata from a file path."""
    stat = file_path.stat()
    return StageFileMetadata(
        filename=file_path.name,
        file_path=str(file_path),
        file_size=stat.st_size,
        created_at=datetime.fromtimestamp(stat.st_mtime),
        stage_type=stage_type,
        content_type=get_content_type(file_path),
        checksum=None  # TODO: Calculate checksum if needed
    )


def get_content_type(file_path: Path) -> str:
    """Get MIME type for a file."""
    import mimetypes
    content_type, _ = mimetypes.guess_type(str(file_path))
    return content_type or "application/octet-stream"


async def send_webhook_notification(webhook_config, stage_result, job_id: Optional[str] = None):
    """Send webhook notification for stage completion."""
    if not webhook_config:
        return False
    
    try:
        import httpx
        
        payload = {
            "job_id": job_id,
            "stage_type": stage_result.stage_type.value,
            "status": stage_result.status.value,
            "success": stage_result.status in [StageStatus.COMPLETED, StageStatus.SKIPPED],
            "output_files": [str(f) for f in stage_result.output_files],
            "execution_time": stage_result.metadata.execution_time,
            "timestamp": datetime.now().isoformat()
        }
        
        headers = {"Content-Type": "application/json"}
        if webhook_config.auth_token:
            headers["Authorization"] = f"Bearer {webhook_config.auth_token}"
        if webhook_config.headers:
            headers.update(webhook_config.headers)
        
        async with httpx.AsyncClient(timeout=webhook_config.timeout) as client:
            response = await client.post(
                webhook_config.url,
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            return True
            
    except Exception as e:
        logger.error("Failed to send webhook notification", error=str(e), webhook_url=webhook_config.url)
        return False


@router.get("/", response_model=StageListResponse)
async def list_stages():
    """List all available stages with their information."""
    try:
        stages_info = []
        
        stage_descriptions = {
            StageTypeEnum.MARKDOWN_CONVERSION: {
                "description": "Convert input files to unified markdown format",
                "input_formats": ["pdf", "docx", "txt", "mp3", "mp4", "wav", "m4a", "flac", "avi", "mov", "mkv"],
                "output_formats": ["md"],
                "required_config": [],
                "optional_config": ["include_timestamps", "speaker_diarization", "topic_segmentation"],
                "dependencies": []
            },
            StageTypeEnum.MARKDOWN_OPTIMIZER: {
                "description": "LLM-based text improvement and transcription error correction",
                "input_formats": ["md"],
                "output_formats": ["md"],
                "required_config": [],
                "optional_config": ["fix_transcription_errors", "improve_readability", "preserve_timestamps"],
                "dependencies": [StageTypeEnum.MARKDOWN_CONVERSION]
            },
            StageTypeEnum.CHUNKER: {
                "description": "Create summary, chunks, and contextual embeddings",
                "input_formats": ["md"],
                "output_formats": ["json"],
                "required_config": [],
                "optional_config": ["chunk_strategy", "chunk_size", "generate_summary"],
                "dependencies": [StageTypeEnum.MARKDOWN_CONVERSION]
            },
            StageTypeEnum.FACT_GENERATOR: {
                "description": "Extract facts, entities, relations, and keywords",
                "input_formats": ["json"],
                "output_formats": ["json"],
                "required_config": [],
                "optional_config": ["extract_entities", "extract_relations", "domain"],
                "dependencies": [StageTypeEnum.CHUNKER]
            },
            StageTypeEnum.INGESTOR: {
                "description": "Database ingestion and storage",
                "input_formats": ["json"],
                "output_formats": ["json"],
                "required_config": [],
                "optional_config": ["databases", "collection_name"],
                "dependencies": [StageTypeEnum.CHUNKER, StageTypeEnum.FACT_GENERATOR]
            }
        }
        
        for stage_type in StageTypeEnum:
            info = stage_descriptions[stage_type]
            stages_info.append(StageInfoResponse(
                stage_type=stage_type,
                description=info["description"],
                input_formats=info["input_formats"],
                output_formats=info["output_formats"],
                required_config=info["required_config"],
                optional_config=info["optional_config"],
                dependencies=info["dependencies"]
            ))
        
        return StageListResponse(
            stages=stages_info,
            total_count=len(stages_info)
        )
        
    except Exception as e:
        logger.error("Failed to list stages", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{stage_name}/execute", response_model=StageExecutionResponse)
async def execute_stage(
    stage_name: str,
    request: Optional[StageExecutionRequest] = None,
    file: Optional[UploadFile] = File(None),
    # Form data parameters for when file upload is used
    output_dir: Optional[str] = Form("./output"),
    config: Optional[str] = Form(None),
    input_files: Optional[str] = Form(None),
    webhook_url: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Execute a single stage using canonical stage names."""
    try:
        # Validate stage name
        try:
            stage_enum = StageTypeEnum(stage_name)
            stage_type = convert_stage_type(stage_enum)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid stage name: {stage_name}. Valid stages: {[s.value for s in StageTypeEnum]}"
            )

        # Handle request data - either from JSON body or form data
        if request is not None:
            # JSON request body
            request_output_dir = request.output_dir
            request_config = request.config or {}
            request_input_files = request.input_files or []
            request_webhook_url = request.webhook_config.url if request.webhook_config else None
        else:
            # Form data request
            request_output_dir = output_dir or "./output"
            request_config = json.loads(config) if config else {}
            request_input_files = json.loads(input_files) if input_files else []
            request_webhook_url = webhook_url

        # Handle file upload if provided
        input_file_paths = []
        if file:
            upload_handler = get_upload_handler()
            temp_path = await upload_handler.save_upload(file)
            input_file_paths = [temp_path]
        elif request_input_files:
            input_file_paths = [Path(f) for f in request_input_files]
        else:
            raise HTTPException(status_code=400, detail="Either file upload or input_files must be provided")

        # Create stage context
        output_path = Path(request_output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        context = StageContext(
            source_path=input_file_paths[0] if input_file_paths else None,
            output_dir=output_path,
            webhook_url=request_webhook_url,
            config=request_config
        )
        
        # Execute stage
        start_time = datetime.now()
        result = await stage_manager.execute_stage(stage_type, input_file_paths, context)
        end_time = datetime.now()

        # Create response
        output_file_metadata = [
            create_file_metadata(f, stage_enum) for f in result.output_files
        ]

        execution_metadata = StageExecutionMetadata(
            execution_time=result.metadata.execution_time,
            start_time=start_time,
            end_time=end_time,
            input_files=[str(f) for f in input_file_paths],
            config_used=request_config,
            warnings=[]
        )
        
        # Send webhook notification if configured
        webhook_sent = False
        if request is not None and request.webhook_config:
            webhook_sent = await send_webhook_notification(request.webhook_config, result)
        
        return StageExecutionResponse(
            success=result.status in [StageStatus.COMPLETED, StageStatus.SKIPPED],
            stage_type=stage_enum,
            status=convert_stage_status(result.status),
            output_files=output_file_metadata,
            metadata=execution_metadata,
            error_message=result.error_message,
            webhook_sent=webhook_sent
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Stage execution failed", stage=stage_name, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chain", response_model=StageChainResponse)
async def execute_stage_chain(
    request: StageChainRequest,
    file: Optional[UploadFile] = File(None),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Execute a chain of stages using canonical stage names."""
    try:
        # Convert stage enums to internal types
        stage_types = [convert_stage_type(stage_enum) for stage_enum in request.stages]

        # Handle file upload if provided
        input_files = []
        if file:
            upload_handler = get_upload_handler()
            temp_path = await upload_handler.save_upload(file)
            input_files = [temp_path]
        elif request.input_files:
            input_files = [Path(f) for f in request.input_files]
        else:
            raise HTTPException(status_code=400, detail="Either file upload or input_files must be provided")

        # Create stage context
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Merge global and stage-specific configs
        merged_config = request.global_config or {}
        if request.stage_configs:
            for stage_enum, stage_config in request.stage_configs.items():
                stage_key = stage_enum.value
                merged_config[stage_key] = {**(merged_config.get(stage_key, {})), **stage_config}

        context = StageContext(
            source_path=input_files[0] if input_files else None,
            output_dir=output_dir,
            webhook_url=request.webhook_config.url if request and request.webhook_config else None,
            config=merged_config
        )

        # Execute stage chain
        start_time = datetime.now()
        results = await stage_manager.execute_stage_chain(stage_types, input_files, context)
        end_time = datetime.now()

        # Process results
        stage_responses = []
        failed_stage = None
        final_output_files = []

        for i, result in enumerate(results):
            stage_enum = request.stages[i]

            output_file_metadata = [
                create_file_metadata(f, stage_enum) for f in result.output_files
            ]

            execution_metadata = StageExecutionMetadata(
                execution_time=result.metadata.execution_time,
                start_time=start_time,  # TODO: Get actual stage start time
                end_time=end_time,      # TODO: Get actual stage end time
                input_files=[str(f) for f in input_files],
                config_used=merged_config.get(stage_enum.value, {}),
                warnings=[]
            )

            stage_response = StageExecutionResponse(
                success=result.status in [StageStatus.COMPLETED, StageStatus.SKIPPED],
                stage_type=stage_enum,
                status=convert_stage_status(result.status),
                output_files=output_file_metadata,
                metadata=execution_metadata,
                error_message=result.error_message,
                webhook_sent=False  # Individual webhooks not sent in chain mode
            )

            stage_responses.append(stage_response)

            # Track failures
            if result.status not in [StageStatus.COMPLETED, StageStatus.SKIPPED]:
                failed_stage = stage_enum
                if request.stop_on_failure:
                    break

            # Update final output files
            if result.output_files:
                final_output_files = output_file_metadata

        # Calculate total execution time
        total_execution_time = (end_time - start_time).total_seconds()

        # Send webhook notification for chain completion if configured
        if request and request.webhook_config:
            chain_success = failed_stage is None
            await send_webhook_notification_for_chain(
                request.webhook_config, stage_responses, chain_success, total_execution_time
            )

        return StageChainResponse(
            success=failed_stage is None,
            stages_executed=stage_responses,
            total_execution_time=total_execution_time,
            failed_stage=failed_stage,
            final_output_files=final_output_files
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Stage chain execution failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


async def send_webhook_notification_for_chain(webhook_config, stage_responses, success, total_time):
    """Send webhook notification for stage chain completion."""
    try:
        import httpx

        payload = {
            "type": "stage_chain_completion",
            "success": success,
            "total_execution_time": total_time,
            "stages_executed": len(stage_responses),
            "stages_successful": sum(1 for r in stage_responses if r.success),
            "stages_failed": sum(1 for r in stage_responses if not r.success),
            "timestamp": datetime.now().isoformat()
        }

        headers = {"Content-Type": "application/json"}
        if webhook_config.auth_token:
            headers["Authorization"] = f"Bearer {webhook_config.auth_token}"
        if webhook_config.headers:
            headers.update(webhook_config.headers)

        async with httpx.AsyncClient(timeout=webhook_config.timeout) as client:
            response = await client.post(
                webhook_config.url,
                json=payload,
                headers=headers
            )
            response.raise_for_status()

    except Exception as e:
        logger.error("Failed to send chain webhook notification", error=str(e))


@router.get("/status", response_model=StageStatusResponse)
async def get_stage_status(
    output_dir: str = Query("./output", description="Output directory to check"),
    job_id: Optional[str] = Query(None, description="Job ID for background execution")
):
    """Get status of stage execution and available files."""
    try:
        output_path = Path(output_dir)

        # Check what stages have completed based on output files
        stages_completed = []
        available_files = []

        if output_path.exists():
            # Check for stage output files
            md_files = list(output_path.glob("*.md"))
            opt_files = list(output_path.glob("*.opt.md"))
            chunk_files = list(output_path.glob("*.chunks.json"))
            fact_files = list(output_path.glob("*.facts.json"))
            ingestion_files = list(output_path.glob("*.ingestion.json"))

            # Determine completed stages
            if md_files:
                stages_completed.append(StageTypeEnum.MARKDOWN_CONVERSION)
            if opt_files:
                stages_completed.append(StageTypeEnum.MARKDOWN_OPTIMIZER)
            if chunk_files:
                stages_completed.append(StageTypeEnum.CHUNKER)
            if fact_files:
                stages_completed.append(StageTypeEnum.FACT_GENERATOR)
            if ingestion_files:
                stages_completed.append(StageTypeEnum.INGESTOR)

            # Create metadata for all files
            all_files = md_files + opt_files + chunk_files + fact_files + ingestion_files
            for file_path in all_files:
                try:
                    metadata = create_file_metadata(file_path, None)
                    available_files.append(metadata)
                except Exception as e:
                    logger.warning("Failed to create metadata for file", file=str(file_path), error=str(e))

        # Check background job status if job_id provided
        current_stage = None
        overall_status = StageStatusEnum.NOT_STARTED

        if job_id and job_id in background_jobs:
            job_info = background_jobs[job_id]
            current_stage = job_info.get("current_stage")
            overall_status = StageStatusEnum(job_info.get("status", "not_started"))
        else:
            # Determine status based on completed stages
            if len(stages_completed) == 0:
                overall_status = StageStatusEnum.NOT_STARTED
            elif len(stages_completed) == 5:  # All stages completed
                overall_status = StageStatusEnum.COMPLETED
            else:
                overall_status = StageStatusEnum.IN_PROGRESS

        # Calculate progress percentage
        total_stages = 5  # Total number of stages
        progress_percentage = (len(stages_completed) / total_stages) * 100

        return StageStatusResponse(
            job_id=job_id,
            stages_completed=stages_completed,
            current_stage=current_stage,
            available_files=available_files,
            overall_status=overall_status,
            progress_percentage=progress_percentage
        )

    except Exception as e:
        logger.error("Failed to get stage status", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint for the stage-based API."""
    try:
        # Check if stage manager is available
        stages_available = list(StageTypeEnum)

        # Check underlying services
        services_status = {
            "stage_manager": "healthy",
            "file_system": "healthy",
            "background_jobs": "healthy"
        }

        # TODO: Add actual health checks for:
        # - Database connections (Qdrant, Neo4j)
        # - LLM services (Gemini API)
        # - File system permissions
        # - Memory usage

        return HealthCheckResponse(
            status="healthy",
            timestamp=datetime.now(),
            version="1.0.0",  # TODO: Get actual version
            stages_available=stages_available,
            services_status=services_status
        )

    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
