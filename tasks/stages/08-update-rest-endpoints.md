# Task 8: Update REST Endpoints

## Overview
**COMPLETELY REPLACE** REST endpoints with new stage-based processing API. **ALL OLD ENDPOINTS MUST BE REMOVED** - no backwards compatibility.

## Objectives
- Design completely new stage-based REST API using canonical stage names
- Implement individual stage endpoints for each named stage
- Add file download and management endpoints
- **REMOVE ALL OLD ENDPOINTS COMPLETELY** - no backwards compatibility
- **REPLACE ENTIRE API STRUCTURE** - clean slate approach

## Deliverables

### 1. New Stage-Based REST API Design (Complete Replacement)
```python
# packages/morag/src/morag/api_models/endpoints/stages.py
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse
from typing import List, Optional, Dict, Any
from pathlib import Path
import json

from morag_stages import StageManager, StageType
from morag_stages.models import StageContext, StageStatus
from morag.api_models.stage_models import (
    StageExecutionRequest, StageExecutionResponse,
    StageChainRequest, StageChainResponse,
    StageStatusResponse, FileDownloadResponse
)

router = APIRouter(prefix="/api/v1/stages", tags=["stages"])

# Global stage manager instance
stage_manager = StageManager()

@router.post("/markdown-conversion/execute", response_model=StageExecutionResponse)
async def execute_markdown_conversion(
    file: UploadFile = File(...),
    config: Optional[str] = Form(None),
    webhook_url: Optional[str] = Form(None),
    output_dir: Optional[str] = Form("./api_output")
):
    """Execute markdown-conversion stage."""

    try:
        # Save uploaded file
        input_path = await save_uploaded_file(file, "markdown_conversion_input")

        # Parse config
        stage_config = json.loads(config) if config else {}

        # Create context
        context = StageContext(
            source_path=input_path,
            output_dir=Path(output_dir),
            webhook_url=webhook_url,
            config={'markdown_conversion': stage_config}
        )

        # Execute stage
        result = await stage_manager.execute_stage(
            StageType.MARKDOWN_CONVERSION, [input_path], context
        )

        return StageExecutionResponse(
            stage="markdown-conversion",
            status=result.status.value,
            output_files=[str(f) for f in result.output_files],
            metadata=result.metadata,
            execution_time=result.execution_time,
            error_message=result.error_message
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/markdown-optimizer/execute", response_model=StageExecutionResponse)
async def execute_markdown_optimizer(
    markdown_file: UploadFile = File(...),
    config: Optional[str] = Form(None),
    webhook_url: Optional[str] = Form(None),
    output_dir: Optional[str] = Form("./api_output")
):
    """Execute markdown-optimizer stage."""
    
    try:
        # Validate file type
        if not markdown_file.filename.endswith('.md'):
            raise HTTPException(status_code=400, detail="markdown-optimizer requires a .md file")

        # Save uploaded file
        input_path = await save_uploaded_file(markdown_file, "markdown_optimizer_input")

        # Parse config
        stage_config = json.loads(config) if config else {}

        # Create context
        context = StageContext(
            source_path=input_path,
            output_dir=Path(output_dir),
            webhook_url=webhook_url,
            config={'markdown_optimizer': stage_config}
        )

        # Execute stage
        result = await stage_manager.execute_stage(
            StageType.MARKDOWN_OPTIMIZER, [input_path], context
        )

        return StageExecutionResponse(
            stage="markdown-optimizer",
            status=result.status.value,
            output_files=[str(f) for f in result.output_files],
            metadata=result.metadata,
            execution_time=result.execution_time,
            error_message=result.error_message
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chunker/execute", response_model=StageExecutionResponse)
async def execute_chunker(
    markdown_file: UploadFile = File(...),
    config: Optional[str] = Form(None),
    webhook_url: Optional[str] = Form(None),
    output_dir: Optional[str] = Form("./api_output")
):
    """Execute chunker stage."""
    
    try:
        # Validate file type
        if not markdown_file.filename.endswith('.md'):
            raise HTTPException(status_code=400, detail="chunker requires a .md file")

        # Save uploaded file
        input_path = await save_uploaded_file(markdown_file, "chunker_input")

        # Parse config
        stage_config = json.loads(config) if config else {}

        # Create context
        context = StageContext(
            source_path=input_path,
            output_dir=Path(output_dir),
            webhook_url=webhook_url,
            config={'chunker': stage_config}
        )

        # Execute stage
        result = await stage_manager.execute_stage(
            StageType.CHUNKER, [input_path], context
        )

        return StageExecutionResponse(
            stage="chunker",
            status=result.status.value,
            output_files=[str(f) for f in result.output_files],
            metadata=result.metadata,
            execution_time=result.execution_time,
            error_message=result.error_message
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/fact-generator/execute", response_model=StageExecutionResponse)
async def execute_fact_generator(
    chunks_file: UploadFile = File(...),
    config: Optional[str] = Form(None),
    webhook_url: Optional[str] = Form(None),
    output_dir: Optional[str] = Form("./api_output")
):
    """Execute fact-generator stage."""
    
    try:
        # Validate file type
        if not chunks_file.filename.endswith('.chunks.json'):
            raise HTTPException(status_code=400, detail="fact-generator requires a .chunks.json file")

        # Save uploaded file
        input_path = await save_uploaded_file(chunks_file, "fact_generator_input")

        # Parse config
        stage_config = json.loads(config) if config else {}

        # Create context
        context = StageContext(
            source_path=input_path,
            output_dir=Path(output_dir),
            webhook_url=webhook_url,
            config={'fact_generator': stage_config}
        )

        # Execute stage
        result = await stage_manager.execute_stage(
            StageType.FACT_GENERATOR, [input_path], context
        )

        return StageExecutionResponse(
            stage="fact-generator",
            status=result.status.value,
            output_files=[str(f) for f in result.output_files],
            metadata=result.metadata,
            execution_time=result.execution_time,
            error_message=result.error_message
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ingestor/execute", response_model=StageExecutionResponse)
async def execute_ingestor(
    chunks_file: UploadFile = File(...),
    facts_file: UploadFile = File(...),
    config: Optional[str] = Form(None),
    webhook_url: Optional[str] = Form(None),
    output_dir: Optional[str] = Form("./api_output")
):
    """Execute ingestor stage."""
    
    try:
        # Validate file types
        if not chunks_file.filename.endswith('.chunks.json'):
            raise HTTPException(status_code=400, detail="ingestor requires a .chunks.json file")
        if not facts_file.filename.endswith('.facts.json'):
            raise HTTPException(status_code=400, detail="ingestor requires a .facts.json file")

        # Save uploaded files
        chunks_path = await save_uploaded_file(chunks_file, "ingestor_chunks")
        facts_path = await save_uploaded_file(facts_file, "ingestor_facts")

        # Parse config
        stage_config = json.loads(config) if config else {}

        # Create context
        context = StageContext(
            source_path=chunks_path,  # Use chunks file as primary source
            output_dir=Path(output_dir),
            webhook_url=webhook_url,
            config={'ingestor': stage_config}
        )

        # Execute stage
        result = await stage_manager.execute_stage(
            StageType.INGESTOR, [chunks_path, facts_path], context
        )

        return StageExecutionResponse(
            stage="ingestor",
            status=result.status.value,
            output_files=[str(f) for f in result.output_files],
            metadata=result.metadata,
            execution_time=result.execution_time,
            error_message=result.error_message
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chain", response_model=StageChainResponse)
async def execute_stage_chain(request: StageChainRequest):
    """Execute a chain of stages."""
    
    try:
        # Parse stages
        stage_types = [StageType(s) for s in request.stages]
        
        # Create context
        context = StageContext(
            source_path=Path(request.input_file),
            output_dir=Path(request.output_dir),
            webhook_url=request.webhook_url,
            config=request.config or {}
        )
        
        # Execute stage chain
        results = await stage_manager.execute_stage_chain(
            stage_types, [Path(request.input_file)], context
        )
        
        # Convert results to response format
        stage_results = {}
        for stage_type, result in results.items():
            stage_results[stage_type.value] = {
                "status": result.status.value,
                "output_files": [str(f) for f in result.output_files],
                "metadata": result.metadata,
                "execution_time": result.execution_time,
                "error_message": result.error_message
            }

        return StageChainResponse(
            stages_executed=request.stages,
            results=stage_results,
            overall_success=all(r.status == StageStatus.COMPLETED for r in results.values())
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{job_id}", response_model=StageStatusResponse)
async def get_stage_status(job_id: str):
    """Get status of a stage execution job."""
    
    # This would integrate with a job tracking system
    # For now, return a placeholder response
    return StageStatusResponse(
        job_id=job_id,
        status="completed",
        progress=100,
        current_stage=None,
        completed_stages=[1, 2, 3, 4, 5],
        error_message=None
    )

async def save_uploaded_file(file: UploadFile, prefix: str) -> Path:
    """Save uploaded file to temporary location."""
    import tempfile
    import shutil
    
    # Create temporary file
    temp_dir = Path(tempfile.gettempdir()) / "morag_api"
    temp_dir.mkdir(exist_ok=True)
    
    file_path = temp_dir / f"{prefix}_{file.filename}"
    
    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return file_path
```

### 2. File Download and Management Endpoints
```python
# packages/morag/src/morag/api_models/endpoints/files.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
from typing import List

router = APIRouter(prefix="/api/v1/files", tags=["files"])

@router.get("/download/{file_id}")
async def download_file(file_id: str):
    """Download a stage output file by ID."""
    
    # Decode file ID to get actual path
    try:
        file_path = decode_file_id(file_id)
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            path=str(file_path),
            filename=file_path.name,
            media_type='application/octet-stream'
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid file ID: {e}")

@router.get("/list/{output_dir}")
async def list_files(output_dir: str):
    """List all files in an output directory."""
    
    try:
        dir_path = Path(output_dir)
        
        if not dir_path.exists() or not dir_path.is_dir():
            raise HTTPException(status_code=404, detail="Directory not found")
        
        files = []
        for file_path in dir_path.rglob('*'):
            if file_path.is_file():
                files.append({
                    "name": file_path.name,
                    "path": str(file_path),
                    "size": file_path.stat().st_size,
                    "modified": file_path.stat().st_mtime,
                    "file_id": encode_file_id(file_path)
                })
        
        return {"files": files}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/cleanup/{output_dir}")
async def cleanup_files(output_dir: str, max_age_hours: int = 24):
    """Clean up old files in output directory."""
    
    try:
        dir_path = Path(output_dir)
        
        if not dir_path.exists():
            return {"message": "Directory not found", "files_deleted": 0}
        
        import time
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        files_deleted = 0
        for file_path in dir_path.rglob('*'):
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    file_path.unlink()
                    files_deleted += 1
        
        return {"message": f"Cleanup completed", "files_deleted": files_deleted}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def encode_file_id(file_path: Path) -> str:
    """Encode file path as safe file ID."""
    import base64
    return base64.urlsafe_b64encode(str(file_path).encode()).decode()

def decode_file_id(file_id: str) -> Path:
    """Decode file ID back to file path."""
    import base64
    path_str = base64.urlsafe_b64decode(file_id.encode()).decode()
    return Path(path_str)
```

### 3. API Models for Stage Endpoints
```python
# packages/morag/src/morag/api_models/stage_models.py
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class StageExecutionRequest(BaseModel):
    input_file: str
    config: Optional[Dict[str, Any]] = None
    webhook_url: Optional[str] = None
    output_dir: str = "./api_output"

class StageExecutionResponse(BaseModel):
    stage: int
    status: str
    output_files: List[str]
    metadata: Dict[str, Any]
    execution_time: Optional[float]
    error_message: Optional[str]

class StageChainRequest(BaseModel):
    stages: List[int]
    input_file: str
    config: Optional[Dict[str, Any]] = None
    webhook_url: Optional[str] = None
    output_dir: str = "./api_output"

class StageChainResponse(BaseModel):
    stages_executed: List[int]
    results: Dict[str, Dict[str, Any]]
    overall_success: bool

class StageStatusResponse(BaseModel):
    job_id: str
    status: str  # pending, running, completed, failed
    progress: int  # 0-100
    current_stage: Optional[int]
    completed_stages: List[int]
    error_message: Optional[str]

class FileDownloadResponse(BaseModel):
    file_id: str
    filename: str
    size: int
    download_url: str
```

### 4. **NO BACKWARD COMPATIBILITY** - All Old Endpoints Removed
```python
# **ALL OLD ENDPOINTS ARE COMPLETELY REMOVED**
# **NO LEGACY COMPATIBILITY LAYER**
# **NO BACKWARDS COMPATIBILITY**

# The following endpoints are COMPLETELY REMOVED and will return 404:
# - /api/v1/process
# - /api/v1/upload
# - /api/v1/convert
# - /api/v1/extract
# - /api/v1/ingest
# - Any other old endpoints

# Only the new stage-based endpoints exist:
# - /api/v1/stages/markdown-conversion/execute
# - /api/v1/stages/markdown-optimizer/execute
# - /api/v1/stages/chunker/execute
# - /api/v1/stages/fact-generator/execute
# - /api/v1/stages/ingestor/execute
# - /api/v1/stages/chain
# - /api/v1/files/*

# Implementation note: Remove all old endpoint files and router registrations
# Update server.py to only include new stage-based routers
```

## Implementation Steps

1. **REMOVE ALL OLD API ENDPOINTS COMPLETELY**
2. **Create completely new stage-based API endpoints using canonical names**
3. **Implement file upload and download functionality**
4. **Add stage execution endpoints for each named stage**
5. **Create stage chain execution endpoint**
6. **Implement file management endpoints**
7. **Add job status tracking**
8. **UPDATE SERVER.PY TO REMOVE ALL OLD ROUTERS**
9. **Update API documentation for new endpoints only**
10. **Add comprehensive error handling and security**

## Testing Requirements

- Unit tests for all new endpoints
- Integration tests for stage execution
- File upload/download tests
- Stage chain execution tests
- Backward compatibility validation
- Error handling and edge case tests

## Files to Create/Update

- `packages/morag/src/morag/api_models/endpoints/stages.py` (completely new)
- `packages/morag/src/morag/api_models/endpoints/files.py` (completely new)
- `packages/morag/src/morag/api_models/stage_models.py` (completely new)
- **REMOVE** all old endpoint files completely
- **COMPLETELY REPLACE** `packages/morag/src/morag/server.py` with new routers only

## Success Criteria

- All stage endpoints work correctly with file uploads using canonical names
- Stage chain execution works with proper dependency handling
- File download and management endpoints function properly
- **ALL OLD ENDPOINTS ARE COMPLETELY REMOVED** - no backwards compatibility
- Error handling provides clear feedback
- API documentation covers new endpoints only
- Server startup only registers new stage-based routers
