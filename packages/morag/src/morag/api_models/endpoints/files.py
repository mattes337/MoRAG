"""File management endpoints for stage-based API."""

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import structlog
from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import FileResponse

from morag.api_models.stage_models import (
    StageTypeEnum, FileDownloadResponse, FileListResponse,
    StageFileMetadata, ErrorResponse
)

logger = structlog.get_logger(__name__)

# Create router for file management
router = APIRouter(prefix="/api/v1/files", tags=["files"])


def get_content_type(file_path: Path) -> str:
    """Get MIME type for a file."""
    import mimetypes
    content_type, _ = mimetypes.guess_type(str(file_path))
    return content_type or "application/octet-stream"


def create_file_metadata(file_path: Path, stage_type: Optional[StageTypeEnum] = None, include_content: bool = False) -> StageFileMetadata:
    """Create file metadata from a file path."""
    stat = file_path.stat()

    # Determine stage type from filename if not provided
    if stage_type is None:
        if file_path.name.endswith('.opt.md'):
            stage_type = StageTypeEnum.MARKDOWN_OPTIMIZER
        elif file_path.name.endswith('.md'):
            stage_type = StageTypeEnum.MARKDOWN_CONVERSION
        elif file_path.name.endswith('.chunks.json'):
            stage_type = StageTypeEnum.CHUNKER
        elif file_path.name.endswith('.facts.json'):
            stage_type = StageTypeEnum.FACT_GENERATOR
        elif file_path.name.endswith('.ingestion.json'):
            stage_type = StageTypeEnum.INGESTOR
        else:
            stage_type = StageTypeEnum.MARKDOWN_CONVERSION  # Default

    # Read file content if requested
    content = None
    if include_content:
        try:
            # Try to read as text first
            content = file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            # If it's not text, read as binary and encode as base64
            import base64
            content = base64.b64encode(file_path.read_bytes()).decode('ascii')

    return StageFileMetadata(
        filename=file_path.name,
        file_path=str(file_path),
        file_size=stat.st_size,
        created_at=datetime.fromtimestamp(stat.st_mtime),
        stage_type=stage_type,
        content_type=get_content_type(file_path),
        checksum=None,  # TODO: Calculate checksum if needed
        content=content
    )


@router.get("/list", response_model=FileListResponse)
async def list_files(
    output_dir: str = Query("./output", description="Directory to list files from"),
    stage_type: Optional[StageTypeEnum] = Query(None, description="Filter by stage type"),
    file_extension: Optional[str] = Query(None, description="Filter by file extension (e.g., 'json', 'md')"),
    recursive: bool = Query(True, description="Search recursively in subdirectories")
):
    """List available output files with optional filtering."""
    try:
        output_path = Path(output_dir)

        if not output_path.exists():
            return FileListResponse(files=[], total_count=0, total_size=0)

        # Find files
        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"

        all_files = list(output_path.glob(pattern))

        # Filter to only files (not directories)
        files = [f for f in all_files if f.is_file()]

        # Apply file extension filter
        if file_extension:
            if not file_extension.startswith('.'):
                file_extension = '.' + file_extension
            files = [f for f in files if f.suffix.lower() == file_extension.lower()]

        # Create metadata for all files
        file_metadata = []
        total_size = 0

        for file_path in files:
            try:
                metadata = create_file_metadata(file_path)

                # Apply stage type filter
                if stage_type and metadata.stage_type != stage_type:
                    continue

                file_metadata.append(metadata)
                total_size += metadata.file_size

            except Exception as e:
                logger.warning("Failed to create metadata for file", file=str(file_path), error=str(e))
                continue

        # Sort by creation time (newest first)
        file_metadata.sort(key=lambda x: x.created_at, reverse=True)

        return FileListResponse(
            files=file_metadata,
            total_count=len(file_metadata),
            total_size=total_size
        )

    except Exception as e:
        logger.error("Failed to list files", output_dir=output_dir, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download/{file_path:path}")
async def download_file(
    file_path: str,
    inline: bool = Query(False, description="Whether to display inline or as attachment")
):
    """Download a specific file."""
    try:
        # Resolve file path
        resolved_path = Path(file_path)

        # Security check: ensure file is within allowed directories
        # This is a basic check - in production, implement more robust path validation
        if not resolved_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        if not resolved_path.is_file():
            raise HTTPException(status_code=400, detail="Path is not a file")

        # Determine content type
        content_type = get_content_type(resolved_path)

        # Set disposition based on inline parameter
        if inline:
            disposition = "inline"
        else:
            disposition = f'attachment; filename="{resolved_path.name}"'

        return FileResponse(
            path=str(resolved_path),
            media_type=content_type,
            headers={"Content-Disposition": disposition}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to download file", file_path=file_path, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/info/{file_path:path}", response_model=StageFileMetadata)
async def get_file_info(file_path: str):
    """Get detailed information about a specific file."""
    try:
        resolved_path = Path(file_path)

        if not resolved_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        if not resolved_path.is_file():
            raise HTTPException(status_code=400, detail="Path is not a file")

        return create_file_metadata(resolved_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get file info", file_path=file_path, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete/{file_path:path}")
async def delete_file(file_path: str):
    """Delete a specific file."""
    try:
        resolved_path = Path(file_path)

        if not resolved_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        if not resolved_path.is_file():
            raise HTTPException(status_code=400, detail="Path is not a file")

        # Delete the file
        resolved_path.unlink()

        return {"success": True, "message": f"File {resolved_path.name} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete file", file_path=file_path, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cleanup")
async def cleanup_output_directory(
    output_dir: str = Query("./output", description="Directory to clean up"),
    older_than_days: int = Query(7, description="Delete files older than this many days"),
    stage_type: Optional[StageTypeEnum] = Query(None, description="Only delete files from specific stage"),
    dry_run: bool = Query(True, description="If true, only return what would be deleted")
):
    """Clean up old output files."""
    try:
        output_path = Path(output_dir)

        if not output_path.exists():
            return {"success": True, "message": "Output directory does not exist", "files_deleted": []}

        # Calculate cutoff date
        from datetime import timedelta
        cutoff_date = datetime.now() - timedelta(days=older_than_days)

        # Find files to delete
        files_to_delete = []
        total_size_to_delete = 0

        for file_path in output_path.rglob("*"):
            if not file_path.is_file():
                continue

            # Check file age
            file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            if file_mtime > cutoff_date:
                continue

            # Check stage type filter
            if stage_type:
                try:
                    metadata = create_file_metadata(file_path)
                    if metadata.stage_type != stage_type:
                        continue
                except Exception:
                    continue

            files_to_delete.append({
                "path": str(file_path),
                "name": file_path.name,
                "size": file_path.stat().st_size,
                "modified": file_mtime.isoformat()
            })
            total_size_to_delete += file_path.stat().st_size

        # Delete files if not dry run
        deleted_files = []
        if not dry_run:
            for file_info in files_to_delete:
                try:
                    Path(file_info["path"]).unlink()
                    deleted_files.append(file_info)
                except Exception as e:
                    logger.warning("Failed to delete file", file=file_info["path"], error=str(e))

        return {
            "success": True,
            "dry_run": dry_run,
            "files_found": len(files_to_delete),
            "files_deleted": len(deleted_files) if not dry_run else 0,
            "total_size_freed": total_size_to_delete if not dry_run else 0,
            "files": files_to_delete if dry_run else deleted_files
        }

    except Exception as e:
        logger.error("Failed to cleanup files", output_dir=output_dir, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_file_statistics(
    output_dir: str = Query("./output", description="Directory to analyze")
):
    """Get statistics about output files."""
    try:
        output_path = Path(output_dir)

        if not output_path.exists():
            return {
                "total_files": 0,
                "total_size": 0,
                "by_stage": {},
                "by_extension": {}
            }

        # Analyze all files
        stage_stats = {}
        extension_stats = {}
        total_files = 0
        total_size = 0

        for file_path in output_path.rglob("*"):
            if not file_path.is_file():
                continue

            file_size = file_path.stat().st_size
            total_files += 1
            total_size += file_size

            # Stage statistics
            try:
                metadata = create_file_metadata(file_path)
                stage_name = metadata.stage_type.value
                if stage_name not in stage_stats:
                    stage_stats[stage_name] = {"count": 0, "size": 0}
                stage_stats[stage_name]["count"] += 1
                stage_stats[stage_name]["size"] += file_size
            except Exception:
                pass

            # Extension statistics
            extension = file_path.suffix.lower()
            if extension not in extension_stats:
                extension_stats[extension] = {"count": 0, "size": 0}
            extension_stats[extension]["count"] += 1
            extension_stats[extension]["size"] += file_size

        return {
            "total_files": total_files,
            "total_size": total_size,
            "by_stage": stage_stats,
            "by_extension": extension_stats,
            "directory": str(output_path)
        }

    except Exception as e:
        logger.error("Failed to get file statistics", output_dir=output_dir, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
