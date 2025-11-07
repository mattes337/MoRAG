# Task 9: Implement File Management and Webhook Support

## Overview
Add file storage and retrieval system, implement webhook notification system, and create file download endpoints with cleanup policies.

## Objectives
- Create file storage and retrieval system
- Implement webhook notification system for stage completions
- Add file download endpoints with secure access
- Create file cleanup and retention policies
- Support file sharing and persistence across processing runs

## Deliverables

### 1. File Management System
```python
# packages/morag-stages/src/morag_stages/file_manager.py
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import hashlib
import time
from datetime import datetime, timedelta

class FileManager:
    """Manages stage output files with metadata and cleanup policies."""

    def __init__(self, base_storage_dir: Path = Path("./morag_storage")):
        self.base_storage_dir = base_storage_dir
        self.base_storage_dir.mkdir(exist_ok=True)

        # Create subdirectories for different file types
        self.stage_outputs_dir = self.base_storage_dir / "stage_outputs"
        self.temp_files_dir = self.base_storage_dir / "temp_files"
        self.metadata_dir = self.base_storage_dir / "metadata"

        for dir_path in [self.stage_outputs_dir, self.temp_files_dir, self.metadata_dir]:
            dir_path.mkdir(exist_ok=True)

    def store_stage_output(self,
                          stage_result: 'StageResult',
                          source_file: Path,
                          context: 'StageContext') -> Dict[str, str]:
        """Store stage output files and return file IDs."""

        file_ids = {}

        for output_file in stage_result.output_files:
            # Generate unique file ID
            file_id = self._generate_file_id(output_file, stage_result.stage_type)

            # Create storage path
            storage_path = self.stage_outputs_dir / file_id

            # Copy file to storage
            import shutil
            shutil.copy2(output_file, storage_path)

            # Store metadata
            metadata = {
                'file_id': file_id,
                'original_name': output_file.name,
                'original_path': str(output_file),
                'storage_path': str(storage_path),
                'stage': stage_result.stage_type.value,
                'stage_name': stage_result.stage_type.name,
                'source_file': str(source_file),
                'created_at': datetime.utcnow().isoformat(),
                'file_size': storage_path.stat().st_size,
                'checksum': self._calculate_checksum(storage_path),
                'stage_metadata': stage_result.metadata,
                'webhook_url': context.webhook_url
            }

            self._store_file_metadata(file_id, metadata)
            file_ids[output_file.name] = file_id

        return file_ids

    def get_file_path(self, file_id: str) -> Optional[Path]:
        """Get file path from file ID."""
        storage_path = self.stage_outputs_dir / file_id

        if storage_path.exists():
            return storage_path

        return None

    def get_file_metadata(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get file metadata from file ID."""
        metadata_path = self.metadata_dir / f"{file_id}.json"

        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)

        return None

    def list_files_by_source(self, source_file: Path) -> List[Dict[str, Any]]:
        """List all files created from a specific source file."""
        files = []

        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                if Path(metadata['source_file']) == source_file:
                    files.append(metadata)
            except Exception:
                continue

        return sorted(files, key=lambda x: x['stage'])

    def list_files_by_stage(self, stage: int) -> List[Dict[str, Any]]:
        """List all files created by a specific stage."""
        files = []

        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                if metadata['stage'] == stage:
                    files.append(metadata)
            except Exception:
                continue

        return sorted(files, key=lambda x: x['created_at'])

    def cleanup_old_files(self, max_age_hours: int = 24) -> int:
        """Clean up files older than specified age."""
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        files_deleted = 0

        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                created_at = datetime.fromisoformat(metadata['created_at'])

                if created_at < cutoff_time:
                    # Delete actual file
                    file_path = Path(metadata['storage_path'])
                    if file_path.exists():
                        file_path.unlink()

                    # Delete metadata
                    metadata_file.unlink()
                    files_deleted += 1

            except Exception as e:
                logger.warning(f"Failed to process metadata file {metadata_file}: {e}")

        return files_deleted

    def _generate_file_id(self, file_path: Path, stage_type: 'StageType') -> str:
        """Generate unique file ID."""
        # Create hash from file path, stage, and timestamp
        content = f"{file_path.name}_{stage_type.value}_{time.time()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file."""
        sha256_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)

        return sha256_hash.hexdigest()

    def _store_file_metadata(self, file_id: str, metadata: Dict[str, Any]):
        """Store file metadata."""
        metadata_path = self.metadata_dir / f"{file_id}.json"

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

class FileNamingConvention:
    """Standardized file naming conventions for stage outputs."""

    STAGE_EXTENSIONS = {
        1: ".md",
        2: ".opt.md",
        3: ".chunks.json",
        4: ".facts.json",
        5: ".ingestion.json"
    }

    @classmethod
    def get_stage_output_filename(cls, source_path: Path, stage: int) -> str:
        """Get standardized filename for stage output."""
        base_name = source_path.stem
        extension = cls.STAGE_EXTENSIONS.get(stage, ".out")
        return f"{base_name}{extension}"

    @classmethod
    def get_metadata_filename(cls, source_path: Path, stage: int) -> str:
        """Get metadata filename for stage output."""
        base_name = source_path.stem
        return f"{base_name}.stage{stage}.meta.json"

    @classmethod
    def get_report_filename(cls, source_path: Path, stage: int) -> str:
        """Get report filename for stage output."""
        base_name = source_path.stem
        return f"{base_name}.stage{stage}.report.json"
```

### 2. Webhook Notification System
```python
# packages/morag-stages/src/morag_stages/webhook.py
import aiohttp
import asyncio
import json
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class WebhookNotifier:
    """Handles webhook notifications for stage completions."""

    def __init__(self,
                 webhook_url: Optional[str] = None,
                 timeout_seconds: int = 30,
                 retry_attempts: int = 3):
        self.webhook_url = webhook_url
        self.timeout_seconds = timeout_seconds
        self.retry_attempts = retry_attempts

    async def notify_stage_completion(self,
                                    stage_result: 'StageResult',
                                    context: 'StageContext',
                                    file_ids: Dict[str, str]):
        """Send webhook notification for stage completion."""

        if not self.webhook_url:
            return

        payload = self._create_notification_payload(stage_result, context, file_ids)

        success = await self._send_webhook_with_retry(payload)

        if success:
            logger.info(f"Webhook notification sent successfully for stage {stage_result.stage_type.value}")
        else:
            logger.error(f"Failed to send webhook notification for stage {stage_result.stage_type.value}")

    async def notify_stage_chain_completion(self,
                                          results: Dict['StageType', 'StageResult'],
                                          context: 'StageContext',
                                          all_file_ids: Dict[str, Dict[str, str]]):
        """Send webhook notification for stage chain completion."""

        if not self.webhook_url:
            return

        payload = self._create_chain_notification_payload(results, context, all_file_ids)

        success = await self._send_webhook_with_retry(payload)

        if success:
            logger.info("Webhook notification sent successfully for stage chain completion")
        else:
            logger.error("Failed to send webhook notification for stage chain completion")

    def _create_notification_payload(self,
                                   stage_result: 'StageResult',
                                   context: 'StageContext',
                                   file_ids: Dict[str, str]) -> Dict[str, Any]:
        """Create webhook payload for single stage completion."""

        return {
            'event_type': 'stage_completed',
            'timestamp': datetime.utcnow().isoformat(),
            'stage': {
                'number': stage_result.stage_type.value,
                'name': stage_result.stage_type.name,
                'status': stage_result.status.value,
                'execution_time': stage_result.execution_time,
                'error_message': stage_result.error_message
            },
            'source_file': str(context.source_path),
            'output_files': [
                {
                    'filename': filename,
                    'file_id': file_id,
                    'download_url': f"/api/v1/files/download/{file_id}"
                }
                for filename, file_id in file_ids.items()
            ],
            'metadata': stage_result.metadata,
            'context': {
                'output_dir': str(context.output_dir),
                'config': context.config
            }
        }

    def _create_chain_notification_payload(self,
                                         results: Dict['StageType', 'StageResult'],
                                         context: 'StageContext',
                                         all_file_ids: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
        """Create webhook payload for stage chain completion."""

        stages_info = []
        for stage_type, result in results.items():
            file_ids = all_file_ids.get(stage_type.name, {})

            stages_info.append({
                'number': stage_type.value,
                'name': stage_type.name,
                'status': result.status.value,
                'execution_time': result.execution_time,
                'error_message': result.error_message,
                'output_files': [
                    {
                        'filename': filename,
                        'file_id': file_id,
                        'download_url': f"/api/v1/files/download/{file_id}"
                    }
                    for filename, file_id in file_ids.items()
                ]
            })

        overall_success = all(r.status.value == 'completed' for r in results.values())
        total_execution_time = sum(r.execution_time or 0 for r in results.values())

        return {
            'event_type': 'stage_chain_completed',
            'timestamp': datetime.utcnow().isoformat(),
            'source_file': str(context.source_path),
            'overall_success': overall_success,
            'total_execution_time': total_execution_time,
            'stages': stages_info,
            'context': {
                'output_dir': str(context.output_dir),
                'config': context.config
            }
        }

    async def _send_webhook_with_retry(self, payload: Dict[str, Any]) -> bool:
        """Send webhook with retry logic."""

        for attempt in range(self.retry_attempts):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.webhook_url,
                        json=payload,
                        headers={'Content-Type': 'application/json'},
                        timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)
                    ) as response:

                        if response.status == 200:
                            return True
                        else:
                            logger.warning(f"Webhook returned status {response.status}, attempt {attempt + 1}")

            except asyncio.TimeoutError:
                logger.warning(f"Webhook timeout on attempt {attempt + 1}")
            except Exception as e:
                logger.warning(f"Webhook error on attempt {attempt + 1}: {e}")

            # Wait before retry (exponential backoff)
            if attempt < self.retry_attempts - 1:
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)

        return False

class WebhookValidator:
    """Validates webhook URLs and payloads."""

    @staticmethod
    def validate_webhook_url(url: str) -> bool:
        """Validate webhook URL format."""
        import re

        # Basic URL validation
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)

        return url_pattern.match(url) is not None

    @staticmethod
    def validate_payload(payload: Dict[str, Any]) -> bool:
        """Validate webhook payload structure."""
        required_fields = ['event_type', 'timestamp', 'source_file']

        for field in required_fields:
            if field not in payload:
                return False

        return True

# Integration with Stage Manager
class StageManagerWithWebhooks:
    """Extended stage manager with webhook support."""

    def __init__(self, file_manager: FileManager):
        self.file_manager = file_manager

    async def execute_stage_with_notifications(self,
                                             stage_type: 'StageType',
                                             input_files: List[Path],
                                             context: 'StageContext') -> 'StageResult':
        """Execute stage and send webhook notifications."""

        # Execute stage
        result = await self.execute_stage(stage_type, input_files, context)

        # Store files and get file IDs
        file_ids = self.file_manager.store_stage_output(result, context.source_path, context)

        # Send webhook notification
        if context.webhook_url:
            webhook_notifier = WebhookNotifier(context.webhook_url)
            await webhook_notifier.notify_stage_completion(result, context, file_ids)

        return result
```

### 3. File Download API Integration
```python
# packages/morag/src/morag/api_models/endpoints/file_downloads.py
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from typing import List, Optional
import mimetypes
from pathlib import Path

from morag_stages.file_manager import FileManager

router = APIRouter(prefix="/api/v1/files", tags=["file_management"])

# Global file manager instance
file_manager = FileManager()

@router.get("/download/{file_id}")
async def download_file(file_id: str):
    """Download a file by its ID."""

    file_path = file_manager.get_file_path(file_id)

    if not file_path:
        raise HTTPException(status_code=404, detail="File not found")

    metadata = file_manager.get_file_metadata(file_id)
    original_name = metadata.get('original_name', file_path.name) if metadata else file_path.name

    # Determine media type
    media_type, _ = mimetypes.guess_type(str(file_path))
    if not media_type:
        media_type = 'application/octet-stream'

    return FileResponse(
        path=str(file_path),
        filename=original_name,
        media_type=media_type
    )

@router.get("/metadata/{file_id}")
async def get_file_metadata(file_id: str):
    """Get metadata for a file."""

    metadata = file_manager.get_file_metadata(file_id)

    if not metadata:
        raise HTTPException(status_code=404, detail="File metadata not found")

    return metadata

@router.get("/list/by-source")
async def list_files_by_source(source_file: str):
    """List all files created from a specific source file."""

    source_path = Path(source_file)
    files = file_manager.list_files_by_source(source_path)

    return {"source_file": source_file, "files": files}

@router.get("/list/by-stage")
async def list_files_by_stage(stage: int = Query(..., ge=1, le=5)):
    """List all files created by a specific stage."""

    files = file_manager.list_files_by_stage(stage)

    return {"stage": stage, "files": files}

@router.delete("/cleanup")
async def cleanup_old_files(max_age_hours: int = Query(24, ge=1)):
    """Clean up files older than specified age."""

    files_deleted = file_manager.cleanup_old_files(max_age_hours)

    return {"message": f"Cleanup completed", "files_deleted": files_deleted}
```

## Implementation Steps

1. **Create file management system**
2. **Implement webhook notification system**
3. **Add file download endpoints**
4. **Create file metadata tracking**
5. **Implement cleanup policies**
6. **Add webhook validation and retry logic**
7. **Create file sharing capabilities**
8. **Add comprehensive error handling**
9. **Implement security measures**
10. **Add monitoring and logging**

## Testing Requirements

- Unit tests for file management operations
- Webhook notification tests with mock endpoints
- File download and metadata tests
- Cleanup policy validation
- Security and access control tests
- Error handling and edge case tests

## Files to Create

- `packages/morag-stages/src/morag_stages/file_manager.py`
- `packages/morag-stages/src/morag_stages/webhook.py`
- `packages/morag/src/morag/api_models/endpoints/file_downloads.py`
- `packages/morag-stages/tests/test_file_manager.py`
- `packages/morag-stages/tests/test_webhook.py`

## Success Criteria

- Files are stored securely with proper metadata
- Webhook notifications are sent reliably for all stage completions
- File download endpoints work with proper security
- Cleanup policies prevent storage bloat
- File sharing works across processing runs
- All tests pass with good coverage
