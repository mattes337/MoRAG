"""Intermediate file management system for pipeline debugging and continuation."""

import asyncio
import json
import time
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import structlog

from morag_core.config import get_settings
from morag_core.exceptions import ProcessingError

logger = structlog.get_logger(__name__)


@dataclass
class FileMetadata:
    """Metadata for intermediate files."""
    stage: str
    source_id: str
    timestamp: str
    file_format: str
    file_size: int
    checksum: str
    version: int
    pipeline_id: str
    processing_time: float
    metadata: Dict[str, Any]


@dataclass
class StageOutput:
    """Output from a processing stage."""
    data: Any
    metadata: FileMetadata
    file_path: Path


class IntermediateFileManager:
    """Manages intermediate files for pipeline debugging and continuation."""
    
    def __init__(
        self,
        base_dir: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the intermediate file manager.
        
        Args:
            base_dir: Base directory for intermediate files
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.settings = get_settings()
        
        # Directory configuration
        self.base_dir = base_dir or Path("intermediate_files")
        self.base_dir.mkdir(exist_ok=True)
        
        # File management settings
        self.max_versions = self.config.get('max_versions', 5)
        self.retention_days = self.config.get('retention_days', 30)
        self.enable_compression = self.config.get('enable_compression', False)
        self.enable_checksums = self.config.get('enable_checksums', True)
        
        # File format settings
        self.default_format = self.config.get('default_format', 'json')
        self.supported_formats = {'json', 'markdown', 'txt', 'yaml'}
        
        # Performance settings
        self.async_writes = self.config.get('async_writes', True)
        self.buffer_size = self.config.get('buffer_size', 8192)
        
        # Initialize metadata tracking
        self.metadata_file = self.base_dir / "metadata.json"
        self.file_registry = self._load_file_registry()
        
        logger.info(
            "Intermediate file manager initialized",
            base_dir=str(self.base_dir),
            max_versions=self.max_versions,
            retention_days=self.retention_days,
            enable_compression=self.enable_compression
        )
    
    async def save_stage_output(
        self,
        stage: str,
        data: Any,
        source_id: str,
        pipeline_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        file_format: Optional[str] = None
    ) -> Path:
        """Save intermediate output with proper naming and metadata.
        
        Args:
            stage: Processing stage name
            data: Data to save
            source_id: Source identifier
            pipeline_id: Pipeline execution identifier
            metadata: Optional additional metadata
            file_format: Optional file format override
            
        Returns:
            Path to saved file
        """
        try:
            start_time = time.time()
            
            # Determine file format
            format_to_use = file_format or self.default_format
            if format_to_use not in self.supported_formats:
                format_to_use = self.default_format
            
            # Create safe filename
            safe_source_id = self._sanitize_filename(source_id)
            safe_stage = self._sanitize_filename(stage)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            
            # Get next version number
            version = self._get_next_version(stage, safe_source_id, pipeline_id)
            
            filename = f"{safe_source_id}_{safe_stage}_v{version:03d}_{timestamp}.{format_to_use}"
            file_path = self.base_dir / pipeline_id / filename
            
            # Create directory if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Serialize and save data
            serialized_data = await self._serialize_data(data, format_to_use)
            
            if self.async_writes:
                await self._write_file_async(file_path, serialized_data)
            else:
                await self._write_file_sync(file_path, serialized_data)
            
            # Calculate file metadata
            file_size = file_path.stat().st_size
            checksum = await self._calculate_checksum(file_path) if self.enable_checksums else ""
            processing_time = time.time() - start_time
            
            # Create metadata
            file_metadata = FileMetadata(
                stage=stage,
                source_id=source_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                file_format=format_to_use,
                file_size=file_size,
                checksum=checksum,
                version=version,
                pipeline_id=pipeline_id,
                processing_time=processing_time,
                metadata=metadata or {}
            )
            
            # Register file
            await self._register_file(file_path, file_metadata)
            
            logger.info(
                "Stage output saved successfully",
                stage=stage,
                source_id=source_id,
                file_path=str(file_path),
                file_size=file_size,
                version=version
            )
            
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to save stage output: {e}")
            raise ProcessingError(f"Failed to save stage output for {stage}: {e}")
    
    async def load_stage_output(
        self,
        stage: str,
        source_id: str,
        pipeline_id: str,
        version: Optional[int] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Load intermediate output for pipeline continuation.
        
        Args:
            stage: Processing stage name
            source_id: Source identifier
            pipeline_id: Pipeline execution identifier
            version: Optional specific version to load (latest if None)
            
        Returns:
            Tuple of (data, metadata)
        """
        try:
            # Find the file
            file_path, file_metadata = await self._find_stage_file(
                stage, source_id, pipeline_id, version
            )
            
            if not file_path or not file_path.exists():
                raise FileNotFoundError(
                    f"No intermediate file found for stage {stage}, source {source_id}"
                )
            
            # Verify checksum if enabled
            if self.enable_checksums and file_metadata.checksum:
                current_checksum = await self._calculate_checksum(file_path)
                if current_checksum != file_metadata.checksum:
                    logger.warning(
                        "Checksum mismatch for intermediate file",
                        file_path=str(file_path),
                        expected=file_metadata.checksum,
                        actual=current_checksum
                    )
            
            # Load and deserialize data
            data = await self._load_and_deserialize(file_path, file_metadata.file_format)
            
            logger.info(
                "Stage output loaded successfully",
                stage=stage,
                source_id=source_id,
                file_path=str(file_path),
                version=file_metadata.version
            )
            
            return data, asdict(file_metadata)
            
        except Exception as e:
            logger.error(f"Failed to load stage output: {e}")
            raise ProcessingError(f"Failed to load stage output for {stage}: {e}")
    
    async def list_stage_files(
        self,
        stage: Optional[str] = None,
        source_id: Optional[str] = None,
        pipeline_id: Optional[str] = None
    ) -> List[FileMetadata]:
        """List intermediate files matching the criteria.
        
        Args:
            stage: Optional stage filter
            source_id: Optional source ID filter
            pipeline_id: Optional pipeline ID filter
            
        Returns:
            List of file metadata matching the criteria
        """
        matching_files = []
        
        for file_path, metadata in self.file_registry.items():
            if stage and metadata.stage != stage:
                continue
            if source_id and metadata.source_id != source_id:
                continue
            if pipeline_id and metadata.pipeline_id != pipeline_id:
                continue
            
            matching_files.append(metadata)
        
        # Sort by timestamp (newest first)
        matching_files.sort(key=lambda x: x.timestamp, reverse=True)
        
        return matching_files
    
    async def cleanup_old_files(self) -> int:
        """Clean up old intermediate files based on retention policy.
        
        Returns:
            Number of files cleaned up
        """
        try:
            cleanup_count = 0
            cutoff_time = datetime.now(timezone.utc).timestamp() - (self.retention_days * 24 * 3600)
            
            files_to_remove = []
            
            for file_path_str, metadata in self.file_registry.items():
                file_timestamp = datetime.fromisoformat(metadata.timestamp).timestamp()
                
                if file_timestamp < cutoff_time:
                    files_to_remove.append(file_path_str)
            
            # Remove old files
            for file_path_str in files_to_remove:
                try:
                    file_path = Path(file_path_str)
                    if file_path.exists():
                        file_path.unlink()
                    
                    # Remove from registry
                    del self.file_registry[file_path_str]
                    cleanup_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to remove old file {file_path_str}: {e}")
            
            # Save updated registry
            await self._save_file_registry()
            
            logger.info(f"Cleaned up {cleanup_count} old intermediate files")
            return cleanup_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old files: {e}")
            return 0
    
    def _sanitize_filename(self, name: str) -> str:
        """Sanitize a string for use in filenames."""
        import re
        
        # Replace invalid characters with underscores
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)
        
        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        
        # Trim and limit length
        sanitized = sanitized.strip('_')[:50]
        
        return sanitized or "unnamed"
    
    def _get_next_version(self, stage: str, source_id: str, pipeline_id: str) -> int:
        """Get the next version number for a file."""
        max_version = 0
        
        for metadata in self.file_registry.values():
            if (metadata.stage == stage and 
                metadata.source_id == source_id and 
                metadata.pipeline_id == pipeline_id):
                max_version = max(max_version, metadata.version)
        
        return max_version + 1
    
    async def _serialize_data(self, data: Any, file_format: str) -> str:
        """Serialize data to the specified format."""
        if file_format == 'json':
            return json.dumps(data, indent=2, default=str, ensure_ascii=False)
        elif file_format == 'yaml':
            try:
                import yaml
                return yaml.dump(data, default_flow_style=False, allow_unicode=True)
            except ImportError:
                logger.warning("PyYAML not available, falling back to JSON")
                return json.dumps(data, indent=2, default=str, ensure_ascii=False)
        elif file_format == 'markdown':
            if isinstance(data, str):
                return data
            else:
                return f"```json\n{json.dumps(data, indent=2, default=str)}\n```"
        else:  # txt
            return str(data)
    
    async def _write_file_async(self, file_path: Path, content: str) -> None:
        """Write file asynchronously."""
        try:
            import aiofiles
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(content)
        except ImportError:
            # Fallback to sync write if aiofiles not available
            await self._write_file_sync(file_path, content)
    
    async def _write_file_sync(self, file_path: Path, content: str) -> None:
        """Write file synchronously."""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    async def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of a file."""
        hash_sha256 = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(self.buffer_size), b""):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    async def _find_stage_file(
        self,
        stage: str,
        source_id: str,
        pipeline_id: str,
        version: Optional[int]
    ) -> Tuple[Optional[Path], Optional[FileMetadata]]:
        """Find a stage file matching the criteria."""
        matching_files = []
        
        for file_path_str, metadata in self.file_registry.items():
            if (metadata.stage == stage and 
                metadata.source_id == source_id and 
                metadata.pipeline_id == pipeline_id):
                
                if version is None or metadata.version == version:
                    matching_files.append((Path(file_path_str), metadata))
        
        if not matching_files:
            return None, None
        
        # If version not specified, return the latest
        if version is None:
            matching_files.sort(key=lambda x: x[1].version, reverse=True)
        
        return matching_files[0]
    
    async def _load_and_deserialize(self, file_path: Path, file_format: str) -> Any:
        """Load and deserialize data from file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if file_format == 'json':
            return json.loads(content)
        elif file_format == 'yaml':
            try:
                import yaml
                return yaml.safe_load(content)
            except ImportError:
                logger.warning("PyYAML not available, treating as text")
                return content
        else:  # markdown, txt
            return content
    
    async def _register_file(self, file_path: Path, metadata: FileMetadata) -> None:
        """Register a file in the registry."""
        self.file_registry[str(file_path)] = metadata
        await self._save_file_registry()
    
    def _load_file_registry(self) -> Dict[str, FileMetadata]:
        """Load the file registry from disk."""
        if not self.metadata_file.exists():
            return {}
        
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            registry = {}
            for file_path, metadata_dict in data.items():
                registry[file_path] = FileMetadata(**metadata_dict)
            
            return registry
            
        except Exception as e:
            logger.warning(f"Failed to load file registry: {e}")
            return {}
    
    async def _save_file_registry(self) -> None:
        """Save the file registry to disk."""
        try:
            registry_data = {}
            for file_path, metadata in self.file_registry.items():
                registry_data[file_path] = asdict(metadata)
            
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(registry_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.warning(f"Failed to save file registry: {e}")
