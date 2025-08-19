"""File management system for stage outputs with metadata and cleanup policies."""

import hashlib
import json
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


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
                          stage_result,  # StageResult type
                          source_file: Path,
                          context) -> Dict[str, str]:  # StageContext type
        """Store stage output files and return file IDs."""
        
        file_ids = {}
        
        for output_file in stage_result.output_files:
            # Generate unique file ID
            file_id = self._generate_file_id(output_file, stage_result.stage_type)
            
            # Create storage path
            storage_path = self.stage_outputs_dir / file_id
            
            # Copy file to storage
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
                'stage_metadata': stage_result.metadata.__dict__ if hasattr(stage_result.metadata, '__dict__') else {},
                'webhook_url': context.webhook_url if hasattr(context, 'webhook_url') else None
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
    
    def list_files_by_stage(self, stage: str) -> List[Dict[str, Any]]:
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
                logger.warning("Failed to process metadata file", metadata_file=str(metadata_file), error=str(e))
        
        return files_deleted
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        total_files = 0
        total_size = 0
        stage_counts = {}
        
        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                total_files += 1
                total_size += metadata.get('file_size', 0)
                
                stage = metadata.get('stage', 'unknown')
                stage_counts[stage] = stage_counts.get(stage, 0) + 1
                
            except Exception:
                continue
        
        return {
            'total_files': total_files,
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'files_by_stage': stage_counts,
            'storage_directory': str(self.base_storage_dir)
        }
    
    def _generate_file_id(self, file_path: Path, stage_type) -> str:
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
        "markdown-conversion": ".md",
        "markdown-optimizer": ".opt.md",
        "chunker": ".chunks.json",
        "fact-generator": ".facts.json",
        "ingestor": ".ingestion.json"
    }
    
    @classmethod
    def get_stage_output_filename(cls, source_path: Path, stage: str) -> str:
        """Get standardized filename for stage output."""
        base_name = source_path.stem
        extension = cls.STAGE_EXTENSIONS.get(stage, ".out")
        return f"{base_name}{extension}"
    
    @classmethod
    def get_metadata_filename(cls, source_path: Path, stage: str) -> str:
        """Get metadata filename for stage output."""
        base_name = source_path.stem
        return f"{base_name}.{stage}.meta.json"
    
    @classmethod
    def get_report_filename(cls, source_path: Path, stage: str) -> str:
        """Get report filename for stage output."""
        base_name = source_path.stem
        return f"{base_name}.{stage}.report.json"
    
    @classmethod
    def parse_stage_from_filename(cls, filename: str) -> Optional[str]:
        """Parse stage name from filename."""
        for stage, extension in cls.STAGE_EXTENSIONS.items():
            if filename.endswith(extension):
                return stage
        return None


# Global file manager instance
_global_file_manager: Optional[FileManager] = None


def get_file_manager() -> FileManager:
    """Get or create global file manager instance."""
    global _global_file_manager
    if _global_file_manager is None:
        _global_file_manager = FileManager()
    return _global_file_manager


def set_file_manager(file_manager: FileManager):
    """Set global file manager instance."""
    global _global_file_manager
    _global_file_manager = file_manager
