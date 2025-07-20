"""Ingestion state management for resume/retry capabilities."""

import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class IngestionState:
    """Represents the state of an ingestion process."""
    transaction_id: str
    document_id: str
    source_path: str
    state: str  # 'pending', 'validating', 'validated', 'committing', 'committed', 'failed', 'aborted'
    created_at: datetime
    updated_at: datetime
    chunks_processed: int = 0
    entities_extracted: int = 0
    relations_extracted: int = 0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class IngestionStateManager:
    """Manages ingestion state for resume/retry capabilities."""
    
    def __init__(self, state_dir: Optional[Path] = None):
        self.state_dir = state_dir or Path.cwd() / ".morag" / "ingestion_states"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger.bind(component="ingestion_state_manager")
        self._lock = asyncio.Lock()
    
    async def save_state(self, state: IngestionState) -> None:
        """Save ingestion state to disk."""
        async with self._lock:
            state.updated_at = datetime.now(timezone.utc)
            state_file = self.state_dir / f"{state.transaction_id}.json"
            
            # Convert datetime objects to ISO strings for JSON serialization
            state_dict = asdict(state)
            state_dict['created_at'] = state.created_at.isoformat()
            state_dict['updated_at'] = state.updated_at.isoformat()
            
            try:
                with open(state_file, 'w') as f:
                    json.dump(state_dict, f, indent=2)
                
                self.logger.debug(
                    "Saved ingestion state",
                    transaction_id=state.transaction_id,
                    state=state.state
                )
                
            except Exception as e:
                self.logger.error(
                    "Failed to save ingestion state",
                    transaction_id=state.transaction_id,
                    error=str(e)
                )
                raise
    
    async def load_state(self, transaction_id: str) -> Optional[IngestionState]:
        """Load ingestion state from disk."""
        async with self._lock:
            state_file = self.state_dir / f"{transaction_id}.json"
            
            if not state_file.exists():
                return None
            
            try:
                with open(state_file, 'r') as f:
                    state_dict = json.load(f)
                
                # Convert ISO strings back to datetime objects
                state_dict['created_at'] = datetime.fromisoformat(state_dict['created_at'])
                state_dict['updated_at'] = datetime.fromisoformat(state_dict['updated_at'])
                
                state = IngestionState(**state_dict)
                
                self.logger.debug(
                    "Loaded ingestion state",
                    transaction_id=transaction_id,
                    state=state.state
                )
                
                return state
                
            except Exception as e:
                self.logger.error(
                    "Failed to load ingestion state",
                    transaction_id=transaction_id,
                    error=str(e)
                )
                return None
    
    async def update_state(
        self,
        transaction_id: str,
        state: Optional[str] = None,
        chunks_processed: Optional[int] = None,
        entities_extracted: Optional[int] = None,
        relations_extracted: Optional[int] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update specific fields of an ingestion state."""
        current_state = await self.load_state(transaction_id)
        if not current_state:
            self.logger.warning(
                "Cannot update state - transaction not found",
                transaction_id=transaction_id
            )
            return False
        
        # Update fields if provided
        if state is not None:
            current_state.state = state
        if chunks_processed is not None:
            current_state.chunks_processed = chunks_processed
        if entities_extracted is not None:
            current_state.entities_extracted = entities_extracted
        if relations_extracted is not None:
            current_state.relations_extracted = relations_extracted
        if error_message is not None:
            current_state.error_message = error_message
        if metadata is not None:
            current_state.metadata.update(metadata)
        
        await self.save_state(current_state)
        return True
    
    async def delete_state(self, transaction_id: str) -> bool:
        """Delete ingestion state from disk."""
        async with self._lock:
            state_file = self.state_dir / f"{transaction_id}.json"
            
            try:
                if state_file.exists():
                    state_file.unlink()
                    self.logger.debug(
                        "Deleted ingestion state",
                        transaction_id=transaction_id
                    )
                    return True
                return False
                
            except Exception as e:
                self.logger.error(
                    "Failed to delete ingestion state",
                    transaction_id=transaction_id,
                    error=str(e)
                )
                return False
    
    async def list_states(
        self,
        state_filter: Optional[str] = None,
        max_age_hours: Optional[int] = None
    ) -> List[IngestionState]:
        """List ingestion states with optional filtering."""
        async with self._lock:
            states = []
            
            try:
                for state_file in self.state_dir.glob("*.json"):
                    transaction_id = state_file.stem
                    state = await self.load_state(transaction_id)
                    
                    if state is None:
                        continue
                    
                    # Apply filters
                    if state_filter and state.state != state_filter:
                        continue
                    
                    if max_age_hours:
                        age_hours = (datetime.now(timezone.utc) - state.updated_at).total_seconds() / 3600
                        if age_hours > max_age_hours:
                            continue
                    
                    states.append(state)
                
                # Sort by updated_at descending
                states.sort(key=lambda s: s.updated_at, reverse=True)
                
                return states
                
            except Exception as e:
                self.logger.error("Failed to list ingestion states", error=str(e))
                return []
    
    async def cleanup_old_states(self, max_age_hours: int = 168) -> int:  # 7 days default
        """Clean up old ingestion states."""
        cutoff_time = datetime.now(timezone.utc).timestamp() - (max_age_hours * 3600)
        cleaned_count = 0
        
        async with self._lock:
            try:
                for state_file in self.state_dir.glob("*.json"):
                    # Check file modification time
                    if state_file.stat().st_mtime < cutoff_time:
                        state_file.unlink()
                        cleaned_count += 1
                
                if cleaned_count > 0:
                    self.logger.info(
                        "Cleaned up old ingestion states",
                        count=cleaned_count,
                        max_age_hours=max_age_hours
                    )
                
                return cleaned_count
                
            except Exception as e:
                self.logger.error("Failed to cleanup old states", error=str(e))
                return 0
    
    async def get_failed_states(self, max_age_hours: int = 24) -> List[IngestionState]:
        """Get failed ingestion states for retry."""
        return await self.list_states(state_filter="failed", max_age_hours=max_age_hours)
    
    async def get_pending_states(self, max_age_hours: int = 1) -> List[IngestionState]:
        """Get pending ingestion states that might be stuck."""
        return await self.list_states(state_filter="pending", max_age_hours=max_age_hours)
    
    async def create_state(
        self,
        transaction_id: str,
        document_id: str,
        source_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> IngestionState:
        """Create a new ingestion state."""
        now = datetime.now(timezone.utc)
        state = IngestionState(
            transaction_id=transaction_id,
            document_id=document_id,
            source_path=source_path,
            state="pending",
            created_at=now,
            updated_at=now,
            metadata=metadata or {}
        )
        
        await self.save_state(state)
        return state


# Global state manager instance
_state_manager = IngestionStateManager()

def get_state_manager() -> IngestionStateManager:
    """Get the global ingestion state manager instance."""
    return _state_manager
