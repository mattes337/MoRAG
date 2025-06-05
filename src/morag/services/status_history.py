"""Status history service for tracking task lifecycle events."""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import structlog
from dataclasses import dataclass, asdict
from enum import Enum

import redis
from morag_core.config import settings

logger = structlog.get_logger()

@dataclass
class StatusEvent:
    """Represents a status change event."""
    timestamp: datetime
    status: str
    progress: Optional[float]
    message: Optional[str]
    metadata: Optional[Dict[str, Any]]

class StatusHistory:
    """Manages task status history using Redis."""
    
    def __init__(self):
        self.redis_client = redis.from_url(settings.redis_url)
        self.history_ttl = 86400 * 7  # 7 days
    
    def add_status_event(
        self,
        task_id: str,
        status: str,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a status event to task history."""
        
        event = StatusEvent(
            timestamp=datetime.utcnow(),
            status=status,
            progress=progress,
            message=message,
            metadata=metadata
        )
        
        try:
            # Store as JSON in Redis list
            event_data = {
                'timestamp': event.timestamp.isoformat(),
                'status': event.status,
                'progress': event.progress,
                'message': event.message,
                'metadata': event.metadata
            }
            
            key = f"task_history:{task_id}"
            self.redis_client.lpush(key, json.dumps(event_data))
            self.redis_client.expire(key, self.history_ttl)
            
            # Keep only last 100 events
            self.redis_client.ltrim(key, 0, 99)
            
            logger.debug(
                "Status event added",
                task_id=task_id,
                status=status,
                progress=progress,
                message=message
            )
            
        except Exception as e:
            logger.error("Failed to add status event", task_id=task_id, error=str(e))
    
    def get_task_history(self, task_id: str) -> List[StatusEvent]:
        """Get complete status history for a task."""
        
        try:
            key = f"task_history:{task_id}"
            events_data = self.redis_client.lrange(key, 0, -1)
            
            events = []
            for event_json in events_data:
                try:
                    event_dict = json.loads(event_json)
                    event = StatusEvent(
                        timestamp=datetime.fromisoformat(event_dict['timestamp']),
                        status=event_dict['status'],
                        progress=event_dict['progress'],
                        message=event_dict['message'],
                        metadata=event_dict['metadata']
                    )
                    events.append(event)
                except Exception as e:
                    logger.warning("Failed to parse status event", error=str(e))
            
            # Sort by timestamp (newest first)
            events.sort(key=lambda x: x.timestamp, reverse=True)
            return events
            
        except Exception as e:
            logger.error("Failed to get task history", task_id=task_id, error=str(e))
            return []
    
    def get_recent_events(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent status events across all tasks."""
        
        try:
            # This is a simplified implementation
            # In production, you might want to use Redis Streams or a proper time-series DB
            
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            all_events = []
            
            # Get all task history keys
            keys = self.redis_client.keys("task_history:*")
            
            for key in keys:
                task_id = key.decode().split(":", 1)[1]
                events = self.get_task_history(task_id)
                
                for event in events:
                    if event.timestamp >= cutoff_time:
                        event_dict = asdict(event)
                        event_dict['task_id'] = task_id
                        event_dict['timestamp'] = event.timestamp.isoformat()
                        all_events.append(event_dict)
            
            # Sort by timestamp
            all_events.sort(key=lambda x: x['timestamp'], reverse=True)
            return all_events[:100]  # Limit to 100 most recent
            
        except Exception as e:
            logger.error("Failed to get recent events", error=str(e))
            return []
    
    def cleanup_old_history(self, days: int = 7) -> int:
        """Clean up old task history entries."""
        
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=days)
            cleaned_count = 0
            
            # Get all task history keys
            keys = self.redis_client.keys("task_history:*")
            
            for key in keys:
                # Check if key has any recent events
                events_data = self.redis_client.lrange(key, 0, -1)
                has_recent = False
                
                for event_json in events_data:
                    try:
                        event_dict = json.loads(event_json)
                        event_time = datetime.fromisoformat(event_dict['timestamp'])
                        if event_time >= cutoff_time:
                            has_recent = True
                            break
                    except Exception:
                        continue
                
                # Delete key if no recent events
                if not has_recent:
                    self.redis_client.delete(key)
                    cleaned_count += 1
            
            logger.info("Cleaned up old task history", cleaned_count=cleaned_count)
            return cleaned_count
            
        except Exception as e:
            logger.error("Failed to cleanup old history", error=str(e))
            return 0

# Global instance
status_history = StatusHistory()
