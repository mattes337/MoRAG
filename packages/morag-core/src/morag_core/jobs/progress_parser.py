"""Progress event parser for extracting progress information from log messages."""

import re
import json
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timezone
import structlog

logger = structlog.get_logger(__name__)


class ProgressEvent:
    """Represents a parsed progress event."""
    
    def __init__(
        self,
        percentage: int,
        message: str,
        timestamp: Optional[datetime] = None,
        logger_name: Optional[str] = None,
        level: str = "info",
        raw_event: Optional[Dict[str, Any]] = None
    ):
        self.percentage = percentage
        self.message = message
        self.timestamp = timestamp or datetime.now(timezone.utc)
        self.logger_name = logger_name
        self.level = level
        self.raw_event = raw_event or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "percentage": self.percentage,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "logger_name": self.logger_name,
            "level": self.level,
            "raw_event": self.raw_event
        }


class ProgressEventParser:
    """Parser for extracting progress information from various log formats."""
    
    # Regex patterns for different progress message formats
    PROGRESS_PATTERNS = [
        # Pattern: "Processing progress: Audio processing: Initializing audio processing (52%)"
        r"Processing progress:\s*(.+?)\s*\((\d+)%\)",
        
        # Pattern: "Processing... 75%"
        r"Processing\.\.\.\s*(\d+)%",
        
        # Pattern: "Progress: 45% - Converting video"
        r"Progress:\s*(\d+)%\s*-\s*(.+)",
        
        # Pattern: "Stage: transcription (67%)"
        r"Stage:\s*(.+?)\s*\((\d+)%\)",
        
        # Pattern: "Audio processing: 80% complete"
        r"(.+?):\s*(\d+)%\s*complete",
        
        # Pattern: "[75%] Processing audio content"
        r"\[(\d+)%\]\s*(.+)",
    ]
    
    def __init__(self):
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.PROGRESS_PATTERNS]
    
    def parse_json_log(self, log_line: str) -> Optional[ProgressEvent]:
        """Parse a JSON-formatted log line for progress information."""
        try:
            log_data = json.loads(log_line.strip())
            
            # Extract basic fields
            event_message = log_data.get("event", "")
            logger_name = log_data.get("logger", "")
            level = log_data.get("level", "info")
            timestamp_str = log_data.get("timestamp", "")
            
            # Parse timestamp
            timestamp = None
            if timestamp_str:
                try:
                    # Handle ISO format with Z suffix
                    if timestamp_str.endswith('Z'):
                        timestamp_str = timestamp_str[:-1] + '+00:00'
                    timestamp = datetime.fromisoformat(timestamp_str)
                except ValueError:
                    logger.warning("Failed to parse timestamp", timestamp=timestamp_str)
            
            # Try to extract progress information from the event message
            progress_info = self._extract_progress_from_message(event_message)
            if progress_info:
                percentage, message = progress_info
                return ProgressEvent(
                    percentage=percentage,
                    message=message,
                    timestamp=timestamp,
                    logger_name=logger_name,
                    level=level,
                    raw_event=log_data
                )
            
            return None
            
        except json.JSONDecodeError:
            # Not a JSON log line, try plain text parsing
            return self.parse_plain_text_log(log_line)
        except Exception as e:
            logger.warning("Failed to parse JSON log line", error=str(e), log_line=log_line[:100])
            return None
    
    def parse_plain_text_log(self, log_line: str) -> Optional[ProgressEvent]:
        """Parse a plain text log line for progress information."""
        try:
            progress_info = self._extract_progress_from_message(log_line)
            if progress_info:
                percentage, message = progress_info
                return ProgressEvent(
                    percentage=percentage,
                    message=message,
                    raw_event={"original_line": log_line}
                )
            return None
        except Exception as e:
            logger.warning("Failed to parse plain text log line", error=str(e))
            return None
    
    def _extract_progress_from_message(self, message: str) -> Optional[Tuple[int, str]]:
        """Extract progress percentage and description from a message."""
        if not message:
            return None
        
        for pattern in self.compiled_patterns:
            match = pattern.search(message)
            if match:
                groups = match.groups()
                
                if len(groups) == 1:
                    # Only percentage found
                    try:
                        percentage = int(groups[0])
                        # Extract message context around the percentage
                        context = message.replace(f"({percentage}%)", "").replace(f"{percentage}%", "").strip()
                        return percentage, context or f"Processing at {percentage}%"
                    except ValueError:
                        continue
                
                elif len(groups) == 2:
                    # Both message and percentage found
                    try:
                        # Try both orders: (message, percentage) and (percentage, message)
                        if groups[1].isdigit():
                            percentage = int(groups[1])
                            description = groups[0].strip()
                        elif groups[0].isdigit():
                            percentage = int(groups[0])
                            description = groups[1].strip()
                        else:
                            continue
                        
                        # Validate percentage range
                        if 0 <= percentage <= 100:
                            return percentage, description or f"Processing at {percentage}%"
                    except ValueError:
                        continue
        
        return None
    
    def parse_log_stream(self, log_lines: list) -> list[ProgressEvent]:
        """Parse multiple log lines and return all progress events found."""
        events = []
        for line in log_lines:
            if isinstance(line, str) and line.strip():
                event = self.parse_json_log(line)
                if event:
                    events.append(event)
        return events
    
    def get_latest_progress(self, log_lines: list) -> Optional[ProgressEvent]:
        """Get the most recent progress event from a list of log lines."""
        events = self.parse_log_stream(log_lines)
        if events:
            # Return the event with the latest timestamp
            return max(events, key=lambda e: e.timestamp)
        return None
