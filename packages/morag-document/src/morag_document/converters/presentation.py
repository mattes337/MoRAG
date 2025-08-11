"""PowerPoint document converter using markitdown."""

from pathlib import Path
from typing import Set

import structlog

from .markitdown_base import MarkitdownConverter

logger = structlog.get_logger(__name__)


class PresentationConverter(MarkitdownConverter):
    """PowerPoint document converter using markitdown framework."""

    def __init__(self):
        """Initialize PowerPoint converter."""
        super().__init__()
        self.supported_formats: Set[str] = {"powerpoint", "pptx", "ppt"}

    async def supports_format(self, format_type: str) -> bool:
        """Check if format is supported.

        Args:
            format_type: Format type string

        Returns:
            True if format is supported, False otherwise
        """
        return format_type.lower() in {"powerpoint", "pptx", "ppt"}
