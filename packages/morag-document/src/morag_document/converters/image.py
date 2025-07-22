"""Image document converter using markitdown."""

from pathlib import Path
from typing import Set

import structlog

from .markitdown_base import MarkitdownConverter

logger = structlog.get_logger(__name__)


class ImageConverter(MarkitdownConverter):
    """Image document converter using markitdown framework."""

    def __init__(self):
        """Initialize Image converter."""
        super().__init__()
        self.supported_formats: Set[str] = {
            "image", "jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp", "svg"
        }

    async def supports_format(self, format_type: str) -> bool:
        """Check if format is supported.

        Args:
            format_type: Format type string

        Returns:
            True if format is supported, False otherwise
        """
        return (format_type.lower() in self.supported_formats or
                format_type.lower() == "image")
