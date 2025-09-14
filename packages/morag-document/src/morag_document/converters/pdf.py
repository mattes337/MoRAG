"""PDF document converter using markitdown."""

from pathlib import Path
from typing import Set

import structlog

from .markitdown_base import MarkitdownConverter

logger = structlog.get_logger(__name__)


class PDFConverter(MarkitdownConverter):
    """PDF document converter using markitdown framework."""

    def __init__(self):
        """Initialize PDF converter."""
        super().__init__()
        self.supported_formats: Set[str] = {"pdf"}

    async def supports_format(self, format_type: str) -> bool:
        """Check if format is supported.

        Args:
            format_type: Format type string

        Returns:
            True if format is supported, False otherwise
        """
        return format_type.lower() == "pdf"
