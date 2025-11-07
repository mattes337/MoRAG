"""Text document converter using markitdown."""

from pathlib import Path
from typing import Set

import structlog

from .markitdown_base import MarkitdownConverter

logger = structlog.get_logger(__name__)


class TextConverter(MarkitdownConverter):
    """Text document converter using markitdown framework.

    This converter handles plain text, markdown, and HTML files using the
    markitdown framework for consistent processing across all formats.
    """

    def __init__(self):
        """Initialize Text converter."""
        super().__init__()
        self.supported_formats: Set[str] = {"text", "txt", "markdown", "md", "html", "htm"}

    async def supports_format(self, format_type: str) -> bool:
        """Check if format is supported.

        Args:
            format_type: Format type string

        Returns:
            True if format is supported, False otherwise
        """
        return format_type.lower() in {"text", "txt", "markdown", "md", "html", "htm"}
