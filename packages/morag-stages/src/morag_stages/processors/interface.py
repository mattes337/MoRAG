"""Common interface for stage processors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ProcessorResult:
    """Result from a stage processor."""
    content: str
    metadata: Dict[str, Any]
    metrics: Dict[str, Any]
    final_output_file: Optional[Path] = None


class StageProcessor(ABC):
    """Abstract base class for stage processors."""

    @abstractmethod
    def supports_content_type(self, content_type: str) -> bool:
        """Check if this processor supports the given content type.

        Args:
            content_type: Content type to check (e.g., "VIDEO", "AUDIO", "YOUTUBE")

        Returns:
            True if supported, False otherwise
        """
        pass

    @abstractmethod
    async def process(
        self,
        input_file: Path,
        output_file: Path,
        config: Dict[str, Any]
    ) -> ProcessorResult:
        """Process input file and generate markdown output.

        Args:
            input_file: Input file path
            output_file: Output markdown file path
            config: Processing configuration

        Returns:
            ProcessorResult with content, metadata, and metrics

        Raises:
            ProcessingError: If processing fails
        """
        pass

    def create_markdown_with_metadata(self, content: str, metadata: Dict[str, Any]) -> str:
        """Create markdown content with metadata header.

        Args:
            content: Main content
            metadata: Metadata to include in header

        Returns:
            Formatted markdown with metadata header
        """
        # Create YAML front matter
        yaml_lines = ["---"]
        for key, value in metadata.items():
            if value is not None:
                if isinstance(value, str) and ('\n' in value or '"' in value):
                    # Multi-line or quoted strings
                    yaml_lines.append(f'{key}: |')
                    for line in str(value).split('\n'):
                        yaml_lines.append(f'  {line}')
                else:
                    yaml_lines.append(f'{key}: {value}')
        yaml_lines.append("---")
        yaml_lines.append("")

        # Combine with content
        return '\n'.join(yaml_lines) + content
