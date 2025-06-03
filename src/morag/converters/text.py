"""Basic text to Markdown converter."""

import time
from pathlib import Path
from typing import Union
import structlog

from .base import BaseConverter, ConversionOptions, ConversionResult, QualityScore
from .quality import ConversionQualityValidator

logger = structlog.get_logger(__name__)


class TextConverter(BaseConverter):
    """Basic text to Markdown converter for plain text files."""

    def __init__(self):
        super().__init__("Basic Text Converter")
        self.supported_formats = ['text', 'txt', 'md', 'markdown']
        self.quality_validator = ConversionQualityValidator()

    def supports_format(self, format_type: str) -> bool:
        """Check if this converter supports the given format."""
        return format_type.lower() in self.supported_formats

    async def convert(self, file_path: Union[str, Path], options: ConversionOptions) -> ConversionResult:
        """Convert text file to structured markdown.

        Args:
            file_path: Path to text file
            options: Conversion options

        Returns:
            ConversionResult with markdown content
        """
        start_time = time.time()
        file_path = Path(file_path)

        await self.validate_input(file_path)

        logger.info(
            "Starting text conversion",
            file_path=str(file_path),
            chunking_strategy=options.chunking_strategy.value
        )

        try:
            # Read the text file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Create structured markdown
            markdown_content = await self._create_structured_markdown(content, file_path, options)

            # Calculate quality score
            result = ConversionResult(
                content=markdown_content,
                metadata={
                    'file_name': file_path.name,
                    'file_size': file_path.stat().st_size,
                    'original_format': 'text',
                    'converter_version': '1.0.0'
                }
            )

            quality_score = self.quality_validator.validate_conversion(str(file_path), result)

            processing_time = time.time() - start_time

            return ConversionResult(
                content=markdown_content,
                metadata={
                    'parser': 'basic_text',
                    'file_name': file_path.name,
                    'file_size': file_path.stat().st_size,
                    'word_count': len(content.split()),
                    'line_count': len(content.splitlines()),
                    'original_filename': file_path.name,
                    'conversion_format': 'text_to_markdown',
                    'converter_version': '1.0.0'
                },
                quality_score=quality_score,
                processing_time=processing_time,
                success=True,
                original_format='text',
                converter_used=self.name
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error("Text conversion failed", error=str(e), processing_time=processing_time)
            
            return ConversionResult(
                content="",
                metadata={},
                processing_time=processing_time,
                success=False,
                error_message=f"Text conversion failed: {str(e)}",
                original_format='text',
                converter_used=self.name
            )

    async def _create_structured_markdown(self, content: str, file_path: Path, options: ConversionOptions) -> str:
        """Create structured markdown from text content."""
        sections = []

        # Document header
        sections.append(f"# {file_path.stem}")
        sections.append("")

        # Document information
        if options.include_metadata:
            sections.append("## Document Information")
            sections.append("")
            sections.append(f"**Source**: {file_path.name}")
            sections.append(f"**Format**: Text → Markdown")
            sections.append(f"**Lines**: {len(content.splitlines())}")
            sections.append(f"**Words**: {len(content.split())}")
            sections.append("")

        # Content section
        sections.append("## Content")
        sections.append("")

        # Process content based on chunking strategy
        if options.chunking_strategy.value == 'paragraph':
            # Split by paragraphs (double newlines)
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            for i, paragraph in enumerate(paragraphs, 1):
                sections.append(f"### Paragraph {i}")
                sections.append("")
                sections.append(paragraph)
                sections.append("")
        
        elif options.chunking_strategy.value == 'sentence':
            # Split by sentences (simple approach)
            import re
            sentences = re.split(r'[.!?]+', content)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            for i, sentence in enumerate(sentences, 1):
                sections.append(f"### Sentence {i}")
                sections.append("")
                sections.append(sentence + ".")
                sections.append("")
        
        else:
            # Default: treat as single content block
            sections.append(content)
            sections.append("")

        # Processing details
        sections.append("## Processing Details")
        sections.append("")
        sections.append(f"**Conversion Method**: Basic Text Processing")
        sections.append(f"**Chunking Strategy**: {options.chunking_strategy.value}")

        return "\n".join(sections)
