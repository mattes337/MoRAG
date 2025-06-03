"""PDF to Markdown converter using existing MoRAG document processor."""

import time
from pathlib import Path
from typing import Union
import structlog

from .base import BaseConverter, ConversionOptions, ConversionResult, QualityScore
from .quality import ConversionQualityValidator
from ..processors.document import document_processor

logger = structlog.get_logger(__name__)


class PDFConverter(BaseConverter):
    """PDF to Markdown converter using MoRAG's document processor."""
    
    def __init__(self):
        super().__init__("MoRAG PDF Converter")
        self.supported_formats = ['pdf']
        self.quality_validator = ConversionQualityValidator()
    
    def supports_format(self, format_type: str) -> bool:
        """Check if this converter supports the given format."""
        return format_type.lower() in self.supported_formats
    
    async def convert(self, file_path: Union[str, Path], options: ConversionOptions) -> ConversionResult:
        """Convert PDF to structured markdown.
        
        Args:
            file_path: Path to PDF file
            options: Conversion options
            
        Returns:
            ConversionResult with markdown content
        """
        start_time = time.time()
        file_path = Path(file_path)
        
        await self.validate_input(file_path)
        
        logger.info(
            "Starting PDF conversion",
            file_path=str(file_path),
            use_docling=options.format_options.get('use_docling', True),
            chunking_strategy=options.chunking_strategy.value
        )
        
        try:
            # Use existing MoRAG document processor
            use_docling = options.format_options.get('use_docling', True)
            
            parse_result = await document_processor.parse_document(
                file_path,
                use_docling=use_docling,
                chunking_strategy=options.chunking_strategy.value
            )
            
            # Convert chunks to structured markdown
            markdown_content = await self._create_structured_markdown(parse_result, options)
            
            # Calculate quality score
            quality_score = self.quality_validator.validate_conversion(str(file_path), ConversionResult(
                content=markdown_content,
                metadata=parse_result.metadata
            ))
            
            processing_time = time.time() - start_time
            
            result = ConversionResult(
                content=markdown_content,
                metadata=self._enhance_metadata(parse_result.metadata, file_path),
                quality_score=quality_score,
                processing_time=processing_time,
                success=True,
                images=parse_result.images,
                original_format='pdf',
                converter_used=self.name
            )
            
            logger.info(
                "PDF conversion completed",
                processing_time=processing_time,
                quality_score=quality_score.overall_score,
                word_count=result.word_count,
                chunks_count=len(parse_result.chunks)
            )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"PDF conversion failed: {str(e)}"
            
            logger.error(
                "PDF conversion failed",
                error=str(e),
                error_type=type(e).__name__,
                processing_time=processing_time
            )
            
            return ConversionResult(
                content="",
                metadata={},
                processing_time=processing_time,
                success=False,
                error_message=error_msg,
                original_format='pdf',
                converter_used=self.name
            )
    
    async def _create_structured_markdown(self, parse_result, options: ConversionOptions) -> str:
        """Create structured markdown from parse result.
        
        Args:
            parse_result: Document parse result from MoRAG processor
            options: Conversion options
            
        Returns:
            Structured markdown content
        """
        sections = []
        
        # Document header
        title = parse_result.metadata.get('title', 'Document')
        sections.append(f"# {title}")
        sections.append("")
        
        # Metadata section
        if options.include_metadata:
            sections.append("## Document Information")
            sections.append("")
            
            metadata_items = [
                ("**Source**", parse_result.metadata.get('filename', 'Unknown')),
                ("**Pages**", str(parse_result.total_pages)),
                ("**Word Count**", str(parse_result.word_count)),
                ("**Processing Method**", parse_result.metadata.get('parser_used', 'Unknown')),
                ("**Chunking Strategy**", parse_result.metadata.get('chunking_strategy', 'Unknown'))
            ]
            
            for label, value in metadata_items:
                if value and value != 'Unknown':
                    sections.append(f"{label}: {value}")
            
            sections.append("")
        
        # Table of contents
        if options.include_toc and options.chunking_strategy.value == 'page':
            sections.append("## Table of Contents")
            sections.append("")
            for i in range(1, parse_result.total_pages + 1):
                sections.append(f"- [Page {i}](#page-{i})")
            sections.append("")
        
        # Content sections
        sections.append("## Content")
        sections.append("")
        
        if options.chunking_strategy.value == 'page':
            # Page-based organization
            current_page = 1
            for chunk in parse_result.chunks:
                chunk_page = chunk.metadata.get('page_number', current_page)
                if chunk_page != current_page:
                    current_page = chunk_page
                
                sections.append(f"### Page {current_page}")
                sections.append("")
                sections.append(chunk.content.strip())
                sections.append("")
                current_page += 1
        else:
            # Sequential chunks
            for i, chunk in enumerate(parse_result.chunks, 1):
                sections.append(f"### Section {i}")
                sections.append("")
                sections.append(chunk.content.strip())
                sections.append("")
        
        # Images section
        if parse_result.images and options.extract_images:
            sections.append("## Images")
            sections.append("")
            
            for i, image in enumerate(parse_result.images, 1):
                sections.append(f"### Image {i}")
                if image.get('description'):
                    sections.append(f"**Description**: {image['description']}")
                if image.get('page_number'):
                    sections.append(f"**Page**: {image['page_number']}")
                if image.get('text_content'):
                    sections.append(f"**Extracted Text**: {image['text_content']}")
                sections.append("")
        
        # Processing metadata
        sections.append("## Processing Details")
        sections.append("")
        sections.append(f"**Conversion Method**: {parse_result.metadata.get('parser_used', 'Unknown')}")
        sections.append(f"**Chunks Generated**: {len(parse_result.chunks)}")
        sections.append(f"**Images Extracted**: {len(parse_result.images)}")
        
        return "\n".join(sections)
    
    def _enhance_metadata(self, original_metadata: dict, file_path: Path) -> dict:
        """Enhance metadata with additional information.
        
        Args:
            original_metadata: Original metadata from parser
            file_path: Path to original file
            
        Returns:
            Enhanced metadata dictionary
        """
        enhanced = original_metadata.copy()
        
        # Add file information
        enhanced.update({
            'original_filename': file_path.name,
            'file_size': file_path.stat().st_size,
            'conversion_format': 'pdf_to_markdown',
            'converter_version': '1.0.0'
        })
        
        # Add format-specific metadata
        if 'total_pages' not in enhanced and 'page_count' in enhanced:
            enhanced['total_pages'] = enhanced['page_count']
        
        return enhanced
