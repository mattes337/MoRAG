"""Enhanced PDF to Markdown converter with advanced docling features."""

import time
from pathlib import Path
from typing import Union, List, Dict, Any, Optional
import structlog

from .base import BaseConverter, ConversionOptions, ConversionResult, QualityScore
from .quality import ConversionQualityValidator
from ..processors.document import document_processor
from ..utils.text_processing import normalize_text_encoding, clean_pdf_text_encoding

logger = structlog.get_logger(__name__)

try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    logger.warning("Docling not available, falling back to basic PDF processing")

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False


class PDFConverter(BaseConverter):
    """Enhanced PDF to Markdown converter with advanced docling features."""

    def __init__(self):
        super().__init__("Enhanced MoRAG PDF Converter")
        self.supported_formats = ['pdf']
        self.quality_validator = ConversionQualityValidator()

        # Initialize advanced docling converter if available
        self.docling_converter = None
        if DOCLING_AVAILABLE:
            self._initialize_docling_converter()

        # Fallback converters
        self.fallback_converters = []
        if PYMUPDF_AVAILABLE:
            self.fallback_converters.append('pymupdf')
        if PDFPLUMBER_AVAILABLE:
            self.fallback_converters.append('pdfplumber')

    def _initialize_docling_converter(self):
        """Initialize the docling converter with advanced options."""
        try:
            pipeline_options = PdfPipelineOptions(
                do_ocr=True,
                do_table_structure=True,
                table_structure_options=TableStructureOptions(
                    do_cell_matching=True
                )
            )

            self.docling_converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            logger.info("Advanced docling converter initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize advanced docling converter: {e}")
            self.docling_converter = None
    
    def supports_format(self, format_type: str) -> bool:
        """Check if this converter supports the given format."""
        return format_type.lower() in self.supported_formats
    
    async def convert(self, file_path: Union[str, Path], options: ConversionOptions) -> ConversionResult:
        """Convert PDF to structured markdown with advanced features.

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
            "Starting enhanced PDF conversion",
            file_path=str(file_path),
            use_advanced_docling=bool(self.docling_converter),
            chunking_strategy=options.chunking_strategy.value,
            extract_tables=options.format_options.get('extract_tables', True),
            use_ocr=options.format_options.get('use_ocr', True)
        )

        # Try advanced docling first if available and requested
        if (self.docling_converter and
            options.format_options.get('use_advanced_docling', True)):
            try:
                result = await self._convert_with_advanced_docling(file_path, options, start_time)
                if result.success and result.quality_score.overall_score >= options.min_quality_threshold:
                    return result
                else:
                    logger.warning("Advanced docling conversion quality below threshold, trying fallback")
            except Exception as e:
                logger.warning(f"Advanced docling conversion failed: {e}, trying fallback")

        # Fallback to existing MoRAG processor
        try:
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
                converter_used=self.name,
                fallback_used=True
            )

            logger.info(
                "PDF conversion completed with fallback",
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

    async def _convert_with_advanced_docling(self, file_path: Path, options: ConversionOptions, start_time: float) -> ConversionResult:
        """Convert PDF using advanced docling features.

        Args:
            file_path: Path to PDF file
            options: Conversion options
            start_time: Start time for processing measurement

        Returns:
            ConversionResult with advanced docling processing
        """
        logger.info("Using advanced docling converter", file_path=str(file_path))

        # Convert with docling
        docling_result = self.docling_converter.convert(str(file_path))

        if docling_result.status.name != "SUCCESS":
            raise Exception(f"Docling conversion failed with status: {docling_result.status}")

        # Process docling result into structured markdown
        markdown_content = await self._process_docling_result(docling_result, options)

        # Extract enhanced metadata
        metadata = self._extract_docling_metadata(docling_result, file_path)

        # Calculate quality score
        quality_score = self._calculate_advanced_quality_score(docling_result, markdown_content)

        processing_time = time.time() - start_time

        result = ConversionResult(
            content=markdown_content,
            metadata=metadata,
            quality_score=quality_score,
            processing_time=processing_time,
            success=True,
            original_format='pdf',
            converter_used=f"{self.name} (Advanced Docling)"
        )

        # Get page count for logging
        page_count = 0
        if docling_result.document:
            for item, level in docling_result.document.iterate_items():
                if hasattr(item, 'prov') and item.prov:
                    for prov in item.prov:
                        if hasattr(prov, 'page_no'):
                            page_count = max(page_count, prov.page_no)

        logger.info(
            "Advanced docling conversion completed",
            processing_time=processing_time,
            quality_score=quality_score.overall_score,
            word_count=result.word_count,
            pages=page_count
        )

        return result

    async def _process_docling_result(self, docling_result, options: ConversionOptions) -> str:
        """Process docling result into structured markdown.

        Args:
            docling_result: Result from docling conversion
            options: Conversion options

        Returns:
            Structured markdown content
        """
        sections = []

        # Document header
        title = docling_result.document.name or "PDF Document"
        sections.append(f"# {title}")
        sections.append("")

        # Document metadata
        if options.include_metadata:
            sections.append("## Document Information")
            sections.append("")

            # Get page count from document structure
            page_count = 0
            for item, level in docling_result.document.iterate_items():
                if hasattr(item, 'prov') and item.prov:
                    for prov in item.prov:
                        if hasattr(prov, 'page_no'):
                            page_count = max(page_count, prov.page_no)

            metadata_items = [
                ("**Source**", docling_result.input.file.name),
                ("**Pages**", str(page_count)),
                ("**Processing Method**", "Advanced Docling"),
                ("**OCR Used**", "Yes" if options.format_options.get('use_ocr', True) else "No"),
                ("**Table Extraction**", "Yes" if options.format_options.get('extract_tables', True) else "No")
            ]

            for label, value in metadata_items:
                sections.append(f"{label}: {value}")

            sections.append("")

        # Table of contents for page-based chunking
        if options.include_toc and options.chunking_strategy.value == 'page':
            sections.append("## Table of Contents")
            sections.append("")
            for i in range(1, page_count + 1):
                sections.append(f"- [Page {i}](#page-{i})")
            sections.append("")

        # Process content using docling v2 API
        sections.append("## Content")
        sections.append("")

        # Group items by page
        page_items = {}
        for item, level in docling_result.document.iterate_items():
            if hasattr(item, 'text') and item.text.strip():
                # Get page number from provenance
                page_number = 1
                if hasattr(item, 'prov') and item.prov:
                    for prov in item.prov:
                        if hasattr(prov, 'page_no'):
                            page_number = prov.page_no
                            break

                if page_number not in page_items:
                    page_items[page_number] = []

                page_items[page_number].append((item, level))

        # Process each page
        for page_num in sorted(page_items.keys()):
            sections.append(f"### Page {page_num}")
            sections.append("")

            page_content = await self._process_page_items(page_items[page_num], options)
            sections.append(page_content)
            sections.append("")

        return "\n".join(sections)

    async def _process_page_items(self, page_items, options: ConversionOptions) -> str:
        """Process items within a page using docling v2 API.

        Args:
            page_items: List of (item, level) tuples for the page
            options: Conversion options

        Returns:
            Processed page content
        """
        elements = []

        # Process text items
        for item, level in page_items:
            if hasattr(item, 'text') and item.text.strip():
                # Clean text encoding issues
                clean_text = normalize_text_encoding(item.text.strip())

                # Handle different item types based on label
                if hasattr(item, 'label'):
                    if item.label in ['title', 'section_header']:
                        elements.append(f"#### {clean_text}")
                    elif item.label in ['heading', 'subtitle']:
                        elements.append(f"##### {clean_text}")
                    elif item.label == 'table':
                        # For table items, try to export as markdown
                        table_md = await self._convert_docling_item_table(item)
                        if table_md:
                            elements.append(table_md)
                        else:
                            elements.append(clean_text)
                    else:
                        elements.append(clean_text)
                else:
                    elements.append(clean_text)

        return "\n\n".join(elements)

    async def _convert_docling_item_table(self, table_item) -> str:
        """Convert docling table item to markdown format using v2 API.

        Args:
            table_item: Docling table item from iterate_items()

        Returns:
            Markdown table string
        """
        try:
            # Try to export table as dataframe first (docling v2 method)
            if hasattr(table_item, 'export_to_dataframe'):
                import pandas as pd
                df = table_item.export_to_dataframe()
                if df is not None and not df.empty:
                    return df.to_markdown(index=False)

            # Fallback to text representation
            if hasattr(table_item, 'text') and table_item.text:
                return f"*[Table: {normalize_text_encoding(table_item.text)}]*"

        except Exception as e:
            logger.warning(f"Failed to convert table item: {e}")
            return "*[Table conversion failed]*"

        return ""

    async def _convert_docling_table(self, table) -> str:
        """Convert docling table to markdown format (legacy method).

        Args:
            table: Docling table object

        Returns:
            Markdown table string
        """
        try:
            # Extract table data
            if hasattr(table, 'data') and table.data:
                rows = table.data
                if not rows:
                    return ""

                # Create markdown table
                table_lines = []

                # Header row
                if rows:
                    headers = [normalize_text_encoding(str(cell)) for cell in rows[0]]
                    table_lines.append("| " + " | ".join(headers) + " |")
                    table_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

                    # Data rows
                    for row in rows[1:]:
                        cells = [normalize_text_encoding(str(cell)) for cell in row]
                        table_lines.append("| " + " | ".join(cells) + " |")

                return "\n".join(table_lines)
        except Exception as e:
            logger.warning(f"Failed to convert table: {e}")
            return "*[Table conversion failed]*"

        return ""

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
                sections.append(chunk.text.strip())
                sections.append("")
                current_page += 1
        else:
            # Sequential chunks
            for i, chunk in enumerate(parse_result.chunks, 1):
                sections.append(f"### Section {i}")
                sections.append("")
                sections.append(chunk.text.strip())
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

    def _extract_docling_metadata(self, docling_result, file_path: Path) -> Dict[str, Any]:
        """Extract enhanced metadata from docling result.

        Args:
            docling_result: Result from docling conversion
            file_path: Path to original file

        Returns:
            Enhanced metadata dictionary
        """
        metadata = {
            'original_filename': file_path.name,
            'file_size': file_path.stat().st_size,
            'conversion_format': 'pdf_to_markdown',
            'converter_version': '2.0.0-enhanced',
            'processing_method': 'advanced_docling'
        }

        if docling_result.document:
            # Get page count and check for tables/images using v2 API
            page_count = 0
            has_tables = False
            has_images = False

            for item, level in docling_result.document.iterate_items():
                if hasattr(item, 'prov') and item.prov:
                    for prov in item.prov:
                        if hasattr(prov, 'page_no'):
                            page_count = max(page_count, prov.page_no)

                if hasattr(item, 'label'):
                    if item.label == 'table':
                        has_tables = True
                    elif item.label in ['picture', 'figure']:
                        has_images = True

            metadata.update({
                'total_pages': page_count,
                'document_name': docling_result.document.name or file_path.stem,
                'has_tables': has_tables,
                'has_images': has_images,
                'ocr_used': True,  # Advanced docling always uses OCR
                'table_extraction_used': True
            })

        return metadata

    def _calculate_advanced_quality_score(self, docling_result, markdown_content: str) -> QualityScore:
        """Calculate quality score for advanced docling conversion.

        Args:
            docling_result: Result from docling conversion
            markdown_content: Generated markdown content

        Returns:
            Quality score assessment
        """
        # Base scores
        completeness_score = 0.9  # Advanced docling typically has high completeness
        readability_score = 0.85
        structure_score = 0.9
        metadata_preservation = 0.95

        # Adjust based on content analysis
        word_count = len(markdown_content.split())
        if word_count < 50:
            completeness_score *= 0.7
        elif word_count > 1000:
            completeness_score = min(completeness_score * 1.1, 1.0)

        # Check for tables and structure
        if '|' in markdown_content:  # Has tables
            structure_score = min(structure_score * 1.1, 1.0)

        # Check for proper headings
        heading_count = markdown_content.count('#')
        if heading_count > 0:
            structure_score = min(structure_score * (1 + heading_count * 0.02), 1.0)

        overall_score = (
            completeness_score * 0.3 +
            readability_score * 0.25 +
            structure_score * 0.25 +
            metadata_preservation * 0.2
        )

        return QualityScore(
            overall_score=overall_score,
            completeness_score=completeness_score,
            readability_score=readability_score,
            structure_score=structure_score,
            metadata_preservation=metadata_preservation
        )
