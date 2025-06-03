from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import tempfile
import structlog
from dataclasses import dataclass
from enum import Enum

from morag.core.config import settings
from morag.core.exceptions import ProcessingError, ValidationError
from morag.utils.text_processing import prepare_text_for_summary, normalize_text_encoding

logger = structlog.get_logger()

# Import unstructured only when available
try:
    from unstructured.partition.auto import partition
    from unstructured.partition.pdf import partition_pdf
    from unstructured.partition.docx import partition_docx
    from unstructured.partition.md import partition_md
    from unstructured.documents.elements import Element, Text, Title, NarrativeText, Table, Image
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    logger.warning("Unstructured.io not available - using basic text processing")

class DocumentType(Enum):
    """Supported document types."""
    PDF = "pdf"
    DOCX = "docx"
    MARKDOWN = "md"
    TXT = "txt"

@dataclass
class DocumentChunk:
    """Represents a processed document chunk."""
    text: str
    chunk_type: str  # text, table, image_caption
    page_number: Optional[int] = None
    element_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class DocumentParseResult:
    """Result of document parsing."""
    chunks: List[DocumentChunk]
    metadata: Dict[str, Any]
    images: List[Dict[str, Any]]  # Extracted images for processing
    total_pages: Optional[int] = None
    word_count: int = 0

class DocumentProcessor:
    """Handles document parsing and processing."""
    
    def __init__(self):
        self.supported_types = {
            ".pdf": DocumentType.PDF,
            ".docx": DocumentType.DOCX,
            ".doc": DocumentType.DOCX,
            ".md": DocumentType.MARKDOWN,
            ".markdown": DocumentType.MARKDOWN,
            ".txt": DocumentType.TXT,
        }
    
    def detect_document_type(self, file_path: Union[str, Path]) -> DocumentType:
        """Detect document type from file extension."""
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if extension not in self.supported_types:
            raise ValidationError(f"Unsupported file type: {extension}")
        
        return self.supported_types[extension]
    
    async def parse_document(
        self,
        file_path: Union[str, Path],
        use_docling: bool = False,
        chunking_strategy: Optional[str] = None
    ) -> DocumentParseResult:
        """Parse document and extract structured content."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise ValidationError(f"File not found: {file_path}")

        doc_type = self.detect_document_type(file_path)
        chunking_strategy = chunking_strategy or settings.default_chunking_strategy

        logger.info(
            "Starting document parsing",
            file_path=str(file_path),
            doc_type=doc_type.value,
            use_docling=use_docling,
            chunking_strategy=chunking_strategy
        )

        try:
            if use_docling and doc_type == DocumentType.PDF:
                result = await self._parse_with_docling(file_path)
            else:
                result = await self._parse_with_unstructured(file_path, doc_type)

            # Apply page-based chunking if enabled
            if chunking_strategy == "page" and settings.enable_page_based_chunking:
                result = await self._apply_page_based_chunking(result)

            return result

        except Exception as e:
            logger.error("Document parsing failed", error=str(e), file_path=str(file_path))
            raise ProcessingError(f"Failed to parse document: {str(e)}")
    
    async def _parse_with_unstructured(
        self,
        file_path: Path,
        doc_type: DocumentType
    ) -> DocumentParseResult:
        """Parse document using unstructured.io."""

        if not UNSTRUCTURED_AVAILABLE:
            logger.warning("Unstructured.io not available, using basic text parsing",
                          file_path=str(file_path))
            return await self._parse_with_basic_text(file_path, doc_type)

        # Choose appropriate partition function
        partition_func = {
            DocumentType.PDF: partition_pdf,
            DocumentType.DOCX: partition_docx,
            DocumentType.MARKDOWN: partition_md,
            DocumentType.TXT: partition,
        }.get(doc_type, partition)

        logger.info("Starting unstructured.io parsing",
                   file_path=str(file_path),
                   doc_type=doc_type.value,
                   partition_function=partition_func.__name__)

        try:
            # Parse document
            elements = partition_func(
                filename=str(file_path),
                strategy="hi_res" if doc_type == DocumentType.PDF else "fast",
                include_page_breaks=True,
                infer_table_structure=True,
                extract_images_in_pdf=True if doc_type == DocumentType.PDF else False,
            )

            logger.info("Unstructured.io parsing completed",
                       elements_count=len(elements),
                       file_path=str(file_path))

            # Log first few elements for debugging
            for i, element in enumerate(elements[:3]):
                logger.info(f"Element {i}",
                           element_type=type(element).__name__,
                           text_preview=element.text[:100] + "..." if len(element.text) > 100 else element.text,
                           text_length=len(element.text))

            return await self._process_elements(elements, file_path)

        except Exception as e:
            logger.error("Unstructured.io parsing failed",
                        error=str(e),
                        error_type=type(e).__name__,
                        file_path=str(file_path))
            # Fallback to basic text parsing
            return await self._parse_with_basic_text(file_path, doc_type)

    async def _parse_with_basic_text(
        self,
        file_path: Path,
        doc_type: DocumentType
    ) -> DocumentParseResult:
        """Basic text parsing fallback when unstructured is not available."""

        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()

        # Simple text processing
        chunks = []

        if doc_type == DocumentType.MARKDOWN:
            # Basic markdown processing
            lines = content.split('\n')
            current_chunk = ""

            for line in lines:
                if line.strip().startswith('#'):
                    # Header - create new chunk
                    if current_chunk.strip():
                        chunks.append(DocumentChunk(
                            text=current_chunk.strip(),
                            chunk_type="text",
                            page_number=1,
                            element_id=f"chunk_{len(chunks)}",
                            metadata={"basic_parser": True}
                        ))
                        current_chunk = ""

                    chunks.append(DocumentChunk(
                        text=line.strip(),
                        chunk_type="title",
                        page_number=1,
                        element_id=f"title_{len(chunks)}",
                        metadata={"basic_parser": True, "header_level": line.count('#')}
                    ))
                else:
                    current_chunk += line + "\n"

            # Add remaining content
            if current_chunk.strip():
                chunks.append(DocumentChunk(
                    text=current_chunk.strip(),
                    chunk_type="text",
                    page_number=1,
                    element_id=f"chunk_{len(chunks)}",
                    metadata={"basic_parser": True}
                ))

        else:
            # Simple paragraph-based chunking for other formats
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

            for i, paragraph in enumerate(paragraphs):
                chunks.append(DocumentChunk(
                    text=paragraph,
                    chunk_type="text",
                    page_number=1,
                    element_id=f"paragraph_{i}",
                    metadata={"basic_parser": True}
                ))

        # Calculate metadata
        metadata = {
            "parser": "basic_text",
            "file_name": file_path.name,
            "file_size": file_path.stat().st_size,
            "total_chunks": len(chunks),
            "total_images": 0
        }

        return DocumentParseResult(
            chunks=chunks,
            metadata=metadata,
            images=[],
            total_pages=1,
            word_count=len(content.split())
        )
    
    async def _parse_with_docling(self, file_path: Path) -> DocumentParseResult:
        """Parse document using docling (alternative for PDFs)."""
        try:
            # Import docling only when needed
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions

            # Configure pipeline options for better text extraction
            pipeline_options = PdfPipelineOptions(
                do_ocr=True,  # Enable OCR for scanned documents
                do_table_structure=True,  # Extract table structure
                generate_page_images=False,  # Don't generate page images to save memory
                generate_picture_images=False,  # Don't generate picture images to save memory
            )

            # Create converter with optimized settings
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )

            logger.info("Starting docling conversion", file_path=str(file_path))
            result = converter.convert(str(file_path))

            if result.status.name != "SUCCESS":
                raise ProcessingError(f"Docling conversion failed with status: {result.status}")

            # Convert docling result to our format
            chunks = []
            images = []
            total_pages = 0

            # Process docling document structure using the new v2 API
            for item, level in result.document.iterate_items():
                # Handle different item types
                if hasattr(item, 'text') and item.text.strip():
                    # Get page number from provenance if available
                    page_number = 1
                    if hasattr(item, 'prov') and item.prov:
                        for prov in item.prov:
                            if hasattr(prov, 'page_no'):
                                page_number = prov.page_no
                                total_pages = max(total_pages, page_number)
                                break

                    # Determine chunk type based on item type
                    chunk_type = "text"
                    if hasattr(item, 'label'):
                        if item.label in ['title', 'section_header']:
                            chunk_type = "title"
                        elif item.label == 'table':
                            chunk_type = "table"
                        elif item.label in ['list_item', 'list']:
                            chunk_type = "list"

                    # Clean text encoding issues
                    clean_text = normalize_text_encoding(item.text.strip())

                    chunk = DocumentChunk(
                        text=clean_text,
                        chunk_type=chunk_type,
                        page_number=page_number,
                        element_id=getattr(item, 'self_ref', f"item_{len(chunks)}"),
                        metadata={
                            "element_type": type(item).__name__,
                            "docling_source": True,
                            "label": getattr(item, 'label', 'unknown'),
                            "hierarchy_level": level
                        }
                    )
                    chunks.append(chunk)

                # Handle table items specifically
                elif hasattr(item, 'export_to_dataframe'):
                    try:
                        # Export table to DataFrame and convert to markdown
                        df = item.export_to_dataframe()
                        table_markdown = df.to_markdown(index=False)
                        # Clean encoding issues in table content
                        table_markdown = normalize_text_encoding(table_markdown)

                        # Get page number from provenance
                        page_number = 1
                        if hasattr(item, 'prov') and item.prov:
                            for prov in item.prov:
                                if hasattr(prov, 'page_no'):
                                    page_number = prov.page_no
                                    total_pages = max(total_pages, page_number)
                                    break

                        chunk = DocumentChunk(
                            text=f"**Table:**\n{table_markdown}",
                            chunk_type="table",
                            page_number=page_number,
                            element_id=getattr(item, 'self_ref', f"table_{len(chunks)}"),
                            metadata={
                                "element_type": "TableItem",
                                "docling_source": True,
                                "label": "table",
                                "hierarchy_level": level,
                                "table_shape": df.shape
                            }
                        )
                        chunks.append(chunk)
                    except Exception as e:
                        logger.warning("Failed to process table item", error=str(e))

            # If no chunks were extracted, try to get the raw markdown
            if not chunks:
                logger.warning("No chunks extracted from docling, trying markdown export")
                markdown_text = result.document.export_to_markdown()
                if markdown_text.strip():
                    # Clean encoding issues in markdown export
                    clean_markdown = normalize_text_encoding(markdown_text.strip())
                    chunk = DocumentChunk(
                        text=clean_markdown,
                        chunk_type="text",
                        page_number=1,
                        element_id="markdown_export",
                        metadata={
                            "element_type": "MarkdownExport",
                            "docling_source": True,
                            "label": "full_document"
                        }
                    )
                    chunks.append(chunk)

            metadata = {
                "parser": "docling",
                "file_name": file_path.name,
                "file_size": file_path.stat().st_size,
                "total_elements": len(chunks),
                "conversion_status": result.status.name,
                "docling_version": "v2"
            }

            logger.info(
                "Docling conversion completed",
                chunks_extracted=len(chunks),
                total_pages=total_pages,
                status=result.status.name
            )

            return DocumentParseResult(
                chunks=chunks,
                metadata=metadata,
                images=images,
                total_pages=max(total_pages, 1),
                word_count=sum(len(chunk.text.split()) for chunk in chunks)
            )

        except ImportError as e:
            logger.warning("Docling not available, falling back to unstructured.io",
                          error=str(e),
                          error_type=type(e).__name__,
                          file_path=str(file_path))
            return await self._parse_with_unstructured(file_path, DocumentType.PDF)
        except Exception as e:
            logger.error("Docling parsing failed, falling back to unstructured.io",
                        error=str(e),
                        error_type=type(e).__name__,
                        file_path=str(file_path))
            # Fallback to unstructured.io
            return await self._parse_with_unstructured(file_path, DocumentType.PDF)
    
    async def _process_elements(
        self,
        elements,  # List[Element] when unstructured is available
        file_path: Path
    ) -> DocumentParseResult:
        """Process unstructured elements into chunks."""
        if not UNSTRUCTURED_AVAILABLE:
            raise ProcessingError("Cannot process elements without unstructured.io")

        chunks = []
        images = []
        current_page = 1

        for i, element in enumerate(elements):
            # Update page number if available
            if hasattr(element, 'metadata') and element.metadata.page_number:
                current_page = element.metadata.page_number

            # Process different element types
            if isinstance(element, (Text, NarrativeText, Title)):
                if element.text.strip():
                    # Clean text encoding issues
                    clean_text = normalize_text_encoding(element.text.strip())
                    chunk = DocumentChunk(
                        text=clean_text,
                        chunk_type="text",
                        page_number=current_page,
                        element_id=f"element_{i}",
                        metadata={
                            "element_type": type(element).__name__,
                            "category": getattr(element, 'category', 'text')
                        }
                    )
                    chunks.append(chunk)

            elif isinstance(element, Table):
                # Convert table to markdown format
                table_text = self._table_to_markdown(element)
                if table_text:
                    chunk = DocumentChunk(
                        text=table_text,
                        chunk_type="table",
                        page_number=current_page,
                        element_id=f"table_{i}",
                        metadata={
                            "element_type": "Table",
                            "table_html": getattr(element, 'metadata', {}).get('text_as_html', '')
                        }
                    )
                    chunks.append(chunk)

            elif isinstance(element, Image):
                # Queue image for processing
                image_info = {
                    "element_id": f"image_{i}",
                    "page_number": current_page,
                    "metadata": element.metadata.__dict__ if hasattr(element, 'metadata') else {}
                }
                images.append(image_info)

        # Calculate metadata
        metadata = {
            "parser": "unstructured",
            "file_name": file_path.name,
            "file_size": file_path.stat().st_size,
            "total_elements": len(elements),
            "total_chunks": len(chunks),
            "total_images": len(images)
        }

        # Estimate total pages
        total_pages = max((chunk.page_number for chunk in chunks if chunk.page_number), default=1)

        return DocumentParseResult(
            chunks=chunks,
            metadata=metadata,
            images=images,
            total_pages=total_pages,
            word_count=sum(len(chunk.text.split()) for chunk in chunks)
        )
    
    def _table_to_markdown(self, table_element) -> str:
        """Convert table element to markdown format."""
        if not UNSTRUCTURED_AVAILABLE:
            return "**Table:** (content not available without unstructured.io)"

        try:
            # Try to get HTML table and convert to markdown
            if hasattr(table_element, 'metadata') and hasattr(table_element.metadata, 'text_as_html'):
                html_table = table_element.metadata.text_as_html
                # Simple HTML to markdown conversion for tables
                # This is a basic implementation - could be enhanced
                return f"**Table:**\n{table_element.text}\n"
            else:
                return f"**Table:**\n{table_element.text}\n"
        except Exception as e:
            logger.warning("Failed to convert table to markdown", error=str(e))
            return f"**Table:**\n{table_element.text}\n"
    
    def validate_file(self, file_path: Union[str, Path], max_size_mb: int = 100) -> bool:
        """Validate file before processing."""
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists():
            raise ValidationError(f"File not found: {path}")
        
        # Check file size
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > max_size_mb:
            raise ValidationError(f"File too large: {size_mb:.1f}MB (max: {max_size_mb}MB)")
        
        # Check file type
        try:
            self.detect_document_type(path)
        except ValidationError:
            raise
        
        return True

    async def _apply_page_based_chunking(self, parse_result: DocumentParseResult) -> DocumentParseResult:
        """Group chunks by page to create page-based chunks."""
        if not parse_result.chunks:
            return parse_result

        # Group chunks by page number
        page_groups = {}
        for chunk in parse_result.chunks:
            page_num = chunk.page_number or 1
            if page_num not in page_groups:
                page_groups[page_num] = []
            page_groups[page_num].append(chunk)

        # Create new page-based chunks
        new_chunks = []
        max_chunk_size = settings.max_page_chunk_size

        for page_num in sorted(page_groups.keys()):
            page_chunks = page_groups[page_num]

            # Combine all text from the page
            page_text_parts = []
            chunk_types = set()
            all_metadata = {}

            for chunk in page_chunks:
                page_text_parts.append(chunk.text)
                chunk_types.add(chunk.chunk_type)
                if chunk.metadata:
                    all_metadata.update(chunk.metadata)

            # Join text with appropriate separators
            page_text = "\n\n".join(page_text_parts).strip()

            # Check if the combined text is too large
            if len(page_text) > max_chunk_size:
                logger.warning(
                    f"Page {page_num} text too large ({len(page_text)} chars), "
                    f"splitting into smaller chunks"
                )

                # Split large pages into smaller chunks while preserving page context
                text_parts = []

                # If we have multiple parts, try to combine them intelligently
                if len(page_text_parts) > 1:
                    text_parts = page_text_parts
                else:
                    # Single large chunk - split by sentences or paragraphs
                    single_text = page_text_parts[0]
                    # Try to split by double newlines (paragraphs) first
                    paragraphs = [p.strip() for p in single_text.split('\n\n') if p.strip()]
                    if len(paragraphs) > 1:
                        text_parts = paragraphs
                    else:
                        # Split by sentences as fallback
                        import re
                        sentences = [s.strip() + '.' for s in re.split(r'[.!?]+', single_text) if s.strip()]
                        text_parts = sentences

                current_chunk_text = ""
                chunk_index = 0

                for part in text_parts:
                    # Check if adding this part would exceed the limit
                    if len(current_chunk_text) + len(part) + 2 <= max_chunk_size:
                        if current_chunk_text:
                            current_chunk_text += "\n\n" + part
                        else:
                            current_chunk_text = part
                    else:
                        # Create chunk from current text if it's not empty
                        if current_chunk_text:
                            new_chunk = DocumentChunk(
                                text=current_chunk_text,
                                chunk_type="page",
                                page_number=page_num,
                                element_id=f"page_{page_num}_chunk_{chunk_index}",
                                metadata={
                                    **all_metadata,
                                    "original_chunk_types": list(chunk_types),
                                    "page_based_chunking": True,
                                    "chunk_index_on_page": chunk_index,
                                    "is_partial_page": True
                                }
                            )
                            new_chunks.append(new_chunk)
                            chunk_index += 1

                        # Start new chunk with current part
                        current_chunk_text = part

                # Add final chunk
                if current_chunk_text:
                    new_chunk = DocumentChunk(
                        text=current_chunk_text,
                        chunk_type="page",
                        page_number=page_num,
                        element_id=f"page_{page_num}_chunk_{chunk_index}",
                        metadata={
                            **all_metadata,
                            "original_chunk_types": list(chunk_types),
                            "page_based_chunking": True,
                            "chunk_index_on_page": chunk_index,
                            "is_partial_page": True
                        }
                    )
                    new_chunks.append(new_chunk)
            else:
                # Create single chunk for the entire page
                new_chunk = DocumentChunk(
                    text=page_text,
                    chunk_type="page",
                    page_number=page_num,
                    element_id=f"page_{page_num}",
                    metadata={
                        **all_metadata,
                        "original_chunk_types": list(chunk_types),
                        "page_based_chunking": True,
                        "original_chunks_count": len(page_chunks)
                    }
                )
                new_chunks.append(new_chunk)

        # Update metadata
        updated_metadata = {
            **parse_result.metadata,
            "chunking_strategy": "page",
            "original_chunks_count": len(parse_result.chunks),
            "page_based_chunks_count": len(new_chunks),
            "page_based_chunking_applied": True
        }

        logger.info(
            "Applied page-based chunking",
            original_chunks=len(parse_result.chunks),
            new_chunks=len(new_chunks),
            total_pages=parse_result.total_pages
        )

        return DocumentParseResult(
            chunks=new_chunks,
            metadata=updated_metadata,
            images=parse_result.images,
            total_pages=parse_result.total_pages,
            word_count=parse_result.word_count
        )

# Global instance
document_processor = DocumentProcessor()
