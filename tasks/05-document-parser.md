# Task 05: Document Parser Implementation

## Overview
Implement document parsing capabilities using unstructured.io as the primary library and docling as an alternative, with support for PDF, DOCX, and Markdown files.

## Prerequisites
- Task 01: Project Setup completed
- Task 04: Task Queue Setup completed
- Task 14: Gemini Integration completed

## Dependencies
- Task 01: Project Setup
- Task 04: Task Queue Setup
- Task 14: Gemini Integration

## Implementation Steps

### 1. Document Processor Service
Create `src/morag/processors/document.py`:
```python
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import tempfile
import structlog
from dataclasses import dataclass
from enum import Enum

from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.md import partition_md
from unstructured.documents.elements import Element, Text, Title, NarrativeText, Table, Image

from morag.core.config import settings
from morag.core.exceptions import ProcessingError, ValidationError
from morag.utils.text_processing import prepare_text_for_summary

logger = structlog.get_logger()

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
        use_docling: bool = False
    ) -> DocumentParseResult:
        """Parse document and extract structured content."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ValidationError(f"File not found: {file_path}")
        
        doc_type = self.detect_document_type(file_path)
        
        logger.info(
            "Starting document parsing",
            file_path=str(file_path),
            doc_type=doc_type.value,
            use_docling=use_docling
        )
        
        try:
            if use_docling and doc_type == DocumentType.PDF:
                return await self._parse_with_docling(file_path)
            else:
                return await self._parse_with_unstructured(file_path, doc_type)
                
        except Exception as e:
            logger.error("Document parsing failed", error=str(e), file_path=str(file_path))
            raise ProcessingError(f"Failed to parse document: {str(e)}")
    
    async def _parse_with_unstructured(
        self,
        file_path: Path,
        doc_type: DocumentType
    ) -> DocumentParseResult:
        """Parse document using unstructured.io."""
        
        # Choose appropriate partition function
        partition_func = {
            DocumentType.PDF: partition_pdf,
            DocumentType.DOCX: partition_docx,
            DocumentType.MARKDOWN: partition_md,
            DocumentType.TXT: partition,
        }.get(doc_type, partition)
        
        # Parse document
        elements = partition_func(
            filename=str(file_path),
            strategy="hi_res" if doc_type == DocumentType.PDF else "fast",
            include_page_breaks=True,
            infer_table_structure=True,
            extract_images_in_pdf=True if doc_type == DocumentType.PDF else False,
        )
        
        return await self._process_elements(elements, file_path)
    
    async def _parse_with_docling(self, file_path: Path) -> DocumentParseResult:
        """Parse document using docling (alternative for PDFs)."""
        try:
            # Import docling only when needed
            from docling.document_converter import DocumentConverter
            
            converter = DocumentConverter()
            result = converter.convert(str(file_path))
            
            # Convert docling result to our format
            chunks = []
            images = []
            
            # Process docling document structure
            for element in result.document.body:
                if hasattr(element, 'text') and element.text.strip():
                    chunk = DocumentChunk(
                        text=element.text.strip(),
                        chunk_type="text",
                        page_number=getattr(element, 'page', None),
                        metadata={
                            "element_type": type(element).__name__,
                            "docling_source": True
                        }
                    )
                    chunks.append(chunk)
            
            metadata = {
                "parser": "docling",
                "file_name": file_path.name,
                "file_size": file_path.stat().st_size,
                "total_elements": len(chunks)
            }
            
            return DocumentParseResult(
                chunks=chunks,
                metadata=metadata,
                images=images,
                word_count=sum(len(chunk.text.split()) for chunk in chunks)
            )
            
        except ImportError:
            logger.warning("Docling not available, falling back to unstructured.io")
            return await self._parse_with_unstructured(file_path, DocumentType.PDF)
        except Exception as e:
            logger.error("Docling parsing failed", error=str(e))
            # Fallback to unstructured.io
            return await self._parse_with_unstructured(file_path, DocumentType.PDF)
    
    async def _process_elements(
        self,
        elements: List[Element],
        file_path: Path
    ) -> DocumentParseResult:
        """Process unstructured elements into chunks."""
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
                    chunk = DocumentChunk(
                        text=element.text.strip(),
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
    
    def _table_to_markdown(self, table_element: Table) -> str:
        """Convert table element to markdown format."""
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

# Global instance
document_processor = DocumentProcessor()
```

### 2. Document Processing Task
Create `src/morag/tasks/document_tasks.py`:
```python
from typing import Dict, Any, List
import structlog
from pathlib import Path

from morag.core.celery_app import celery_app
from morag.tasks.base import ProcessingTask
from morag.processors.document import document_processor
from morag.services.embedding import gemini_service
from morag.services.chunking import chunking_service
from morag.services.storage import qdrant_service
from morag.utils.text_processing import combine_text_and_summary

logger = structlog.get_logger()

@celery_app.task(bind=True, base=ProcessingTask)
def process_document_task(
    self,
    file_path: str,
    source_type: str,
    metadata: Dict[str, Any],
    use_docling: bool = False
) -> Dict[str, Any]:
    """Process a document file through the complete pipeline."""
    
    try:
        self.log_step("Starting document processing", file_path=file_path)
        self.update_progress(0.1, "Validating file")
        
        # Validate file
        document_processor.validate_file(file_path)
        
        self.update_progress(0.2, "Parsing document")
        
        # Parse document
        parse_result = await document_processor.parse_document(
            file_path,
            use_docling=use_docling
        )
        
        self.log_step(
            "Document parsed",
            chunks_count=len(parse_result.chunks),
            images_count=len(parse_result.images),
            word_count=parse_result.word_count
        )
        
        self.update_progress(0.4, "Processing chunks")
        
        # Process chunks for embedding
        processed_chunks = []
        
        for i, chunk in enumerate(parse_result.chunks):
            # Generate summary for chunk
            self.update_progress(
                0.4 + (0.3 * i / len(parse_result.chunks)),
                f"Generating summary for chunk {i+1}/{len(parse_result.chunks)}"
            )
            
            try:
                summary_result = await gemini_service.generate_summary(
                    chunk.text,
                    max_length=100,
                    style="concise"
                )
                summary = summary_result.summary
            except Exception as e:
                logger.warning("Failed to generate summary for chunk", error=str(e))
                summary = chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text
            
            # Combine text and summary for embedding
            combined_text = combine_text_and_summary(chunk.text, summary)
            
            processed_chunk = {
                "text": chunk.text,
                "summary": summary,
                "combined_text": combined_text,
                "source": file_path,
                "source_type": source_type,
                "chunk_index": i,
                "chunk_type": chunk.chunk_type,
                "metadata": {
                    **metadata,
                    **parse_result.metadata,
                    "page_number": chunk.page_number,
                    "element_id": chunk.element_id,
                    "chunk_metadata": chunk.metadata or {}
                }
            }
            processed_chunks.append(processed_chunk)
        
        self.update_progress(0.7, "Generating embeddings")
        
        # Generate embeddings
        texts_for_embedding = [chunk["combined_text"] for chunk in processed_chunks]
        
        embedding_results = await gemini_service.generate_embeddings_batch(
            texts_for_embedding,
            batch_size=5
        )
        
        embeddings = [result.embedding for result in embedding_results]
        
        self.update_progress(0.9, "Storing in vector database")
        
        # Store in vector database
        point_ids = await qdrant_service.store_chunks(processed_chunks, embeddings)
        
        self.update_progress(1.0, "Document processing completed")
        
        result = {
            "status": "success",
            "file_path": file_path,
            "chunks_processed": len(processed_chunks),
            "images_found": len(parse_result.images),
            "word_count": parse_result.word_count,
            "total_pages": parse_result.total_pages,
            "point_ids": point_ids,
            "metadata": parse_result.metadata
        }
        
        self.log_step("Document processing completed", **result)
        return result
        
    except Exception as e:
        error_msg = f"Document processing failed: {str(e)}"
        logger.error("Document processing task failed", error=str(e), file_path=file_path)
        self.update_progress(0.0, error_msg)
        raise
```

### 3. Chunking Service Placeholder
Create `src/morag/services/chunking.py`:
```python
from typing import List, Dict, Any
import structlog

logger = structlog.get_logger()

class ChunkingService:
    """Service for semantic chunking of text. (Placeholder - will be implemented in Task 06)"""
    
    def __init__(self):
        self.max_chunk_size = 1000  # Default chunk size
    
    async def semantic_chunk(self, text: str, chunk_size: int = None) -> List[str]:
        """Perform semantic chunking of text. (Placeholder implementation)"""
        chunk_size = chunk_size or self.max_chunk_size
        
        # Simple sentence-based chunking as placeholder
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

# Global instance
chunking_service = ChunkingService()
```

## Testing Instructions

### 1. Test Document Processor
Create `tests/unit/test_document_processor.py`:
```python
import pytest
import tempfile
from pathlib import Path
from morag.processors.document import document_processor, DocumentType

def test_document_type_detection():
    """Test document type detection."""
    assert document_processor.detect_document_type("test.pdf") == DocumentType.PDF
    assert document_processor.detect_document_type("test.docx") == DocumentType.DOCX
    assert document_processor.detect_document_type("test.md") == DocumentType.MARKDOWN
    
    with pytest.raises(Exception):
        document_processor.detect_document_type("test.xyz")

@pytest.mark.asyncio
async def test_markdown_parsing():
    """Test parsing of markdown files."""
    # Create a test markdown file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write("""
# Test Document

This is a test document with multiple sections.

## Section 1

Some content in section 1.

## Section 2

Some content in section 2.
        """)
        temp_path = f.name
    
    try:
        result = await document_processor.parse_document(temp_path)
        
        assert len(result.chunks) > 0
        assert result.word_count > 0
        assert result.metadata["parser"] == "unstructured"
        
    finally:
        Path(temp_path).unlink()
```

### 2. Test Document Task
Create `tests/integration/test_document_tasks.py`:
```python
import pytest
import tempfile
from pathlib import Path
from morag.tasks.document_tasks import process_document_task

@pytest.mark.asyncio
async def test_document_processing_task():
    """Test complete document processing task."""
    # Create a test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write("# Test\nThis is a test document for processing.")
        temp_path = f.name
    
    try:
        # Process document
        result = process_document_task.delay(
            file_path=temp_path,
            source_type="document",
            metadata={"test": True}
        )
        
        # Wait for completion (in real scenario, would check status)
        task_result = result.get(timeout=60)
        
        assert task_result["status"] == "success"
        assert task_result["chunks_processed"] > 0
        
    finally:
        Path(temp_path).unlink()
```

### 3. Manual Testing Script
Create `scripts/test_document_processing.py`:
```python
#!/usr/bin/env python3
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morag.processors.document import document_processor

async def main():
    # Test with a sample markdown file
    test_content = """
# Sample Document

This is a sample document for testing the document processor.

## Introduction

The document processor handles various file formats including PDF, DOCX, and Markdown.

## Features

- Document parsing
- Text extraction
- Metadata extraction
- Image detection

## Conclusion

This completes the sample document.
    """
    
    # Create temporary file
    with open("test_sample.md", "w") as f:
        f.write(test_content)
    
    try:
        print("Testing document processor...")
        result = await document_processor.parse_document("test_sample.md")
        
        print(f"Chunks found: {len(result.chunks)}")
        print(f"Word count: {result.word_count}")
        print(f"Images found: {len(result.images)}")
        
        for i, chunk in enumerate(result.chunks[:3]):  # Show first 3 chunks
            print(f"\nChunk {i+1}:")
            print(f"Type: {chunk.chunk_type}")
            print(f"Text: {chunk.text[:100]}...")
            
    finally:
        Path("test_sample.md").unlink(missing_ok=True)

if __name__ == "__main__":
    asyncio.run(main())
```

## Success Criteria
- [ ] Document processor can handle PDF, DOCX, and Markdown files
- [ ] Unstructured.io integration works correctly
- [ ] Docling fallback works for PDFs (if installed)
- [ ] Document chunks are properly extracted
- [ ] Tables are converted to markdown format
- [ ] Images are detected and queued for processing
- [ ] Document processing task completes successfully
- [ ] Processed chunks are stored in vector database
- [ ] Error handling works for invalid files
- [ ] Unit and integration tests pass

## Next Steps
- Task 06: Semantic Chunking (enhances chunking_service)
- Task 07: Summary Generation (enhances summarization)
- Task 10: Image Processing (processes extracted images)
- Task 17: Ingestion API (uses document processing tasks)
