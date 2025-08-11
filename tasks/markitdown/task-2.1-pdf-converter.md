# Task 2.1: Implement PDF Converter with Markitdown

## Objective
Replace the existing PDFConverter with a markitdown-based implementation that provides superior PDF processing capabilities including better table extraction, OCR, and markdown formatting.

## Scope
- Replace `packages/morag-document/src/morag_document/converters/pdf.py` with markitdown implementation
- Maintain API compatibility with existing PDFConverter interface
- Implement enhanced features available through markitdown
- **MANDATORY**: Test thoroughly and validate before proceeding to Task 2.2

## Implementation Details

### 1. Replace PDF Converter Implementation

**File**: `packages/morag-document/src/morag_document/converters/pdf.py`

```python
"""PDF document converter using markitdown."""

import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
import structlog

from morag_core.interfaces.converter import ConversionOptions, ConversionError
from morag_core.models.document import Document, DocumentMetadata, DocumentType
from morag_document.services.markitdown_service import MarkitdownService
from .base import DocumentConverter

logger = structlog.get_logger(__name__)

class PDFConverter(DocumentConverter):
    """PDF document converter using markitdown framework."""

    def __init__(self):
        """Initialize PDF converter."""
        super().__init__()
        self.supported_formats = {"pdf"}
        self.markitdown_service = None
    
    def _get_markitdown_service(self) -> MarkitdownService:
        """Get or create markitdown service instance."""
        if self.markitdown_service is None:
            # Configure markitdown for PDF processing
            config = {
                'enable_plugins': False,  # Start with basic functionality
                'use_azure_doc_intel': False,  # Will be enabled in Task 3.1
                'use_llm_image_description': False  # Will be enabled in Task 3.2
            }
            self.markitdown_service = MarkitdownService(config)
        return self.markitdown_service

    async def _extract_text(self, file_path: Path, document: Document, options: ConversionOptions) -> Document:
        """Extract text from PDF using markitdown.
        
        Args:
            file_path: Path to PDF file
            document: Document to update
            options: Conversion options
            
        Returns:
            Updated document with extracted text
            
        Raises:
            ConversionError: If text extraction fails
        """
        try:
            logger.info("Processing PDF with markitdown", file_path=str(file_path))
            
            # Get markitdown service
            service = self._get_markitdown_service()
            
            # Convert PDF to markdown
            result = await service.convert_file(file_path)
            
            if not result['success']:
                raise ConversionError("Markitdown conversion failed")
            
            # Extract markdown content
            markdown_content = result['text_content']
            document.raw_text = markdown_content
            
            # Update metadata from markitdown result
            markitdown_metadata = result.get('metadata', {})
            if markitdown_metadata:
                # Map markitdown metadata to document metadata
                if 'title' in markitdown_metadata:
                    document.metadata.title = markitdown_metadata['title']
                if 'author' in markitdown_metadata:
                    document.metadata.author = markitdown_metadata['author']
                if 'creation_date' in markitdown_metadata:
                    document.metadata.created_at = markitdown_metadata['creation_date']
                if 'modification_date' in markitdown_metadata:
                    document.metadata.modified_at = markitdown_metadata['modification_date']
            
            # Estimate word count from markdown content
            # Remove markdown formatting for more accurate count
            import re
            text_only = re.sub(r'[#*`\[\]()_~]', '', markdown_content)
            document.metadata.word_count = len(text_only.split())
            
            # Set document type
            document.metadata.document_type = DocumentType.PDF
            
            logger.info(
                "PDF processing completed",
                file_path=str(file_path),
                word_count=document.metadata.word_count,
                content_length=len(markdown_content)
            )
            
            return document
            
        except Exception as e:
            logger.error(
                "PDF text extraction failed",
                error=str(e),
                error_type=e.__class__.__name__,
                file_path=str(file_path),
            )
            raise ConversionError(f"Failed to extract text from PDF: {str(e)}")

    async def supports_format(self, format_type: str) -> bool:
        """Check if format is supported."""
        return format_type.lower() == "pdf"
```

### 2. Update Document Processor Registration

**File**: `packages/morag-document/src/morag_document/processor.py`

Update the PDF converter registration to use the new markitdown-based implementation:

```python
def _register_converters(self) -> None:
    """Register document converters."""
    # Register PDF converter (now using markitdown)
    pdf_converter = PDFConverter()
    for format_type in pdf_converter.supported_formats:
        self.converters[format_type] = pdf_converter
    
    # ... rest of converters remain unchanged for now
```

## Testing Requirements - MANDATORY

### Unit Tests

**File**: `packages/morag-document/tests/test_pdf_converter_markitdown.py`

```python
"""Tests for markitdown-based PDF converter."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from morag_document.converters.pdf import PDFConverter
from morag_core.interfaces.converter import ConversionOptions, ConversionError
from morag_core.models.document import Document, DocumentMetadata, DocumentType

class TestPDFConverterMarkitdown:
    
    def test_initialization(self):
        """Test PDF converter initialization."""
        converter = PDFConverter()
        assert converter.supported_formats == {"pdf"}
        assert converter.markitdown_service is None
    
    @pytest.mark.asyncio
    async def test_supports_format(self):
        """Test format support checking."""
        converter = PDFConverter()
        assert await converter.supports_format("pdf")
        assert not await converter.supports_format("docx")
        assert not await converter.supports_format("txt")
    
    @pytest.mark.asyncio
    async def test_extract_text_success(self, tmp_path):
        """Test successful PDF text extraction."""
        # Create test PDF file path
        pdf_file = tmp_path / "test.pdf"
        pdf_file.touch()  # Create empty file for testing
        
        # Create document and options
        document = Document(metadata=DocumentMetadata())
        options = ConversionOptions()
        
        converter = PDFConverter()
        
        # Mock markitdown service
        mock_service = Mock()
        mock_service.convert_file = AsyncMock(return_value={
            'success': True,
            'text_content': '# Test Document\n\nThis is a test PDF content.',
            'metadata': {
                'title': 'Test Document',
                'author': 'Test Author'
            }
        })
        
        with patch.object(converter, '_get_markitdown_service', return_value=mock_service):
            result = await converter._extract_text(pdf_file, document, options)
            
            assert result.raw_text == '# Test Document\n\nThis is a test PDF content.'
            assert result.metadata.title == 'Test Document'
            assert result.metadata.author == 'Test Author'
            assert result.metadata.document_type == DocumentType.PDF
            assert result.metadata.word_count > 0
    
    @pytest.mark.asyncio
    async def test_extract_text_failure(self, tmp_path):
        """Test PDF text extraction failure."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.touch()
        
        document = Document(metadata=DocumentMetadata())
        options = ConversionOptions()
        
        converter = PDFConverter()
        
        # Mock markitdown service to raise exception
        mock_service = Mock()
        mock_service.convert_file = AsyncMock(side_effect=Exception("Conversion failed"))
        
        with patch.object(converter, '_get_markitdown_service', return_value=mock_service):
            with pytest.raises(ConversionError, match="Failed to extract text from PDF"):
                await converter._extract_text(pdf_file, document, options)
```

### Integration Tests

**File**: `packages/morag-document/tests/integration/test_pdf_converter_integration.py`

```python
"""Integration tests for PDF converter with markitdown."""

import pytest
from pathlib import Path

from morag_document.converters.pdf import PDFConverter
from morag_core.interfaces.converter import ConversionOptions
from morag_core.models.document import Document, DocumentMetadata

class TestPDFConverterIntegration:
    
    @pytest.mark.asyncio
    async def test_real_pdf_conversion(self, sample_pdf_file):
        """Test conversion with a real PDF file."""
        converter = PDFConverter()
        document = Document(metadata=DocumentMetadata())
        options = ConversionOptions()
        
        result = await converter._extract_text(sample_pdf_file, document, options)
        
        assert result.raw_text is not None
        assert len(result.raw_text) > 0
        assert result.metadata.word_count > 0
        assert result.metadata.document_type.value == "pdf"
    
    @pytest.mark.asyncio
    async def test_pdf_with_tables(self, sample_pdf_with_tables):
        """Test PDF conversion with tables."""
        converter = PDFConverter()
        document = Document(metadata=DocumentMetadata())
        options = ConversionOptions()
        
        result = await converter._extract_text(sample_pdf_with_tables, document, options)
        
        # Check that markdown table formatting is present
        assert '|' in result.raw_text  # Markdown table indicator
        assert result.raw_text is not None
        assert len(result.raw_text) > 0
```

### End-to-End Tests

**File**: `packages/morag-document/tests/e2e/test_pdf_e2e.py`

```python
"""End-to-end tests for PDF processing."""

import pytest
from pathlib import Path

from morag_document.processor import DocumentProcessor
from morag_core.models.config import ProcessingConfig

class TestPDFEndToEnd:
    
    @pytest.mark.asyncio
    async def test_pdf_processing_pipeline(self, sample_pdf_file):
        """Test complete PDF processing pipeline."""
        processor = DocumentProcessor()
        
        config = ProcessingConfig(
            file_path=str(sample_pdf_file),
            extract_metadata=True
        )
        
        result = await processor.process(config)
        
        assert result.success is True
        assert result.document is not None
        assert result.document.raw_text is not None
        assert len(result.document.raw_text) > 0
        assert result.document.metadata.document_type.value == "pdf"
```

## Testing Checklist - MUST COMPLETE BEFORE TASK 2.2

- [ ] PDF converter initializes correctly with markitdown service
- [ ] Basic PDF conversion works with sample documents
- [ ] Metadata extraction works correctly
- [ ] Table extraction produces proper markdown tables
- [ ] Error handling works for corrupted/invalid PDFs
- [ ] All unit tests pass (>90% coverage)
- [ ] Integration tests pass with real PDF files
- [ ] End-to-end tests pass through document processor
- [ ] Performance is acceptable (compare with baseline)
- [ ] No regressions in other document formats
- [ ] API compatibility maintained

## Acceptance Criteria

- [ ] PDFConverter completely replaced with markitdown implementation
- [ ] All existing PDF processing functionality maintained
- [ ] Enhanced table extraction and formatting
- [ ] Proper markdown output with structure preservation
- [ ] Comprehensive test coverage (>90%)
- [ ] All tests pass before proceeding to Task 2.2
- [ ] Performance meets or exceeds current implementation
- [ ] Error handling and logging maintained

## Dependencies
- Task 1.1: Markitdown service must be implemented and tested
- Task 1.2: Converter interface must be ready
- Task 1.3: Configuration mapping must be complete

## Estimated Effort
- **Development**: 6-8 hours
- **Testing**: 4-6 hours
- **Validation**: 2-3 hours
- **Total**: 12-17 hours

## Notes
- **CRITICAL**: Do not proceed to Task 2.2 until all tests pass
- Focus on maintaining API compatibility while leveraging markitdown's superior PDF processing
- Pay special attention to table extraction and formatting
- Ensure proper error handling for edge cases
