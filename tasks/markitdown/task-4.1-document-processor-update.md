# Task 4.1: Update Document Processor Registry (Big Bang Switch)

## Objective
Update the document processor to use only markitdown-based converters, completing the big bang migration from old converter implementations to the new markitdown framework.

## Scope
- Update converter registration in DocumentProcessor
- Ensure all format mappings use markitdown converters
- Remove any fallback mechanisms to old converters
- **MANDATORY**: Full system testing before proceeding to cleanup

## Implementation Details

### 1. Update Document Processor Registration

**File**: `packages/morag-document/src/morag_document/processor.py`

```python
"""Document processor with markitdown-based converters."""

import structlog
from typing import Dict, Optional, Union
from pathlib import Path

from morag_core.models.document import Document, DocumentType
from morag_core.utils.file_handling import get_file_info, detect_format
from morag_core.exceptions import ValidationError, ProcessingError
from morag_core.config import get_settings, validate_configuration_and_log

# Import all markitdown-based converters
from .converters.base import DocumentConverter
from .converters.pdf import PDFConverter
from .converters.word import WordConverter
from .converters.text import TextConverter
from .converters.excel import ExcelConverter
from .converters.presentation import PresentationConverter
from .converters.image import ImageConverter
from .converters.audio import AudioConverter
from .converters.archive import ArchiveConverter

logger = structlog.get_logger(__name__)

class DocumentProcessor:
    """Document processor using markitdown-based converters."""

    def __init__(self):
        """Initialize document processor."""
        self.converters: Dict[str, DocumentConverter] = {}
        self._register_converters()
        logger.info("Document processor initialized with markitdown converters")

    def _register_converters(self) -> None:
        """Register all markitdown-based document converters."""
        
        # Register PDF converter (markitdown-based)
        pdf_converter = PDFConverter()
        for format_type in pdf_converter.supported_formats:
            self.converters[format_type] = pdf_converter
            logger.debug("Registered PDF converter", format=format_type)

        # Register Word converter (markitdown-based)
        word_converter = WordConverter()
        for format_type in word_converter.supported_formats:
            self.converters[format_type] = word_converter
            logger.debug("Registered Word converter", format=format_type)

        # Register Excel converter (markitdown-based)
        excel_converter = ExcelConverter()
        for format_type in excel_converter.supported_formats:
            self.converters[format_type] = excel_converter
            logger.debug("Registered Excel converter", format=format_type)
            
        # Register PowerPoint converter (markitdown-based)
        presentation_converter = PresentationConverter()
        for format_type in presentation_converter.supported_formats:
            self.converters[format_type] = presentation_converter
            logger.debug("Registered PowerPoint converter", format=format_type)

        # Register Text converter (markitdown-based)
        text_converter = TextConverter()
        for format_type in text_converter.supported_formats:
            self.converters[format_type] = text_converter
            logger.debug("Registered Text converter", format=format_type)

        # Register Image converter (markitdown-based)
        image_converter = ImageConverter()
        for format_type in image_converter.supported_formats:
            self.converters[format_type] = image_converter
            logger.debug("Registered Image converter", format=format_type)

        # Register Audio converter (markitdown-based)
        audio_converter = AudioConverter()
        for format_type in audio_converter.supported_formats:
            self.converters[format_type] = audio_converter
            logger.debug("Registered Audio converter", format=format_type)

        # Register Archive converter (markitdown-based)
        archive_converter = ArchiveConverter()
        for format_type in archive_converter.supported_formats:
            self.converters[format_type] = archive_converter
            logger.debug("Registered Archive converter", format=format_type)

        logger.info(
            "All markitdown converters registered", 
            total_formats=len(self.converters),
            supported_formats=list(self.converters.keys())
        )

    async def get_supported_formats(self) -> set:
        """Get all supported document formats.
        
        Returns:
            Set of supported file extensions
        """
        return set(self.converters.keys())

    async def supports_format(self, format_type: str) -> bool:
        """Check if format is supported.

        Args:
            format_type: Format type string

        Returns:
            True if format is supported, False otherwise
        """
        return format_type.lower() in self.converters

    async def process(self, config: ProcessingConfig) -> ProcessingResult:
        """Process document using markitdown-based converters.

        Args:
            config: Processing configuration

        Returns:
            Processing result

        Raises:
            ProcessingError: If processing fails
            ValidationError: If input is invalid
        """
        # Validate input
        await self.validate_input(config)

        file_path = Path(config.file_path)
        
        try:
            # Detect format
            format_type = detect_format(file_path)
            
            # Get appropriate converter
            converter = self.converters.get(format_type.lower())
            if not converter:
                raise ProcessingError(f"No converter available for format: {format_type}")

            logger.info(
                "Processing document with markitdown converter",
                file_path=str(file_path),
                format_type=format_type,
                converter=converter.__class__.__name__
            )

            # Convert document
            conversion_options = ConversionOptions(
                format_type=format_type,
                chunking_strategy=getattr(config, 'chunking_strategy', ChunkingStrategy.PARAGRAPH),
                chunk_size=getattr(config, 'chunk_size', None),
                chunk_overlap=getattr(config, 'chunk_overlap', None),
                extract_metadata=getattr(config, 'extract_metadata', True),
                extract_images=getattr(config, 'extract_images', True),
                extract_tables=getattr(config, 'extract_tables', True),
            )

            conversion_result = await converter.convert(file_path, conversion_options)

            if not conversion_result.success:
                raise ProcessingError(f"Document conversion failed: {conversion_result.error}")

            # Create processing result
            result = ProcessingResult(
                success=True,
                document=conversion_result.document,
                processing_time=0.0,  # Will be calculated by caller
                metadata={
                    'converter_used': converter.__class__.__name__,
                    'format_type': format_type,
                    'markitdown_based': True,
                    **conversion_result.metadata
                },
                warnings=conversion_result.warnings,
                quality_score=conversion_result.quality_score
            )

            logger.info(
                "Document processing completed successfully",
                file_path=str(file_path),
                converter=converter.__class__.__name__,
                word_count=conversion_result.document.metadata.word_count if conversion_result.document else 0,
                quality_score=conversion_result.quality_score
            )

            return result

        except Exception as e:
            logger.error(
                "Document processing failed",
                error=str(e),
                error_type=e.__class__.__name__,
                file_path=str(file_path),
            )
            
            return ProcessingResult(
                success=False,
                document=None,
                processing_time=0.0,
                metadata={'error': str(e), 'error_type': e.__class__.__name__},
                warnings=[],
                quality_score=0.0
            )
```

### 2. Update Service Integration

**File**: `packages/morag-services/src/morag_services/document_service.py`

Ensure the service layer properly integrates with the updated document processor:

```python
"""Document service with markitdown integration."""

import structlog
from typing import Optional, Dict, Any

from morag_document.processor import DocumentProcessor
from morag_core.models.config import ProcessingConfig, ProcessingResult

logger = structlog.get_logger(__name__)

class DocumentService:
    """Document service using markitdown-based processing."""

    def __init__(self):
        """Initialize document service."""
        self.processor = DocumentProcessor()
        logger.info("Document service initialized with markitdown processor")

    async def process_document(self, file_path: str, **kwargs) -> ProcessingResult:
        """Process document using markitdown-based converters.
        
        Args:
            file_path: Path to document file
            **kwargs: Additional processing options
            
        Returns:
            Processing result
        """
        config = ProcessingConfig(
            file_path=file_path,
            **kwargs
        )
        
        logger.info("Processing document", file_path=file_path, config=kwargs)
        
        result = await self.processor.process(config)
        
        if result.success:
            logger.info(
                "Document processed successfully",
                file_path=file_path,
                converter_used=result.metadata.get('converter_used'),
                markitdown_based=result.metadata.get('markitdown_based', False)
            )
        else:
            logger.error(
                "Document processing failed",
                file_path=file_path,
                error=result.metadata.get('error')
            )
        
        return result

    async def get_supported_formats(self) -> set:
        """Get supported document formats."""
        return await self.processor.get_supported_formats()

    async def supports_format(self, format_type: str) -> bool:
        """Check if format is supported."""
        return await self.processor.supports_format(format_type)
```

## Testing Requirements - MANDATORY

### System Integration Tests

**File**: `tests/test_system_integration_markitdown.py`

```python
"""System integration tests for markitdown-based processing."""

import pytest
from pathlib import Path

from morag_document.processor import DocumentProcessor
from morag_services.document_service import DocumentService
from morag_core.models.config import ProcessingConfig

class TestSystemIntegrationMarkitdown:
    
    @pytest.mark.asyncio
    async def test_all_formats_through_processor(self, sample_documents):
        """Test all formats work through document processor."""
        processor = DocumentProcessor()
        
        test_files = {
            'pdf': 'test.pdf',
            'docx': 'test.docx', 
            'xlsx': 'test.xlsx',
            'pptx': 'test.pptx',
            'txt': 'test.txt',
            'jpg': 'test.jpg',
            'wav': 'test.wav',
            'zip': 'test.zip'
        }
        
        for format_type, filename in test_files.items():
            file_path = sample_documents / filename
            if file_path.exists():
                config = ProcessingConfig(file_path=str(file_path))
                result = await processor.process(config)
                
                assert result.success, f"Processing failed for {format_type}"
                assert result.metadata.get('markitdown_based') is True
                assert result.document is not None
                assert len(result.document.raw_text) > 0
    
    @pytest.mark.asyncio
    async def test_service_layer_integration(self, sample_documents):
        """Test service layer integration."""
        service = DocumentService()
        
        # Test PDF processing through service
        pdf_file = sample_documents / 'test.pdf'
        if pdf_file.exists():
            result = await service.process_document(str(pdf_file))
            
            assert result.success is True
            assert result.metadata.get('markitdown_based') is True
            assert result.document is not None
    
    @pytest.mark.asyncio
    async def test_supported_formats(self):
        """Test supported formats reporting."""
        processor = DocumentProcessor()
        service = DocumentService()
        
        processor_formats = await processor.get_supported_formats()
        service_formats = await service.get_supported_formats()
        
        # Should have all markitdown formats
        expected_formats = {
            'pdf', 'docx', 'pptx', 'xlsx', 'xls',
            'html', 'htm', 'csv', 'json', 'xml',
            'txt', 'md', 'markdown', 'zip',
            'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff',
            'wav', 'mp3', 'mp4', 'avi', 'mov',
            'epub', 'msg'
        }
        
        assert processor_formats >= expected_formats
        assert service_formats == processor_formats
    
    @pytest.mark.asyncio
    async def test_no_old_converters_used(self, sample_documents):
        """Verify no old converters are being used."""
        processor = DocumentProcessor()
        
        # Process various formats and check metadata
        for file_path in sample_documents.glob("*"):
            if file_path.is_file():
                config = ProcessingConfig(file_path=str(file_path))
                result = await processor.process(config)
                
                if result.success:
                    # Verify markitdown-based converter was used
                    assert result.metadata.get('markitdown_based') is True
                    converter_name = result.metadata.get('converter_used', '')
                    assert 'markitdown' in converter_name.lower() or 'Converter' in converter_name
```

### Performance Tests

**File**: `tests/test_performance_big_bang.py`

```python
"""Performance tests for big bang migration."""

import pytest
import time
from pathlib import Path

from morag_document.processor import DocumentProcessor

class TestPerformanceBigBang:
    
    @pytest.mark.asyncio
    async def test_processing_performance(self, sample_documents, performance_baseline):
        """Test processing performance after big bang switch."""
        processor = DocumentProcessor()
        
        start_time = time.time()
        processed_count = 0
        
        for file_path in sample_documents.glob("*"):
            if file_path.is_file():
                config = ProcessingConfig(file_path=str(file_path))
                result = await processor.process(config)
                
                if result.success:
                    processed_count += 1
        
        end_time = time.time()
        total_time = end_time - start_time
        
        if processed_count > 0:
            avg_time_per_doc = total_time / processed_count
            logger.info(f"Processed {processed_count} documents in {total_time:.2f}s")
            logger.info(f"Average time per document: {avg_time_per_doc:.2f}s")
            
            # Performance should be reasonable
            assert avg_time_per_doc < 30.0, "Processing too slow after migration"
```

## Testing Checklist - MUST COMPLETE BEFORE TASK 4.2

- [ ] Document processor uses only markitdown converters
- [ ] All supported formats work correctly
- [ ] Service layer integration works
- [ ] No old converters are being used
- [ ] Performance is acceptable
- [ ] All format mappings are correct
- [ ] Error handling works properly
- [ ] Logging indicates markitdown usage
- [ ] API compatibility maintained
- [ ] Full system tests pass

## Acceptance Criteria

- [ ] DocumentProcessor completely updated to use markitdown converters
- [ ] All format registrations use markitdown-based implementations
- [ ] No fallback to old converters
- [ ] Service layer properly integrated
- [ ] All supported formats work end-to-end
- [ ] Performance meets requirements
- [ ] Comprehensive test coverage
- [ ] All tests pass before proceeding to Task 4.2
- [ ] System logging clearly indicates markitdown usage

## Dependencies
- All format-specific tasks (2.1-2.6) must be complete
- All advanced feature tasks (3.1-3.4) must be complete
- All converter implementations must be tested and working

## Estimated Effort
- **Development**: 4-6 hours
- **Testing**: 6-8 hours
- **Integration**: 2-3 hours
- **Validation**: 2-3 hours
- **Total**: 14-20 hours

## Notes
- **CRITICAL**: This is the "big bang" moment - all converters switch at once
- Ensure comprehensive testing before this step
- Have rollback plan ready in case of issues
- Monitor system performance closely
- This task completes the core migration to markitdown
