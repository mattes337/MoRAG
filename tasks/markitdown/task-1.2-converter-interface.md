# Task 1.2: Create Markitdown-Based Converter Interface

## Objective
Create a base converter interface that integrates markitdown functionality while maintaining compatibility with the existing MoRAG converter architecture.

## Scope
- Create base markitdown converter class
- Implement common markitdown integration patterns
- Ensure API compatibility with existing converter interface
- Set up quality assessment for markitdown output
- **MANDATORY**: Test thoroughly before proceeding to Task 1.3

## Implementation Details

### 1. Create Base Markitdown Converter

**File**: `packages/morag-document/src/morag_document/converters/markitdown_base.py`

```python
"""Base converter class for markitdown-based document processing."""

import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, Union
import structlog

from morag_core.interfaces.converter import (
    BaseConverter,
    ConversionOptions,
    ConversionResult,
    QualityScore,
    ConversionError,
    UnsupportedFormatError,
)
from morag_core.models.document import Document, DocumentMetadata, DocumentType
from morag_core.utils.file_handling import get_file_info, detect_format
from morag_document.services.markitdown_service import MarkitdownService

logger = structlog.get_logger(__name__)

class MarkitdownBaseConverter(BaseConverter):
    """Base converter class using markitdown framework."""

    def __init__(self, supported_formats: set, converter_name: str):
        """Initialize markitdown-based converter.
        
        Args:
            supported_formats: Set of supported file extensions
            converter_name: Name of the converter for logging
        """
        self.supported_formats = supported_formats
        self.converter_name = converter_name
        self.markitdown_service = None
        logger.info(f"{converter_name} initialized", supported_formats=list(supported_formats))

    def _get_markitdown_service(self) -> MarkitdownService:
        """Get or create markitdown service instance."""
        if self.markitdown_service is None:
            # Get converter-specific configuration
            config = self._get_markitdown_config()
            self.markitdown_service = MarkitdownService(config)
        return self.markitdown_service

    def _get_markitdown_config(self) -> Dict[str, Any]:
        """Get markitdown configuration for this converter.
        
        Override in subclasses to provide converter-specific configuration.
        
        Returns:
            Configuration dictionary for markitdown service
        """
        return {
            'enable_plugins': False,
            'use_azure_doc_intel': False,
            'use_llm_image_description': False
        }

    async def convert(self, file_path: Union[str, Path], options: Optional[ConversionOptions] = None) -> ConversionResult:
        """Convert document using markitdown.
        
        Args:
            file_path: Path to document file
            options: Conversion options
            
        Returns:
            Conversion result with markdown content
            
        Raises:
            ConversionError: If conversion fails
            UnsupportedFormatError: If format is not supported
        """
        file_path = Path(file_path)
        options = options or ConversionOptions()

        # Validate input
        if not file_path.exists():
            raise ConversionError(f"File not found: {file_path}")

        # Check format support
        format_type = options.format_type or detect_format(file_path)
        if not self.supports_format(format_type):
            raise UnsupportedFormatError(f"Format '{format_type}' not supported by {self.converter_name}")

        try:
            logger.info(
                f"Converting document with {self.converter_name}",
                file_path=str(file_path),
                format_type=format_type
            )

            # Get markitdown service
            service = self._get_markitdown_service()

            # Convert document
            markitdown_result = await service.convert_file(file_path)

            if not markitdown_result['success']:
                raise ConversionError("Markitdown conversion failed")

            # Create document from markitdown result
            document = await self._create_document_from_markitdown(
                markitdown_result, file_path, options
            )

            # Apply post-processing
            document = await self._post_process_document(document, options)

            # Assess quality
            quality_score = await self._assess_quality(document, markitdown_result)

            # Create conversion result
            result = ConversionResult(
                success=True,
                content=document.raw_text,
                metadata={
                    'converter': self.converter_name,
                    'markitdown_based': True,
                    'format_type': format_type,
                    **markitdown_result.get('metadata', {})
                },
                quality_score=quality_score,
                document=document,
                warnings=[]
            )

            logger.info(
                f"Document conversion completed with {self.converter_name}",
                file_path=str(file_path),
                word_count=document.metadata.word_count,
                quality_score=quality_score
            )

            return result

        except Exception as e:
            logger.error(
                f"Document conversion failed with {self.converter_name}",
                error=str(e),
                error_type=e.__class__.__name__,
                file_path=str(file_path),
            )
            raise ConversionError(f"{self.converter_name} conversion failed: {str(e)}")

    async def _create_document_from_markitdown(
        self, 
        markitdown_result: Dict[str, Any], 
        file_path: Path, 
        options: ConversionOptions
    ) -> Document:
        """Create Document object from markitdown result.
        
        Args:
            markitdown_result: Result from markitdown conversion
            file_path: Original file path
            options: Conversion options
            
        Returns:
            Document object with extracted content
        """
        # Create document metadata
        metadata = DocumentMetadata()
        metadata.file_path = str(file_path)
        metadata.file_name = file_path.name
        metadata.file_size = file_path.stat().st_size if file_path.exists() else 0

        # Extract metadata from markitdown result
        md_metadata = markitdown_result.get('metadata', {})
        if 'title' in md_metadata:
            metadata.title = md_metadata['title']
        if 'author' in md_metadata:
            metadata.author = md_metadata['author']
        if 'creation_date' in md_metadata:
            metadata.created_at = md_metadata['creation_date']
        if 'modification_date' in md_metadata:
            metadata.modified_at = md_metadata['modification_date']

        # Set document type based on file extension
        format_type = detect_format(file_path)
        metadata.document_type = self._get_document_type(format_type)

        # Create document
        document = Document(metadata=metadata)
        document.raw_text = markitdown_result['text_content']

        # Calculate word count
        if document.raw_text:
            # Remove markdown formatting for more accurate count
            import re
            text_only = re.sub(r'[#*`\[\]()_~]', '', document.raw_text)
            metadata.word_count = len(text_only.split())

        return document

    async def _post_process_document(self, document: Document, options: ConversionOptions) -> Document:
        """Post-process document after markitdown conversion.
        
        Override in subclasses for format-specific post-processing.
        
        Args:
            document: Document to post-process
            options: Conversion options
            
        Returns:
            Post-processed document
        """
        # Default implementation - no post-processing
        return document

    async def _assess_quality(self, document: Document, markitdown_result: Dict[str, Any]) -> float:
        """Assess quality of markitdown conversion.
        
        Args:
            document: Converted document
            markitdown_result: Original markitdown result
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        quality_score = 0.8  # Default quality score

        # Basic quality checks
        if document.raw_text and len(document.raw_text) > 0:
            quality_score += 0.1

        if document.metadata.word_count and document.metadata.word_count > 0:
            quality_score += 0.1

        # Check for markdown structure
        if document.raw_text:
            # Look for markdown elements
            markdown_indicators = ['#', '|', '*', '-', '```', '[', ']']
            found_indicators = sum(1 for indicator in markdown_indicators if indicator in document.raw_text)
            structure_score = min(found_indicators / len(markdown_indicators), 1.0) * 0.1
            quality_score += structure_score

        return min(quality_score, 1.0)

    def _get_document_type(self, format_type: str) -> DocumentType:
        """Get document type from format.
        
        Args:
            format_type: File format
            
        Returns:
            DocumentType enum value
        """
        format_mapping = {
            'pdf': DocumentType.PDF,
            'docx': DocumentType.WORD,
            'xlsx': DocumentType.EXCEL,
            'pptx': DocumentType.POWERPOINT,
            'txt': DocumentType.TEXT,
            'md': DocumentType.MARKDOWN,
            'html': DocumentType.HTML,
            'jpg': DocumentType.IMAGE,
            'jpeg': DocumentType.IMAGE,
            'png': DocumentType.IMAGE,
            'wav': DocumentType.AUDIO,
            'mp3': DocumentType.AUDIO,
        }
        
        return format_mapping.get(format_type.lower(), DocumentType.OTHER)

    def supports_format(self, format_type: str) -> bool:
        """Check if format is supported.
        
        Args:
            format_type: Format type to check
            
        Returns:
            True if format is supported
        """
        return format_type.lower() in self.supported_formats
```

### 2. Update Base Converter Import

**File**: `packages/morag-document/src/morag_document/converters/base.py`

Add import for the new markitdown base converter:

```python
# Add to existing imports
from .markitdown_base import MarkitdownBaseConverter

# Export the new base class
__all__ = ['DocumentConverter', 'MarkitdownBaseConverter']
```

## Testing Requirements - MANDATORY

### Unit Tests

**File**: `packages/morag-document/tests/test_markitdown_base_converter.py`

```python
"""Tests for markitdown base converter."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from morag_document.converters.markitdown_base import MarkitdownBaseConverter
from morag_core.interfaces.converter import ConversionOptions, ConversionError
from morag_core.models.document import DocumentType

class TestMarkitdownBaseConverter:
    
    def test_initialization(self):
        """Test base converter initialization."""
        converter = MarkitdownBaseConverter({'pdf'}, 'TestConverter')
        assert converter.supported_formats == {'pdf'}
        assert converter.converter_name == 'TestConverter'
        assert converter.markitdown_service is None
    
    def test_supports_format(self):
        """Test format support checking."""
        converter = MarkitdownBaseConverter({'pdf', 'docx'}, 'TestConverter')
        assert converter.supports_format('pdf')
        assert converter.supports_format('docx')
        assert not converter.supports_format('txt')
    
    @pytest.mark.asyncio
    async def test_convert_success(self, tmp_path):
        """Test successful conversion."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("test content")
        
        converter = MarkitdownBaseConverter({'pdf'}, 'TestConverter')
        
        # Mock markitdown service
        mock_service = Mock()
        mock_service.convert_file = AsyncMock(return_value={
            'success': True,
            'text_content': '# Test Document\n\nTest content.',
            'metadata': {'title': 'Test Document'}
        })
        
        with patch.object(converter, '_get_markitdown_service', return_value=mock_service):
            options = ConversionOptions(format_type='pdf')
            result = await converter.convert(test_file, options)
            
            assert result.success is True
            assert result.document is not None
            assert result.document.raw_text == '# Test Document\n\nTest content.'
            assert result.metadata['markitdown_based'] is True
    
    @pytest.mark.asyncio
    async def test_convert_file_not_found(self):
        """Test conversion with non-existent file."""
        converter = MarkitdownBaseConverter({'pdf'}, 'TestConverter')
        
        with pytest.raises(ConversionError, match="File not found"):
            await converter.convert("nonexistent.pdf")
    
    @pytest.mark.asyncio
    async def test_quality_assessment(self):
        """Test quality assessment."""
        converter = MarkitdownBaseConverter({'pdf'}, 'TestConverter')
        
        # Create mock document
        from morag_core.models.document import Document, DocumentMetadata
        document = Document(metadata=DocumentMetadata())
        document.raw_text = "# Test\n\n| Table | Data |\n|-------|------|\n| A | B |"
        document.metadata.word_count = 5
        
        markitdown_result = {'text_content': document.raw_text}
        
        quality = await converter._assess_quality(document, markitdown_result)
        assert 0.0 <= quality <= 1.0
        assert quality > 0.8  # Should be high quality with markdown structure
```

## Testing Checklist - MUST COMPLETE BEFORE TASK 1.3

- [ ] MarkitdownBaseConverter initializes correctly
- [ ] Format support checking works
- [ ] Document conversion pipeline works
- [ ] Quality assessment functions properly
- [ ] Error handling works for various scenarios
- [ ] Metadata extraction and mapping works
- [ ] Document type detection works
- [ ] All unit tests pass (>90% coverage)
- [ ] Integration with MarkitdownService works
- [ ] API compatibility with existing converter interface

## Acceptance Criteria

- [ ] MarkitdownBaseConverter class implemented and tested
- [ ] API compatibility with existing converter interface maintained
- [ ] Quality assessment for markitdown output implemented
- [ ] Proper error handling and logging
- [ ] Metadata extraction and mapping functionality
- [ ] Document type detection and assignment
- [ ] Comprehensive unit tests with >90% coverage
- [ ] All tests pass before proceeding to Task 1.3
- [ ] Clean integration with MarkitdownService

## Dependencies
- Task 1.1: MarkitdownService must be implemented and tested

## Estimated Effort
- **Development**: 6-8 hours
- **Testing**: 4-5 hours
- **Integration**: 2-3 hours
- **Validation**: 1-2 hours
- **Total**: 13-18 hours

## Notes
- **CRITICAL**: Do not proceed to Task 1.3 until all tests pass
- This base class will be used by all format-specific converters
- Focus on clean abstraction and reusability
- Ensure proper error handling and quality assessment
- Maintain compatibility with existing converter patterns
