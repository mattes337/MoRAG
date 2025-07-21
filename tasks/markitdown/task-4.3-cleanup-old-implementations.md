# Task 4.3: Remove Old Converter Implementations and Cleanup

## Objective
Remove all old converter implementations and related code that has been replaced by markitdown-based converters. Clean up dependencies, imports, and any remaining references to the old system.

## Scope
- Remove old converter implementation files
- Clean up imports and dependencies
- Remove obsolete configuration options
- Update documentation
- **MANDATORY**: Ensure no functionality is broken after cleanup

## Implementation Details

### 1. Files to Remove

#### Old Converter Implementations
```bash
# Remove old converter files (after confirming markitdown replacements work)
packages/morag-document/src/morag_document/converters/pdf.py.old
packages/morag-document/src/morag_document/converters/word.py.old
packages/morag-document/src/morag_document/converters/excel.py.old
packages/morag-document/src/morag_document/converters/presentation.py.old
packages/morag-document/src/morag_document/converters/text.py.old
```

#### Old Dependencies (if no longer needed)
**File**: `requirements.txt`
```txt
# Remove these if they're only used by old converters:
# python-docx>=1.1.2,<2.0.0  # Only if not used elsewhere
# openpyxl>=3.1.5,<4.0.0     # Only if not used elsewhere
# pypdf>=3.0.0               # Only if not used elsewhere
```

#### Old Configuration Options
**File**: `packages/morag-core/src/morag_core/config.py`
```python
# Remove old converter-specific configurations that are no longer needed
# Example:
# OLD_PDF_USE_OCR: bool = Field(default=False)  # Remove if markitdown handles this
# OLD_WORD_EXTRACT_IMAGES: bool = Field(default=True)  # Remove if obsolete
```

### 2. Update Imports and References

#### Document Processor Updates
**File**: `packages/morag-document/src/morag_document/processor.py`

```python
# Remove old imports
# from .converters.pdf import PDFConverter  # OLD - now using markitdown
# from .converters.word import WordConverter  # OLD - now using markitdown

# Keep only markitdown-based imports
from .converters.pdf import PDFConverter  # NEW - markitdown-based
from .converters.word import WordConverter  # NEW - markitdown-based
from .converters.excel import ExcelConverter  # NEW - markitdown-based
from .converters.presentation import PresentationConverter  # NEW - markitdown-based
from .converters.text import TextConverter  # NEW - markitdown-based
from .converters.image import ImageConverter  # NEW - markitdown-based
```

#### Test File Cleanup
```bash
# Remove old test files
packages/morag-document/tests/test_old_pdf_converter.py
packages/morag-document/tests/test_old_word_converter.py
packages/morag-document/tests/test_old_excel_converter.py
packages/morag-document/tests/test_old_presentation_converter.py
```

### 3. Documentation Updates

#### README Updates
**File**: `packages/morag-document/README.md`

```markdown
## Features

- Support for multiple document formats:
  - PDF documents (via markitdown)
  - Microsoft Word documents (.docx) (via markitdown)
  - Microsoft Excel spreadsheets (.xlsx) (via markitdown)
  - Microsoft PowerPoint presentations (.pptx) (via markitdown)
  - HTML documents (via markitdown)
  - Markdown files (via markitdown)
  - Plain text files (via markitdown)
  - Images with OCR (via markitdown)
  - Audio transcription (via markitdown)
  - ZIP files and archives (via markitdown)
  - EPUB documents (via markitdown)
- Enhanced text extraction with markdown formatting
- Superior table extraction and preservation
- Document chunking with configurable strategies
- Language detection
- Quality assessment
- Document summarization
```

#### API Documentation Updates
**File**: `docs/UNIVERSAL_DOCUMENT_CONVERSION.md`

```markdown
## Supported Formats

| Format | Converter | Status | Features |
|--------|-----------|--------|----------|
| PDF | MarkitdownPDFConverter | ✅ Active | Enhanced table extraction, OCR, markdown formatting |
| Word | MarkitdownWordConverter | ✅ Active | Structure preservation, table extraction |
| Excel | MarkitdownExcelConverter | ✅ Active | Spreadsheet to markdown conversion |
| PowerPoint | MarkitdownPresentationConverter | ✅ Active | Slide content extraction |
| Images | MarkitdownImageConverter | ✅ Active | OCR, metadata extraction, LLM descriptions |
| Audio | MarkitdownAudioConverter | ✅ Active | Transcription, metadata extraction |
| Web | MarkitdownWebConverter | ✅ Active | HTML parsing, content extraction |
| Archives | MarkitdownArchiveConverter | ✅ Active | ZIP, EPUB support |
```

### 4. Dependency Cleanup

#### Check for Unused Dependencies
```python
# Script to identify unused dependencies
# File: scripts/check_unused_dependencies.py

import ast
import os
from pathlib import Path

def find_unused_dependencies():
    """Find dependencies that might no longer be needed."""
    
    # Dependencies that were primarily used by old converters
    old_converter_deps = {
        'python-docx': 'Used by old WordConverter',
        'openpyxl': 'Used by old ExcelConverter', 
        'pypdf': 'Used by old PDFConverter',
        'python-pptx': 'Used by old PresentationConverter'
    }
    
    # Scan codebase for usage
    for dep, description in old_converter_deps.items():
        print(f"Checking usage of {dep} ({description})")
        # Implementation to scan for imports and usage
        # If not found in new code, mark for removal
```

### 5. Configuration Cleanup

#### Remove Obsolete Settings
**File**: `packages/morag-core/src/morag_core/config.py`

```python
# Remove old converter-specific settings that are now handled by markitdown
# Examples of settings to remove:

# OLD PDF settings (now handled by markitdown configuration)
# PDF_USE_DOCLING: bool = Field(default=True)
# PDF_FALLBACK_TO_PYPDF: bool = Field(default=True)
# PDF_EXTRACT_IMAGES: bool = Field(default=True)

# OLD Word settings (now handled by markitdown)
# WORD_EXTRACT_TABLES: bool = Field(default=True)
# WORD_PRESERVE_FORMATTING: bool = Field(default=True)

# OLD Excel settings (now handled by markitdown)
# EXCEL_SHEET_PROCESSING: str = Field(default="all")
# EXCEL_INCLUDE_FORMULAS: bool = Field(default=False)
```

## Testing Requirements - MANDATORY

### Cleanup Validation Tests

**File**: `tests/test_cleanup_validation.py`

```python
"""Tests to validate cleanup was successful."""

import pytest
import importlib
from pathlib import Path

class TestCleanupValidation:
    
    def test_old_converters_removed(self):
        """Verify old converter files are removed."""
        old_files = [
            "packages/morag-document/src/morag_document/converters/pdf.py.old",
            "packages/morag-document/src/morag_document/converters/word.py.old",
            # Add other old files
        ]
        
        for file_path in old_files:
            assert not Path(file_path).exists(), f"Old file still exists: {file_path}"
    
    def test_imports_work(self):
        """Verify all imports still work after cleanup."""
        try:
            from morag_document.processor import DocumentProcessor
            from morag_document.converters.pdf import PDFConverter
            from morag_document.converters.word import WordConverter
            # Test all converter imports
        except ImportError as e:
            pytest.fail(f"Import failed after cleanup: {e}")
    
    def test_no_old_references(self):
        """Verify no references to old implementations remain."""
        # Scan codebase for references to old converter classes
        # This would be a more comprehensive implementation
        pass
    
    @pytest.mark.asyncio
    async def test_all_formats_still_work(self, sample_documents):
        """Verify all document formats still work after cleanup."""
        from morag_document.processor import DocumentProcessor
        
        processor = DocumentProcessor()
        
        # Test each format
        test_files = {
            'pdf': sample_documents / 'test.pdf',
            'docx': sample_documents / 'test.docx',
            'xlsx': sample_documents / 'test.xlsx',
            'pptx': sample_documents / 'test.pptx',
        }
        
        for format_type, file_path in test_files.items():
            if file_path.exists():
                result = await processor.process_file(file_path)
                assert result.success, f"Processing failed for {format_type} after cleanup"
```

### Performance Validation

**File**: `tests/test_performance_after_cleanup.py`

```python
"""Performance tests after cleanup."""

import pytest
import time
from pathlib import Path

from morag_document.processor import DocumentProcessor

class TestPerformanceAfterCleanup:
    
    @pytest.mark.asyncio
    async def test_processing_performance(self, sample_documents, performance_baseline):
        """Verify performance hasn't degraded after cleanup."""
        processor = DocumentProcessor()
        
        start_time = time.time()
        
        # Process multiple documents
        for doc_file in sample_documents.glob("*"):
            if doc_file.is_file():
                result = await processor.process_file(doc_file)
                assert result.success
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Compare with baseline (should be similar or better)
        assert processing_time <= performance_baseline * 1.1, "Performance degraded after cleanup"
```

## Testing Checklist - MUST COMPLETE

- [ ] All old converter files removed
- [ ] All imports updated and working
- [ ] No references to old implementations remain
- [ ] All document formats still work correctly
- [ ] Performance hasn't degraded
- [ ] Configuration is clean and consistent
- [ ] Documentation is updated
- [ ] Dependencies are optimized
- [ ] All tests pass after cleanup
- [ ] No regressions introduced

## Acceptance Criteria

- [ ] All old converter implementations removed
- [ ] Codebase is clean with no obsolete references
- [ ] All functionality preserved with markitdown implementations
- [ ] Dependencies optimized (unused ones removed)
- [ ] Documentation updated to reflect new architecture
- [ ] Configuration simplified and consistent
- [ ] All tests pass after cleanup
- [ ] Performance maintained or improved
- [ ] Code quality improved (less complexity, better maintainability)

## Dependencies
- Task 4.1: Document processor must be fully updated
- Task 4.2: All testing must be complete and passing
- All format-specific tasks (2.1-2.6, 3.1-3.4) must be complete

## Estimated Effort
- **Cleanup**: 4-6 hours
- **Testing**: 3-4 hours
- **Documentation**: 2-3 hours
- **Validation**: 2-3 hours
- **Total**: 11-16 hours

## Notes
- **CRITICAL**: Only remove code after confirming markitdown replacements work perfectly
- Keep backups of removed files until fully validated
- Be conservative - if unsure about removing something, keep it
- Focus on improving code maintainability and reducing complexity
- Ensure no functionality is lost in the cleanup process
