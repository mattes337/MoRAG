# Markitdown Framework Integration

## Overview

This directory contains tasks for integrating Microsoft's markitdown framework into the MoRAG document processing pipeline. Markitdown is a lightweight Python utility for converting various files to Markdown for use with LLMs and text analysis pipelines.

## Why Markitdown?

- **LLM-Optimized**: Mainstream LLMs natively "speak" Markdown and understand it well
- **Token-Efficient**: Markdown conventions are highly token-efficient
- **Structure Preservation**: Maintains important document structure (headings, lists, tables, links)
- **Comprehensive Format Support**: PDF, PowerPoint, Word, Excel, Images, Audio, HTML, and more
- **Quality Focus**: Designed for text analysis tools rather than human-readable output

## Integration Strategy - Big Bang Migration

### Phase 1: Foundation Setup (Tasks 1.1-1.3)
- [ ] **Task 1.1**: Add markitdown dependency and create wrapper service
- [ ] **Task 1.2**: Create markitdown-based converter interface
- [ ] **Task 1.3**: Implement configuration and options mapping
- [ ] **Test Phase 1**: Verify foundation components work correctly

### Phase 2: Format Implementation (Tasks 2.1-2.6)
Each task includes implementation + testing + validation before proceeding:

- [ ] **Task 2.1**: Implement PDF converter with markitdown + **TEST & VALIDATE**
- [ ] **Task 2.2**: Implement Word converter with markitdown + **TEST & VALIDATE**
- [ ] **Task 2.3**: Implement Excel converter with markitdown + **TEST & VALIDATE**
- [ ] **Task 2.4**: Implement PowerPoint converter with markitdown + **TEST & VALIDATE**
- [ ] **Task 2.5**: Implement image processing with markitdown + **TEST & VALIDATE**
- [ ] **Task 2.6**: Implement additional formats (ZIP, EPUB, etc.) + **TEST & VALIDATE**

### Phase 3: Advanced Features (Tasks 3.1-3.4)
- [ ] **Task 3.1**: Implement Azure Document Intelligence integration + **TEST & VALIDATE**
- [ ] **Task 3.2**: Add LLM-based image description support + **TEST & VALIDATE**
- [ ] **Task 3.3**: Implement audio transcription capabilities + **TEST & VALIDATE**
- [ ] **Task 3.4**: Add quality assessment and validation + **TEST & VALIDATE**

### Phase 4: Integration & Cleanup (Tasks 4.1-4.3)
- [ ] **Task 4.1**: Update document processor registry (big bang switch) + **FULL SYSTEM TEST**
- [ ] **Task 4.2**: Comprehensive end-to-end testing and validation
- [ ] **Task 4.3**: Remove old converter implementations and cleanup

## Affected Code Files

### Files to Modify
```
requirements.txt                                           # Add markitdown dependency
packages/morag-core/src/morag_core/config.py              # Add markitdown configuration
packages/morag-document/pyproject.toml                    # Add markitdown dependency
packages/morag-document/src/morag_document/processor.py   # Update converter registration
packages/morag-services/src/morag_services/document_service.py # Update service integration
```

### Files to Create
```
packages/morag-document/src/morag_document/services/markitdown_service.py
packages/morag-document/src/morag_document/converters/pdf.py              # Replace with markitdown
packages/morag-document/src/morag_document/converters/word.py             # Replace with markitdown
packages/morag-document/src/morag_document/converters/excel.py            # Replace with markitdown
packages/morag-document/src/morag_document/converters/presentation.py     # Replace with markitdown
packages/morag-document/src/morag_document/converters/text.py             # Replace with markitdown
packages/morag-document/src/morag_document/converters/image.py            # New markitdown-based
packages/morag-document/src/morag_document/converters/audio.py            # New markitdown-based
packages/morag-document/src/morag_document/converters/archive.py          # New markitdown-based
```

### Test Files to Create
```
packages/morag-document/tests/test_markitdown_service.py
packages/morag-document/tests/test_pdf_converter_markitdown.py
packages/morag-document/tests/test_word_converter_markitdown.py
packages/morag-document/tests/test_excel_converter_markitdown.py
packages/morag-document/tests/test_presentation_converter_markitdown.py
packages/morag-document/tests/test_image_converter_markitdown.py
packages/morag-document/tests/test_audio_converter_markitdown.py
packages/morag-document/tests/test_archive_converter_markitdown.py
packages/morag-document/tests/integration/test_markitdown_integration.py
tests/test_system_integration_markitdown.py
tests/test_performance_big_bang.py
tests/test_cleanup_validation.py
```

### Files to Remove (After Task 4.3)
```
# Old converter implementations (backup first)
packages/morag-document/src/morag_document/converters/pdf.py.old
packages/morag-document/src/morag_document/converters/word.py.old
packages/morag-document/src/morag_document/converters/excel.py.old
packages/morag-document/src/morag_document/converters/presentation.py.old
packages/morag-document/src/morag_document/converters/text.py.old

# Old test files
packages/morag-document/tests/test_old_*_converter.py
```

### Documentation to Update
```
packages/morag-document/README.md                         # Update features and usage
docs/UNIVERSAL_DOCUMENT_CONVERSION.md                     # Update converter table
API_USAGE_GUIDE.md                                        # Update examples if needed
```

## Progress Tracking

### Completed ✅
- Analysis of markitdown capabilities and integration requirements
- Task structure and planning

### In Progress 🔄
- Creating task directory structure and documentation

### Pending ⏳
- All implementation tasks (1.1 through 4.3)

## Dependencies

### Required
- `markitdown[all]`: Core markitdown package with all optional dependencies
- Existing MoRAG dependencies (maintained)

### Optional Enhancements
- Azure Document Intelligence endpoint configuration
- OpenAI client for LLM-based image descriptions
- Additional format-specific dependencies as needed

## Configuration Changes

### New Settings
```python
# Markitdown configuration
MARKITDOWN_ENABLED: bool = True
MARKITDOWN_USE_AZURE_DOC_INTEL: bool = False
MARKITDOWN_AZURE_ENDPOINT: Optional[str] = None
MARKITDOWN_USE_LLM_IMAGE_DESCRIPTION: bool = False
MARKITDOWN_LLM_MODEL: str = "gpt-4o"
MARKITDOWN_ENABLE_PLUGINS: bool = False
```

### Converter Strategy
- **Big Bang Migration**: Complete replacement of existing converters with markitdown
- No fallback mechanisms - markitdown becomes the single source of truth
- Thorough testing at each step to ensure stability before proceeding

## Testing Strategy - Test After Each Step

### Per-Task Testing Requirements
Each implementation task must include:
- **Unit Tests**: Individual converter functionality, configuration handling, error scenarios
- **Integration Tests**: End-to-end workflow with the specific format
- **Quality Validation**: Compare output with sample documents
- **Performance Testing**: Ensure no significant performance degradation
- **API Compatibility**: Verify existing API contracts are maintained

### System-Wide Testing
After each format implementation:
- Run full test suite to ensure no regressions
- Test document processing pipeline end-to-end
- Validate that other formats still work correctly
- Performance benchmarking against baseline

### Final Integration Testing
Before cleanup phase:
- Comprehensive end-to-end testing with all formats
- Load testing with various document types and sizes
- API compatibility validation
- Performance comparison with original implementation

## Migration Plan - Big Bang Approach

1. **Foundation Setup**: Implement core markitdown infrastructure and test thoroughly
2. **Format-by-Format Implementation**: Implement each format converter with markitdown
   - After each format: Run comprehensive tests, validate output quality
   - Fix any issues before proceeding to next format
   - No parallel implementation - direct replacement
3. **Advanced Features**: Add Azure Document Intelligence, LLM integration, etc.
   - Test each feature thoroughly before proceeding
4. **Big Bang Switch**: Update document processor registry to use only markitdown converters
   - Full system testing with all formats
   - Performance validation
5. **Cleanup**: Remove all old converter implementations and related code

## Success Criteria

- [ ] All supported document formats work with markitdown
- [ ] Quality metrics meet or exceed current converter performance
- [ ] API compatibility maintained
- [ ] Performance impact is minimal or positive
- [ ] Comprehensive test coverage achieved
- [ ] Documentation updated and complete

## Next Steps

1. Start with Task 1.1: Add markitdown dependency and create wrapper service
2. Implement core infrastructure before format-specific converters
3. Test each component thoroughly before moving to the next
4. Maintain backward compatibility throughout the process

---

**Last Updated**: 2025-01-21
**Status**: Planning Complete, Implementation Starting
