# Task 1: Fix PDF Chunking to Preserve Word Integrity

## Problem Analysis

### Current Issues
1. **Character-based chunking splits words mid-character**: The current chunking algorithm in `packages/morag-document/src/morag_document/converters/base.py` uses simple character-based splitting that can break words in the middle.

2. **Sentence chunking has poor word boundary detection**: The sentence-based chunking uses basic regex `(?<=[.!?])\s+` which doesn't handle complex sentence structures well.

3. **No semantic chunking at document structure boundaries**: Current implementation doesn't leverage document structure like sections, chapters, or headings for intelligent chunking.

4. **Overlap calculation doesn't respect word boundaries**: The overlap mechanism counts characters rather than preserving complete words in overlapping regions.

### Current Implementation Analysis
From `packages/morag-document/src/morag_document/converters/base.py`:
- CHARACTER strategy: `chunk_text = text[i:i + chunk_size]` - splits at arbitrary character positions
- SENTENCE strategy: Basic regex splitting without proper sentence boundary detection
- WORD strategy: Exists but has simplistic overlap calculation
- No semantic or structure-aware chunking

## Solution Approach

### 1. Enhanced Word Boundary Detection
- Implement intelligent word boundary detection using regex patterns
- Ensure chunks never split words mid-character
- Handle punctuation and special characters properly

### 2. Improved Sentence Boundary Detection
- Use more sophisticated sentence boundary detection
- Handle abbreviations, decimal numbers, and complex punctuation
- Consider using NLTK or spaCy for better sentence segmentation

### 3. Semantic Chunking Implementation
- Detect document structure (headings, sections, chapters)
- Implement semantic chunking that respects document hierarchy
- Use heading levels to determine chunk boundaries

### 4. Smart Overlap Calculation
- Calculate overlap based on complete words, not characters
- Ensure overlapping regions contain meaningful content
- Preserve sentence boundaries in overlap regions

## Implementation Plan

### Phase 1: Word Boundary Preservation
1. **Update CHARACTER strategy** to never split words
   - Find word boundaries near chunk size limit
   - Implement `_find_word_boundary()` helper method
   - Ensure chunks end at complete words

2. **Enhance WORD strategy** with better overlap
   - Calculate overlap in words, not characters
   - Preserve complete sentences in overlap regions
   - Add word count validation

### Phase 2: Sentence Boundary Improvement
1. **Improve sentence detection regex**
   - Handle abbreviations (Dr., Mr., etc.)
   - Handle decimal numbers (3.14, $1.50)
   - Handle quotations and parentheses
   - Add support for multiple languages

2. **Implement sentence-aware chunking**
   - Ensure chunks end at sentence boundaries
   - Handle long sentences that exceed chunk size
   - Preserve paragraph structure

### Phase 3: Semantic Chunking
1. **Document structure detection**
   - Detect headings using various patterns
   - Identify section boundaries
   - Extract table of contents information

2. **Implement SEMANTIC strategy**
   - Chunk by document sections
   - Preserve heading-content relationships
   - Handle nested section structures

### Phase 4: Integration and Testing
1. **Update chunking options**
   - Add new chunking strategies to enum
   - Update configuration options
   - Ensure backward compatibility

2. **Comprehensive testing**
   - Test with various document types
   - Validate word boundary preservation
   - Test semantic chunking accuracy

## Technical Implementation

### New Helper Methods
```python
def _find_word_boundary(self, text: str, position: int, direction: str = "backward") -> int:
    """Find the nearest word boundary from a given position."""
    
def _detect_sentence_boundaries(self, text: str) -> List[int]:
    """Detect sentence boundaries using improved regex patterns."""
    
def _detect_document_structure(self, text: str) -> List[Dict[str, Any]]:
    """Detect document structure including headings and sections."""
    
def _chunk_semantically(self, document: Document, options: ConversionOptions) -> None:
    """Implement semantic chunking based on document structure."""
```

### Enhanced Chunking Strategies
1. **CHARACTER_WORD_SAFE**: Character-based but respects word boundaries
2. **SENTENCE_IMPROVED**: Better sentence boundary detection
3. **SEMANTIC**: Structure-aware chunking
4. **HYBRID**: Combination of strategies based on content

### Configuration Options
- `respect_word_boundaries`: Boolean flag for word boundary preservation
- `sentence_detection_method`: Choose between basic, advanced, or NLP-based
- `semantic_chunking_depth`: Control how deep to go in document structure
- `min_chunk_size`: Minimum chunk size to prevent tiny chunks

## Files to Modify

### Core Files
1. `packages/morag-document/src/morag_document/converters/base.py`
   - Update `_chunk_document()` method
   - Add new helper methods
   - Implement enhanced chunking strategies

2. `packages/morag-core/src/morag_core/interfaces/converter.py`
   - Add new chunking strategy enum values
   - Update ConversionOptions class

3. `packages/morag-core/src/morag_core/config.py`
   - Add new configuration options for chunking

### Converter-Specific Files
1. `packages/morag-document/src/morag_document/converters/pdf.py`
   - Enhance chapter detection for semantic chunking
   - Improve page-based chunking with word boundaries

2. `packages/morag-document/src/morag_document/converters/word.py`
   - Leverage Word document structure for semantic chunking
   - Use heading styles for better structure detection

3. `packages/morag-document/src/morag_document/converters/text.py`
   - Implement markdown heading detection
   - Add HTML structure parsing for semantic chunking

## Testing Strategy

### Unit Tests
1. **Word boundary preservation tests**
   - Test that no words are split mid-character
   - Validate chunk boundaries are at word boundaries
   - Test with various languages and character sets

2. **Sentence boundary detection tests**
   - Test with complex sentences containing abbreviations
   - Validate handling of quotations and parentheses
   - Test with different punctuation patterns

3. **Semantic chunking tests**
   - Test with documents containing clear structure
   - Validate heading-content relationships
   - Test with nested section structures

### Integration Tests
1. **End-to-end chunking tests**
   - Test with real PDF documents
   - Validate chunk quality and coherence
   - Test with various document types and sizes

2. **Performance tests**
   - Measure chunking performance with large documents
   - Compare performance of different strategies
   - Validate memory usage with large files

## Success Criteria

### Functional Requirements
1. ✅ No words are split mid-character in any chunking strategy
2. ✅ Sentence boundaries are properly detected and preserved
3. ✅ Document structure is leveraged for semantic chunking
4. ✅ Overlap regions contain complete words and meaningful content
5. ✅ Backward compatibility with existing chunking strategies

### Quality Requirements
1. ✅ Chunk coherence improved (measured by semantic similarity within chunks)
2. ✅ Better retrieval performance (measured by search accuracy)
3. ✅ Reduced chunk fragmentation (fewer tiny or oversized chunks)
4. ✅ Preserved document context (headings associated with content)

### Performance Requirements
1. ✅ Chunking performance within 20% of current implementation
2. ✅ Memory usage remains reasonable for large documents
3. ✅ Processing time scales linearly with document size

## Dependencies

### Required Libraries
- `nltk` or `spaCy` for advanced sentence boundary detection (optional)
- `regex` for enhanced pattern matching
- Existing document processing libraries (docling, python-docx, etc.)

### Configuration Dependencies
- Environment variables for chunking strategy defaults
- Configuration options for semantic chunking parameters
- Backward compatibility with existing chunk size settings

## Risk Mitigation

### Potential Issues
1. **Performance degradation**: Enhanced chunking might be slower
   - Mitigation: Implement caching and optimize algorithms
   - Fallback: Provide option to use legacy chunking

2. **Increased complexity**: More sophisticated chunking is harder to debug
   - Mitigation: Comprehensive logging and debugging tools
   - Fallback: Clear error messages and validation

3. **Compatibility issues**: Changes might break existing workflows
   - Mitigation: Maintain backward compatibility
   - Fallback: Version chunking strategies and allow selection

### Testing Risks
1. **Insufficient test coverage**: Complex chunking needs extensive testing
   - Mitigation: Create comprehensive test suite with real documents
   - Validation: Test with various document types and languages

2. **Performance regression**: Enhanced features might slow down processing
   - Mitigation: Performance benchmarking and optimization
   - Monitoring: Continuous performance testing in CI/CD

## Implementation Timeline

### Week 1: Analysis and Design
- [ ] Analyze current chunking implementation
- [ ] Design enhanced word boundary detection
- [ ] Create detailed technical specifications

### Week 2: Core Implementation
- [ ] Implement word boundary preservation
- [ ] Enhance sentence boundary detection
- [ ] Update CHARACTER and SENTENCE strategies

### Week 3: Semantic Chunking
- [ ] Implement document structure detection
- [ ] Create SEMANTIC chunking strategy
- [ ] Add configuration options

### Week 4: Testing and Integration
- [ ] Create comprehensive test suite
- [ ] Performance testing and optimization
- [ ] Documentation and examples

## Next Steps

1. **Start with Phase 1**: Implement word boundary preservation in CHARACTER strategy
2. **Create test cases**: Develop tests that validate word integrity
3. **Benchmark performance**: Establish baseline performance metrics
4. **Iterate and improve**: Refine implementation based on test results

This task focuses on the fundamental issue of preserving word integrity while maintaining good chunking performance and quality.
