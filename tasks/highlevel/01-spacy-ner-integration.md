# Task 1: Complete SpaCy NER Integration

## Overview

Implement comprehensive SpaCy NER integration with proper entity extraction, normalization, and integration with the existing entity extraction pipeline. Ensure entities are normalized to singular, non-conjugated forms.

## Current Status

- ✅ Basic SpaCy integration exists in `morag-graph/normalizers/`
- ✅ Entity normalization framework in place
- ❌ Dedicated SpaCy entity extractor missing
- ❌ Full pipeline integration incomplete
- ❌ Multi-language model support needs enhancement

## Subtasks

### 1.1 Create SpaCy Entity Extractor

**File**: `packages/morag-graph/src/morag_graph/extraction/spacy_extractor.py`

**Requirements**:
- Implement `SpacyEntityExtractor` class inheriting from `BaseExtractor`
- Support multiple language models (en_core_web_lg, de_core_news_lg, es_core_news_lg)
- Automatic language detection and model selection
- Confidence scoring for extracted entities
- Integration with existing entity models

**Implementation Steps**:
1. Create base SpaCy extractor class
2. Implement language detection logic
3. Add model loading and caching
4. Implement entity extraction with confidence scores
5. Add proper error handling and logging

**Expected Output**:
```python
class SpacyEntityExtractor(BaseExtractor):
    async def extract(self, text: str, language: Optional[str] = None) -> List[Entity]:
        # Returns normalized entities with confidence scores
```

### 1.2 Implement Entity Normalization for SpaCy

**File**: `packages/morag-graph/src/morag_graph/normalizers/spacy_normalizer.py`

**Requirements**:
- Normalize entities to singular, non-conjugated forms
- Handle proper nouns correctly
- Support multiple languages (English, Spanish, German)
- Remove unnecessary articles and prepositions
- Maintain entity type information

**Implementation Steps**:
1. Create language-specific normalization rules
2. Implement lemmatization and stemming
3. Add proper noun detection and preservation
4. Create multi-language support
5. Add validation and quality checks

**Expected Output**:
- 'brains' → 'brain'
- 'pilotinnen' → 'pilot'
- 'los doctores' → 'doctor'
- Proper nouns preserved: 'Einstein' → 'Einstein'

### 1.3 Integrate SpaCy with Existing Pipeline

**Files**:
- `packages/morag-graph/src/morag_graph/builders/enhanced_graph_builder.py`
- `packages/morag/src/morag/graph_extractor_wrapper.py`

**Requirements**:
- Integrate SpaCy extractor with existing graph builder
- Coordinate with LLM-based extraction
- Ensure proper entity deduplication
- Maintain backward compatibility
- Add configuration options

**Implementation Steps**:
1. Modify graph builder to include SpaCy extraction
2. Implement entity merging logic
3. Add configuration for SpaCy vs LLM extraction
4. Update ingestion coordinator
5. Add comprehensive testing

## Acceptance Criteria

### Functional
- [ ] SpaCy extractor processes text and returns normalized entities
- [ ] Multi-language support works correctly
- [ ] Entity normalization follows user preferences (singular, base forms)
- [ ] Integration with existing pipeline maintains all functionality
- [ ] Confidence scores are accurate and meaningful

### Quality
- [ ] Entity extraction accuracy > 90% on test datasets
- [ ] Normalization consistency > 95%
- [ ] Language detection accuracy > 98%
- [ ] Processing speed < 2s for 1000-word documents
- [ ] Memory usage remains reasonable for large documents

### Technical
- [ ] Proper error handling for missing language models
- [ ] Graceful fallback to available models
- [ ] Comprehensive logging for debugging
- [ ] Unit tests cover all major functionality
- [ ] Integration tests validate end-to-end workflow

## Dependencies

### External
- `spacy` >= 3.7.0
- Language models: `en_core_web_lg`, `de_core_news_lg`, `es_core_news_lg`
- `langdetect` for language detection

### Internal
- `morag-graph.extraction.base.BaseExtractor`
- `morag-graph.models.Entity`
- `morag-graph.normalizers.entity_normalizer.EntityNormalizer`

## Testing Strategy

### Unit Tests
- Test entity extraction for each supported language
- Test normalization rules for various entity types
- Test confidence scoring accuracy
- Test error handling for edge cases

### Integration Tests
- Test full pipeline with SpaCy integration
- Test coordination with LLM-based extraction
- Test performance with large documents
- Test multi-language document processing

### Test Data
- Create test documents in English, Spanish, German
- Include various entity types (PERSON, ORG, LOC, MISC)
- Test edge cases (mixed languages, technical terms, proper nouns)

## Implementation Notes

### Language Model Management
- Lazy loading of language models to reduce startup time
- Caching of loaded models for performance
- Fallback strategy when preferred model unavailable

### Entity Deduplication
- Compare normalized forms between SpaCy and LLM entities
- Use confidence scores to resolve conflicts
- Preserve highest-quality entity information

### Performance Considerations
- Batch processing for multiple documents
- Parallel processing for different languages
- Memory management for large document collections

## Files to Create/Modify

### New Files
- `packages/morag-graph/src/morag_graph/extraction/spacy_extractor.py`
- `packages/morag-graph/src/morag_graph/normalizers/spacy_normalizer.py`
- `packages/morag-graph/tests/extraction/test_spacy_extractor.py`
- `packages/morag-graph/tests/normalizers/test_spacy_normalizer.py`

### Modified Files
- `packages/morag-graph/src/morag_graph/extraction/__init__.py`
- `packages/morag-graph/src/morag_graph/builders/enhanced_graph_builder.py`
- `packages/morag/src/morag/graph_extractor_wrapper.py`
- `packages/morag-graph/pyproject.toml` (add spacy dependencies)

## Estimated Timeline

- **Week 1**: SpaCy extractor implementation and testing
- **Week 2**: Entity normalization and pipeline integration
- **Total**: 2 weeks for complete implementation and testing
