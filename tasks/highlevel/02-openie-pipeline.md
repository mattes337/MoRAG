# Task 2: Enhanced OpenIE Pipeline Integration

## Overview

Complete the OpenIE pipeline integration to work seamlessly with SpaCy NER, ensuring proper relation extraction and entity linking between OpenIE triplets and SpaCy entities.

## Current Status

- ✅ OpenIE extractor framework exists in `morag-graph/extractors/`
- ✅ Entity linking system partially implemented
- ✅ Triplet processing and validation in place
- ❌ Full integration with SpaCy entities incomplete
- ❌ Relation validation and scoring needs enhancement
- ❌ Pipeline coordination requires improvement

## Subtasks

### 2.1 Enhance OpenIE Service Integration

**File**: `packages/morag-graph/src/morag_graph/services/openie_service.py`

**Requirements**:
- Improve OpenIE service reliability and performance
- Better sentence processing and triplet extraction
- Enhanced error handling and recovery
- Integration with document chunking strategy
- Support for different OpenIE backends

**Implementation Steps**:
1. Review and optimize existing OpenIE service
2. Improve sentence segmentation and processing
3. Add support for multiple OpenIE backends (Stanford, AllenNLP)
4. Implement better error handling and timeouts
5. Add performance monitoring and logging

**Expected Output**:
```python
class OpenIEService:
    async def extract_triplets(self, text: str, chunk_info: Optional[Dict] = None) -> List[OpenIETriplet]:
        # Returns high-quality triplets with source tracking
```

### 2.2 Implement Entity Linking Between OpenIE and SpaCy

**File**: `packages/morag-graph/src/morag_graph/normalizers/entity_linker.py`

**Requirements**:
- Link OpenIE triplet entities to SpaCy NER entities
- Use canonical forms for matching
- Handle fuzzy matching and semantic similarity
- Maintain confidence scores for links
- Support multi-language entity matching

**Implementation Steps**:
1. Enhance existing entity linking logic
2. Implement canonical form matching
3. Add fuzzy string matching capabilities
4. Integrate semantic similarity scoring
5. Add comprehensive validation and testing

**Expected Output**:
- OpenIE "the pilot" → SpaCy "pilot" (canonical: "pilot")
- OpenIE "Einstein's theory" → SpaCy "Einstein" + "theory"
- Confidence scores for each link

### 2.3 Build Relation Validation and Scoring

**File**: `packages/morag-graph/src/morag_graph/processors/relation_validator.py`

**Requirements**:
- Validate extracted relations for quality and relevance
- Score relations based on confidence and context
- Filter out low-quality or nonsensical relations
- Preserve context information in relations
- Support domain-specific validation rules

**Implementation Steps**:
1. Create relation validation framework
2. Implement quality scoring algorithms
3. Add context preservation logic
4. Create domain-agnostic validation rules
5. Add LLM-based validation for complex cases

**Expected Output**:
```python
class RelationValidator:
    async def validate_relations(self, relations: List[Relation]) -> List[ValidatedRelation]:
        # Returns filtered and scored relations with context
```

## Acceptance Criteria

### Functional
- [ ] OpenIE service extracts high-quality triplets consistently
- [ ] Entity linking connects OpenIE and SpaCy entities accurately
- [ ] Relation validation filters out low-quality relations
- [ ] Context information is preserved in relations
- [ ] Multi-language support works correctly

### Quality
- [ ] Triplet extraction precision > 85%
- [ ] Entity linking accuracy > 90%
- [ ] Relation validation reduces noise by > 70%
- [ ] Processing speed < 5s for 1000-word documents
- [ ] Memory usage scales linearly with document size

### Technical
- [ ] Robust error handling for OpenIE failures
- [ ] Graceful degradation when services unavailable
- [ ] Comprehensive logging for debugging
- [ ] Unit and integration tests cover all components
- [ ] Performance monitoring and metrics

## Dependencies

### External
- OpenIE backend (Stanford CoreNLP or AllenNLP)
- `fuzzywuzzy` for string matching
- `sentence-transformers` for semantic similarity

### Internal
- `morag-graph.extraction.spacy_extractor.SpacyEntityExtractor`
- `morag-graph.models.Entity` and `morag-graph.models.Relation`
- `morag-graph.services.openie_service.OpenIEService`

## Testing Strategy

### Unit Tests
- Test OpenIE service with various text types
- Test entity linking with different entity pairs
- Test relation validation with good and bad relations
- Test error handling and edge cases

### Integration Tests
- Test full OpenIE pipeline with SpaCy integration
- Test performance with large documents
- Test multi-language document processing
- Test coordination with graph building

### Test Data
- Create test documents with known entity-relation patterns
- Include challenging cases (ambiguous entities, complex relations)
- Test with different domains (scientific, news, technical)

## Implementation Notes

### Entity Linking Strategy
- Primary: Exact canonical form matching
- Secondary: Fuzzy string matching (>80% similarity)
- Tertiary: Semantic similarity using embeddings
- Fallback: Create new entity if no good match

### Relation Quality Scoring
- Confidence from OpenIE extraction
- Entity linking quality scores
- Context relevance to document
- Grammatical correctness
- Domain-specific validation

### Performance Optimization
- Batch processing for multiple sentences
- Caching of entity embeddings
- Parallel processing where possible
- Memory-efficient data structures

## Files to Create/Modify

### New Files
- `packages/morag-graph/src/morag_graph/processors/relation_validator.py`
- `packages/morag-graph/tests/processors/test_relation_validator.py`
- `packages/morag-graph/tests/normalizers/test_entity_linker_enhanced.py`

### Modified Files
- `packages/morag-graph/src/morag_graph/services/openie_service.py`
- `packages/morag-graph/src/morag_graph/normalizers/entity_linker.py`
- `packages/morag-graph/src/morag_graph/extractors/openie_extractor.py`
- `packages/morag-graph/src/morag_graph/builders/enhanced_graph_builder.py`

## Estimated Timeline

- **Week 1**: OpenIE service enhancement and entity linking improvement
- **Week 2**: Relation validation and pipeline integration
- **Total**: 2 weeks for complete implementation and testing

## Success Metrics

### Before Enhancement
- Triplet extraction: ~70% precision
- Entity linking: ~75% accuracy
- Relation noise: ~40% low-quality relations

### After Enhancement
- Triplet extraction: >85% precision
- Entity linking: >90% accuracy
- Relation noise: <15% low-quality relations
- Processing speed: 2x improvement
- Error rate: <5% for typical documents
