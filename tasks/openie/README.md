# OpenIE Relation Extraction Integration

## Overview

This directory contains tasks for integrating Open Information Extraction (OpenIE) into the MoRAG knowledge graph pipeline. OpenIE automatically extracts structured (subject, predicate, object) triplets from text to enrich the knowledge graph with semantic relationships beyond simple entity mentions.

## Why OpenIE?

- **Automatic Relation Discovery**: Extracts relationships without predefined schemas
- **Rich Semantic Triplets**: Provides (subject, predicate, object) structured data
- **Language Agnostic**: Works across multiple languages with proper preprocessing
- **Knowledge Graph Enhancement**: Enriches entity connections with meaningful relationships
- **Scalable Processing**: Operates on sentence-level for efficient batch processing

## Integration Strategy

### Pipeline Position
OpenIE integrates between spaCy NER and Neo4j ingestion:
```
Input → Conversion → Text Extraction → spaCy NER → OpenIE → Neo4j
```

### Processing Flow
1. **Input Processing**: Audio, Video, PDF, etc. converted to clean text
2. **Entity Recognition**: spaCy NER identifies and normalizes entities
3. **Relation Extraction**: OpenIE extracts triplets from sentences
4. **Post-Processing**: Link OpenIE entities to spaCy entities, normalize predicates
5. **Graph Ingestion**: Create nodes and relationships in Neo4j

## Task Breakdown

### Phase 1: Foundation Setup (Tasks 1.1-1.3)
- [ ] **Task 1.1**: OpenIE dependency integration and service wrapper
- [ ] **Task 1.2**: Sentence segmentation and preprocessing pipeline
- [ ] **Task 1.3**: Basic triplet extraction and validation

### Phase 2: Entity Integration (Tasks 2.1-2.3)
- [ ] **Task 2.1**: Entity linking between OpenIE and spaCy NER
- [ ] **Task 2.2**: Entity normalization and canonical mapping
- [ ] **Task 2.3**: Confidence scoring and filtering mechanisms

### Phase 3: Predicate Processing (Tasks 3.1-3.3)
- [ ] **Task 3.1**: Predicate normalization and standardization
- [ ] **Task 3.2**: Relationship type mapping and categorization
- [ ] **Task 3.3**: Quality assessment and validation rules

### Phase 4: Neo4j Integration (Tasks 4.1-4.3)
- [ ] **Task 4.1**: Graph schema extension for OpenIE relationships
- [ ] **Task 4.2**: Batch ingestion pipeline for triplets
- [ ] **Task 4.3**: Provenance tracking and metadata storage

### Phase 5: Advanced Features (Tasks 5.1-5.4)
- [ ] **Task 5.1**: Integration testing and validation
- [ ] **Task 5.2**: Multilingual normalization with German support
- [ ] **Task 5.3**: Temporal relationship extraction
- [ ] **Task 5.4**: Performance optimization and caching

## Affected Code Files

### Files to Modify
```
requirements.txt                                           # Add OpenIE dependencies
packages/morag-core/src/morag_core/config.py              # Add OpenIE configuration
packages/morag-graph/src/morag_graph/extractors/          # Add OpenIE extractor
packages/morag-graph/src/morag_graph/services/            # Update graph service
packages/morag-services/src/morag_services/graph_service.py # Update service integration
```

### Files to Create
```
packages/morag-graph/src/morag_graph/services/openie_service.py
packages/morag-graph/src/morag_graph/extractors/openie_extractor.py
packages/morag-graph/src/morag_graph/processors/sentence_processor.py
packages/morag-graph/src/morag_graph/processors/triplet_processor.py
packages/morag-graph/src/morag_graph/normalizers/entity_linker.py
packages/morag-graph/src/morag_graph/normalizers/predicate_normalizer.py
packages/morag-graph/src/morag_graph/normalizers/german_normalizer.py
packages/morag-graph/src/morag_graph/validators/triplet_validator.py
```

### Test Files to Create
```
packages/morag-graph/tests/test_openie_service.py
packages/morag-graph/tests/test_openie_extractor.py
packages/morag-graph/tests/test_sentence_processor.py
packages/morag-graph/tests/test_triplet_processor.py
packages/morag-graph/tests/test_entity_linker.py
packages/morag-graph/tests/test_predicate_normalizer.py
packages/morag-graph/tests/test_german_normalizer.py
packages/morag-graph/tests/integration/test_openie_integration.py
tests/test_openie_pipeline_integration.py
tests/test_multilingual_openie_integration.py
```

## Dependencies

### Required
- `openie`: Python wrapper for Stanford OpenIE
- `stanford-openie`: Java-based OpenIE implementation
- `spacy`: For entity recognition integration
- `nltk`: For sentence segmentation
- Existing MoRAG dependencies (maintained)

### Optional Enhancements
- `transformers`: For advanced predicate normalization
- `sentence-transformers`: For semantic similarity in entity linking
- Additional NLP libraries as needed

## Configuration Changes

### New Settings
```python
# OpenIE configuration
OPENIE_ENABLED: bool = True
OPENIE_IMPLEMENTATION: str = "stanford"  # stanford, openie5, etc.
OPENIE_CONFIDENCE_THRESHOLD: float = 0.7
OPENIE_MAX_TRIPLETS_PER_SENTENCE: int = 10
OPENIE_ENABLE_ENTITY_LINKING: bool = True
OPENIE_ENABLE_PREDICATE_NORMALIZATION: bool = True
OPENIE_BATCH_SIZE: int = 100
OPENIE_TIMEOUT_SECONDS: int = 30
```

## Processing Strategy

### Sentence-Level Processing
- Clean text segmentation into individual sentences
- Parallel processing for scalability
- Error handling for malformed sentences

### Entity Linking Strategy
- Map OpenIE entities to spaCy canonical entities
- Handle partial matches and fuzzy matching
- Maintain entity confidence scores

### Predicate Normalization
- Standardize verbose predicates to consistent forms
- Create predicate taxonomy for relationship types
- Support multi-language predicate mapping (Spanish, German, English)

## Success Criteria

- [ ] OpenIE successfully extracts triplets from processed text
- [ ] Entity linking achieves >85% accuracy with spaCy entities
- [ ] Predicate normalization reduces redundancy by >70%
- [ ] Neo4j integration maintains performance with triplet ingestion
- [ ] Multi-language support for Spanish, German, and English
- [ ] Comprehensive test coverage >90%
- [ ] Documentation complete and examples provided

## Testing Strategy

### Per-Task Testing
- **Unit Tests**: Individual component functionality
- **Integration Tests**: End-to-end triplet extraction workflow
- **Quality Validation**: Triplet accuracy and relevance
- **Performance Testing**: Processing speed and memory usage

### System-Wide Testing
- Full pipeline testing with various document types
- Multi-language processing validation (Spanish, German, English)
- Neo4j integration and query performance
- Scalability testing with large document sets

## Migration Plan

1. **Foundation Setup**: Implement core OpenIE infrastructure
2. **Entity Integration**: Connect with existing spaCy NER pipeline
3. **Predicate Processing**: Add normalization and categorization
4. **Graph Integration**: Extend Neo4j schema and ingestion
5. **Advanced Features**: Multi-language (Spanish, German, English) and optimization

## Next Steps

1. Start with Task 1.1: OpenIE dependency integration and service wrapper
2. Implement sentence processing before triplet extraction
3. Test each component thoroughly before moving to the next
4. Maintain compatibility with existing graph extraction pipeline

---

**Last Updated**: 2025-07-26
**Status**: Planning Phase - Tasks Defined
