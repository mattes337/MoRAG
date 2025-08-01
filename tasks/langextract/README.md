# LangExtract Big Bang Migration

## Overview

This task folder contains the complete migration plan to replace MoRAG's current entity and relation extraction system with Google's LangExtract library. This is a **big bang replacement** that will:

1. Replace all current extraction components with LangExtract
2. Remove obsolete dependencies and files
3. Maintain compatibility with existing MoRAG interfaces
4. Improve extraction quality and performance

## Migration Strategy

### What Gets Replaced
- **Entity Extraction**: Current PydanticAI-based EntityExtractor → LangExtract
- **Relation Extraction**: Current PydanticAI-based RelationExtractor → LangExtract  
- **OpenIE System**: Complete OpenIE pipeline → LangExtract (answers your question: **YES, we replace OpenIE**)
- **Hybrid Extraction**: Pattern matching + AI hybrid → Pure LangExtract
- **SpaCy Integration**: SpaCy entity extraction → LangExtract
- **Custom Normalization**: Complex normalization logic → LangExtract built-in normalization

### What Gets Preserved
- **Neo4j Storage**: Keep existing graph storage with LangExtract results
- **API Interfaces**: Maintain existing API contracts
- **Configuration System**: Adapt existing config to LangExtract parameters
- **Monitoring/Logging**: Keep existing observability

## Task Breakdown

| Task | File | Status | Description |
|------|------|--------|-------------|
| 1 | `01-install-langextract.md` | ⏳ | Install LangExtract and configure dependencies |
| 2 | `02-create-langextract-wrapper.md` | ⏳ | Create MoRAG-compatible LangExtract wrapper |
| 3 | `03-domain-examples.md` | ⏳ | Create few-shot examples for different domains |
| 4 | `04-replace-entity-extractor.md` | ⏳ | Replace EntityExtractor with LangExtract |
| 5 | `05-replace-relation-extractor.md` | ⏳ | Replace RelationExtractor with LangExtract |
| 6 | `06-remove-openie-system.md` | ⏳ | Remove entire OpenIE pipeline and dependencies |
| 7 | `07-remove-hybrid-extractor.md` | ⏳ | Remove HybridEntityExtractor and pattern matching |
| 8 | `08-remove-spacy-extractor.md` | ⏳ | Remove SpaCy entity extraction |
| 9 | `09-remove-normalization.md` | ⏳ | Remove custom entity normalization system |
| 10 | `10-update-graph-builders.md` | ⏳ | Update GraphBuilder and EnhancedGraphBuilder |
| 11 | `11-update-storage-layer.md` | ⏳ | Update Neo4j storage for LangExtract results |
| 12 | `12-update-configuration.md` | ⏳ | Replace extraction configs with LangExtract configs |
| 13 | `13-update-api-layer.md` | ⏳ | Update API endpoints and request handling |
| 14 | `14-add-visualization.md` | ⏳ | Integrate LangExtract's HTML visualization |
| 15 | `15-cleanup-dependencies.md` | ⏳ | Remove obsolete dependencies from requirements |
| 16 | `16-cleanup-files.md` | ⏳ | Delete all obsolete files and directories |
| 17 | `17-update-tests.md` | ⏳ | Update test suite for LangExtract integration |
| 18 | `18-update-documentation.md` | ⏳ | Update documentation and examples |

## Progress Tracking

### Phase 1: Setup and Core Replacement (Tasks 1-5)
- [ ] LangExtract installation and configuration
- [ ] Core wrapper implementation
- [ ] Domain-specific examples
- [ ] Entity and relation extractor replacement

### Phase 2: System Cleanup (Tasks 6-9)
- [ ] OpenIE system removal
- [ ] Hybrid extractor removal  
- [ ] SpaCy extractor removal
- [ ] Normalization system removal

### Phase 3: Integration Updates (Tasks 10-14)
- [ ] Graph builder updates
- [ ] Storage layer updates
- [ ] Configuration updates
- [ ] API layer updates
- [ ] Visualization integration

### Phase 4: Cleanup and Testing (Tasks 15-18)
- [ ] Dependency cleanup
- [ ] File cleanup
- [ ] Test updates
- [ ] Documentation updates

## Key Benefits Expected

### Performance Improvements
- **Parallel Processing**: 20 concurrent workers vs current 3
- **Optimized Chunking**: Better context windows (1000 chars vs 4000)
- **Multiple Passes**: Improved recall through sequential extraction

### Quality Improvements
- **Source Grounding**: Exact character offsets for all extractions
- **Rich Context**: Attributes and metadata for entities/relations
- **Domain Adaptation**: Few-shot learning for specialized domains
- **Built-in Visualization**: Interactive HTML for quality assessment

### Complexity Reduction
- **Simplified Configuration**: Few-shot examples vs complex prompts
- **Reduced Code**: Remove ~15 extraction-related files
- **Unified Pipeline**: Single extraction system vs multiple approaches
- **Less Maintenance**: Leverage Google's maintained library

## Dependencies to Remove

### Python Packages
- `stanford-openie` (if used)
- `allennlp` (if used for OpenIE)
- `spacy` (entity extraction models)
- Custom pattern matching dependencies

### Internal Modules
- `morag_graph.extractors.openie_extractor`
- `morag_graph.extraction.hybrid_extractor`
- `morag_graph.extraction.spacy_extractor`
- `morag_graph.extraction.pattern_matcher`
- `morag_graph.normalizers.entity_normalizer`
- `morag_graph.services.openie_service`
- All OpenIE-related operations and storage

## Risk Mitigation

### Backup Strategy
- Create feature branch for migration
- Keep current system in separate branch
- Comprehensive testing before merge

### Rollback Plan
- Document all removed components
- Keep dependency versions for quick restoration
- Maintain API compatibility for easy rollback

### Testing Strategy
- Unit tests for all new components
- Integration tests with existing systems
- Performance benchmarks vs current system
- Quality assessment on sample documents

## Success Criteria

- [ ] All extraction functionality working with LangExtract
- [ ] Performance equal or better than current system
- [ ] Quality metrics improved (precision/recall)
- [ ] All obsolete code and dependencies removed
- [ ] Documentation updated
- [ ] Tests passing
- [ ] API compatibility maintained

## Timeline Estimate

- **Phase 1**: 1-2 weeks
- **Phase 2**: 1 week  
- **Phase 3**: 1-2 weeks
- **Phase 4**: 1 week

**Total**: 4-6 weeks for complete migration

## Notes

This is a **destructive migration** - we are completely replacing the extraction system, not adding LangExtract alongside it. The goal is to simplify the codebase while improving quality and performance.

**Answer to your question**: **YES, we replace OpenIE completely**. OpenIE will be removed entirely as LangExtract provides superior relation extraction capabilities with better performance and quality.
