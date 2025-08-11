# Task 4: Fact Gathering and Scoring System

## Overview

Build a comprehensive fact extraction system that gathers relevant facts from graph traversal, scores them for relevance and confidence, and maintains citations and source references.

## Current Status

- ✅ Basic fact extraction exists in `morag-reasoning/fact_extraction.py`
- ✅ Graph traversal provides entity-relation paths
- ✅ Source tracking through DocumentChunk relationships
- ❌ Comprehensive fact gathering from traversal missing
- ❌ Advanced relevance scoring system incomplete
- ❌ Citation and source tracking needs enhancement

## Subtasks

### 4.1 Build Fact Extraction from Graph Traversal

**File**: `packages/morag-reasoning/src/morag_reasoning/graph_fact_extractor.py`

**Requirements**:
- Extract facts from graph traversal results
- Maintain context and relationship chains
- Preserve source document information
- Support different fact types (direct, inferred, contextual)
- Handle multi-hop relationship facts

**Implementation Steps**:
1. Create graph-based fact extraction framework
2. Implement relationship chain analysis
3. Add context preservation mechanisms
4. Create fact type classification
5. Add comprehensive validation and testing

**Expected Output**:
```python
class GraphFactExtractor:
    async def extract_facts(self, traversal_result: TraversalResult, 
                           query_context: QueryContext) -> List[ExtractedFact]:
        # Returns facts with full context and source information
```

### 4.2 Implement Fact Relevance Scoring

**File**: `packages/morag-reasoning/src/morag_reasoning/fact_scorer.py`

**Requirements**:
- Score facts based on relevance to user query
- Consider confidence, source quality, and recency
- Weight direct vs. inferred facts appropriately
- Support domain-specific scoring rules
- Provide explainable scoring rationale

**Implementation Steps**:
1. Create multi-dimensional scoring framework
2. Implement query relevance algorithms
3. Add source quality assessment
4. Create confidence aggregation methods
5. Add explainability and debugging features

**Expected Output**:
```python
class FactRelevanceScorer:
    async def score_facts(self, facts: List[ExtractedFact], 
                         query: str) -> List[ScoredFact]:
        # Returns facts with relevance scores and explanations
```

### 4.3 Create Citation and Source Tracking

**File**: `packages/morag-reasoning/src/morag_reasoning/citation_manager.py`

**Requirements**:
- Track original documents for all facts
- Maintain timestamps and chapter information
- Support different citation formats
- Handle multi-source facts
- Provide citation validation and verification

**Implementation Steps**:
1. Create comprehensive citation tracking system
2. Implement source metadata preservation
3. Add citation format generation
4. Create multi-source fact handling
5. Add validation and verification mechanisms

**Expected Output**:
```python
class CitationManager:
    async def track_citations(self, facts: List[ScoredFact]) -> List[CitedFact]:
        # Returns facts with complete citation information
```

## Acceptance Criteria

### Functional
- [ ] Fact extraction captures all relevant information from traversal
- [ ] Relevance scoring accurately ranks facts by importance
- [ ] Citation tracking maintains complete source information
- [ ] Multi-hop facts preserve relationship context
- [ ] System handles complex queries with multiple fact types

### Quality
- [ ] Fact extraction completeness > 95%
- [ ] Relevance scoring accuracy > 90%
- [ ] Citation accuracy 100%
- [ ] Processing speed < 5s for typical fact sets
- [ ] Memory usage scales linearly with fact count

### Technical
- [ ] Robust handling of incomplete or missing data
- [ ] Configurable scoring weights and parameters
- [ ] Comprehensive logging for debugging
- [ ] Full test coverage for all components
- [ ] Performance monitoring and optimization

## Dependencies

### External
- LLM service (Gemini API) for relevance assessment
- Graph database (Neo4j) for source tracking
- Vector similarity for semantic relevance

### Internal
- `morag-graph.traversal.RecursiveTraversalEngine`
- `morag-graph.models.Entity` and `morag-graph.models.Relation`
- `morag-reasoning.fact_extraction.FactExtractor`

## Testing Strategy

### Unit Tests
- Test fact extraction with various traversal results
- Test scoring algorithms with different fact types
- Test citation tracking with complex source chains
- Test edge cases and error conditions

### Integration Tests
- Test full fact gathering pipeline
- Test performance with large fact sets
- Test multi-language fact processing
- Test citation format generation

### Test Data
- Create test graphs with known fact patterns
- Design queries with expected fact outcomes
- Include challenging cases (conflicting facts, sparse sources)

## Implementation Notes

### Fact Types
- **Direct Facts**: Single entity-relation-entity triplets
- **Chain Facts**: Multi-hop relationship chains
- **Contextual Facts**: Facts requiring document context
- **Inferred Facts**: Facts derived from multiple sources
- **Temporal Facts**: Facts with time-dependent information

### Scoring Dimensions
- **Query Relevance**: Semantic similarity to user query
- **Source Quality**: Document authority and reliability
- **Confidence**: Extraction and validation confidence
- **Recency**: Temporal relevance of information
- **Completeness**: Availability of supporting evidence

### Citation Format
```json
{
  "fact_id": "uuid",
  "content": "Einstein developed the theory of relativity",
  "sources": [
    {
      "document_id": "doc_123",
      "document_title": "Physics History",
      "chunk_id": "chunk_456",
      "page": 42,
      "timestamp": "1:23:45",
      "confidence": 0.95
    }
  ],
  "relevance_score": 0.92,
  "fact_type": "direct"
}
```

## Files to Create/Modify

### New Files
- `packages/morag-reasoning/src/morag_reasoning/graph_fact_extractor.py`
- `packages/morag-reasoning/src/morag_reasoning/fact_scorer.py`
- `packages/morag-reasoning/src/morag_reasoning/citation_manager.py`
- `packages/morag-reasoning/src/morag_reasoning/models/fact_models.py`
- `packages/morag-reasoning/tests/test_graph_fact_extractor.py`
- `packages/morag-reasoning/tests/test_fact_scorer.py`
- `packages/morag-reasoning/tests/test_citation_manager.py`

### Modified Files
- `packages/morag-reasoning/src/morag_reasoning/fact_extraction.py`
- `packages/morag-reasoning/src/morag_reasoning/recursive_fact_retrieval_service.py`
- `packages/morag-reasoning/src/morag_reasoning/__init__.py`

## Estimated Timeline

- **Week 1**: Graph fact extraction and scoring implementation
- **Week 2**: Citation management and pipeline integration
- **Total**: 2 weeks for complete implementation and testing

## Success Metrics

### Fact Quality
- Fact extraction recall: >95%
- Fact extraction precision: >90%
- Relevance ranking accuracy: >90%
- Citation completeness: 100%

### Performance Targets
- Fact processing: <5s for 100 facts
- Scoring computation: <2s for 50 facts
- Citation generation: <1s for 20 sources
- Memory usage: <100MB for typical fact sets

### User Experience
- Fact relevance satisfaction: >90%
- Citation usefulness: >95%
- Response completeness: >90%
- Source trustworthiness: >85%
