# Task 3: Recursive Multi-hop Graph Resolution

## Overview

Implement intelligent recursive graph traversal that can follow entity relationships across multiple hops, with LLM-guided path selection and relevance scoring for user queries.

## Current Status

- ✅ Basic graph traversal operations exist in `morag-graph/operations/`
- ✅ Path finding algorithms implemented
- ✅ Query entity extraction framework in place
- ❌ LLM-guided path selection missing
- ❌ Recursive traversal engine incomplete
- ❌ Query-relevant entity discovery needs enhancement

## Subtasks

### 3.1 Implement LLM-Guided Path Selection

**File**: `packages/morag-graph/src/morag_graph/traversal/path_selector.py`

**Requirements**:
- Use LLM to evaluate path relevance to user queries
- Score potential paths before full traversal
- Avoid irrelevant or circular paths
- Support different traversal strategies
- Maintain context throughout path selection

**Implementation Steps**:
1. Create LLM-based path evaluation agent
2. Implement path scoring algorithms
3. Add relevance filtering mechanisms
4. Create traversal strategy selection
5. Add comprehensive testing and validation

**Expected Output**:
```python
class LLMPathSelector:
    async def select_paths(self, query: str, starting_entities: List[Entity], 
                          available_paths: List[GraphPath]) -> List[ScoredPath]:
        # Returns paths ranked by relevance to query
```

### 3.2 Build Recursive Graph Traversal Engine

**File**: `packages/morag-graph/src/morag_graph/traversal/recursive_engine.py`

**Requirements**:
- Implement intelligent recursive traversal
- Avoid infinite loops and cycles
- Maintain traversal context and depth limits
- Support breadth-first and depth-first strategies
- Track visited nodes and relationship types

**Implementation Steps**:
1. Create recursive traversal framework
2. Implement cycle detection and prevention
3. Add depth and breadth limiting
4. Create context preservation mechanisms
5. Add performance optimization and monitoring

**Expected Output**:
```python
class RecursiveTraversalEngine:
    async def traverse(self, starting_entities: List[Entity], query_context: QueryContext,
                      max_depth: int = 3) -> TraversalResult:
        # Returns comprehensive traversal results with context
```

### 3.3 Create Query-Relevant Entity Discovery

**File**: `packages/morag-graph/src/morag_graph/discovery/entity_discovery.py`

**Requirements**:
- Identify starting entities from user queries
- Expand entity search based on query intent
- Use semantic similarity for entity matching
- Support multi-entity queries
- Rank entities by relevance to query

**Implementation Steps**:
1. Create query analysis and entity extraction
2. Implement semantic entity matching
3. Add entity expansion algorithms
4. Create relevance ranking system
5. Add query intent classification

**Expected Output**:
```python
class QueryEntityDiscovery:
    async def discover_entities(self, query: str) -> List[RankedEntity]:
        # Returns entities ranked by relevance to query
```

## Acceptance Criteria

### Functional
- [ ] LLM path selector chooses relevant paths for queries
- [ ] Recursive engine traverses graph without infinite loops
- [ ] Entity discovery finds relevant starting points
- [ ] Multi-hop traversal maintains context and relevance
- [ ] System handles complex queries with multiple entities

### Quality
- [ ] Path relevance accuracy > 85%
- [ ] Entity discovery precision > 90%
- [ ] Traversal completeness > 95% for relevant paths
- [ ] Query response time < 10s for typical queries
- [ ] Memory usage scales reasonably with graph size

### Technical
- [ ] Robust cycle detection prevents infinite loops
- [ ] Configurable depth and breadth limits
- [ ] Comprehensive error handling and recovery
- [ ] Performance monitoring and optimization
- [ ] Full test coverage for all components

## Dependencies

### External
- LLM service (Gemini API) for path evaluation
- Graph database (Neo4j) for traversal operations
- Vector similarity for entity matching

### Internal
- `morag-graph.operations.GraphTraversal`
- `morag-graph.query.QueryEntityExtractor`
- `morag-graph.models.Entity` and `morag-graph.models.Relation`

## Testing Strategy

### Unit Tests
- Test path selection with various query types
- Test recursive traversal with different graph structures
- Test entity discovery with complex queries
- Test cycle detection and depth limiting

### Integration Tests
- Test full resolution pipeline with real queries
- Test performance with large knowledge graphs
- Test multi-language query processing
- Test edge cases and error conditions

### Test Data
- Create test knowledge graphs with known structures
- Design queries with known expected paths
- Include challenging cases (ambiguous queries, sparse graphs)

## Implementation Notes

### Path Selection Strategy
- Use LLM to evaluate path relevance before full traversal
- Score paths based on entity relevance and relation strength
- Prefer shorter paths when relevance is equal
- Consider query intent (factual, exploratory, comparative)

### Traversal Optimization
- Implement early termination for irrelevant paths
- Use caching for frequently accessed subgraphs
- Parallel traversal for independent paths
- Memory-efficient data structures for large graphs

### Context Preservation
- Maintain query context throughout traversal
- Track relationship chains and their meanings
- Preserve source document information
- Support context-dependent path evaluation

## Files to Create/Modify

### New Files
- `packages/morag-graph/src/morag_graph/traversal/path_selector.py`
- `packages/morag-graph/src/morag_graph/traversal/recursive_engine.py`
- `packages/morag-graph/src/morag_graph/discovery/entity_discovery.py`
- `packages/morag-graph/src/morag_graph/traversal/__init__.py`
- `packages/morag-graph/tests/traversal/test_path_selector.py`
- `packages/morag-graph/tests/traversal/test_recursive_engine.py`
- `packages/morag-graph/tests/discovery/test_entity_discovery.py`

### Modified Files
- `packages/morag-graph/src/morag_graph/operations/traversal.py`
- `packages/morag-graph/src/morag_graph/query/entity_extractor.py`
- `packages/morag-graph/src/morag_graph/retrieval/coordinator.py`

## Estimated Timeline

- **Week 1**: LLM path selector and entity discovery
- **Week 2**: Recursive traversal engine and integration
- **Total**: 2 weeks for complete implementation and testing

## Success Metrics

### Query Resolution Quality
- Simple queries (1 entity): >95% accuracy
- Complex queries (2+ entities): >85% accuracy
- Multi-hop queries (3+ hops): >80% accuracy
- Response completeness: >90%

### Performance Targets
- Single-hop queries: <2s response time
- Multi-hop queries: <10s response time
- Memory usage: <500MB for 100K entity graphs
- Concurrent queries: Support 10+ simultaneous users

### Traversal Efficiency
- Path pruning: Reduce search space by >70%
- Cycle prevention: 100% effectiveness
- Relevant path discovery: >90% recall
- False positive paths: <10%
