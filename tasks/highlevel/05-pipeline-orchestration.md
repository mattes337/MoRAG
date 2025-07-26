# Task 5: Agent Pipeline Orchestration

## Overview

Create a unified agent pipeline that orchestrates the complete MoRAG workflow from document ingestion through final response generation, with proper intermediate file generation and debugging capabilities.

## Current Status

- ✅ Basic pipeline framework exists in `morag-services/pipeline.py`
- ✅ Individual components (ingestion, extraction, storage) implemented
- ✅ Intermediate file generation partially implemented
- ❌ Unified agent pipeline missing
- ❌ State management and recovery incomplete
- ❌ End-to-end orchestration needs implementation

## Subtasks

### 5.1 Create Unified MoRAG Agent Pipeline

**File**: `packages/morag/src/morag/agents/morag_pipeline_agent.py`

**Requirements**:
- Orchestrate complete MoRAG workflow
- Support both ingestion and resolution modes
- Coordinate all processing stages
- Handle different content types seamlessly
- Provide progress tracking and monitoring

**Implementation Steps**:
1. Create main pipeline orchestration agent
2. Implement stage coordination logic
3. Add content type routing
4. Create progress tracking system
5. Add comprehensive error handling

**Expected Output**:
```python
class MoRAGPipelineAgent:
    async def process_ingestion(self, source: str, options: IngestionOptions) -> IngestionResult:
        # Complete ingestion pipeline from source to knowledge graph
    
    async def process_resolution(self, query: str, options: ResolutionOptions) -> ResolutionResult:
        # Complete resolution pipeline from query to final response
```

### 5.2 Implement Intermediate File Generation System

**File**: `packages/morag/src/morag/pipeline/intermediate_manager.py`

**Requirements**:
- Generate intermediate files at each processing stage
- Support debugging and pipeline continuation
- Maintain file versioning and metadata
- Handle different file formats (JSON, Markdown, etc.)
- Provide file cleanup and management

**Implementation Steps**:
1. Create intermediate file management system
2. Implement stage-specific file generation
3. Add versioning and metadata tracking
4. Create file cleanup and retention policies
5. Add debugging and inspection tools

**Expected Output**:
```python
class IntermediateFileManager:
    async def save_stage_output(self, stage: str, data: Any, metadata: Dict) -> Path:
        # Saves intermediate output with proper naming and metadata
    
    async def load_stage_output(self, stage: str, source_id: str) -> Tuple[Any, Dict]:
        # Loads intermediate output for pipeline continuation
```

### 5.3 Build Pipeline State Management

**File**: `packages/morag/src/morag/pipeline/state_manager.py`

**Requirements**:
- Track pipeline execution state
- Support resumption from any stage
- Handle failures and recovery
- Maintain execution history and logs
- Provide state inspection and debugging

**Implementation Steps**:
1. Create pipeline state tracking system
2. Implement checkpoint and recovery mechanisms
3. Add failure handling and retry logic
4. Create state inspection tools
5. Add comprehensive logging and monitoring

**Expected Output**:
```python
class PipelineStateManager:
    async def save_checkpoint(self, pipeline_id: str, stage: str, state: Dict) -> None:
        # Saves pipeline state for recovery
    
    async def resume_pipeline(self, pipeline_id: str) -> PipelineState:
        # Resumes pipeline from last successful checkpoint
```

## Acceptance Criteria

### Functional
- [ ] Pipeline orchestrates complete ingestion workflow
- [ ] Pipeline orchestrates complete resolution workflow
- [ ] Intermediate files generated at each stage
- [ ] State management supports resumption from any point
- [ ] Error handling and recovery work correctly

### Quality
- [ ] Pipeline success rate > 95% for typical inputs
- [ ] Recovery success rate > 90% after failures
- [ ] Processing speed within acceptable limits
- [ ] Memory usage remains reasonable
- [ ] Intermediate files are useful for debugging

### Technical
- [ ] Modular design allows easy component replacement
- [ ] Comprehensive error handling and logging
- [ ] Performance monitoring and metrics
- [ ] Full test coverage for all components
- [ ] Documentation for pipeline configuration

## Dependencies

### External
- All MoRAG component packages
- File system for intermediate storage
- Database connections for state persistence

### Internal
- `morag-services.pipeline.Pipeline`
- `morag-graph.builders.EnhancedGraphBuilder`
- `morag-reasoning.recursive_fact_retrieval_service`

## Testing Strategy

### Unit Tests
- Test individual pipeline stages
- Test state management operations
- Test intermediate file generation
- Test error handling and recovery

### Integration Tests
- Test complete ingestion pipeline
- Test complete resolution pipeline
- Test pipeline resumption scenarios
- Test performance with various content types

### Test Data
- Create test documents of different types
- Design test queries with known expected outcomes
- Include failure scenarios for recovery testing

## Implementation Notes

### Pipeline Stages

#### Ingestion Pipeline
1. **Content Conversion** → Markdown via markitdown
2. **Entity Extraction** → SpaCy NER + LLM extraction
3. **Relation Extraction** → OpenIE + LLM relations
4. **Graph Building** → Neo4j storage with embeddings
5. **Validation** → Quality checks and verification

#### Resolution Pipeline
1. **Query Analysis** → Entity extraction and intent analysis
2. **Graph Traversal** → Multi-hop recursive exploration
3. **Fact Gathering** → Relevant fact extraction and scoring
4. **Response Generation** → LLM synthesis with citations
5. **Quality Assessment** → Response validation and scoring

### Intermediate Files
- `{source}_01_conversion.md` - Converted markdown content
- `{source}_02_entities.json` - Extracted entities with metadata
- `{source}_03_relations.json` - Extracted relations with context
- `{source}_04_graph.json` - Graph structure and embeddings
- `{source}_05_validation.json` - Quality metrics and issues

### State Management
- Pipeline execution tracking
- Stage completion status
- Error and retry information
- Performance metrics
- Resource usage statistics

## Files to Create/Modify

### New Files
- `packages/morag/src/morag/agents/morag_pipeline_agent.py`
- `packages/morag/src/morag/pipeline/intermediate_manager.py`
- `packages/morag/src/morag/pipeline/state_manager.py`
- `packages/morag/src/morag/pipeline/__init__.py`
- `packages/morag/src/morag/agents/__init__.py`
- `packages/morag/tests/agents/test_morag_pipeline_agent.py`
- `packages/morag/tests/pipeline/test_intermediate_manager.py`
- `packages/morag/tests/pipeline/test_state_manager.py`

### Modified Files
- `packages/morag/src/morag/orchestrator.py`
- `packages/morag/src/morag/api.py`
- `packages/morag/src/morag/__init__.py`
- `packages/morag-services/src/morag_services/pipeline.py`

## Estimated Timeline

- **Week 1**: Pipeline agent and intermediate file management
- **Week 2**: State management and integration testing
- **Total**: 2 weeks for complete implementation and testing

## Success Metrics

### Pipeline Reliability
- Ingestion success rate: >95%
- Resolution success rate: >95%
- Recovery success rate: >90%
- Error handling coverage: 100%

### Performance Targets
- Document ingestion: <60s for typical documents
- Query resolution: <30s for typical queries
- State save/load: <2s operations
- Memory usage: <1GB for typical workloads

### User Experience
- Clear progress indication: 100%
- Useful error messages: >95%
- Debugging information: Comprehensive
- Pipeline resumption: Seamless

### Monitoring and Observability
- Stage-level performance metrics
- Error rate tracking by component
- Resource usage monitoring
- Quality metrics for each stage
