# Task 6: System Integration

## Objective
Integrate the new fact-based pipeline with the existing MoRAG system, ensuring compatibility with current document processing and vector storage while providing migration path from entity-based approach.

## Integration Architecture

### 6.1 Pipeline Integration Points

```
Existing Pipeline:
Document → Formatter → Analyzer → Chunker → Entity Extractor → Graph Builder → Storage

New Fact Pipeline:
Document → Formatter → Analyzer → Chunker → Fact Extractor → Fact Graph Builder → Fact Storage
                                                    ↓
                                            Vector Embeddings → Qdrant
```

### 6.2 Hybrid Processing Mode

**File**: `packages/morag-graph/src/morag_graph/pipeline/hybrid_processor.py`

```python
class HybridProcessor:
    """Process documents using both entity and fact extraction."""
    
    def __init__(
        self,
        entity_extractor: Optional[EntityExtractor] = None,
        fact_extractor: Optional[FactExtractor] = None,
        processing_mode: str = "hybrid"  # "entity", "fact", "hybrid"
    ):
        self.entity_extractor = entity_extractor
        self.fact_extractor = fact_extractor
        self.processing_mode = processing_mode
        
    async def process_document(self, document: Document) -> ProcessingResult:
        """Process document using configured extraction methods."""
        
    async def _process_with_entities(self, chunks: List[DocumentChunk]) -> EntityResult:
        """Process using entity extraction."""
        
    async def _process_with_facts(self, chunks: List[DocumentChunk]) -> FactResult:
        """Process using fact extraction."""
        
    async def _process_hybrid(self, chunks: List[DocumentChunk]) -> HybridResult:
        """Process using both entity and fact extraction."""
```

### 6.3 Configuration Management

**File**: `packages/morag-graph/src/morag_graph/config/extraction_config.py`

```python
class ExtractionConfig(BaseModel):
    """Configuration for extraction pipeline."""
    
    # Processing mode
    extraction_mode: str = Field(default="hybrid", description="entity|fact|hybrid")
    
    # Entity extraction settings
    enable_entity_extraction: bool = Field(default=True)
    entity_min_confidence: float = Field(default=0.6)
    entity_max_per_chunk: int = Field(default=20)
    
    # Fact extraction settings
    enable_fact_extraction: bool = Field(default=True)
    fact_min_confidence: float = Field(default=0.7)
    fact_max_per_chunk: int = Field(default=10)
    
    # Hybrid mode settings
    prefer_facts_over_entities: bool = Field(default=True)
    fact_entity_overlap_threshold: float = Field(default=0.8)
    
    # Performance settings
    parallel_processing: bool = Field(default=True)
    max_workers: int = Field(default=10)
    batch_size: int = Field(default=50)
```

### 6.4 API Integration

**File**: `packages/morag-api/src/morag_api/endpoints/fact_endpoints.py`

```python
@router.post("/facts/extract")
async def extract_facts_from_text(
    request: FactExtractionRequest,
    extraction_service: FactExtractionService = Depends()
) -> FactExtractionResponse:
    """Extract facts from provided text."""
    
@router.get("/facts/search")
async def search_facts(
    query: str,
    fact_type: Optional[str] = None,
    domain: Optional[str] = None,
    limit: int = 20,
    fact_service: FactService = Depends()
) -> FactSearchResponse:
    """Search for facts using various criteria."""
    
@router.get("/facts/{fact_id}/related")
async def get_related_facts(
    fact_id: str,
    relationship_types: Optional[List[str]] = Query(None),
    max_depth: int = 2,
    fact_service: FactService = Depends()
) -> RelatedFactsResponse:
    """Get facts related to a specific fact."""
    
@router.get("/facts/{fact_id}/chain/{target_fact_id}")
async def get_fact_chain(
    fact_id: str,
    target_fact_id: str,
    fact_service: FactService = Depends()
) -> FactChainResponse:
    """Find chain of facts connecting two facts."""
```

### 6.5 Migration Utilities

**File**: `packages/morag-graph/src/morag_graph/migration/entity_to_fact_migrator.py`

```python
class EntityToFactMigrator:
    """Migrate existing entity-based graphs to fact-based approach."""
    
    def __init__(
        self,
        entity_storage: EntityStorage,
        fact_storage: FactStorage,
        migration_strategy: str = "parallel"  # "parallel", "replace", "hybrid"
    ):
        self.entity_storage = entity_storage
        self.fact_storage = fact_storage
        self.migration_strategy = migration_strategy
        
    async def migrate_document(self, document_id: str) -> MigrationResult:
        """Migrate a single document from entity to fact representation."""
        
    async def migrate_all_documents(self, batch_size: int = 100) -> MigrationSummary:
        """Migrate all documents in batches."""
        
    async def _convert_entities_to_facts(self, entities: List[Entity], relations: List[Relation]) -> List[Fact]:
        """Convert entity-relation pairs to structured facts."""
        
    async def validate_migration(self, document_id: str) -> ValidationResult:
        """Validate that migration preserved important information."""
```

## Implementation Tasks

### Task 6.1: Pipeline Integration
- [ ] Create hybrid processor that supports both extraction methods
- [ ] Integrate fact extraction into existing document processing pipeline
- [ ] Add configuration management for extraction modes
- [ ] Implement parallel processing for both entity and fact extraction

### Task 6.2: API Extensions
- [ ] Add fact-specific API endpoints
- [ ] Extend existing endpoints to support fact queries
- [ ] Implement fact-based intelligent retrieval
- [ ] Add fact visualization and exploration endpoints

### Task 6.3: Storage Integration
- [ ] Integrate fact storage with existing Qdrant vector storage
- [ ] Create unified ID system for facts and entities
- [ ] Implement cross-reference between facts and existing entities
- [ ] Add fact embeddings to vector database

### Task 6.4: Migration Tools
- [ ] Create entity-to-fact migration utilities
- [ ] Implement validation tools for migration quality
- [ ] Add rollback capabilities for failed migrations
- [ ] Create comparison tools for entity vs fact approaches

### Task 6.5: Configuration and Deployment
- [ ] Add fact extraction configuration to environment variables
- [ ] Update Docker configurations for new dependencies
- [ ] Create deployment scripts for fact-enabled systems
- [ ] Add monitoring and logging for fact extraction pipeline

## Environment Variables

```bash
# Fact extraction configuration
MORAG_EXTRACTION_MODE=hybrid  # entity|fact|hybrid
MORAG_ENABLE_FACT_EXTRACTION=true
MORAG_FACT_MIN_CONFIDENCE=0.7
MORAG_FACT_MAX_PER_CHUNK=10

# Fact storage configuration
MORAG_FACT_STORAGE_ENABLED=true
MORAG_FACT_VECTOR_INDEXING=true

# Migration settings
MORAG_MIGRATION_MODE=parallel  # parallel|replace|hybrid
MORAG_MIGRATION_BATCH_SIZE=100
```

## API Request/Response Models

```python
class FactExtractionRequest(BaseModel):
    text: str = Field(description="Text to extract facts from")
    domain: Optional[str] = Field(description="Domain context")
    language: str = Field(default="en", description="Text language")
    max_facts: int = Field(default=10, description="Maximum facts to extract")

class FactExtractionResponse(BaseModel):
    facts: List[Fact] = Field(description="Extracted facts")
    extraction_time_ms: float = Field(description="Processing time")
    confidence_stats: Dict[str, float] = Field(description="Confidence statistics")

class FactSearchResponse(BaseModel):
    facts: List[Fact] = Field(description="Found facts")
    total_count: int = Field(description="Total matching facts")
    query_time_ms: float = Field(description="Query execution time")
    facets: Dict[str, List[str]] = Field(description="Available facets for filtering")

class RelatedFactsResponse(BaseModel):
    source_fact: Fact = Field(description="Source fact")
    related_facts: List[RelatedFact] = Field(description="Related facts with relationships")
    relationship_summary: Dict[str, int] = Field(description="Count by relationship type")

class RelatedFact(BaseModel):
    fact: Fact = Field(description="Related fact")
    relationship_type: str = Field(description="Type of relationship")
    confidence: float = Field(description="Relationship confidence")
    path_length: int = Field(description="Distance in graph")
```

## Testing Strategy

### Integration Tests
```python
class TestFactIntegration:
    """Test fact extraction integration with existing system."""
    
    async def test_hybrid_processing(self):
        """Test processing document with both entity and fact extraction."""
        
    async def test_api_endpoints(self):
        """Test new fact-based API endpoints."""
        
    async def test_migration_process(self):
        """Test migration from entity to fact representation."""
        
    async def test_performance_comparison(self):
        """Compare performance of entity vs fact approaches."""
```

### Performance Benchmarks
- Document processing time comparison
- Graph size and complexity metrics
- Query performance for different retrieval patterns
- Memory usage and resource consumption

## Rollout Strategy

### Phase 1: Parallel Implementation
- Deploy fact extraction alongside existing entity extraction
- Run both systems in parallel for comparison
- Collect performance and quality metrics

### Phase 2: Gradual Migration
- Start using fact-based retrieval for new documents
- Migrate high-value documents to fact representation
- Maintain backward compatibility with entity-based queries

### Phase 3: Full Transition
- Default to fact-based extraction for all new documents
- Complete migration of existing documents
- Deprecate entity-based extraction (optional)

## Success Criteria

1. **Seamless Integration**: Fact extraction works alongside existing pipeline
2. **Performance**: No degradation in document processing speed
3. **Quality**: Improved retrieval quality with fact-based approach
4. **Compatibility**: Existing API contracts remain functional
5. **Migration**: Smooth transition from entity to fact representation
