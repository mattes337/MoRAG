# Task 5: Fact Storage and Retrieval Implementation

## Objective
Implement storage mechanisms for facts in Neo4j and retrieval systems that can query facts semantically and follow fact-based relationships.

## Implementation Plan

### 5.1 Enhanced Fact Storage

**File**: `packages/morag-graph/src/morag_graph/storage/fact_storage.py`

```python
class FactStorage:
    """Comprehensive fact storage and management system."""
    
    def __init__(
        self,
        neo4j_driver,
        vector_storage: Optional[VectorStorage] = None,
        enable_vector_indexing: bool = True
    ):
        self.neo4j_driver = neo4j_driver
        self.vector_storage = vector_storage
        self.enable_vector_indexing = enable_vector_indexing
        
    async def store_fact_with_vectors(self, fact: Fact) -> str:
        """Store fact in both Neo4j and vector database."""
        
    async def store_facts_batch(self, facts: List[Fact]) -> BatchResult:
        """Efficiently store multiple facts with relationships."""
        
    async def update_fact(self, fact_id: str, updates: Dict[str, Any]) -> bool:
        """Update existing fact with new information."""
        
    async def delete_fact(self, fact_id: str, cascade: bool = False) -> bool:
        """Delete fact and optionally its relationships."""
        
    async def get_fact_statistics(self) -> FactStatistics:
        """Get statistics about stored facts."""
```

### 5.2 Fact Retrieval System

**File**: `packages/morag-graph/src/morag_graph/retrieval/fact_retrieval.py`

```python
class FactRetrieval:
    """Advanced fact retrieval with multiple query strategies."""
    
    def __init__(
        self,
        fact_storage: FactStorage,
        embedding_model: Optional[EmbeddingModel] = None,
        default_limit: int = 20
    ):
        self.fact_storage = fact_storage
        self.embedding_model = embedding_model
        self.default_limit = default_limit
        
    async def search_facts(self, query: FactQuery) -> FactSearchResult:
        """Search facts using multiple strategies."""
        
    async def get_related_facts(
        self, 
        fact_id: str, 
        relationship_types: Optional[List[str]] = None,
        max_depth: int = 2
    ) -> List[Fact]:
        """Get facts related through graph relationships."""
        
    async def find_supporting_facts(self, fact_id: str) -> List[Fact]:
        """Find facts that support or elaborate on a given fact."""
        
    async def find_contradicting_facts(self, fact_id: str) -> List[Fact]:
        """Find facts that contradict a given fact."""
        
    async def get_fact_chain(self, start_fact_id: str, end_fact_id: str) -> Optional[List[Fact]]:
        """Find chain of facts connecting two facts."""
        
    async def semantic_fact_search(self, query_text: str, limit: int = 10) -> List[Fact]:
        """Search facts using semantic similarity."""
```

### 5.3 Query Models and Types

**File**: `packages/morag-graph/src/morag_graph/models/fact_query.py`

```python
class FactQuery(BaseModel):
    """Comprehensive fact query specification."""
    
    # Text-based search
    text_query: Optional[str] = Field(description="Free text search query")
    semantic_search: bool = Field(default=True, description="Use semantic similarity")
    
    # Structured filters
    fact_type: Optional[str] = Field(description="Filter by fact type")
    domain: Optional[str] = Field(description="Filter by domain")
    subject_contains: Optional[str] = Field(description="Subject must contain text")
    object_contains: Optional[str] = Field(description="Object must contain text")
    
    # Confidence and quality filters
    min_confidence: float = Field(default=0.0, description="Minimum confidence score")
    max_confidence: float = Field(default=1.0, description="Maximum confidence score")
    
    # Source filters
    source_document: Optional[str] = Field(description="Filter by source document")
    source_chunk: Optional[str] = Field(description="Filter by source chunk")
    
    # Temporal filters
    created_after: Optional[datetime] = Field(description="Facts created after date")
    created_before: Optional[datetime] = Field(description="Facts created before date")
    
    # Result configuration
    limit: int = Field(default=20, description="Maximum results to return")
    offset: int = Field(default=0, description="Results offset for pagination")
    sort_by: str = Field(default="confidence", description="Sort field")
    sort_order: str = Field(default="desc", description="Sort order")
    
    # Relationship expansion
    include_related: bool = Field(default=False, description="Include related facts")
    max_relationship_depth: int = Field(default=1, description="Max depth for related facts")

class FactSearchResult(BaseModel):
    """Result of fact search operation."""
    
    facts: List[Fact] = Field(description="Found facts")
    total_count: int = Field(description="Total matching facts")
    query_time_ms: float = Field(description="Query execution time")
    search_strategy: str = Field(description="Strategy used for search")
    related_facts: Optional[Dict[str, List[Fact]]] = Field(description="Related facts by ID")
```

### 5.4 Advanced Query Strategies

**File**: `packages/morag-graph/src/morag_graph/retrieval/fact_query_strategies.py`

```python
class FactQueryStrategy(ABC):
    """Abstract base for fact query strategies."""
    
    @abstractmethod
    async def execute(self, query: FactQuery) -> FactSearchResult:
        """Execute the query strategy."""

class TextSearchStrategy(FactQueryStrategy):
    """Full-text search on fact content."""
    
    async def execute(self, query: FactQuery) -> FactSearchResult:
        """Execute text-based search using Neo4j full-text indexes."""

class SemanticSearchStrategy(FactQueryStrategy):
    """Semantic search using vector embeddings."""
    
    async def execute(self, query: FactQuery) -> FactSearchResult:
        """Execute semantic search using vector similarity."""

class StructuredSearchStrategy(FactQueryStrategy):
    """Structured search using fact schema fields."""
    
    async def execute(self, query: FactQuery) -> FactSearchResult:
        """Execute structured search on fact properties."""

class HybridSearchStrategy(FactQueryStrategy):
    """Combine multiple search strategies."""
    
    def __init__(self, strategies: List[FactQueryStrategy], weights: List[float]):
        self.strategies = strategies
        self.weights = weights
        
    async def execute(self, query: FactQuery) -> FactSearchResult:
        """Execute hybrid search combining multiple strategies."""
```

### 5.5 Neo4j Query Optimization

**File**: `packages/morag-graph/src/morag_graph/storage/neo4j_queries/fact_queries.py`

```python
class FactQueries:
    """Optimized Neo4j queries for fact operations."""
    
    # Basic fact queries
    FIND_FACTS_BY_TEXT = """
    CALL db.index.fulltext.queryNodes('fact_content', $query) 
    YIELD node, score
    WHERE node.confidence >= $min_confidence
    RETURN node, score
    ORDER BY score DESC, node.confidence DESC
    LIMIT $limit
    """
    
    FIND_FACTS_BY_FILTERS = """
    MATCH (f:Fact)
    WHERE ($fact_type IS NULL OR f.fact_type = $fact_type)
      AND ($domain IS NULL OR f.domain = $domain)
      AND f.confidence >= $min_confidence
      AND f.confidence <= $max_confidence
    RETURN f
    ORDER BY f.confidence DESC
    LIMIT $limit
    """
    
    # Relationship queries
    FIND_RELATED_FACTS = """
    MATCH (f:Fact {id: $fact_id})-[r]->(related:Fact)
    WHERE ($relationship_types IS NULL OR type(r) IN $relationship_types)
    RETURN related, type(r) as relationship_type, r.confidence as rel_confidence
    ORDER BY r.confidence DESC
    LIMIT $limit
    """
    
    FIND_FACT_CHAIN = """
    MATCH path = shortestPath((start:Fact {id: $start_id})-[*1..5]-(end:Fact {id: $end_id}))
    WHERE ALL(r IN relationships(path) WHERE r.confidence >= 0.5)
    RETURN [node IN nodes(path) | node] as fact_chain,
           [rel IN relationships(path) | {type: type(rel), confidence: rel.confidence}] as relationships
    """
    
    # Aggregation queries
    FACT_STATISTICS = """
    MATCH (f:Fact)
    RETURN 
        count(f) as total_facts,
        avg(f.confidence) as avg_confidence,
        collect(DISTINCT f.fact_type) as fact_types,
        collect(DISTINCT f.domain) as domains,
        min(f.created_at) as earliest_fact,
        max(f.created_at) as latest_fact
    """
```

## Implementation Tasks

### Task 5.1: Core Storage Implementation
- [ ] Implement FactStorage class with Neo4j integration
- [ ] Add vector storage integration for semantic search
- [ ] Create batch processing for efficient fact storage
- [ ] Implement fact update and deletion operations

### Task 5.2: Query System Development
- [ ] Implement FactQuery model with comprehensive filtering
- [ ] Create multiple query strategies (text, semantic, structured)
- [ ] Add hybrid search combining multiple strategies
- [ ] Implement result ranking and relevance scoring

### Task 5.3: Relationship Traversal
- [ ] Implement related fact discovery
- [ ] Add fact chain finding algorithms
- [ ] Create supporting/contradicting fact detection
- [ ] Implement graph traversal with depth limits

### Task 5.4: Performance Optimization
- [ ] Create Neo4j indexes for fact properties
- [ ] Implement query caching for common searches
- [ ] Add query performance monitoring
- [ ] Optimize batch operations and transactions

### Task 5.5: Advanced Features
- [ ] Implement fact clustering and categorization
- [ ] Add fact quality scoring and ranking
- [ ] Create fact recommendation system
- [ ] Implement fact validation and verification

## Neo4j Index Strategy

```cypher
-- Full-text search indexes
CREATE FULLTEXT INDEX fact_content_index IF NOT EXISTS 
FOR (f:Fact) ON EACH [f.subject, f.object, f.approach, f.solution, f.remarks];

CREATE FULLTEXT INDEX keyword_search_index IF NOT EXISTS 
FOR (k:Keyword) ON EACH [k.name];

-- Property indexes for filtering
CREATE INDEX fact_type_confidence_index IF NOT EXISTS 
FOR (f:Fact) ON (f.fact_type, f.confidence);

CREATE INDEX fact_domain_created_index IF NOT EXISTS 
FOR (f:Fact) ON (f.domain, f.created_at);

-- Composite indexes for common query patterns
CREATE INDEX fact_search_composite IF NOT EXISTS 
FOR (f:Fact) ON (f.fact_type, f.domain, f.confidence);
```

## Caching Strategy

```python
class FactCache:
    """Caching layer for fact retrieval."""
    
    def __init__(self, cache_backend: str = "redis"):
        self.cache = self._initialize_cache(cache_backend)
        
    async def get_cached_facts(self, query_hash: str) -> Optional[FactSearchResult]:
        """Get cached search results."""
        
    async def cache_facts(self, query_hash: str, result: FactSearchResult, ttl: int = 3600):
        """Cache search results with TTL."""
        
    def _generate_query_hash(self, query: FactQuery) -> str:
        """Generate cache key from query parameters."""
```

## Success Criteria

1. **Query Performance**: Sub-second response times for most fact queries
2. **Scalability**: Handle millions of facts with consistent performance
3. **Flexibility**: Support diverse query patterns and use cases
4. **Accuracy**: Relevant results with proper ranking and scoring
5. **Integration**: Seamless integration with existing MoRAG components
